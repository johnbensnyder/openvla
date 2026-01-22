"""
finetune_libero_demo.py

Demo script for LoRA fine-tuning OpenVLA on LIBERO with periodic validation rollouts
and TensorBoard video logging. Replicates Appendix E of the OpenVLA paper.

Usage:
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_libero_demo.py \
        --data_root_dir ./datasets/modified_libero_rlds \
        --task_suites libero_spatial \
        --val_frequency 500 \
        --run_root_dir ./runs
"""

import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

# Add experiments to path for libero imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.training.trackers import TensorBoardTracker
from experiments.robot.libero.libero_eval_utils import run_libero_rollouts

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Map task suite to dataset name
SUITE_TO_DATASET = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


@dataclass
class FinetuneConfig:
    # Model
    vla_path: str = "openvla/openvla-7b"
    
    # Data
    data_root_dir: Path = Path("datasets/modified_libero_rlds")
    task_suites: str = "libero_spatial"  # Comma-separated list: libero_spatial,libero_object,libero_goal,libero_10
    
    # Output
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")
    
    # Training (from paper Appendix E)
    batch_size: int = 4
    max_steps: int = 50000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100000
    lora_rank: int = 32
    lora_dropout: float = 0.0
    
    # Validation
    val_frequency: int = 1000
    val_episodes: int = 10
    val_loss_batches: int = 50
    num_videos: int = 5
    center_crop: bool = True
    
    # Training rollouts (0 = disabled)
    train_rollout_frequency: int = 0
    train_rollout_episodes: int = 5
    train_rollout_videos: int = 2
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "openvla-libero"
    wandb_entity: Optional[str] = None


def log_videos(tracker: TensorBoardTracker, videos_dict: dict, step: int, prefix: str = "val"):
    """Log rollout videos to TensorBoard."""
    video_idx = 0
    for task_name, task_videos in videos_dict.items():
        for frames, success in task_videos:
            tag = f"{prefix}_videos/{video_idx}_{task_name[:30]}_{'success' if success else 'fail'}"
            tracker.write_video(tag, frames, step, fps=30)
            video_idx += 1


def compute_val_loss(model, val_dataloader, device_id, action_tokenizer, get_model_fn, num_batches: int = 50):
    """Compute validation loss over a fixed number of batches."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            if count >= num_batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
            total_loss += output.loss.item()
            
            # Compute accuracy
            action_logits = output.logits[:, get_model_fn().vision_backbone.featurizer.patch_embed.num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            acc = ((action_preds == action_gt) & mask).sum().float() / mask.sum().float()
            total_acc += acc.item()
            count += 1
    
    model.train()
    return total_loss / max(count, 1), total_acc / max(count, 1)


def run_validation(model, processor, cfg, task_suites, tracker, step, device_id, dataset_stats):
    """Run validation rollouts and log results."""
    print(f"\n[Step {step}] Running validation rollouts...")
    
    # Unwrap DDP/PEFT model for inference
    eval_model = model.module if hasattr(model, 'module') else model
    if hasattr(eval_model, 'get_base_model'):
        eval_model = eval_model.get_base_model()
    
    # Inject dataset statistics for action un-normalization
    eval_model.norm_stats = dataset_stats
    
    all_success_rates = {}
    all_videos = {}
    
    for suite in task_suites:
        unnorm_key = SUITE_TO_DATASET[suite]
        success_rate, videos = run_libero_rollouts(
            eval_model, processor, suite, unnorm_key,
            num_episodes=cfg.val_episodes,
            num_videos=cfg.num_videos,
            center_crop=cfg.center_crop,
        )
        all_success_rates[suite] = success_rate
        all_videos.update(videos)
        print(f"  {suite}: {success_rate:.1%} success rate")
    
    # Log metrics
    metrics = {f"val/success_rate/{suite}": rate for suite, rate in all_success_rates.items()}
    metrics["val/success_rate/mean"] = sum(all_success_rates.values()) / len(all_success_rates)
    tracker.write(step, metrics)
    
    # Log videos
    log_videos(tracker, all_videos, step, prefix="val")
    tracker.flush()
    
    # Set model back to training mode
    model.train()
    return metrics["val/success_rate/mean"]


def run_train_rollouts(model, processor, cfg, task_suites, tracker, step, device_id, dataset_stats):
    """Run training rollouts (lighter weight than validation)."""
    print(f"\n[Step {step}] Running training rollouts...")
    
    eval_model = model.module if hasattr(model, 'module') else model
    if hasattr(eval_model, 'get_base_model'):
        eval_model = eval_model.get_base_model()
    
    # Inject dataset statistics
    eval_model.norm_stats = dataset_stats
    
    # Just run on first task suite for training rollouts
    suite = task_suites[0]
    unnorm_key = SUITE_TO_DATASET[suite]
    
    success_rate, videos = run_libero_rollouts(
        eval_model, processor, suite, unnorm_key,
        num_episodes=cfg.train_rollout_episodes,
        num_videos=cfg.train_rollout_videos,
        center_crop=cfg.center_crop,
    )
    
    tracker.write(step, {f"train_rollout/success_rate/{suite}": success_rate})
    log_videos(tracker, videos, step, prefix="train_rollout")
    tracker.flush()
    
    model.train()


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # Parse task suites from comma-separated string
    task_suites = [s.strip() for s in cfg.task_suites.split(",")]
    
    # Validate task suites
    for suite in task_suites:
        assert suite in SUITE_TO_DATASET, f"Unknown task suite: {suite}"
    
    print(f"Fine-tuning OpenVLA on LIBERO: {task_suites}")
    
    # Setup distributed
    assert torch.cuda.is_available(), "CUDA required!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()
    
    # Check if running in distributed mode
    use_ddp = distributed_state.num_processes > 1
    
    # Experiment ID
    suites_str = "+".join(task_suites)
    exp_id = f"libero-{suites_str}+lora-r{cfg.lora_rank}+lr-{cfg.learning_rate}"
    if cfg.image_aug:
        exp_id += "+aug"
    
    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)
    
    # Register OpenVLA
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    # Load model
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device_id)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()
    
    # Wrap in DDP only if distributed
    if use_ddp:
        vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    
    # Optimizer
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    
    # Action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # Helper to get underlying model (handles DDP wrapping)
    def get_model():
        return vla.module if use_ddp else vla
    
    # Load dataset (use first task suite for training)
    dataset_name = SUITE_TO_DATASET[task_suites[0]]
    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir, dataset_name, batch_transform,
        resize_resolution=tuple(get_model().config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    # Save dataset statistics
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
    
    # DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(vla_dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)
    
    # Validation dataloader (no augmentation, smaller shuffle buffer)
    val_batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    val_dataset = RLDSDataset(
        cfg.data_root_dir, dataset_name, val_batch_transform,
        resize_resolution=tuple(get_model().config.image_sizes),
        shuffle_buffer_size=1000,
        image_aug=False,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)
    
    # Initialize trackers
    hparams = draccus.encode(cfg)
    tracker = TensorBoardTracker(exp_id, run_dir, hparams)
    tracker.write_hyperparameters()
    
    wandb_tracker = None
    if cfg.use_wandb and distributed_state.is_main_process:
        import wandb
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=exp_id, config=hparams)
    
    # Training metrics
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Train
    print(f"\nStarting training for {cfg.max_steps} steps...")
    with tqdm.tqdm(total=cfg.max_steps, disable=not distributed_state.is_main_process) as pbar:
        vla.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss
            
            # Backward
            (loss / cfg.grad_accumulation_steps).backward()
            
            # Compute accuracy
            action_logits = output.logits[:, get_model().vision_backbone.featurizer.patch_embed.num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            accuracy = ((action_preds == action_gt) & mask).sum().float() / mask.sum().float()
            
            recent_losses.append(loss.item())
            recent_accuracies.append(accuracy.item())
            
            step = batch_idx // cfg.grad_accumulation_steps
            
            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # Compute gradient norm before optimizer step
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float('inf'))
                
                optimizer.step()
                optimizer.zero_grad()
                pbar.update()
                
                # Log metrics
                if step % 10 == 0 and distributed_state.is_main_process:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_acc = sum(recent_accuracies) / len(recent_accuracies)
                    tracker.write(step, {
                        "train/loss": avg_loss,
                        "train/action_accuracy": avg_acc,
                        "train/grad_norm": grad_norm.item(),
                    })
                    if cfg.use_wandb:
                        import wandb
                        wandb.log({"train/loss": avg_loss, "train/action_accuracy": avg_acc, "train/grad_norm": grad_norm.item()}, step=step)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2%}")
            
            # Training rollouts
            if cfg.train_rollout_frequency > 0 and step > 0 and step % cfg.train_rollout_frequency == 0:
                if distributed_state.is_main_process:
                    run_train_rollouts(vla, processor, cfg, task_suites, tracker, step, device_id, vla_dataset.dataset_statistics)
            
            # Validation
            if step > 0 and step % cfg.val_frequency == 0:
                if distributed_state.is_main_process:
                    # Compute validation loss
                    val_loss, val_acc = compute_val_loss(
                        vla, val_dataloader, device_id, action_tokenizer, get_model, cfg.val_loss_batches
                    )
                    tracker.write(step, {
                        "val/loss": val_loss,
                        "val/action_accuracy": val_acc,
                    })
                    if cfg.use_wandb:
                        import wandb
                        wandb.log({"val/loss": val_loss, "val/action_accuracy": val_acc}, step=step)
                    print(f"\n[Step {step}] Val loss: {val_loss:.4f}, Val acc: {val_acc:.2%}")
                    
                    # Run validation rollouts
                    run_validation(vla, processor, cfg, task_suites, tracker, step, device_id, vla_dataset.dataset_statistics)
                if use_ddp:
                    dist.barrier()
            
            # Save checkpoint
            if step > 0 and step % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"\n[Step {step}] Saving checkpoint...")
                    processor.save_pretrained(run_dir)
                    get_model().save_pretrained(adapter_dir)
                
                if use_ddp:
                    dist.barrier()
                
                # Merge LoRA weights
                base_vla = AutoModelForVision2Seq.from_pretrained(
                    cfg.vla_path, torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True, trust_remote_code=True,
                )
                merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                merged_vla = merged_vla.merge_and_unload()
                if distributed_state.is_main_process:
                    merged_vla.save_pretrained(run_dir)
                    print(f"Saved checkpoint to {run_dir}")
                
                if use_ddp:
                    dist.barrier()
            
            # Check termination
            if step >= cfg.max_steps:
                break
    
    # Final validation
    if distributed_state.is_main_process:
        print("\nRunning final validation...")
        val_loss, val_acc = compute_val_loss(
            vla, val_dataloader, device_id, action_tokenizer, get_model, cfg.val_loss_batches
        )
        tracker.write(cfg.max_steps, {
            "val/loss": val_loss,
            "val/action_accuracy": val_acc,
        })
        if cfg.use_wandb:
            import wandb
            wandb.log({"val/loss": val_loss, "val/action_accuracy": val_acc}, step=cfg.max_steps)
        print(f"Final val loss: {val_loss:.4f}, Val acc: {val_acc:.2%}")
        run_validation(vla, processor, cfg, task_suites, tracker, cfg.max_steps, device_id, vla_dataset.dataset_statistics)
    
    tracker.finalize()
    if cfg.use_wandb and distributed_state.is_main_process:
        import wandb
        wandb.finish()
    
    print(f"\nTraining complete! Logs at: {run_dir}/tensorboard")
    print(f"View with: tensorboard --logdir {run_dir}/tensorboard")


if __name__ == "__main__":
    finetune()
