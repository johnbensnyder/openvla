"""
finetune_libero_trainium.py

LoRA fine-tuning OpenVLA on LIBERO using AWS Trainium with NeuronX Distributed.
Uses tensor parallelism for the 7B model and runs validation on CPU.

Usage:
    torchrun --nproc_per_node=8 vla-scripts/trainium/finetune_libero_trainium.py \
        --data_root_dir ./datasets/modified_libero_rlds \
        --task_suites libero_spatial \
        --run_root_dir ./runs_trainium

    # With YAML config
    torchrun --nproc_per_node=8 vla-scripts/trainium/finetune_libero_trainium.py \
        --config vla-scripts/trainium/finetune_trainium_config.yaml
"""

import os
import sys
import gc
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import tqdm

# XLA imports for Trainium
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend

import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import parallel_state

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # For openvla_nxd

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.training.trackers import TensorBoardTracker
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Task suite to dataset mapping
SUITE_TO_DATASET = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


@dataclass
class TrainiumFinetuneConfig:
    # Config file (optional)
    config: Optional[Path] = None
    
    # Model
    vla_path: str = "openvla/openvla-7b"
    
    # Data
    data_root_dir: Path = Path("datasets/modified_libero_rlds")
    task_suites: str = "libero_spatial"
    
    # Output
    run_root_dir: Path = Path("runs_trainium")
    
    # Trainium-specific
    tensor_parallel_size: int = 8
    neuron_compile_cache_dir: Optional[str] = None
    
    # Training
    batch_size: int = 4
    max_steps: int = 50000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100000
    
    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    
    # Validation
    val_frequency: int = 1000
    val_episodes: int = 10
    center_crop: bool = True
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "openvla-libero-trainium"
    wandb_entity: Optional[str] = None


def run_cpu_validation(
    checkpoint_dir: Path,
    processor,
    task_suites: list,
    cfg: TrainiumFinetuneConfig,
    step: int,
    dataset_stats: dict,
):
    """Run validation rollouts on CPU using saved checkpoint."""
    from experiments.robot.libero.libero_eval_utils import run_libero_rollouts
    from transformers import AutoModelForVision2Seq
    from peft import PeftModel
    
    print(f"\n[Step {step}] Running CPU validation rollouts...")
    
    # Load base model on CPU
    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter if checkpoint exists
    adapter_path = checkpoint_dir / "lora_adapter"
    if adapter_path.exists():
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model = model.merge_and_unload()
    else:
        model = base_model
    
    model.eval()
    model.norm_stats = dataset_stats
    
    all_success_rates = {}
    for suite in task_suites:
        unnorm_key = SUITE_TO_DATASET[suite]
        success_rate, _ = run_libero_rollouts(
            model, processor, suite, unnorm_key,
            num_episodes=cfg.val_episodes,
            num_videos=0,  # Skip video recording for speed
            center_crop=cfg.center_crop,
        )
        all_success_rates[suite] = success_rate
        print(f"  {suite}: {success_rate:.1%}")
    
    # Cleanup
    del model, base_model
    gc.collect()
    
    return all_success_rates


def save_checkpoint(model, optimizer, run_dir: Path, step: int):
    """Save NxD checkpoint with LoRA weights."""
    checkpoint_dir = run_dir / f"checkpoint-{step}"
    
    # Save using NxD checkpoint utilities
    nxd.save_checkpoint(
        checkpoint_dir_str=str(checkpoint_dir),
        tag="model",
        model=model,
        optimizer=optimizer,
    )
    
    if xm.is_master_ordinal():
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    return checkpoint_dir


@draccus.wrap()
def finetune(cfg: TrainiumFinetuneConfig) -> None:
    # Set Neuron compile cache if specified
    if cfg.neuron_compile_cache_dir:
        os.environ["NEURON_COMPILE_CACHE_URL"] = cfg.neuron_compile_cache_dir
    
    # Parse task suites
    task_suites = [s.strip() for s in cfg.task_suites.split(",")]
    for suite in task_suites:
        assert suite in SUITE_TO_DATASET, f"Unknown task suite: {suite}"
    
    # Initialize XLA distributed
    torch.distributed.init_process_group("xla")
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    device = xm.xla_device()
    
    if xm.is_master_ordinal():
        print(f"Fine-tuning OpenVLA on Trainium: {task_suites}")
        print(f"World size: {world_size}, TP size: {cfg.tensor_parallel_size}")
    
    # Initialize NxD parallel state
    nxd.parallel_layers.initialize_model_parallel(
        tensor_model_parallel_size=cfg.tensor_parallel_size,
    )
    
    # Experiment ID
    suites_str = "+".join(task_suites)
    exp_id = f"trainium-libero-{suites_str}+lora-r{cfg.lora_rank}+tp{cfg.tensor_parallel_size}"
    if cfg.image_aug:
        exp_id += "+aug"
    
    run_dir = cfg.run_root_dir / exp_id
    if xm.is_master_ordinal():
        os.makedirs(run_dir, exist_ok=True)
    xm.rendezvous("mkdir")
    
    # Register OpenVLA config
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    
    # Load model with NxD tensor parallelism and LoRA
    from openvla_nxd import load_openvla_for_trainium
    
    model = load_openvla_for_trainium(
        vla_path=cfg.vla_path,
        tensor_parallel_size=cfg.tensor_parallel_size,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )
    model = model.to(device)
    
    if xm.is_master_ordinal():
        model.print_trainable_parameters()
    
    # Optimizer - only trainable (LoRA) params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    
    # Action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # Load dataset
    dataset_name = SUITE_TO_DATASET[task_suites[0]]
    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    
    vla_dataset = RLDSDataset(
        cfg.data_root_dir, dataset_name, batch_transform,
        resize_resolution=tuple(model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    # Save dataset statistics
    if xm.is_master_ordinal():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
    
    # DataLoader with XLA parallel loader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    
    dataloader = torch.utils.data.DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=0,
    )
    
    # Wrap with XLA parallel loader for efficient data transfer
    para_loader = pl.ParallelLoader(dataloader, [device])
    device_loader = para_loader.per_device_loader(device)
    
    # Initialize tracker
    if xm.is_master_ordinal():
        hparams = draccus.encode(cfg)
        tracker = TensorBoardTracker(exp_id, run_dir, hparams)
        tracker.write_hyperparameters()
        
        if cfg.use_wandb:
            import wandb
            wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=exp_id, config=hparams)
    
    # Training metrics
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Training loop
    if xm.is_master_ordinal():
        print(f"\nStarting training for {cfg.max_steps} steps...")
    
    model.train()
    optimizer.zero_grad()
    
    pbar = tqdm.tqdm(total=cfg.max_steps, disable=not xm.is_master_ordinal())
    
    for batch_idx, batch in enumerate(device_loader):
        # Forward pass
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"].to(torch.bfloat16),
            labels=batch["labels"],
        )
        loss = output.loss
        
        # Backward
        scaled_loss = loss / cfg.grad_accumulation_steps
        scaled_loss.backward()
        
        # Compute accuracy (on CPU to avoid XLA graph issues)
        with torch.no_grad():
            num_patches = model.vision_backbone.featurizer.patch_embed.num_patches
            action_logits = output.logits[:, num_patches:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            correct = ((action_preds == action_gt) & mask).sum()
            total = mask.sum()
            accuracy = (correct.float() / total.float()).item()
        
        recent_losses.append(loss.item())
        recent_accuracies.append(accuracy)
        
        step = batch_idx // cfg.grad_accumulation_steps
        
        # Optimizer step
        if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
            # XLA mark_step to execute accumulated graph
            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            xm.mark_step()
            
            pbar.update()
            
            # Log metrics
            if step % 10 == 0 and xm.is_master_ordinal():
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_acc = sum(recent_accuracies) / len(recent_accuracies)
                tracker.write(step, {
                    "train/loss": avg_loss,
                    "train/action_accuracy": avg_acc,
                })
                if cfg.use_wandb:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/action_accuracy": avg_acc}, step=step)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2%}")
        
        # Validation
        if step > 0 and step % cfg.val_frequency == 0 and (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
            xm.rendezvous("pre_validation")
            
            if xm.is_master_ordinal():
                # Save temporary checkpoint for CPU validation
                temp_ckpt = save_checkpoint(model, optimizer, run_dir, step)
                
                # Run validation on CPU
                success_rates = run_cpu_validation(
                    temp_ckpt, processor, task_suites, cfg, step, vla_dataset.dataset_statistics
                )
                
                # Log validation metrics
                metrics = {f"val/success_rate/{suite}": rate for suite, rate in success_rates.items()}
                metrics["val/success_rate/mean"] = sum(success_rates.values()) / len(success_rates)
                tracker.write(step, metrics)
                
                if cfg.use_wandb:
                    import wandb
                    wandb.log(metrics, step=step)
            
            xm.rendezvous("post_validation")
            model.train()
        
        # Save checkpoint
        if step > 0 and step % cfg.save_steps == 0 and (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
            xm.rendezvous("pre_checkpoint")
            save_checkpoint(model, optimizer, run_dir, step)
            xm.rendezvous("post_checkpoint")
        
        # Check termination
        if step >= cfg.max_steps:
            break
    
    pbar.close()
    
    # Final checkpoint
    xm.rendezvous("final_checkpoint")
    save_checkpoint(model, optimizer, run_dir, cfg.max_steps)
    
    # Final validation
    if xm.is_master_ordinal():
        print("\nRunning final validation...")
        final_ckpt = run_dir / f"checkpoint-{cfg.max_steps}"
        success_rates = run_cpu_validation(
            final_ckpt, processor, task_suites, cfg, cfg.max_steps, vla_dataset.dataset_statistics
        )
        
        tracker.write(cfg.max_steps, {
            f"val/success_rate/{suite}": rate for suite, rate in success_rates.items()
        })
        tracker.finalize()
        
        if cfg.use_wandb:
            import wandb
            wandb.finish()
        
        print(f"\nTraining complete! Logs at: {run_dir}/tensorboard")


if __name__ == "__main__":
    finetune()
