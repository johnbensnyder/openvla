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
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, get_cosine_schedule_with_warmup
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
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)
from experiments.robot.openvla_utils import get_vla_action
from libero.libero import benchmark

# Max steps per task suite
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Map task suite to dataset name
SUITE_TO_DATASET = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


# === SageMaker Environment Detection ===

def is_sagemaker_training() -> bool:
    """Check if running inside a SageMaker training job."""
    return "SM_MODEL_DIR" in os.environ


def get_sagemaker_paths() -> dict:
    """Get SageMaker paths when running in a training job."""
    return {
        "model_dir": os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        "output_dir": os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        "training_data": os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        "s3_output": os.environ.get("SM_OUTPUT_DATA_DIR", ""),
    }


@dataclass
class FinetuneConfig:
    # Model
    vla_path: str = "openvla/openvla-7b"
    
    # Data
    data_root_dir: str = "datasets/modified_libero_rlds"
    task_suites: str = "libero_spatial"  # Comma-separated list: libero_spatial,libero_object,libero_goal,libero_10
    
    # Output
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")
    
    # Training (from paper Appendix E)
    batch_size: int = 4
    max_steps: int = 50000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
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
    
    # Distributed rollouts (lightweight, runs on all GPUs in parallel)
    dist_rollout_frequency: int = 500
    
    # Training rollouts (0 = disabled)
    train_rollout_frequency: int = 0
    train_rollout_episodes: int = 5
    train_rollout_videos: int = 2
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "openvla-libero"
    wandb_entity: Optional[str] = None
    
    # SageMaker S3 sync
    s3_sync_frequency: int = 500  # Sync TensorBoard to S3 every N steps (0 = disabled)
    s3_output_path: Optional[str] = None  # S3 path for outputs (auto-detected in SageMaker)


def sync_tensorboard_to_s3(local_dir: Path, s3_path: str, step: int):
    """Sync TensorBoard logs directory to S3."""
    import boto3
    import glob
    
    s3 = boto3.client("s3")
    # Parse s3://bucket/prefix
    s3_path = s3_path.rstrip("/")
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    bucket, *prefix_parts = s3_path.split("/")
    prefix = "/".join(prefix_parts) if prefix_parts else ""
    
    tb_dir = local_dir / "tensorboard"
    if not tb_dir.exists():
        return
    
    for filepath in glob.glob(str(tb_dir / "**/*"), recursive=True):
        if os.path.isfile(filepath):
            rel_path = os.path.relpath(filepath, local_dir)
            s3_key = f"{prefix}/{rel_path}" if prefix else rel_path
            s3.upload_file(filepath, bucket, s3_key)
    print(f"[Step {step}] Synced TensorBoard to s3://{bucket}/{prefix}")


def log_videos(tracker: TensorBoardTracker, videos_dict: dict, step: int, prefix: str = "val"):
    """Log rollout videos to TensorBoard."""
    video_idx = 0
    for task_name, task_videos in videos_dict.items():
        for frames, success in task_videos:
            tag = f"{prefix}_videos/{video_idx}_{task_name[:30]}_{'success' if success else 'fail'}"
            tracker.write_video(tag, frames, step, fps=30)
            video_idx += 1


def run_distributed_rollout(model, processor, cfg, task_suites, step, device_id, dataset_stats, rank, world_size):
    """Run a single rollout on each GPU with a random task. Returns (frames, success, task_desc, rank)."""
    from PIL import Image
    
    eval_model = model.module if hasattr(model, 'module') else model
    if hasattr(eval_model, 'get_base_model'):
        eval_model = eval_model.get_base_model()
    eval_model.norm_stats = dataset_stats
    eval_model.eval()
    
    # Use numpy random with unique seed per GPU
    rng = np.random.RandomState(step * 1000 + rank * 123 + 42)
    suite = task_suites[rng.randint(len(task_suites))]
    unnorm_key = SUITE_TO_DATASET[suite]
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    task_id = rng.randint(task_suite.n_tasks)
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    max_steps = MAX_STEPS.get(suite, 300)
    
    env.reset()
    obs = env.set_init_state(initial_states[0])
    
    frames = []
    done = False
    num_steps_wait = 10
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            continue
        
        img = get_libero_image(obs, 224)
        frames.append(get_libero_image(obs, 256))
        
        image = Image.fromarray(img).convert("RGB")
        inputs = processor(prompt, image).to(device_id, dtype=torch.bfloat16)
        
        with torch.no_grad():
            action = eval_model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        
        action[-1] = -np.sign(2 * action[-1] - 1)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
    
    env.close()
    return np.stack(frames) if frames else None, done, task_description, rank


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
    
    # Override paths if running in SageMaker
    if is_sagemaker_training():
        sm_paths = get_sagemaker_paths()
        print(f"Running in SageMaker environment")
        print(f"  Training data: {sm_paths['training_data']}")
        print(f"  Model output: {sm_paths['model_dir']}")
        print(f"  Artifacts: {sm_paths['output_dir']}")
        cfg.data_root_dir = sm_paths["training_data"]
        cfg.run_root_dir = Path(sm_paths["output_dir"])
        cfg.adapter_tmp_dir = Path(sm_paths["output_dir"]) / "adapter-tmp"
    
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
    
    # For SageMaker, also save final model to SM_MODEL_DIR
    if is_sagemaker_training():
        sm_model_dir = Path(get_sagemaker_paths()["model_dir"])
        os.makedirs(sm_model_dir, exist_ok=True)
    
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
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )
    
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
                # Clip gradients and compute norm
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.update()
                
                # Log metrics
                if step % 10 == 0 and distributed_state.is_main_process:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_acc = sum(recent_accuracies) / len(recent_accuracies)
                    current_lr = scheduler.get_last_lr()[0]
                    tracker.write(step, {
                        "train/loss": avg_loss,
                        "train/action_accuracy": avg_acc,
                        "train/grad_norm": grad_norm.item(),
                        "train/learning_rate": current_lr,
                    })
                    if cfg.use_wandb:
                        import wandb
                        wandb.log({"train/loss": avg_loss, "train/action_accuracy": avg_acc, "train/grad_norm": grad_norm.item(), "train/learning_rate": current_lr}, step=step)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2%}")
                
                # S3 sync for TensorBoard (SageMaker only)
                if (is_sagemaker_training() and cfg.s3_sync_frequency > 0 and 
                    step > 0 and step % cfg.s3_sync_frequency == 0 and distributed_state.is_main_process):
                    s3_path = cfg.s3_output_path or os.environ.get("SM_OUTPUT_DATA_DIR", "")
                    if s3_path:
                        tracker.flush()
                        sync_tensorboard_to_s3(run_dir, s3_path, step)
            
                # Distributed rollouts (all GPUs run in parallel)
                if cfg.dist_rollout_frequency > 0 and step > 0 and step % cfg.dist_rollout_frequency == 0:
                    frames, success, task_desc, rank = run_distributed_rollout(
                        vla, processor, cfg, task_suites, step, device_id,
                        vla_dataset.dataset_statistics, distributed_state.local_process_index,
                        distributed_state.num_processes
                    )
                    # Each GPU saves its own video to a temp file, main process logs them
                    if frames is not None:
                        video_path = run_dir / f"tmp_video_gpu{rank}_step{step}.npy"
                        np.save(video_path, {"frames": frames, "success": success, "task": task_desc, "rank": rank})
                    if use_ddp:
                        dist.barrier()
                    # Main process collects and logs all videos
                    if distributed_state.is_main_process:
                        import imageio
                        from PIL import Image, ImageDraw, ImageFont
                        video_dir = run_dir / "videos"
                        video_dir.mkdir(exist_ok=True)
                        for r in range(distributed_state.num_processes):
                            vpath = run_dir / f"tmp_video_gpu{r}_step{step}.npy"
                            if vpath.exists():
                                data = np.load(vpath, allow_pickle=True).item()
                                status = "success" if data["success"] else "fail"
                                tag = f"dist_rollout/gpu{data['rank']}_{data['task'][:30]}_{status}"
                                
                                # Create title frames
                                h, w = data["frames"].shape[1:3]
                                title_img = Image.new("RGB", (w, h), color=(0, 0, 0))
                                draw = ImageDraw.Draw(title_img)
                                try:
                                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                                except:
                                    font = ImageFont.load_default()
                                text = data["task"]
                                # Wrap text
                                words = text.split()
                                lines, line = [], ""
                                for word in words:
                                    if len(line + " " + word) < 25:
                                        line = (line + " " + word).strip()
                                    else:
                                        lines.append(line)
                                        line = word
                                lines.append(line)
                                y = h // 2 - len(lines) * 10
                                for ln in lines:
                                    bbox = draw.textbbox((0, 0), ln, font=font)
                                    x = (w - (bbox[2] - bbox[0])) // 2
                                    draw.text((x, y), ln, fill=(255, 255, 255), font=font)
                                    y += 20
                                title_frame = np.array(title_img)
                                title_frames = np.stack([title_frame] * 90)  # 3 seconds at 30fps
                                
                                frames_with_title = np.concatenate([title_frames, data["frames"]], axis=0)
                                tracker.write_video(tag, frames_with_title, step, fps=30)
                                # Save MP4
                                mp4_path = video_dir / f"step{step}_gpu{r}_{status}.mp4"
                                imageio.mimwrite(mp4_path, frames_with_title, fps=30)
                                # Upload to S3 if in SageMaker
                                if is_sagemaker_training():
                                    import boto3
                                    s3_path = cfg.s3_output_path or os.environ.get("SM_OUTPUT_DATA_DIR", "")
                                    if s3_path:
                                        s3_path = s3_path.rstrip("/")
                                        if s3_path.startswith("s3://"):
                                            s3_path = s3_path[5:]
                                        bucket, *prefix_parts = s3_path.split("/")
                                        prefix = "/".join(prefix_parts) if prefix_parts else ""
                                        s3_key = f"{prefix}/videos/{mp4_path.name}" if prefix else f"videos/{mp4_path.name}"
                                        boto3.client("s3").upload_file(str(mp4_path), bucket, s3_key)
                                vpath.unlink()
                        tracker.flush()
                    if use_ddp:
                        dist.barrier()
                    vla.train()
                
                # Training rollouts
                if cfg.train_rollout_frequency > 0 and step > 0 and step % cfg.train_rollout_frequency == 0:
                    if distributed_state.is_main_process:
                        run_train_rollouts(vla, processor, cfg, task_suites, tracker, step, device_id, vla_dataset.dataset_statistics)
                
                # Validation
                if cfg.val_frequency > 0 and step > 0 and step % cfg.val_frequency == 0:
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
    
    # Save final model to SageMaker model directory
    if is_sagemaker_training() and distributed_state.is_main_process:
        sm_model_dir = Path(get_sagemaker_paths()["model_dir"])
        print(f"Saving final model to {sm_model_dir}")
        # Save adapter first
        get_model().save_pretrained(adapter_dir)
        # Merge and save
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        final_model = PeftModel.from_pretrained(base_vla, adapter_dir)
        final_model = final_model.merge_and_unload()
        final_model.save_pretrained(sm_model_dir)
        processor.save_pretrained(sm_model_dir)
        # Final S3 sync
        s3_path = cfg.s3_output_path or os.environ.get("SM_OUTPUT_DATA_DIR", "")
        if s3_path:
            sync_tensorboard_to_s3(run_dir, s3_path, cfg.max_steps)
    
    print(f"\nTraining complete! Logs at: {run_dir}/tensorboard")
    print(f"View with: tensorboard --logdir {run_dir}/tensorboard")


if __name__ == "__main__":
    finetune()
