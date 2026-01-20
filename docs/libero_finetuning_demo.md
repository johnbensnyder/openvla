# LIBERO Fine-tuning Demo

This demo replicates the LoRA fine-tuning of OpenVLA on the LIBERO dataset as described in Appendix E of the [OpenVLA paper](https://arxiv.org/abs/2406.09246).

## Features

- LoRA fine-tuning with paper hyperparameters (rank=32, lr=5e-4)
- Periodic validation via LIBERO simulation rollouts
- Optional training rollouts to visualize learning progress
- Video logging to TensorBoard
- Support for all four LIBERO task suites

## Prerequisites

1. **GPU**: NVIDIA GPU with at least 24GB VRAM (48GB+ recommended for batch_size=16)
2. **Conda environment**: The `openvla` conda environment should be set up

## Setup

### 1. Install Dependencies and Download Dataset

```bash
cd openvla
bash scripts/setup_libero.sh ./datasets
```

This will:
- Install LIBERO and its dependencies
- Download the modified LIBERO RLDS datasets (~10GB)

### 2. Verify Installation

```bash
conda activate openvla
python -c "from libero.libero import benchmark; print('LIBERO installed successfully')"
```

## Usage

### Basic Training

```bash
python vla-scripts/finetune_libero_demo.py \
    --data_root_dir ./datasets/modified_libero_rlds \
    --task_suites libero_spatial \
    --val_frequency 500 \
    --batch_size 16
```

### Quick Demo (smaller batch size for limited GPU memory)

```bash
python vla-scripts/finetune_libero_demo.py \
    --data_root_dir ./datasets/modified_libero_rlds \
    --task_suites libero_spatial \
    --val_frequency 100 \
    --val_episodes 5 \
    --num_videos 2 \
    --max_steps 500 \
    --batch_size 4
```

### Training on Multiple Task Suites

```bash
python vla-scripts/finetune_libero_demo.py \
    --data_root_dir ./datasets/modified_libero_rlds \
    --task_suites libero_spatial,libero_object \
    --val_frequency 500
```

### With Training Rollouts

```bash
python vla-scripts/finetune_libero_demo.py \
    --data_root_dir ./datasets/modified_libero_rlds \
    --task_suites libero_spatial \
    --val_frequency 500 \
    --train_rollout_frequency 100 \
    --train_rollout_videos 2
```

### With Weights & Biases Logging

```bash
python vla-scripts/finetune_libero_demo.py \
    --data_root_dir ./datasets/modified_libero_rlds \
    --task_suites libero_spatial \
    --val_frequency 500 \
    --use_wandb \
    --wandb_project my-project \
    --wandb_entity my-entity
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root_dir` | `datasets/modified_libero_rlds` | Path to LIBERO RLDS datasets |
| `--task_suites` | `libero_spatial` | Comma-separated list of task suites |
| `--val_frequency` | `1000` | Steps between validation rollouts |
| `--val_episodes` | `10` | Episodes per validation |
| `--num_videos` | `5` | Videos to save per validation |
| `--train_rollout_frequency` | `0` | Steps between training rollouts (0=disabled) |
| `--train_rollout_episodes` | `5` | Episodes per training rollout |
| `--train_rollout_videos` | `2` | Videos per training rollout |
| `--batch_size` | `16` | Training batch size |
| `--max_steps` | `50000` | Maximum training steps |
| `--save_steps` | `5000` | Steps between checkpoint saves |
| `--learning_rate` | `5e-4` | Learning rate |
| `--lora_rank` | `32` | LoRA rank |
| `--image_aug` | `True` | Enable image augmentation |
| `--center_crop` | `True` | Center crop during validation |
| `--use_wandb` | `False` | Enable W&B logging |

## Available Task Suites

- `libero_spatial` - 10 tasks focusing on spatial reasoning
- `libero_object` - 10 tasks focusing on object manipulation
- `libero_goal` - 10 tasks focusing on goal-directed behavior
- `libero_10` - 10 long-horizon tasks (also called LIBERO-Long)

## Viewing Results

### TensorBoard

```bash
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser.

TensorBoard will show:
- `train/loss` - Training loss
- `train/action_accuracy` - Action token accuracy
- `val/success_rate/*` - Validation success rates per task suite
- `val_videos/*` - Rollout videos from validation
- `train_rollout_videos/*` - Rollout videos from training (if enabled)

### Checkpoints

Checkpoints are saved to `runs/<experiment_id>/` and include:
- Merged LoRA weights (ready for inference)
- `dataset_statistics.json` (required for action un-normalization)

## Expected Results

Based on the OpenVLA paper (Appendix E), after full training you should expect:

| Task Suite | Success Rate |
|------------|--------------|
| LIBERO-Spatial | ~84.7% |
| LIBERO-Object | ~88.4% |
| LIBERO-Goal | ~79.2% |
| LIBERO-10 | ~53.7% |

Note: Results may vary based on random seed and exact training configuration.

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch_size 4 --grad_accumulation_steps 4
```

### Slow Validation

Reduce validation episodes:
```bash
--val_episodes 5 --num_videos 2
```

### Missing Dataset Statistics Error

Ensure you're using the modified LIBERO RLDS datasets from HuggingFace:
```bash
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds datasets/modified_libero_rlds
```
