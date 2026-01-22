#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Sync data from S3 if not present locally
DATA_DIR="datasets/modified_libero_rlds"
S3_PATH="s3://jbsnyder-datasets/libero"

if [ ! -d "$DATA_DIR" ]; then
    echo "Syncing data from S3..."
    mkdir -p "$DATA_DIR"
    aws s3 sync "$S3_PATH" "$DATA_DIR"
fi

torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/finetune_libero_demo.py \
  --data_root_dir "$DATA_DIR" \
  --task_suites libero_spatial \
  --val_frequency 5000 \
  --val_episodes 10 \
  --num_videos 5 \
  --train_rollout_frequency 0 \
  --batch_size 4 \
  --save_steps 25000
