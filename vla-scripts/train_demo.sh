#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)

torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/finetune_libero_demo.py \
  --data_root_dir datasets/modified_libero_rlds \
  --task_suites libero_spatial \
  --val_frequency 5000 \
  --val_episodes 10 \
  --num_videos 5 \
  --train_rollout_frequency 0 \
  --batch_size 4 \
  --save_steps 25000
