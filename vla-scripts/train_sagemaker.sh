#!/bin/bash
# train_sagemaker.sh - SageMaker training entrypoint for OpenVLA
# This script installs dlimp (with --no-deps to avoid TF version conflict),
# then launches the distributed training job.

set -e

echo "=== OpenVLA SageMaker Training Entrypoint ==="

# Install dlimp without dependencies (avoids TensorFlow version conflict)
echo "Installing dlimp (no dependencies)..."
pip install --no-deps git+https://github.com/kvablack/dlimp

# Get number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Launch distributed training with torchrun
# SageMaker sets SM_NUM_GPUS, but we detect directly to be safe
echo "Starting training..."
torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS \
    /opt/ml/code/vla-scripts/finetune_libero_demo.py "$@"

echo "=== Training Complete ==="
