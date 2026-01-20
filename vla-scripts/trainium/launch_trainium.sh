#!/bin/bash
# Launch script for OpenVLA LIBERO fine-tuning on AWS Trainium (trn1.32xlarge)
#
# Usage:
#   ./launch_trainium.sh                    # Use default config
#   ./launch_trainium.sh --config my.yaml   # Use custom config
#   ./launch_trainium.sh --task_suites libero_object --batch_size 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Default values
NPROC=${NPROC:-8}  # Number of processes (TP ranks)
NEURON_CACHE=${NEURON_CACHE:-/home/ubuntu/neuron_cache}

# Set Neuron environment variables
export NEURON_COMPILE_CACHE_URL="$NEURON_CACHE"
export NEURON_RT_NUM_CORES=32  # Use all 32 NeuronCores on trn1.32xlarge
export XLA_USE_BF16=1  # Enable BF16 for better performance

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "OpenVLA LIBERO Fine-tuning on Trainium"
echo "=========================================="
echo "Processes: $NPROC"
echo "Neuron cache: $NEURON_CACHE"
echo "Arguments: $@"
echo "=========================================="

# Launch training
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NPROC \
    vla-scripts/trainium/finetune_libero_trainium.py \
    "$@"
