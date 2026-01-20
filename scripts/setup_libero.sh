#!/bin/bash
# Setup script for LIBERO fine-tuning demo
# Downloads modified LIBERO RLDS datasets and installs dependencies
#
# Usage: bash scripts/setup_libero.sh <data_root_dir>
# Example: bash scripts/setup_libero.sh ./datasets

set -e

DATA_ROOT_DIR="${1:-./datasets}"

echo "=== LIBERO Fine-tuning Demo Setup ==="
echo "Data directory: $DATA_ROOT_DIR"

# Install LIBERO dependencies
echo ""
echo "[1/3] Installing LIBERO dependencies..."
pip install -r experiments/robot/libero/libero_requirements.txt

# Install LIBERO package
echo ""
echo "[2/3] Installing LIBERO package..."
pip install libero

# Download modified LIBERO RLDS datasets from HuggingFace
echo ""
echo "[3/3] Downloading modified LIBERO RLDS datasets..."
mkdir -p "$DATA_ROOT_DIR"

if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    sudo apt-get update && sudo apt-get install -y git-lfs
    git lfs install
fi

cd "$DATA_ROOT_DIR"
if [ -d "modified_libero_rlds" ]; then
    echo "Dataset directory already exists, pulling latest..."
    cd modified_libero_rlds && git pull && cd ..
else
    git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Datasets available at: $DATA_ROOT_DIR/modified_libero_rlds"
echo "  - libero_spatial_no_noops"
echo "  - libero_object_no_noops"
echo "  - libero_goal_no_noops"
echo "  - libero_10_no_noops"
echo ""
echo "To run fine-tuning demo:"
echo "  python vla-scripts/finetune_libero_demo.py --data_root_dir $DATA_ROOT_DIR/modified_libero_rlds --help"
