#!/bin/bash
# Setup script for LIBERO fine-tuning on Trainium
# Downloads modified LIBERO RLDS datasets and installs dependencies
#
# Usage: bash vla-scripts/trainium/setup_libero.sh <data_root_dir>
# Example: bash vla-scripts/trainium/setup_libero.sh ./datasets

set -e

DATA_ROOT_DIR="${1:-./datasets}"
VENV_PATH="${VIRTUAL_ENV:-$HOME/custom_trainium_venv}"

echo "=== LIBERO Setup for Trainium ==="
echo "Data directory: $DATA_ROOT_DIR"
echo "Virtual env: $VENV_PATH"

# Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating $VENV_PATH..."
    source "$VENV_PATH/bin/activate"
fi

# Install LIBERO dependencies (robosuite/mujoco already in requirements_trainium.txt)
echo ""
echo "[1/3] Installing LIBERO dependencies..."
pip install bddl easydict cloudpickle gym libero

# Download modified LIBERO RLDS datasets from HuggingFace
echo ""
echo "[2/3] Installing git-lfs if needed..."
if ! command -v git-lfs &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y git-lfs
    git lfs install
fi

echo ""
echo "[3/3] Downloading modified LIBERO RLDS datasets..."
mkdir -p "$DATA_ROOT_DIR"
cd "$DATA_ROOT_DIR"
if [ -d "modified_libero_rlds" ]; then
    echo "Dataset directory exists, pulling latest..."
    cd modified_libero_rlds && git pull && cd ..
else
    git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
fi

echo ""
echo "=== Setup Complete ==="
echo "Datasets: $DATA_ROOT_DIR/modified_libero_rlds"
