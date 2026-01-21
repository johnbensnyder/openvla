#!/bin/bash
set -e

VENV_PATH="${1:-$HOME/custom_trainium_venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating virtual environment at $VENV_PATH..."
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

pip install --upgrade pip -q

# Install dlimp with --no-deps to bypass tensorflow==2.15.0 pin
echo "Installing dlimp..."
pip install --no-deps "dlimp @ git+https://github.com/kvablack/dlimp"

# Install remaining requirements
echo "Installing requirements..."
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com -r "$SCRIPT_DIR/requirements_trainium.txt"

echo "Environment ready at $VENV_PATH"
echo "Activate with: source $VENV_PATH/bin/activate"
