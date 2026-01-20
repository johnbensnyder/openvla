# OpenVLA LIBERO Fine-tuning on AWS Trainium

This directory contains scripts for fine-tuning OpenVLA on AWS Trainium 1 instances (trn1.32xlarge).

## Prerequisites

- AWS trn1.32xlarge instance with Neuron DLAMI (Ubuntu 22)
- Neuron SDK 2.18+ with PyTorch NeuronX

## Environment Setup

```bash
# 1. Launch trn1.32xlarge with Neuron DLAMI (Ubuntu 22)

# 2. Activate PyTorch NeuronX environment
source /opt/aws_neuronx_venv_pytorch_2_1/bin/activate

# 3. Install OpenVLA and dependencies
cd /path/to/openvla
pip install -e .
pip install -r vla-scripts/trainium/requirements_trainium.txt

# 4. Install neuronx-distributed from Neuron repo
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com neuronx-distributed

# 5. Set up Neuron compile cache (optional but recommended)
export NEURON_COMPILE_CACHE_URL="/home/ubuntu/neuron_cache"
```

## Download Dataset

```bash
# Download modified LIBERO datasets in RLDS format
git clone git@hf.co:datasets/openvla/modified_libero_rlds datasets/modified_libero_rlds
```

## Running Fine-tuning

```bash
# Single-node training on trn1.32xlarge (32 NeuronCores, TP=8)
torchrun --nproc_per_node=8 vla-scripts/trainium/finetune_libero_trainium.py \
    --data_root_dir ./datasets/modified_libero_rlds \
    --task_suites libero_spatial \
    --run_root_dir ./runs_trainium \
    --tensor_parallel_size 8 \
    --batch_size 4 \
    --max_steps 50000

# Or with YAML config
torchrun --nproc_per_node=8 vla-scripts/trainium/finetune_libero_trainium.py \
    --config vla-scripts/trainium/finetune_trainium_config.yaml
```

## Key Differences from GPU Script

| Feature | GPU Script | Trainium Script |
|---------|-----------|-----------------|
| Parallelism | DDP | Tensor Parallelism (NxD) |
| LoRA | HuggingFace PEFT | NxD Core LoRA |
| Attention | Flash Attention 2 | Neuron SDPA |
| Device | CUDA | XLA (torch_xla) |
| Validation | GPU | CPU (model copied) |

## Checkpoints

Checkpoints are saved in NxD format under `run_root_dir/`. To convert for HuggingFace deployment:

```python
# See checkpoint conversion in the training script
# Merged checkpoints are saved alongside LoRA adapters
```

## Troubleshooting

1. **OOM errors**: Reduce `batch_size` or increase `grad_accumulation_steps`
2. **Slow compilation**: Set `NEURON_COMPILE_CACHE_URL` for persistent cache
3. **XLA errors**: Ensure all tensor ops are XLA-compatible (no in-place ops on XLA tensors)
