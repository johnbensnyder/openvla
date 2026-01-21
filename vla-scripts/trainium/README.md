# OpenVLA Fine-tuning on AWS Trainium

This directory contains scripts for fine-tuning OpenVLA on LIBERO tasks using AWS Trainium accelerators.

## Quick Start (EC2)

```bash
# 1. Create and activate virtual environment
./setup_trainium_venv.sh

# 2. Download LIBERO data
source ~/custom_trainium_venv/bin/activate
./setup_libero.sh ./datasets

# 3. Run training
./launch_trainium.sh --task_suites libero_spatial
```

## SageMaker Pipeline

For production workloads, use the SageMaker pipeline which provides:
- **Ahead-of-time compilation**: Compile once, reuse across training runs
- **S3 caching**: Compiled NEFFs cached to S3 for fast startup
- **Managed infrastructure**: No EC2 instance management

### Prerequisites

1. SageMaker execution role with S3 access
2. S3 bucket for data and artifacts
3. Service quota for `ml.trn1.32xlarge` instances

### Running the Pipeline

1. Open `sagemaker/openvla_trainium_pipeline.ipynb` in SageMaker Studio or a notebook instance

2. Configure your S3 bucket and settings in the first cell

3. Run the cells in order:
   - **Data Preparation**: Downloads LIBERO data to S3
   - **Compilation Job**: Short ~100 step run to extract and compile XLA graphs (~30 min)
   - **Training Job**: Full training using cached compilation (~8 hours for 50k steps)

### Pipeline Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Data Preparation  │     │   Compilation Job   │     │    Training Job     │
│                     │     │                     │     │                     │
│  HuggingFace → S3   │────▶│  trn1.32xlarge     │────▶│  trn1.32xlarge     │
│                     │     │  ~100 steps         │     │  50k steps          │
│                     │     │  Cache → S3         │     │  Uses S3 cache      │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

### Cost Estimates

| Job | Instance | Duration | Est. Cost |
|-----|----------|----------|-----------|
| Compilation | ml.trn1.32xlarge | ~30 min | ~$10 |
| Training (50k steps) | ml.trn1.32xlarge | ~8 hours | ~$160 |

*Based on us-west-2 on-demand pricing.*

## Files

| File | Description |
|------|-------------|
| `launch_trainium.sh` | Launch script for EC2 training |
| `setup_trainium_venv.sh` | Create Python virtual environment |
| `setup_libero.sh` | Download LIBERO datasets |
| `finetune_libero_trainium.py` | Main training script |
| `openvla_nxd.py` | OpenVLA model with NxD tensor parallelism |
| `background_workers.py` | Non-blocking checkpoint/validation workers |
| `requirements_trainium.txt` | Python dependencies |
| `sagemaker/` | SageMaker pipeline components |

## Background Checkpointing & Validation

By default, checkpointing and validation run in background processes without blocking training:

```
Training Loop (Trainium)
        │
        │ (every save_steps)
        ▼
   Gather Weights ──────► Shared CPU Memory
        │                        │
        │              ┌─────────┴─────────┐
        │              ▼                   ▼
        │      Checkpointer          Validator
        │      (background)          (LIBERO rollouts)
        │              │                   │
        │              ▼                   ▼
        │        checkpoint/          videos/
        │                             metrics
        ▼
   Continue Training (no blocking)
```

### Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `--background_workers` | Enable background checkpoint/validation | `True` |
| `--val_episodes` | Episodes per validation | `1` |
| `--val_videos` | Videos to capture per validation | `1` |
| `--save_steps` | Checkpoint/validation interval | `5000` |

### Cancellation Behavior

If validation is still running when new weights arrive, the old validation is cancelled with a warning. To avoid this:
- Increase `--save_steps` to give validation more time
- Reduce `--val_episodes` for faster validation

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURON_COMPILE_CACHE_URL` | Neuron cache location (local path or S3 URI) | `/home/ubuntu/neuron_cache` |
| `NEURON_RT_NUM_CORES` | Number of NeuronCores to use | `32` |
| `NPROC` | Number of training processes (TP ranks) | `8` |

## Troubleshooting

### Compilation takes too long
- Use the SageMaker pipeline with ahead-of-time compilation
- Set `NEURON_COMPILE_CACHE_URL` to an S3 bucket to persist cache

### Out of memory errors
- Reduce `batch_size`
- Reduce `tensor_parallel_size` (requires recompilation)

### Model loading errors
- Ensure `safetensors` is installed
- Check HuggingFace Hub connectivity

## References

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [Neuron Persistent Cache](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/neuron-caching.html)
- [SageMaker Trainium Training](https://docs.aws.amazon.com/sagemaker/latest/dg/trainium.html)
