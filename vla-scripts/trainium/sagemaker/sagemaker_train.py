#!/usr/bin/env python3
"""
SageMaker entry point for OpenVLA Trainium training.
Adapts the training script for SageMaker environment.
"""
import os
import sys
import subprocess

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


def main():
    # SageMaker environment variables
    data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))
    num_neuron_cores = int(os.environ.get("SM_NUM_NEURONS", "32"))
    
    # Neuron-specific setup
    os.environ.setdefault("NEURON_RT_NUM_CORES", str(num_neuron_cores))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # Parse hyperparameters from command line (SageMaker passes them as args)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suites", type=str, default="libero_spatial")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b")
    args, unknown = parser.parse_known_args()
    
    # Build training command
    nproc = args.tensor_parallel_size
    train_script = os.path.join(os.path.dirname(__file__), "..", "finetune_libero_trainium.py")
    
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        train_script,
        f"--data_root_dir={data_dir}",
        f"--run_root_dir={model_dir}",
        f"--task_suites={args.task_suites}",
        f"--batch_size={args.batch_size}",
        f"--max_steps={args.max_steps}",
        f"--learning_rate={args.learning_rate}",
        f"--lora_rank={args.lora_rank}",
        f"--tensor_parallel_size={args.tensor_parallel_size}",
        f"--save_interval={args.save_interval}",
        f"--vla_path={args.vla_path}",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
