#!/usr/bin/env python3
"""
launch_sagemaker_job.py

Launch OpenVLA finetuning as a SageMaker training job.

Usage:
    python launch_sagemaker_job.py \
        --s3-data s3://your-bucket/datasets/modified_libero_rlds \
        --s3-output s3://your-bucket/openvla-training \
        --instance-type ml.p4d.24xlarge

Requirements:
    pip install sagemaker boto3
"""

import argparse
from datetime import datetime

import sagemaker
from sagemaker.pytorch import PyTorch


def main():
    parser = argparse.ArgumentParser(description="Launch OpenVLA SageMaker training job")
    parser.add_argument("--s3-data", required=True, help="S3 path to training data (RLDS format)")
    parser.add_argument("--s3-output", required=True, help="S3 path for output artifacts")
    parser.add_argument("--instance-type", default="ml.p4d.24xlarge", help="SageMaker instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN (auto-detected if not provided)")
    parser.add_argument("--region", default=None, help="AWS region")
    parser.add_argument("--job-name", default=None, help="Training job name (auto-generated if not provided)")
    parser.add_argument("--max-steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--task-suites", default="libero_spatial", help="Task suites (comma-separated)")
    parser.add_argument("--dist-rollout-frequency", type=int, default=500, help="Distributed rollout frequency")
    parser.add_argument("--s3-sync-frequency", type=int, default=500, help="S3 sync frequency for TensorBoard")
    parser.add_argument("--max-run-hours", type=int, default=72, help="Maximum training time in hours")
    args = parser.parse_args()

    # Setup SageMaker session
    sess = sagemaker.Session()
    region = args.region or sess.boto_region_name
    role = args.role or sagemaker.get_execution_role()

    # Generate job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"openvla-libero-{timestamp}"

    print(f"Launching SageMaker training job: {job_name}")
    print(f"  Instance: {args.instance_type} x {args.instance_count}")
    print(f"  Data: {args.s3_data}")
    print(f"  Output: {args.s3_output}")

    # Training hyperparameters (passed as command line args to train_sagemaker.sh)
    hyperparameters = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "task_suites": args.task_suites,
        "dist_rollout_frequency": args.dist_rollout_frequency,
        "s3_sync_frequency": args.s3_sync_frequency,
        "val_frequency": 0,  # Disable validation, use distributed rollouts only
    }

    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point="vla-scripts/train_sagemaker.sh",
        source_dir=".",  # Upload entire repo
        role=role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version="2.9.0",
        py_version="py312",
        image_uri=f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker",
        output_path=args.s3_output,
        sagemaker_session=sess,
        hyperparameters=hyperparameters,
        max_run=args.max_run_hours * 3600,
        keep_alive_period_in_seconds=0,
        distribution={"torch_distributed": {"enabled": True}},
        environment={
            "TOKENIZERS_PARALLELISM": "false",
        },
    )

    # Configure input data channel with Fast File Mode
    training_input = sagemaker.inputs.TrainingInput(
        s3_data=args.s3_data,
        input_mode="FastFile",
    )

    # Launch training
    estimator.fit(
        inputs={"training": training_input},
        job_name=job_name,
        wait=False,  # Don't block, return immediately
    )

    print(f"\nTraining job submitted: {job_name}")
    print(f"Monitor at: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    print(f"\nTo stream logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --follow --filter-pattern {job_name}")


if __name__ == "__main__":
    main()
