#!/usr/bin/env python3
"""
Download LIBERO data from HuggingFace and upload to S3.
Run from a SageMaker notebook instance or any environment with AWS credentials.
"""
import os
import subprocess
import argparse
import boto3
from pathlib import Path


def download_libero_data(local_dir: str = "./datasets") -> str:
    """Download modified LIBERO RLDS datasets from HuggingFace."""
    local_path = Path(local_dir) / "modified_libero_rlds"
    
    if local_path.exists():
        print(f"Dataset already exists at {local_path}, pulling latest...")
        subprocess.run(["git", "-C", str(local_path), "pull"], check=True)
    else:
        print("Downloading modified LIBERO RLDS datasets from HuggingFace...")
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone",
            "https://huggingface.co/datasets/openvla/modified_libero_rlds",
            str(local_path)
        ], check=True)
    
    return str(local_path)


def upload_to_s3(local_path: str, s3_bucket: str, s3_prefix: str = "openvla/data") -> str:
    """Upload local directory to S3."""
    s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"Uploading {local_path} to {s3_uri}...")
    
    subprocess.run([
        "aws", "s3", "sync",
        local_path, s3_uri,
        "--quiet"
    ], check=True)
    
    print(f"Upload complete: {s3_uri}")
    return s3_uri


def prepare_data(s3_bucket: str, s3_prefix: str = "openvla/data", 
                 local_dir: str = "./datasets") -> str:
    """Download LIBERO data and upload to S3. Returns S3 URI."""
    local_path = download_libero_data(local_dir)
    s3_uri = upload_to_s3(local_path, s3_bucket, s3_prefix)
    return s3_uri


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LIBERO data for SageMaker training")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--s3-prefix", default="openvla/data", help="S3 prefix for data")
    parser.add_argument("--local-dir", default="./datasets", help="Local download directory")
    args = parser.parse_args()
    
    s3_uri = prepare_data(args.s3_bucket, args.s3_prefix, args.local_dir)
    print(f"\nData ready at: {s3_uri}")
    print(f"Use this URI as the 'training' channel in your SageMaker training job.")
