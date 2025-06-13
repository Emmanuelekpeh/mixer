#!/usr/bin/env python3
"""
📤 Model Upload Helper for Cloud Deployment
=========================================

This script helps upload models to cloud storage (S3, etc.) for Render deployment.
Only needed if you want to store models externally rather than in the Git repo.
"""

import os
import sys
import json
from pathlib import Path
import boto3
import time
import argparse

def upload_models_to_cloud():
    """Upload model files to cloud storage for Render deployment"""
    print("📤 Model Upload Helper for Cloud Deployment")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="Upload models to cloud storage")
    parser.add_argument("--service", choices=["s3", "gcs", "azure"], default="s3", 
                        help="Cloud storage service to use")
    parser.add_argument("--bucket", help="Bucket/container name")
    parser.add_argument("--prefix", default="models/", help="Prefix/folder for uploaded models")
    args = parser.parse_args()
    
    if not args.bucket:
        print("❌ No bucket specified. Use --bucket to specify a storage bucket.")
        return
    
    # Get model files
    models_dir = Path(__file__).resolve().parent / "models"
    deployment_dir = models_dir / "deployment"
    
    if not deployment_dir.exists():
        print(f"❌ Deployment directory not found: {deployment_dir}")
        print("Run prepare_models_for_deployment.py first.")
        return
    
    model_files = list(deployment_dir.glob("*.pth")) + list(deployment_dir.glob("*.json"))
    
    if not model_files:
        print("❌ No model files found in deployment directory.")
        return
    
    print(f"Found {len(model_files)} model files to upload")
    
    # Upload to cloud storage
    if args.service == "s3":
        try:
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            for model_file in model_files:
                print(f"Uploading {model_file.name}...")
                object_key = f"{args.prefix}{model_file.name}"
                
                # Upload file
                s3_client.upload_file(
                    str(model_file),
                    args.bucket,
                    object_key
                )
                
                print(f"✅ Uploaded to s3://{args.bucket}/{object_key}")
            
            # Generate config file with URLs
            config = {
                "storage_service": "s3",
                "bucket": args.bucket,
                "prefix": args.prefix,
                "models": {}
            }
            
            for model_file in model_files:
                if model_file.suffix == ".pth":
                    model_name = model_file.stem
                    config["models"][model_name] = f"s3://{args.bucket}/{args.prefix}{model_file.name}"
            
            # Save config
            config_path = deployment_dir / "cloud_storage_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"\n✅ Configuration saved to {config_path}")
            print("\nTo use these models with Render:")
            print("1. Install boto3 in your requirements.txt")
            print("2. Set up AWS credentials as environment variables in Render:")
            print("   - AWS_ACCESS_KEY_ID")
            print("   - AWS_SECRET_ACCESS_KEY")
            print("   - AWS_DEFAULT_REGION")
            print("3. Modify your code to load models from S3 using the config file")
            
        except Exception as e:
            print(f"❌ Error uploading to S3: {str(e)}")
    
    elif args.service == "gcs":
        print("Google Cloud Storage upload not implemented yet.")
        print("Please implement or use S3 instead.")
    
    elif args.service == "azure":
        print("Azure Blob Storage upload not implemented yet.")
        print("Please implement or use S3 instead.")

if __name__ == "__main__":
    upload_models_to_cloud()
