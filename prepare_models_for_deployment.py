#!/usr/bin/env python3
"""
📦 Prepare Models for Render Deployment
=====================================

This script creates deployment-friendly versions of models for hosting
on Render.com and other platforms with storage limitations.

Options:
1. Create small example models for testing
2. Compress existing models
3. Set up on-demand loading
"""

import os
import sys
import shutil
from pathlib import Path
import json
import torch

def prepare_models_for_deployment():
    """Prepare models for deployment by creating smaller versions or examples"""
    print("📦 Preparing Models for Render Deployment")
    print("=" * 50)
    
    models_dir = Path(__file__).resolve().parent / "models"
    deployment_dir = models_dir / "deployment"
    deployment_dir.mkdir(exist_ok=True)
    
    # List available models
    available_models = list(models_dir.glob("*.pth"))
    if not available_models:
        print("❌ No models found in models/ directory")
        return
    
    print(f"Found {len(available_models)} models")
    
    # Create small example/placeholder models
    print("\n📊 Creating deployment-friendly model versions...")
    
    model_info = {}
    
    for model_path in available_models:
        model_name = model_path.stem
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        print(f"Processing {model_name} ({model_size_mb:.2f} MB)")
        
        # Store model info
        model_info[model_name] = {
            "original_size_mb": model_size_mb,
            "deployment_strategy": "full" if model_size_mb < 50 else "placeholder"
        }
        
        # For large models, create placeholder or compressed version
        if model_size_mb >= 50:  # Models larger than 50MB
            print(f"  ⚠️ Model too large for direct deployment, creating placeholder")
            
            # Create a small placeholder model with metadata
            placeholder = {
                "model_name": model_name,
                "original_size_mb": model_size_mb,
                "is_placeholder": True,
                "deployment_note": "Full model too large for deployment, download separately"
            }
            
            # Save placeholder
            with open(deployment_dir / f"{model_name}_placeholder.json", "w") as f:
                json.dump(placeholder, f, indent=2)
        else:
            # Copy smaller models directly
            print(f"  ✅ Model size acceptable, copying directly")
            shutil.copy(model_path, deployment_dir / model_path.name)
    
    # Save model info summary
    with open(deployment_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("\n✅ Model preparation complete!")
    print(f"Deployment-ready models available in: {deployment_dir}")
    print("\nInstructions for Render deployment:")
    print("1. Include these smaller models in your Git repository")
    print("2. For larger models, implement on-demand downloading in your code")
    print("3. Update your application to check for placeholders and handle accordingly")

if __name__ == "__main__":
    prepare_models_for_deployment()
