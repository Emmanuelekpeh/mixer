#!/usr/bin/env python3
"""
📂 Model Metadata Generator
========================

Creates standardized metadata files for models to be used in the tournament system.
This utility helps maintain consistent model information and facilitates proper
model loading and management in the tournament.

Usage:
    python create_model_metadata.py model_path [--name NAME] [--desc DESCRIPTION]
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import glob

def create_model_metadata(
    model_path: str,
    name: Optional[str] = None,
    architecture: str = "cnn",
    description: Optional[str] = None,
    specializations: Optional[List[str]] = None,
    performance_metrics: Optional[Dict[str, float]] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata for a model file
    
    Args:
        model_path: Path to the model file
        name: Name of the model (defaults to filename if not provided)
        architecture: Architecture type (cnn, transformer, etc.)
        description: Description of the model
        specializations: List of specializations (bass, vocals, etc.)
        performance_metrics: Dictionary of performance metrics
        parameters: Additional model parameters
        
    Returns:
        Dictionary containing the metadata
    """
    model_file = Path(model_path)
    
    # Default values
    if name is None:
        name = model_file.stem
    
    if description is None:
        description = f"Mixing model for audio processing"
    
    if specializations is None:
        specializations = []
    
    if performance_metrics is None:
        performance_metrics = {}
    
    if parameters is None:
        parameters = {}
    
    # Load model to extract info if possible
    model_info = {}
    try:
        model_data = torch.load(model_path, map_location="cpu")
        if isinstance(model_data, dict) and "model_config" in model_data:
            model_info = model_data["model_config"]
    except Exception as e:
        print(f"Warning: Could not extract model info: {str(e)}")
    
    # Create metadata
    metadata = {
        "name": name,
        "architecture": architecture,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "file_path": str(model_file),
        "file_size_mb": model_file.stat().st_size / (1024 * 1024),
        "specializations": specializations,
        "performance_metrics": performance_metrics,
        "parameters": parameters,
        "model_info": model_info
    }
    
    return metadata

def save_metadata(model_path: str, metadata: Dict[str, Any]) -> str:
    """
    Save metadata to a JSON file next to the model file
    
    Args:
        model_path: Path to the model file
        metadata: Dictionary containing the metadata
        
    Returns:
        Path to the saved metadata file
    """
    model_file = Path(model_path)
    metadata_file = model_file.with_suffix('.json')
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_file}")
    return str(metadata_file)

def main():
    parser = argparse.ArgumentParser(description="Create metadata for model files or directories")
    parser.add_argument("path", help="Path to model file or directory")
    parser.add_argument("--name", help="Name of the model")
    parser.add_argument("--arch", default="cnn", help="Architecture type")
    parser.add_argument("--desc", help="Description")
    parser.add_argument("--specializations", help="Comma-separated specializations")
    parser.add_argument("--dir", action='store_true', help="Process directory")
    
    args = parser.parse_args()
    
    if args.dir:
        model_files = glob.glob(os.path.join(args.path, '*.pth'))
        for model_path in model_files:
            model_file = Path(model_path)
            metadata_file = model_file.with_suffix('.json')
            if metadata_file.exists():
                print(f"Skipping {model_file}, metadata exists")
                continue
            
            name = model_file.stem.replace('_', ' ').title()
            arch = infer_arch_from_name(model_file.stem)
            desc = f"{arch.upper()} model for audio mixing"
            specs = infer_specs_from_name(model_file.stem)
            
            metadata = create_model_metadata(
                model_path,
                name=name,
                architecture=arch,
                description=desc,
                specializations=specs
            )
            save_metadata(model_path, metadata)
    else:
        # existing code
        specializations = None
        if args.specializations:
            specializations = [s.strip() for s in args.specializations.split(",")]
        
        metadata = create_model_metadata(
            model_path=args.path,
            name=args.name,
            architecture=args.arch,
            description=args.desc,
            specializations=specializations
        )
        save_metadata(args.path, metadata)

def infer_arch_from_name(name: str) -> str:
    name_lower = name.lower()
    if 'lstm' in name_lower:
        return 'lstm'
    elif 'gan' in name_lower:
        return 'gan'
    elif 'vae' in name_lower:
        return 'vae'
    elif 'transformer' in name_lower:
        return 'transformer'
    elif 'resnet' in name_lower:
        return 'resnet'
    return 'cnn'

def infer_specs_from_name(name: str) -> List[str]:
    name_lower = name.lower()
    specs = []
    if 'lstm' in name_lower:
        specs = ['temporal_processing', 'sequence_aware']
    elif 'gan' in name_lower:
        specs = ['generative_mixing', 'creative_effects']
    elif 'vae' in name_lower:
        specs = ['latent_space_mixing', 'smooth_transitions']
    elif 'transformer' in name_lower:
        specs = ['attention_based', 'context_aware']
    elif 'resnet' in name_lower:
        specs = ['deep_processing', 'residual_learning']
    return specs

if __name__ == "__main__":
    main()
