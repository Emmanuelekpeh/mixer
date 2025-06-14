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
    parser = argparse.ArgumentParser(description="Create metadata for model files")
    parser.add_argument("model_path", help="Path to the model file")
    parser.add_argument("--name", help="Name of the model")
    parser.add_argument("--arch", default="cnn", help="Architecture type (cnn, transformer, etc.)")
    parser.add_argument("--desc", help="Description of the model")
    parser.add_argument("--specializations", help="Comma-separated list of specializations")
    
    args = parser.parse_args()
    
    # Parse specializations
    specializations = None
    if args.specializations:
        specializations = [s.strip() for s in args.specializations.split(",")]
    
    # Create metadata
    metadata = create_model_metadata(
        model_path=args.model_path,
        name=args.name,
        architecture=args.arch,
        description=args.desc,
        specializations=specializations
    )
    
    # Save metadata
    save_metadata(args.model_path, metadata)

if __name__ == "__main__":
    main()
