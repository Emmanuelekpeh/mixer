#!/usr/bin/env python3
"""
Model Integration Script
========================

Integrates all available AI mixing models into the tournament webapp system.
This script discovers models from the main models directory and registers them
in the tournament database.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "tournament_webapp" / "backend"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def discover_models(models_dir: Path) -> List[Dict[str, Any]]:
    """Discover all model files in the models directory"""
    models = []
    
    if not models_dir.exists():
        logger.error(f"Models directory does not exist: {models_dir}")
        return models
    
    # Find all .pth files
    for model_file in models_dir.glob("*.pth"):
        if model_file.name.endswith("_best.pth"):
            # Extract model name (remove _best.pth suffix)
            model_name = model_file.stem.replace("_best", "")
            
            # Look for corresponding metadata file
            metadata_file = models_dir / f"{model_name}_best.json"
            metadata = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_name}: {e}")
            
            # Get file stats
            file_stats = model_file.stat()
            
            model_info = {
                'id': model_name,
                'name': model_name.replace('_', ' ').title(),
                'file_path': str(model_file),
                'metadata_path': str(metadata_file) if metadata_file.exists() else None,
                'size_mb': file_stats.st_size / (1024 * 1024),
                'created_at': datetime.fromtimestamp(file_stats.st_mtime),
                'architecture': infer_architecture(model_name),
                'metadata': metadata
            }
            
            models.append(model_info)
            logger.info(f"Discovered model: {model_name}")
    
    return models

def infer_architecture(model_name: str) -> str:
    """Infer model architecture from name"""
    name_lower = model_name.lower()
    
    if 'transformer' in name_lower:
        return 'Transformer'
    elif 'ast' in name_lower:
        return 'AST'
    elif 'cnn' in name_lower:
        return 'CNN'
    elif 'lstm' in name_lower:
        return 'LSTM'
    elif 'resnet' in name_lower:
        return 'ResNet'
    elif 'vae' in name_lower:
        return 'VAE'
    elif 'gan' in name_lower:
        return 'GAN'
    else:
        return 'Unknown'

def register_models_in_database(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Register discovered models in the tournament database"""
    try:
        # Import database components
        from database import SessionLocal, AIModel
        
        db = SessionLocal()
        stats = {
            'total_discovered': len(models),
            'registered': 0,
            'updated': 0,
            'errors': []
        }
        
        for model_info in models:
            try:
                # Check if model already exists
                existing_model = db.query(AIModel).filter(AIModel.id == model_info['id']).first()
                
                if existing_model:
                    # Update existing model
                    existing_model.name = model_info['name']
                    existing_model.architecture = model_info['architecture']
                    existing_model.model_file_path = model_info['file_path']
                    existing_model.updated_at = datetime.now()
                    existing_model.is_active = True
                    
                    # Update metadata if available
                    if model_info['metadata']:
                        existing_model.capabilities = model_info['metadata'].get('capabilities', {})
                        existing_model.specializations = model_info['metadata'].get('specializations', [])
                    
                    stats['updated'] += 1
                    logger.info(f"Updated existing model: {model_info['id']}")
                else:
                    # Create new model
                    new_model = AIModel(
                        id=model_info['id'],
                        name=model_info['name'],
                        architecture=model_info['architecture'],
                        model_file_path=model_info['file_path'],
                        is_active=True,
                        created_at=model_info['created_at'],
                        updated_at=datetime.now(),
                        tier='Challenger',  # Default tier
                        elo_rating=1200,    # Default ELO
                        capabilities=model_info['metadata'].get('capabilities', {}),
                        specializations=model_info['metadata'].get('specializations', []),
                        description=f"AI mixing model using {model_info['architecture']} architecture"
                    )
                    
                    db.add(new_model)
                    stats['registered'] += 1
                    logger.info(f"Registered new model: {model_info['id']}")
                
            except Exception as e:
                error_msg = f"Failed to register model {model_info['id']}: {str(e)}"
                stats['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Commit all changes
        db.commit()
        db.close()
        
        return stats
        
    except Exception as e:
        error_msg = f"Database operation failed: {str(e)}"
        logger.error(error_msg)
        return {
            'total_discovered': len(models),
            'registered': 0,
            'updated': 0,
            'errors': [error_msg]
        }

def validate_model_files(models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that model files can be loaded"""
    validation_stats = {
        'total': len(models),
        'valid': 0,
        'invalid': 0,
        'errors': []
    }
    
    try:
        import torch
        
        for model_info in models:
            try:
                # Try to load the model file
                model_path = Path(model_info['file_path'])
                if model_path.exists():
                    torch.load(model_path, map_location='cpu')
                    validation_stats['valid'] += 1
                    logger.info(f"✓ Model validation passed: {model_info['id']}")
                else:
                    validation_stats['invalid'] += 1
                    error_msg = f"Model file not found: {model_info['file_path']}"
                    validation_stats['errors'].append(error_msg)
                    logger.error(f"✗ {error_msg}")
                    
            except Exception as e:
                validation_stats['invalid'] += 1
                error_msg = f"Model validation failed for {model_info['id']}: {str(e)}"
                validation_stats['errors'].append(error_msg)
                logger.error(f"✗ {error_msg}")
                
    except ImportError:
        logger.warning("PyTorch not available, skipping model validation")
        validation_stats = {
            'total': len(models),
            'valid': 0,
            'invalid': 0,
            'errors': ['PyTorch not available for validation']
        }
    
    return validation_stats

def main():
    """Main integration function"""
    logger.info("🚀 Starting model integration process...")
    
    # Define models directory
    models_dir = Path("models")
    
    # Step 1: Discover models
    logger.info("📡 Discovering models...")
    discovered_models = discover_models(models_dir)
    
    if not discovered_models:
        logger.error("❌ No models discovered. Check that the models directory exists and contains .pth files.")
        return
    
    logger.info(f"✅ Discovered {len(discovered_models)} models")
    
    # Step 2: Validate models
    logger.info("🔍 Validating model files...")
    validation_stats = validate_model_files(discovered_models)
    
    logger.info(f"✅ Validation complete: {validation_stats['valid']}/{validation_stats['total']} models valid")
    
    if validation_stats['errors']:
        logger.warning("⚠️  Validation errors:")
        for error in validation_stats['errors']:
            logger.warning(f"  - {error}")
    
    # Step 3: Register in database
    logger.info("💾 Registering models in database...")
    registration_stats = register_models_in_database(discovered_models)
    
    # Print final summary
    logger.info("🎉 Model integration complete!")
    logger.info("=" * 50)
    logger.info("INTEGRATION SUMMARY:")
    logger.info(f"  Models discovered: {registration_stats['total_discovered']}")
    logger.info(f"  Models registered: {registration_stats['registered']}")
    logger.info(f"  Models updated: {registration_stats['updated']}")
    logger.info(f"  Validation passed: {validation_stats['valid']}")
    logger.info(f"  Validation failed: {validation_stats['invalid']}")
    
    if registration_stats['errors']:
        logger.warning("⚠️  Registration errors:")
        for error in registration_stats['errors']:
            logger.warning(f"  - {error}")
    
    # List integrated models
    logger.info("\n📋 INTEGRATED MODELS:")
    for model in discovered_models:
        status = "✅" if Path(model['file_path']).exists() else "❌"
        logger.info(f"  {status} {model['name']} ({model['architecture']}) - {model['size_mb']:.1f}MB")

if __name__ == "__main__":
    main()