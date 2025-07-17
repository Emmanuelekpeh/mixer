#!/usr/bin/env python3
"""
Prepare Models for Deployment
=============================

Optimizes and prepares AI models for cloud deployment by:
1. Compressing model files if needed
2. Creating model metadata summaries
3. Validating all models work correctly
4. Generating deployment-ready model registry
"""

import os
import sys
import json
import gzip
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compress_large_models(models_dir: Path, size_threshold_mb: int = 50) -> Dict[str, Any]:
    """Compress models larger than threshold"""
    compression_stats = {
        'compressed_models': [],
        'total_size_before': 0,
        'total_size_after': 0,
        'space_saved': 0
    }
    
    for model_file in models_dir.glob("*.pth"):
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        compression_stats['total_size_before'] += file_size_mb
        
        if file_size_mb > size_threshold_mb:
            logger.info(f"Compressing large model: {model_file.name} ({file_size_mb:.1f}MB)")
            
            # Create compressed version
            compressed_file = model_file.with_suffix('.pth.gz')
            
            with open(model_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            compressed_size_mb = compressed_file.stat().st_size / (1024 * 1024)
            space_saved = file_size_mb - compressed_size_mb
            
            compression_stats['compressed_models'].append({
                'original_file': model_file.name,
                'compressed_file': compressed_file.name,
                'original_size_mb': round(file_size_mb, 2),
                'compressed_size_mb': round(compressed_size_mb, 2),
                'space_saved_mb': round(space_saved, 2),
                'compression_ratio': round(compressed_size_mb / file_size_mb, 2)
            })
            
            compression_stats['total_size_after'] += compressed_size_mb
            compression_stats['space_saved'] += space_saved
            
            logger.info(f"✅ Compressed {model_file.name}: {file_size_mb:.1f}MB → {compressed_size_mb:.1f}MB (saved {space_saved:.1f}MB)")
        else:
            compression_stats['total_size_after'] += file_size_mb
    
    return compression_stats

def create_deployment_model_registry(models_dir: Path) -> Dict[str, Any]:
    """Create a comprehensive model registry for deployment"""
    registry = {
        'created_at': datetime.now().isoformat(),
        'total_models': 0,
        'models': {},
        'architectures': {},
        'deployment_ready': True,
        'total_size_mb': 0
    }
    
    for model_file in models_dir.glob("*_best.pth"):
        try:
            # Extract model name
            model_name = model_file.stem.replace('_best', '')
            
            # Get file stats
            file_stats = model_file.stat()
            size_mb = file_stats.st_size / (1024 * 1024)
            
            # Load metadata if available
            metadata_file = models_dir / f"{model_name}_best.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Test model loading
            try:
                torch.load(model_file, map_location='cpu')
                loadable = True
                load_error = None
            except Exception as e:
                loadable = False
                load_error = str(e)
                registry['deployment_ready'] = False
            
            # Infer architecture
            architecture = infer_architecture(model_name)
            
            # Add to registry
            registry['models'][model_name] = {
                'file_path': str(model_file),
                'size_mb': round(size_mb, 2),
                'architecture': architecture,
                'loadable': loadable,
                'load_error': load_error,
                'metadata': metadata,
                'created_at': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                'deployment_priority': get_deployment_priority(model_name, size_mb, metadata)
            }
            
            # Update architecture counts
            if architecture not in registry['architectures']:
                registry['architectures'][architecture] = 0
            registry['architectures'][architecture] += 1
            
            registry['total_models'] += 1
            registry['total_size_mb'] += size_mb
            
            logger.info(f"✅ Registered model: {model_name} ({architecture}, {size_mb:.1f}MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to process model {model_file}: {e}")
            registry['deployment_ready'] = False
    
    return registry

def infer_architecture(model_name: str) -> str:
    """Infer model architecture from name"""
    name_lower = model_name.lower()
    
    if 'transformer' in name_lower:
        return 'transformer'
    elif 'ast' in name_lower:
        return 'ast'
    elif 'cnn' in name_lower:
        return 'cnn'
    elif 'lstm' in name_lower:
        return 'lstm'
    elif 'resnet' in name_lower:
        return 'resnet'
    elif 'vae' in name_lower:
        return 'vae'
    elif 'gan' in name_lower:
        return 'gan'
    elif 'hybrid' in name_lower or 'dual' in name_lower:
        return 'hybrid'
    else:
        return 'unknown'

def get_deployment_priority(model_name: str, size_mb: float, metadata: Dict) -> int:
    """Calculate deployment priority (higher = more important)"""
    priority = 100  # Base priority
    
    # Prioritize smaller models for faster deployment
    if size_mb < 10:
        priority += 20
    elif size_mb < 50:
        priority += 10
    
    # Prioritize models with good performance metrics
    if metadata and 'mae' in metadata:
        mae = metadata['mae']
        if mae < 0.06:
            priority += 30
        elif mae < 0.10:
            priority += 20
        elif mae < 0.15:
            priority += 10
    
    # Prioritize certain architectures
    if 'ast' in model_name.lower():
        priority += 25  # AST models are usually efficient
    elif 'transformer' in model_name.lower():
        priority += 15
    elif 'baseline' in model_name.lower():
        priority += 10  # Good fallback models
    
    return priority

def validate_deployment_readiness(registry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that all models are ready for deployment"""
    validation = {
        'ready_for_deployment': True,
        'total_models': registry['total_models'],
        'loadable_models': 0,
        'failed_models': [],
        'warnings': [],
        'recommendations': []
    }
    
    for model_name, model_info in registry['models'].items():
        if model_info['loadable']:
            validation['loadable_models'] += 1
        else:
            validation['failed_models'].append({
                'name': model_name,
                'error': model_info['load_error']
            })
            validation['ready_for_deployment'] = False
    
    # Check total size
    total_size_mb = registry['total_size_mb']
    if total_size_mb > 500:  # 500MB threshold
        validation['warnings'].append(f"Large total model size: {total_size_mb:.1f}MB")
        validation['recommendations'].append("Consider compressing large models or using cloud storage")
    
    # Check architecture diversity
    if len(registry['architectures']) < 3:
        validation['warnings'].append("Limited architecture diversity")
        validation['recommendations'].append("Consider adding more diverse model architectures")
    
    # Check for essential models
    essential_architectures = ['cnn', 'transformer', 'ast']
    missing_architectures = []
    for arch in essential_architectures:
        if arch not in registry['architectures']:
            missing_architectures.append(arch)
    
    if missing_architectures:
        validation['warnings'].append(f"Missing essential architectures: {missing_architectures}")
    
    return validation

def create_deployment_summary(models_dir: Path) -> Dict[str, Any]:
    """Create comprehensive deployment summary"""
    logger.info("🚀 Creating deployment summary...")
    
    # Create model registry
    registry = create_deployment_model_registry(models_dir)
    
    # Validate deployment readiness
    validation = validate_deployment_readiness(registry)
    
    # Compress large models if needed
    compression_stats = compress_large_models(models_dir, size_threshold_mb=50)
    
    # Create final summary
    summary = {
        'deployment_summary': {
            'created_at': datetime.now().isoformat(),
            'models_directory': str(models_dir),
            'deployment_ready': validation['ready_for_deployment'],
            'total_models': registry['total_models'],
            'total_size_mb': round(registry['total_size_mb'], 2),
            'architectures': registry['architectures']
        },
        'model_registry': registry,
        'validation': validation,
        'compression': compression_stats,
        'deployment_recommendations': []
    }
    
    # Add deployment recommendations
    if validation['ready_for_deployment']:
        summary['deployment_recommendations'].append("✅ All models are ready for deployment")
        summary['deployment_recommendations'].append("🚀 Recommended platform: Railway or Render")
        summary['deployment_recommendations'].append("💾 Database: PostgreSQL (included with platform)")
    else:
        summary['deployment_recommendations'].append("❌ Fix model loading issues before deployment")
        summary['deployment_recommendations'].append("🔧 Check failed models and resolve errors")
    
    if compression_stats['compressed_models']:
        summary['deployment_recommendations'].append(f"📦 {len(compression_stats['compressed_models'])} models compressed for deployment")
    
    return summary

def main():
    """Main deployment preparation function"""
    logger.info("🚀 PREPARING MODELS FOR DEPLOYMENT")
    logger.info("=" * 50)
    
    # Define models directory
    models_dir = Path("models")
    
    if not models_dir.exists():
        logger.error("❌ Models directory not found!")
        return
    
    # Create deployment summary
    summary = create_deployment_summary(models_dir)
    
    # Save summary to file
    summary_file = Path("deployment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    logger.info("📊 DEPLOYMENT PREPARATION COMPLETE!")
    logger.info("=" * 50)
    
    deployment_info = summary['deployment_summary']
    logger.info(f"✅ Models Ready: {deployment_info['total_models']}")
    logger.info(f"📦 Total Size: {deployment_info['total_size_mb']}MB")
    logger.info(f"🏗️  Architectures: {list(deployment_info['architectures'].keys())}")
    logger.info(f"🚀 Deployment Ready: {deployment_info['deployment_ready']}")
    
    if summary['compression']['compressed_models']:
        logger.info(f"📦 Compressed Models: {len(summary['compression']['compressed_models'])}")
        logger.info(f"💾 Space Saved: {summary['compression']['space_saved']:.1f}MB")
    
    logger.info("\n🎯 DEPLOYMENT RECOMMENDATIONS:")
    for rec in summary['deployment_recommendations']:
        logger.info(f"   {rec}")
    
    logger.info(f"\n📄 Full summary saved to: {summary_file}")
    
    if deployment_info['deployment_ready']:
        logger.info("\n🎉 YOUR MODELS ARE READY FOR DEPLOYMENT!")
        logger.info("Next steps:")
        logger.info("1. Choose deployment platform (Railway recommended)")
        logger.info("2. Set up database (PostgreSQL)")
        logger.info("3. Configure environment variables")
        logger.info("4. Deploy and test!")
    else:
        logger.info("\n⚠️  DEPLOYMENT ISSUES DETECTED")
        logger.info("Please fix the following issues before deployment:")
        for issue in summary['validation']['failed_models']:
            logger.info(f"   - {issue['name']}: {issue['error']}")

if __name__ == "__main__":
    main()