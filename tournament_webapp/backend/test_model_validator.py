"""
Test script for ModelValidator

This script tests the ModelValidator functionality with sample model files.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model_validator import ModelValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_validator():
    """Test the ModelValidator with sample model files"""
    
    # Initialize validator
    validator = ModelValidator()
    logger.info("ModelValidator initialized")
    
    # Get paths to model files
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return False
    
    # Find model files
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        logger.error("No model files found")
        return False
    
    logger.info(f"Found {len(model_files)} model files")
    
    # Test each model file
    success_count = 0
    for model_path in model_files:
        logger.info(f"Testing model: {model_path.name}")
        
        # Test loading
        can_load, load_error = validator.test_model_loading(model_path)
        if can_load:
            logger.info(f"✅ Model loading successful: {model_path.name}")
        else:
            logger.error(f"❌ Model loading failed: {model_path.name} - {load_error}")
            continue
        
        # Extract architecture
        architecture = validator.extract_model_architecture(model_path)
        logger.info(f"📊 Architecture: {architecture}")
        
        # Full validation
        result = validator.validate_model_file(model_path)
        
        if result.is_valid:
            logger.info(f"✅ Model validation successful: {model_path.name}")
            logger.info(f"   Architecture: {result.model_architecture}")
            logger.info(f"   Compatibility score: {result.compatibility_score:.2f}")
            
            if result.warnings:
                logger.warning(f"   Warnings: {len(result.warnings)}")
                for warning in result.warnings:
                    logger.warning(f"   - {warning}")
            
            if result.recommendations:
                logger.info(f"   Recommendations: {len(result.recommendations)}")
                for rec in result.recommendations:
                    logger.info(f"   - {rec}")
            
            success_count += 1
        else:
            logger.error(f"❌ Model validation failed: {model_path.name}")
            logger.error(f"   Error: {result.error_message}")
    
    # Summary
    logger.info(f"Validation complete: {success_count}/{len(model_files)} models passed validation")
    return success_count > 0

if __name__ == "__main__":
    logger.info("Starting ModelValidator test")
    success = test_model_validator()
    if success:
        logger.info("✅ ModelValidator test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ ModelValidator test failed")
        sys.exit(1)