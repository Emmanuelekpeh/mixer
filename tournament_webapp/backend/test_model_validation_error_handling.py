"""
Test script for model validation error handling

This script tests the error handling capabilities of the ModelValidator
and ModelValidationErrorHandler classes.
"""

import os
import sys
import logging
import time
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model_validator_fixed import ModelValidator
from backend.model_validation_error_handler import (
    ModelValidationErrorHandler,
    ModelValidationError,
    ModelValidationErrorCategory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_model_files():
    """Create test model files for error handling tests"""
    test_dir = Path("test_models")
    test_dir.mkdir(exist_ok=True)
    
    # Create a valid model file
    valid_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 32 * 32, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    torch.save(valid_model, test_dir / "valid_model.pth")
    
    # Create a corrupted model file
    with open(test_dir / "corrupted_model.pth", "wb") as f:
        f.write(b"This is not a valid PyTorch model file")
    
    # Create an empty model file
    with open(test_dir / "empty_model.pth", "wb") as f:
        pass
    
    # Create a model file that's too large
    try:
        # Create a model with large tensors
        large_model = torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 1000)
        )
        # Add some large tensors to state dict
        state_dict = large_model.state_dict()
        state_dict["large_tensor"] = torch.randn(500, 500, 500)
        torch.save(state_dict, test_dir / "large_model.pth")
    except Exception as e:
        logger.warning(f"Could not create large model: {e}")
    
    return test_dir

def test_error_handling():
    """Test error handling for model validation"""
    # Create test model files
    test_dir = create_test_model_files()
    
    # Create error handler
    error_handler = ModelValidationErrorHandler(log_dir="test_logs")
    
    # Create validator with error handler
    validator = ModelValidator(timeout_seconds=5, log_dir="test_logs")
    
    # Test files
    test_files = [
        (test_dir / "valid_model.pth", "Valid model"),
        (test_dir / "corrupted_model.pth", "Corrupted model"),
        (test_dir / "empty_model.pth", "Empty model"),
        (test_dir / "nonexistent_model.pth", "Nonexistent model"),
    ]
    
    if (test_dir / "large_model.pth").exists():
        test_files.append((test_dir / "large_model.pth", "Large model"))
    
    # Test each file
    results = []
    for file_path, description in test_files:
        logger.info(f"Testing {description}: {file_path}")
        
        try:
            # Validate model
            result = validator.validate_model_file(file_path)
            
            # Log result
            if result.is_valid:
                logger.info(f"✅ {description} validation successful")
            else:
                logger.warning(f"❌ {description} validation failed: {result.error_message}")
                
                # Check warnings
                if result.warnings:
                    logger.warning(f"Warnings: {len(result.warnings)}")
                    for warning in result.warnings:
                        logger.warning(f"- {warning}")
            
            results.append((description, result.is_valid, result.error_message))
            
        except Exception as e:
            logger.error(f"Error validating {description}: {e}")
            results.append((description, False, str(e)))
    
    # Get error summary
    if hasattr(validator, "error_handler") and validator.error_handler:
        error_summary = validator.error_handler.get_error_summary()
        logger.info(f"Error summary: {error_summary}")
    
    # Print results
    logger.info("\n=== TEST RESULTS ===")
    for description, is_valid, error_message in results:
        status = "✅ PASSED" if is_valid else "❌ FAILED"
        logger.info(f"{status} - {description}: {error_message if not is_valid else ''}")
    
    # Check if error log file was created
    log_file = Path("test_logs") / "model_validation_errors.log"
    if log_file.exists():
        logger.info(f"Error log file created: {log_file}")
        logger.info(f"Log file size: {log_file.stat().st_size} bytes")
    else:
        logger.warning(f"Error log file not created: {log_file}")

def test_error_categories():
    """Test error categorization"""
    logger.info("\n=== TESTING ERROR CATEGORIES ===")
    
    # Create error handler
    error_handler = ModelValidationErrorHandler(log_dir="test_logs")
    
    # Test different error types
    test_errors = [
        (FileNotFoundError("File not found"), ModelValidationErrorCategory.FILE_SYSTEM),
        (PermissionError("Permission denied"), ModelValidationErrorCategory.FILE_SYSTEM),
        (RuntimeError("CUDA out of memory"), ModelValidationErrorCategory.MEMORY),
        (ValueError("Expected 4D tensor"), ModelValidationErrorCategory.INFERENCE),
        (TimeoutError("Operation timed out"), ModelValidationErrorCategory.TIMEOUT),
        (Exception("Unknown error"), ModelValidationErrorCategory.UNKNOWN),
    ]
    
    # Test each error
    for error, expected_category in test_errors:
        # Create a dummy model path
        model_path = Path("dummy_model.pth")
        
        # Handle error
        error_info = error_handler.handle_error(error, model_path)
        
        # Check category
        actual_category = error_info.get("category", "")
        if actual_category == expected_category:
            logger.info(f"✅ Correctly categorized {error.__class__.__name__} as {actual_category}")
        else:
            logger.warning(f"❌ Incorrectly categorized {error.__class__.__name__} as {actual_category}, expected {expected_category}")

if __name__ == "__main__":
    logger.info("Starting model validation error handling tests")
    
    # Run tests
    test_error_handling()
    test_error_categories()
    
    logger.info("Tests completed")