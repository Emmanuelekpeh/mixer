"""
Model Validator Wrapper for Tournament Web Application

This module provides a wrapper around the ModelValidator to add robust error handling,
logging, and recovery mechanisms for model validation operations.
"""

import os
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from model_validator import ModelValidator, ValidationResult
from model_validation_error_handler import (
    ModelValidationErrorHandler, 
    ModelValidationError,
    ModelValidationErrorCategory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidatorWrapper:
    """
    Wrapper for ModelValidator with enhanced error handling and recovery.
    
    This class provides functionality to:
    1. Safely execute validation operations with error handling
    2. Implement retry mechanisms for transient failures
    3. Log detailed error information
    4. Provide graceful degradation for validation failures
    """
    
    def __init__(self, validator: Optional[ModelValidator] = None, max_retries: int = 2, 
                retry_delay: float = 1.0, log_dir: str = "logs"):
        """
        Initialize the ModelValidatorWrapper.
        
        Args:
            validator: ModelValidator instance or None to create a new one
            max_retries: Maximum number of retry attempts for transient errors
            retry_delay: Delay between retry attempts in seconds
            log_dir: Directory for error logs
        """
        self.validator = validator or ModelValidator()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_handler = ModelValidationErrorHandler(log_dir=log_dir)
        
        logger.info(f"ModelValidatorWrapper initialized with max_retries={max_retries}")
    
    def validate_model_safely(self, model_path: Path) -> ValidationResult:
        """
        Validate a model with robust error handling and retry mechanism.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            ValidationResult object with validation details
        """
        # Set up validation context
        validation_context = {
            "operation": "model_validation",
            "model_name": model_path.name if model_path else "unknown",
            "timestamp": time.time(),
            "attempt": 0
        }
        
        # Try validation with retries for transient errors
        for attempt in range(self.max_retries + 1):
            validation_context["attempt"] = attempt + 1
            
            try:
                # Attempt validation
                logger.info(f"Validating model {model_path.name} (attempt {attempt + 1}/{self.max_retries + 1})")
                result = self.validator.validate_model_file(model_path)
                
                # If successful, return result
                if result.is_valid:
                    logger.info(f"Model {model_path.name} validation successful")
                    return result
                
                # If validation failed but not due to transient error, don't retry
                if result.error_message is None or not self._is_transient_error(result.error_message):
                    logger.warning(f"Model {model_path.name} validation failed with non-transient error: {result.error_message}")
                    return result
                
                # For transient errors, retry if attempts remain
                if attempt < self.max_retries:
                    logger.info(f"Transient error detected, retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(f"Max retries reached for model {model_path.name}")
                    return result
                    
            except Exception as e:
                # Handle unexpected exceptions
                error_info = self.error_handler.handle_error(e, model_path, validation_context)
                
                # If it's a transient error and we have attempts left, retry
                if self._is_transient_exception(e) and attempt < self.max_retries:
                    logger.info(f"Transient exception detected, retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    # Create failure result
                    logger.error(f"Validation failed with exception: {str(e)}")
                    return ValidationResult(
                        is_valid=False,
                        model_architecture="unknown",
                        error_message=f"Validation error: {str(e)}",
                        can_load=False,
                        inference_test_passed=False,
                        compatibility_score=0.0,
                        warnings=[f"Exception during validation: {str(e)}"],
                        recommendations=["Check model file integrity"],
                        details={"error_info": error_info}
                    )
        
        # This should not be reached, but just in case
        return ValidationResult(
            is_valid=False,
            model_architecture="unknown",
            error_message="Validation failed after retries",
            can_load=False,
            inference_test_passed=False,
            compatibility_score=0.0
        )
    
    def batch_validate_models(self, model_paths: List[Path]) -> Dict[str, ValidationResult]:
        """
        Validate multiple models with error isolation.
        
        Args:
            model_paths: List of paths to model files
            
        Returns:
            Dictionary mapping model names to validation results
        """
        results = {}
        
        for model_path in model_paths:
            try:
                # Validate each model safely
                model_name = model_path.stem
                logger.info(f"Batch validating model: {model_name}")
                
                result = self.validate_model_safely(model_path)
                results[model_name] = result
                
            except Exception as e:
                # Ensure one model failure doesn't affect others
                logger.error(f"Unexpected error in batch validation for {model_path.name}: {e}")
                results[model_path.stem] = ValidationResult(
                    is_valid=False,
                    model_architecture="unknown",
                    error_message=f"Batch validation error: {str(e)}",
                    can_load=False,
                    inference_test_passed=False,
                    compatibility_score=0.0
                )
        
        # Log summary
        valid_count = sum(1 for result in results.values() if result.is_valid)
        logger.info(f"Batch validation complete: {valid_count}/{len(model_paths)} models passed validation")
        
        return results
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation errors.
        
        Returns:
            Error summary dictionary
        """
        return self.error_handler.get_error_summary()
    
    def _is_transient_error(self, error_message: str) -> bool:
        """
        Check if an error message indicates a transient error.
        
        Args:
            error_message: Error message to check
            
        Returns:
            True if the error appears to be transient
        """
        if not error_message:
            return False
            
        # Common transient error patterns
        transient_patterns = [
            "timeout",
            "connection",
            "temporarily",
            "retry",
            "cuda out of memory",
            "resource",
            "busy",
            "locked"
        ]
        
        error_lower = error_message.lower()
        return any(pattern in error_lower for pattern in transient_patterns)
    
    def _is_transient_exception(self, exception: Exception) -> bool:
        """
        Check if an exception indicates a transient error.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if the exception appears to be transient
        """
        # Check exception type
        transient_types = [
            "TimeoutError",
            "ConnectionError",
            "ResourceError",
            "TemporaryError",
            "RetryError",
            "CudaError"
        ]
        
        exception_type = exception.__class__.__name__
        if any(t in exception_type for t in transient_types):
            return True
            
        # Check exception message
        return self._is_transient_error(str(exception))