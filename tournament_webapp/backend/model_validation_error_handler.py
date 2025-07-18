"""
Error Handler for Model Validation

This module provides specialized error handling for model validation operations,
with detailed error reporting, logging, and recovery mechanisms.
"""

import os
import sys
import logging
import traceback
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define error categories
class ModelValidationErrorCategory:
    FILE_SYSTEM = "file_system"
    MODEL_LOADING = "model_loading"
    INFERENCE = "inference"
    COMPATIBILITY = "compatibility"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class ModelValidationError(Exception):
    """
    Specialized exception for model validation errors.
    
    Attributes:
        model_path: Path to the model file
        category: Error category
        message: Error message
        details: Additional error details
    """
    
    def __init__(self, model_path: Path, category: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.category = category
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        
        # Format the error message
        formatted_message = f"[{category.upper()}] {message} (model: {model_path.name})"
        super().__init__(formatted_message)

class ModelValidationErrorHandler:
    """
    Handles errors during model validation with detailed reporting and recovery.
    
    This class provides functionality to:
    1. Categorize and log validation errors
    2. Generate detailed error reports
    3. Implement recovery strategies
    4. Track error patterns
    """
    
    def __init__(self, log_dir: str = "logs", error_log_file: str = "model_validation_errors.log"):
        """
        Initialize the error handler.
        
        Args:
            log_dir: Directory for error logs
            error_log_file: File name for error logs
        """
        self.log_dir = Path(log_dir)
        self.error_log_file = self.log_dir / error_log_file
        self.error_history = []
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler for error logging
        self._setup_file_logging()
    
    def _setup_file_logging(self) -> None:
        """Set up file logging for validation errors."""
        try:
            file_handler = logging.FileHandler(self.error_log_file)
            file_handler.setLevel(logging.ERROR)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
        except Exception as e:
            logger.error(f"Failed to set up error logging: {e}")
    
    def handle_error(self, error: Exception, model_path: Path, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a validation error with appropriate logging and recovery.
        
        Args:
            error: The exception that occurred
            model_path: Path to the model file
            context: Additional context information
            
        Returns:
            Error report dictionary
        """
        context = context or {}
        error_info = self._create_error_info(error, model_path, context)
        
        # Log the error
        self._log_error(error_info)
        
        # Add to error history
        self.error_history.append(error_info)
        if len(self.error_history) > 100:  # Limit history size
            self.error_history = self.error_history[-100:]
        
        # Attempt recovery based on error category
        recovery_action = self._attempt_recovery(error_info)
        error_info["recovery_action"] = recovery_action
        
        return error_info
    
    def _create_error_info(self, error: Exception, model_path: Path, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed error information dictionary.
        
        Args:
            error: The exception that occurred
            model_path: Path to the model file
            context: Additional context information
            
        Returns:
            Error information dictionary
        """
        # Get stack trace
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
        
        # Determine error category
        if isinstance(error, ModelValidationError):
            category = error.category
            message = error.message
            details = error.details
        else:
            category, message = self._categorize_error(error)
            details = {}
        
        # Create error info
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_path.name,
            "model_path": str(model_path),
            "category": category,
            "message": message,
            "exception_type": error.__class__.__name__,
            "exception_args": [str(arg) for arg in getattr(error, 'args', [])],
            "stack_trace": stack_trace,
            "context": context,
            "details": details
        }
        
        return error_info
    
    def _categorize_error(self, error: Exception) -> Tuple[str, str]:
        """
        Categorize an error based on its type and message.
        
        Args:
            error: The exception to categorize
            
        Returns:
            Tuple of (category, message)
        """
        error_message = str(error)
        error_type = error.__class__.__name__
        
        # File system errors
        if isinstance(error, (FileNotFoundError, PermissionError, IOError, OSError)):
            return ModelValidationErrorCategory.FILE_SYSTEM, error_message
        
        # PyTorch specific errors
        if "CUDA" in error_type or "cuda" in error_message.lower():
            return ModelValidationErrorCategory.MODEL_LOADING, f"CUDA error: {error_message}"
            
        if "out of memory" in error_message.lower():
            return ModelValidationErrorCategory.MEMORY, f"Out of memory: {error_message}"
            
        if "shape" in error_message.lower() or "size" in error_message.lower():
            return ModelValidationErrorCategory.INFERENCE, f"Shape mismatch: {error_message}"
            
        if "timeout" in error_message.lower():
            return ModelValidationErrorCategory.TIMEOUT, f"Operation timed out: {error_message}"
            
        # Generic model loading errors
        if "load" in error_message.lower():
            return ModelValidationErrorCategory.MODEL_LOADING, f"Loading error: {error_message}"
        
        # Default category
        return ModelValidationErrorCategory.UNKNOWN, error_message
    
    def _log_error(self, error_info: Dict[str, Any]) -> None:
        """
        Log error information with appropriate detail level.
        
        Args:
            error_info: Error information dictionary
        """
        model_name = error_info["model_name"]
        category = error_info["category"]
        message = error_info["message"]
        
        # Log to console
        logger.error(f"Model validation error [{category}] for {model_name}: {message}")
        
        # Log detailed information to file
        try:
            with open(self.error_log_file, 'a') as f:
                f.write(f"\n--- ERROR REPORT: {datetime.now().isoformat()} ---\n")
                json.dump(error_info, f, indent=2, default=str)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write to error log file: {e}")
    
    def _attempt_recovery(self, error_info: Dict[str, Any]) -> str:
        """
        Attempt to recover from the error based on its category.
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            Description of recovery action taken
        """
        category = error_info["category"]
        model_path = error_info["model_path"]
        
        if category == ModelValidationErrorCategory.FILE_SYSTEM:
            # Nothing to recover for file system errors
            return "Model marked as invalid due to file system error"
            
        elif category == ModelValidationErrorCategory.MODEL_LOADING:
            # For loading errors, we could try alternative loading methods
            return "Model marked as invalid due to loading failure"
            
        elif category == ModelValidationErrorCategory.MEMORY:
            # For memory errors, suggest optimization
            return "Model marked as requiring optimization due to memory usage"
            
        elif category == ModelValidationErrorCategory.TIMEOUT:
            # For timeout errors, suggest optimization
            return "Model marked as requiring optimization due to timeout"
            
        elif category == ModelValidationErrorCategory.INFERENCE:
            # For inference errors, mark as incompatible
            return "Model marked as incompatible with inference system"
            
        else:
            # Default recovery action
            return "Model marked as invalid due to unknown error"
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation errors.
        
        Returns:
            Error summary dictionary
        """
        if not self.error_history:
            return {"total_errors": 0, "categories": {}}
        
        # Count errors by category
        categories = {}
        for error in self.error_history:
            category = error["category"]
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        # Get most recent errors
        recent_errors = self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        
        return {
            "total_errors": len(self.error_history),
            "categories": categories,
            "recent_errors": [
                {
                    "timestamp": e["timestamp"],
                    "model_name": e["model_name"],
                    "category": e["category"],
                    "message": e["message"]
                }
                for e in recent_errors
            ]
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history = []