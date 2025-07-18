"""
Fix for ModelValidator class to add missing _determine_error_category method
"""

def _determine_error_category(self, validation_result: ValidationResult) -> str:
    """
    Determine the error category based on validation result.
    
    Args:
        validation_result: The validation result to categorize
        
    Returns:
        Error category string
    """
    if not validation_result.can_load:
        return ModelValidationErrorCategory.MODEL_LOADING
    
    if not validation_result.inference_test_passed:
        return ModelValidationErrorCategory.INFERENCE
    
    if validation_result.compatibility_score < 0.5:
        return ModelValidationErrorCategory.COMPATIBILITY
    
    # Check for memory issues in details
    details = validation_result.details or {}
    memory_stats = details.get("memory_stats_inference", {})
    if memory_stats.get("possible_leak", False) or memory_stats.get("peak_mb", 0) > 2000:
        return ModelValidationErrorCategory.MEMORY
    
    # Check for timeout
    if details.get("timeout", False):
        return ModelValidationErrorCategory.TIMEOUT
    
    # Default category
    return ModelValidationErrorCategory.UNKNOWN