#!/usr/bin/env python3
"""
🔍 Error Handling & Monitoring
============================

Provides centralized error handling, logging, and monitoring for the AI mixer.
Includes structured logging, error tracking, and performance monitoring.
"""

import logging
import sys
import traceback
import time
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import uuid

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger for this module
logger = logging.getLogger("mixer")

# Setup file logging if logs directory exists
logs_dir = Path("logs")
if not logs_dir.exists():
    logs_dir.mkdir(parents=True, exist_ok=True)

if logs_dir.exists():
    # Create rotating file handler
    file_handler = logging.FileHandler(
        logs_dir / f"mixer_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

# Performance metrics storage
performance_metrics = {}

# Error tracking
error_counts = {}
last_errors = {}


class StructuredLogRecord:
    """Structured log record for JSON logging"""
    
    def __init__(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        **kwargs
    ):
        self.timestamp = datetime.now().isoformat()
        self.level = level
        self.message = message
        self.context = context or {}
        self.exception = None
        
        if exception:
            self.exception = {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # Add any additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "context": self.context
        }
        
        if self.exception:
            result["exception"] = self.exception
        
        # Add any additional fields
        for key, value in self.__dict__.items():
            if key not in ["timestamp", "level", "message", "context", "exception"]:
                result[key] = value
        
        return result
    
    def __str__(self) -> str:
        """Return string representation"""
        return json.dumps(self.to_dict())


def log_structured(
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    exception: Optional[Exception] = None,
    **kwargs
) -> None:
    """
    Log a structured message
    
    Args:
        level: Log level (info, warning, error, critical)
        message: Log message
        context: Additional context information
        exception: Exception object if this is an error log
        **kwargs: Additional fields to include in the log record
    """
    record = StructuredLogRecord(level, message, context, exception, **kwargs)
    
    # Log to file
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(str(record))
    
    # If this is an error, track it
    if level.lower() in ["error", "critical"] and exception:
        track_error(exception)


def track_error(exception: Exception) -> None:
    """
    Track an error for monitoring
    
    Args:
        exception: The exception to track
    """
    error_type = exception.__class__.__name__
    
    # Update error count
    if error_type not in error_counts:
        error_counts[error_type] = 0
    error_counts[error_type] += 1
    
    # Store last error
    last_errors[error_type] = {
        "message": str(exception),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat()
    }


def track_performance(
    name: str,
    duration: float,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Track a performance metric
    
    Args:
        name: Name of the operation
        duration: Duration in seconds
        metadata: Additional metadata about the operation
    """
    if name not in performance_metrics:
        performance_metrics[name] = {
            "count": 0,
            "total_duration": 0,
            "min_duration": float('inf'),
            "max_duration": 0,
            "samples": []
        }
    
    metrics = performance_metrics[name]
    metrics["count"] += 1
    metrics["total_duration"] += duration
    metrics["min_duration"] = min(metrics["min_duration"], duration)
    metrics["max_duration"] = max(metrics["max_duration"], duration)
    
    # Store sample with timestamp and metadata
    sample = {
        "duration": duration,
        "timestamp": datetime.now().isoformat()
    }
    
    if metadata:
        sample["metadata"] = metadata
    
    # Keep only the last 100 samples
    metrics["samples"].append(sample)
    if len(metrics["samples"]) > 100:
        metrics["samples"] = metrics["samples"][-100:]


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get all performance metrics
    
    Returns:
        Dictionary with performance metrics
    """
    result = {}
    
    for name, metrics in performance_metrics.items():
        avg_duration = metrics["total_duration"] / metrics["count"] if metrics["count"] > 0 else 0
        
        result[name] = {
            "count": metrics["count"],
            "avg_duration": avg_duration,
            "min_duration": metrics["min_duration"] if metrics["min_duration"] != float('inf') else 0,
            "max_duration": metrics["max_duration"],
            "last_10_samples": metrics["samples"][-10:] if metrics["samples"] else []
        }
    
    return result


def get_error_metrics() -> Dict[str, Any]:
    """
    Get all error metrics
    
    Returns:
        Dictionary with error metrics
    """
    return {
        "counts": error_counts,
        "last_errors": last_errors
    }


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor the performance of a function
    
    Args:
        func: Function to monitor
    
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Track performance
            track_performance(
                name=func.__name__,
                duration=duration,
                metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Track performance and error
            track_performance(
                name=func.__name__,
                duration=duration,
                metadata={
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "error": str(e)
                }
            )
            
            # Re-raise the exception
            raise
    
    return wrapper


def async_performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor the performance of an async function
    
    Args:
        func: Async function to monitor
    
    Returns:
        Wrapped async function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Track performance
            track_performance(
                name=func.__name__,
                duration=duration,
                metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Track performance and error
            track_performance(
                name=func.__name__,
                duration=duration,
                metadata={
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                    "error": str(e)
                }
            )
            
            # Re-raise the exception
            raise
    
    return wrapper


def safe_execute(func: Callable, *args, **kwargs) -> Union[Any, Dict[str, Any]]:
    """
    Safely execute a function and handle errors
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Result of the function or error information
    """
    try:
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # Track performance
        track_performance(
            name=func.__name__,
            duration=duration,
            metadata={"args_count": len(args), "kwargs_count": len(kwargs)}
        )
        
        return result
        
    except Exception as e:
        # Track error
        track_error(e)
        
        # Log error
        log_structured(
            "error",
            f"Error executing {func.__name__}: {str(e)}",
            context={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_count": len(kwargs)
            },
            exception=e
        )
        
        # Return error information
        return {
            "error": str(e),
            "error_type": e.__class__.__name__,
            "function": func.__name__
        }


# Initialize performance monitoring
log_structured(
    "info",
    "Error handling and monitoring system initialized",
    context={"logs_dir": str(logs_dir) if logs_dir.exists() else None}
)
