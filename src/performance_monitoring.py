#!/usr/bin/env python3
"""
📊 Performance Monitoring
========================

Middleware and utilities for monitoring API performance and gathering metrics.
"""

import time
import logging
import functools
from typing import Callable, Dict, List, Optional, Any, Union
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

TASK_DURATION = Histogram(
    "task_duration_seconds",
    "Background task duration in seconds",
    ["task_type"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

MODEL_INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_name"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
)

# Performance monitoring middleware
class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API performance"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process a request and measure performance"""
        # Start timer
        start_time = time.time()
        
        # Get request method and path
        method = request.method
        endpoint = request.url.path
        
        # Process the request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Log the exception
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            # Update Prometheus metrics
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            
            # Log request info
            logger.info(
                f"Request: {method} {endpoint} {status_code} - Duration: {duration:.3f}s",
                extra={
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "duration": duration
                }
            )
        
        return response


# Decorator for monitoring async function performance
def async_performance_monitor(func):
    """Decorator for monitoring async function performance"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Get function name
        func_name = func.__name__
        
        # Start timer
        start_time = time.time()
        
        # Log start
        logger.info(f"Starting {func_name}")
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log completion
            logger.info(
                f"Completed {func_name} in {duration:.3f}s",
                extra={
                    "function": func_name,
                    "duration": duration,
                    "success": True
                }
            )
            
            return result
        
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log failure
            logger.error(
                f"Failed {func_name} after {duration:.3f}s: {str(e)}",
                exc_info=True,
                extra={
                    "function": func_name,
                    "duration": duration,
                    "success": False,
                    "error": str(e)
                }
            )
            
            # Re-raise the exception
            raise
    
    return wrapper


# Function for tracking model inference time
def track_model_inference(model_name: str, duration: float):
    """
    Track model inference time
    
    Args:
        model_name: Name of the model
        duration: Inference duration in seconds
    """
    MODEL_INFERENCE_DURATION.labels(model_name=model_name).observe(duration)
    
    logger.info(
        f"Model inference: {model_name} - Duration: {duration:.3f}s",
        extra={
            "model_name": model_name,
            "duration": duration
        }
    )


# Function for tracking task duration
def track_task_duration(task_type: str, duration: float):
    """
    Track task duration
    
    Args:
        task_type: Type of task
        duration: Task duration in seconds
    """
    TASK_DURATION.labels(task_type=task_type).observe(duration)
    
    logger.info(
        f"Task completion: {task_type} - Duration: {duration:.3f}s",
        extra={
            "task_type": task_type,
            "duration": duration
        }
    )


# Start Prometheus metrics server
def start_metrics_server(port: int = 9090):
    """
    Start Prometheus metrics server
    
    Args:
        port: Port to listen on
    """
    start_http_server(port)
    logger.info(f"Started Prometheus metrics server on port {port}")
