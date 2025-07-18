"""
Memory Monitor for Model Validation

This module provides utilities to monitor memory usage during model validation,
helping to identify models that might cause memory issues.
"""

import os
import logging
import psutil
import threading
import time
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Monitors memory usage during model operations.
    
    This class provides functionality to:
    1. Track memory usage before and after operations
    2. Monitor memory usage in a separate thread
    3. Detect memory leaks and spikes
    """
    
    def __init__(self, interval_seconds: float = 0.5):
        """
        Initialize the MemoryMonitor.
        
        Args:
            interval_seconds: Interval between memory measurements in seconds
        """
        self.interval_seconds = interval_seconds
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = []
        self.peak_memory = 0
        self.baseline_memory = self._get_current_memory_usage()
        
    def start_monitoring(self) -> None:
        """Start monitoring memory usage in a separate thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.memory_history = []
        self.peak_memory = 0
        self.baseline_memory = self._get_current_memory_usage()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.debug("Memory monitoring started")
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop monitoring memory usage and return statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not self.monitoring:
            return self._get_empty_stats()
            
        self.monitoring = False
        
        # Wait for thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        # Calculate statistics
        current_memory = self._get_current_memory_usage()
        memory_increase = current_memory - self.baseline_memory
        
        stats = {
            "baseline_mb": self.baseline_memory,
            "final_mb": current_memory,
            "peak_mb": self.peak_memory,
            "increase_mb": memory_increase,
            "possible_leak": memory_increase > 10,  # More than 10MB increase might indicate a leak
            "measurements": len(self.memory_history),
            "history": self.memory_history[-10:] if len(self.memory_history) > 10 else self.memory_history
        }
        
        logger.debug(f"Memory monitoring stopped. Peak: {self.peak_memory:.2f}MB, Increase: {memory_increase:.2f}MB")
        
        return stats
        
    def _monitor_memory(self) -> None:
        """Monitor memory usage at regular intervals."""
        while self.monitoring:
            try:
                memory_usage = self._get_current_memory_usage()
                timestamp = time.time()
                
                self.memory_history.append((timestamp, memory_usage))
                self.peak_memory = max(self.peak_memory, memory_usage)
                
                time.sleep(self.interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                break
                
    def _get_current_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            # Convert to MB
            memory_mb = memory_info.rss / (1024 * 1024)
            return memory_mb
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
            
    def _get_empty_stats(self) -> Dict[str, Any]:
        """
        Get empty statistics when monitoring wasn't active.
        
        Returns:
            Empty statistics dictionary
        """
        return {
            "baseline_mb": 0,
            "final_mb": 0,
            "peak_mb": 0,
            "increase_mb": 0,
            "possible_leak": False,
            "measurements": 0,
            "history": []
        }
        
    def measure_operation(self, operation_func, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Measure memory usage during an operation.
        
        Args:
            operation_func: Function to execute and monitor
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (operation result, memory statistics)
        """
        # Start monitoring
        self.start_monitoring()
        
        # Execute operation
        try:
            start_time = time.time()
            result = operation_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Stop monitoring and get stats
            stats = self.stop_monitoring()
            stats["execution_time"] = execution_time
            
            return result, stats
            
        except Exception as e:
            # Stop monitoring even if operation fails
            stats = self.stop_monitoring()
            logger.error(f"Operation failed during memory monitoring: {e}")
            raise e