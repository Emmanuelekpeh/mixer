#!/usr/bin/env python3
"""
🔄 Asynchronous Task Manager
===========================

Handles background processing for the AI mixer application.
Provides task queuing, progress tracking, and notifications
for long-running operations like audio processing and model training.
"""

import asyncio
import logging
import uuid
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    """Task status enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult:
    """Task result object"""
    def __init__(self, 
                 task_id: str, 
                 status: TaskStatus, 
                 result: Any = None, 
                 error: Optional[str] = None,
                 progress: float = 0.0,
                 message: Optional[str] = None):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.progress = progress
        self.message = message
        self.updated_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to dictionary"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "message": self.message,
            "updated_at": self.updated_at
        }


class AsyncTaskManager:
    """
    Manages asynchronous tasks for the AI Mixer application.
    Provides task queuing, progress tracking, and notifications.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the task manager
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.tasks: Dict[str, TaskResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self.notification_callbacks: Dict[str, List[Callable[[TaskResult], Awaitable[None]]]] = {}
        
        # Create a directory for persisting task results
        self.tasks_dir = Path("task_results")
        self.tasks_dir.mkdir(exist_ok=True)
        
        logger.info(f"AsyncTaskManager initialized with {max_workers} workers")
    
    async def create_task(self, 
                          task_func: Callable[..., Any], 
                          *args, 
                          task_id: Optional[str] = None, 
                          **kwargs) -> str:
        """
        Create a new task and queue it for execution
        
        Args:
            task_func: Function to execute
            *args: Arguments to pass to the function
            task_id: Optional task ID (generated if not provided)
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Create task result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0.0,
            message="Task queued"
        )
        
        # Store task result
        with self.lock:
            self.tasks[task_id] = task_result
        
        # Create progress callback
        def progress_callback(progress: float, message: Optional[str] = None):
            self.update_task_progress(task_id, progress, message)
        
        # Queue task for execution
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor, 
            self._run_task, 
            task_func, 
            task_id, 
            progress_callback,
            args, 
            kwargs
        )
        
        # Create task to handle completion
        asyncio.create_task(self._handle_task_completion(task_id, future))
        
        logger.info(f"Task {task_id} created")
        return task_id
    
    def _run_task(self, 
                 task_func: Callable[..., Any], 
                 task_id: str, 
                 progress_callback: Callable[[float, Optional[str]], None],
                 args: tuple, 
                 kwargs: dict) -> Any:
        """
        Run a task in a worker thread
        
        Args:
            task_func: Function to execute
            task_id: Task ID
            progress_callback: Callback for progress updates
            args: Arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
            
        Returns:
            Task result
        """
        try:
            # Update task status
            with self.lock:
                task_result = self.tasks[task_id]
                task_result.status = TaskStatus.RUNNING
                task_result.message = "Task started"
                task_result.updated_at = datetime.now().isoformat()
            
            # Notify of status change
            self._notify_task_update(task_result)
            
            # Add progress callback to kwargs
            kwargs["progress_callback"] = progress_callback
            
            # Execute task
            start_time = time.time()
            result = task_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            
            # Update task status
            with self.lock:
                task_result = self.tasks[task_id]
                task_result.status = TaskStatus.FAILED
                task_result.error = str(e)
                task_result.message = f"Task failed: {str(e)}"
                task_result.updated_at = datetime.now().isoformat()
            
            # Notify of status change
            self._notify_task_update(task_result)
            
            # Re-raise exception
            raise
    
    async def _handle_task_completion(self, task_id: str, future: asyncio.Future):
        """
        Handle task completion
        
        Args:
            task_id: Task ID
            future: Future for the task
        """
        try:
            # Wait for task to complete
            result = await future
            
            # Update task status
            with self.lock:
                task_result = self.tasks[task_id]
                task_result.status = TaskStatus.COMPLETED
                task_result.result = result
                task_result.progress = 1.0
                task_result.message = "Task completed successfully"
                task_result.updated_at = datetime.now().isoformat()
            
            # Save task result
            self._persist_task_result(task_id, task_result)
            
            # Notify of status change
            await self._notify_task_update_async(task_result)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed in completion handler: {str(e)}", exc_info=True)
            
            # Update task status if not already failed
            with self.lock:
                task_result = self.tasks[task_id]
                if task_result.status != TaskStatus.FAILED:
                    task_result.status = TaskStatus.FAILED
                    task_result.error = str(e)
                    task_result.message = f"Task failed: {str(e)}"
                    task_result.updated_at = datetime.now().isoformat()
            
            # Save task result
            self._persist_task_result(task_id, task_result)
            
            # Notify of status change
            await self._notify_task_update_async(task_result)
    
    def update_task_progress(self, 
                             task_id: str, 
                             progress: float, 
                             message: Optional[str] = None):
        """
        Update task progress
        
        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            message: Optional status message
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for progress update")
            return
        
        # Update task status
        with self.lock:
            task_result = self.tasks[task_id]
            task_result.progress = max(0.0, min(1.0, progress))  # Clamp to 0-1 range
            if message:
                task_result.message = message
            task_result.updated_at = datetime.now().isoformat()
        
        # Notify of status change
        self._notify_task_update(task_result)
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """
        Get task status
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if not found
        """
        with self.lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for cancellation")
            return False
        
        # Update task status
        with self.lock:
            task_result = self.tasks[task_id]
            if task_result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                logger.warning(f"Task {task_id} already finished, cannot cancel")
                return False
            
            task_result.status = TaskStatus.CANCELLED
            task_result.message = "Task cancelled"
            task_result.updated_at = datetime.now().isoformat()
        
        # Notify of status change
        self._notify_task_update(task_result)
        
        logger.info(f"Task {task_id} cancelled")
        return True
    
    def register_notification_callback(self, 
                                      task_id: str, 
                                      callback: Callable[[TaskResult], Awaitable[None]]):
        """
        Register a callback for task notifications
        
        Args:
            task_id: Task ID
            callback: Async callback function that takes a TaskResult
        """
        with self.lock:
            if task_id not in self.notification_callbacks:
                self.notification_callbacks[task_id] = []
            
            self.notification_callbacks[task_id].append(callback)
    
    def _notify_task_update(self, task_result: TaskResult):
        """
        Notify task update via callbacks (non-async version)
        
        Args:
            task_result: Task result
        """
        # Run async notification in background
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self._notify_task_update_async(task_result))
        else:
            # We're not in an async context, so just log
            logger.debug(f"Task {task_result.task_id} updated: {task_result.status}, {task_result.progress:.0%}")
    
    async def _notify_task_update_async(self, task_result: TaskResult):
        """
        Notify task update via callbacks (async version)
        
        Args:
            task_result: Task result
        """
        task_id = task_result.task_id
        
        # Get callbacks
        callbacks = []
        with self.lock:
            callbacks = self.notification_callbacks.get(task_id, []).copy()
        
        # Call callbacks
        for callback in callbacks:
            try:
                await callback(task_result)
            except Exception as e:
                logger.error(f"Error in task notification callback: {str(e)}", exc_info=True)
        
        # Remove callbacks for completed or failed tasks
        if task_result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            with self.lock:
                self.notification_callbacks.pop(task_id, None)
    
    def _persist_task_result(self, task_id: str, task_result: TaskResult):
        """
        Persist task result to disk
        
        Args:
            task_id: Task ID
            task_result: Task result
        """
        try:
            # Convert result to JSON-serializable dict
            result_dict = task_result.to_dict()
            
            # Remove large results to avoid disk space issues
            if isinstance(result_dict.get("result"), (dict, list)) and len(str(result_dict["result"])) > 1000:
                result_dict["result"] = {"message": "Result too large to persist, access via API"}
            
            # Save to file
            file_path = self.tasks_dir / f"{task_id}.json"
            with open(file_path, "w") as f:
                json.dump(result_dict, f)
                
        except Exception as e:
            logger.error(f"Error persisting task result: {str(e)}", exc_info=True)
    
    def load_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Load task result from disk
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if not found
        """
        try:
            file_path = self.tasks_dir / f"{task_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, "r") as f:
                result_dict = json.load(f)
            
            # Create TaskResult from dict
            return TaskResult(
                task_id=result_dict["task_id"],
                status=TaskStatus(result_dict["status"]),
                result=result_dict["result"],
                error=result_dict["error"],
                progress=result_dict["progress"],
                message=result_dict["message"]
            )
                
        except Exception as e:
            logger.error(f"Error loading task result: {str(e)}", exc_info=True)
            return None
    
    def get_all_tasks(self) -> List[TaskResult]:
        """
        Get all active tasks
        
        Returns:
            List of task results
        """
        with self.lock:
            return list(self.tasks.values())


# Global task manager instance
task_manager = AsyncTaskManager()


# Example usage functions
async def example_task(iterations: int, delay: float = 0.5, progress_callback=None):
    """Example task that simulates long processing with progress updates"""
    result = {"processed_items": []}
    
    for i in range(iterations):
        # Simulate processing
        await asyncio.sleep(delay)
        
        # Update progress
        if progress_callback:
            progress = (i + 1) / iterations
            progress_callback(progress, f"Processed item {i+1}/{iterations}")
        
        result["processed_items"].append(f"Item {i+1}")
    
    return result


async def example_usage():
    """Example usage of the AsyncTaskManager"""
    # Create a task
    task_id = await task_manager.create_task(example_task, 10, 0.2)
    
    # Register notification callback
    async def on_task_update(task_result):
        print(f"Task {task_result.task_id} updated: {task_result.status}, {task_result.progress:.0%}, {task_result.message}")
    
    task_manager.register_notification_callback(task_id, on_task_update)
    
    # Poll task status
    while True:
        task_result = task_manager.get_task_status(task_id)
        if task_result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            break
        await asyncio.sleep(0.5)
    
    # Get final result
    task_result = task_manager.get_task_status(task_id)
    print(f"Task completed with status: {task_result.status}")
    print(f"Result: {task_result.result}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
