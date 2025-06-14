#!/usr/bin/env python3
"""
🔄 Asynchronous Task Management
============================

Implementation of the async task system for the tournament API
Enables asynchronous processing for audio mixing and tournament battles
"""

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from fastapi import BackgroundTasks
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task status constants
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"

# Task storage
tasks = {}
task_results = {}
task_progress = {}

# Progress tracking
class TaskProgress(BaseModel):
    task_id: str
    status: str = TASK_PENDING
    progress: float = 0.0
    message: Optional[str] = None
    created_at: str = None
    updated_at: str = None
    completed_at: Optional[str] = None


def get_task_status(task_id: str) -> Optional[TaskProgress]:
    """Get the status of a task"""
    return task_progress.get(task_id)


def create_task_record(task_id: str, task_type: str) -> TaskProgress:
    """Create a new task record"""
    now = datetime.now().isoformat()
    progress = TaskProgress(
        task_id=task_id,
        status=TASK_PENDING,
        progress=0.0,
        message=f"Task {task_type} created",
        created_at=now,
        updated_at=now
    )
    task_progress[task_id] = progress
    return progress


def update_task_progress(task_id: str, progress: float, message: str = None, status: str = None):
    """Update task progress"""
    if task_id not in task_progress:
        logger.warning(f"Task {task_id} not found for progress update")
        return

    task = task_progress[task_id]
    task.progress = progress
    task.updated_at = datetime.now().isoformat()
    
    if message:
        task.message = message
    
    if status:
        task.status = status
        if status == TASK_COMPLETED or status == TASK_FAILED:
            task.completed_at = datetime.now().isoformat()
    
    logger.info(f"Task {task_id} progress: {progress:.0%} - {message}")


def create_progress_callback(task_id: str) -> Callable:
    """Create a progress callback function for a task"""
    def progress_callback(progress: float, message: str = None):
        update_task_progress(task_id, progress, message)
    return progress_callback


async def run_task_async(
    task_id: str,
    task_func: Callable,
    *args,
    **kwargs
) -> Any:
    """Run a task asynchronously"""
    try:
        # Update task status to running
        update_task_progress(task_id, 0.0, "Task started", TASK_RUNNING)
        
        # Create progress callback
        progress_callback = create_progress_callback(task_id)
        kwargs["progress_callback"] = progress_callback
        
        # Start task execution time
        start_time = time.time()
        
        # Run the task
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: task_func(*args, **kwargs)
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Update task status to completed
        update_task_progress(
            task_id,
            1.0,
            f"Task completed in {execution_time:.2f}s",
            TASK_COMPLETED
        )
        
        # Store result
        task_results[task_id] = result
        
        return result
    
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
        
        # Update task status to failed
        update_task_progress(
            task_id,
            1.0,
            f"Task failed: {str(e)}",
            TASK_FAILED
        )
        
        # Store error
        task_results[task_id] = {"error": str(e)}
        
        # Re-raise the exception
        raise


def schedule_task(
    background_tasks: BackgroundTasks,
    task_func: Callable,
    task_type: str,
    *args,
    task_id: Optional[str] = None,
    **kwargs
) -> str:
    """Schedule a task to run in the background"""
    # Generate task ID if not provided
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    # Create task record
    create_task_record(task_id, task_type)
    
    # Schedule task
    background_tasks.add_task(
        run_task_async,
        task_id,
        task_func,
        *args,
        **kwargs
    )
    
    logger.info(f"Scheduled {task_type} task {task_id}")
    
    return task_id


# Test utilities
if __name__ == "__main__":
    async def main():
        # Test task
        def test_task(iterations=10, delay=0.1, progress_callback=None):
            result = {"steps": []}
            for i in range(iterations):
                time.sleep(delay)
                if progress_callback:
                    progress = (i + 1) / iterations
                    progress_callback(progress, f"Step {i+1}/{iterations}")
                result["steps"].append(f"Step {i+1}")
            return result
        
        # Create BackgroundTasks
        background_tasks = BackgroundTasks()
        
        # Schedule task
        task_id = schedule_task(
            background_tasks,
            test_task,
            "test",
            iterations=5,
            delay=0.2
        )
        
        # Execute tasks
        await background_tasks()
        
        # Get result
        result = task_results.get(task_id)
        print(f"Task {task_id} result: {result}")
    
    asyncio.run(main())
