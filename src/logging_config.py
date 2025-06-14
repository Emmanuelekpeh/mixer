#!/usr/bin/env python3
"""
📝 Centralized Logging
====================

Centralized logging configuration with structured logs and standardized formats.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pythonjsonlogger.jsonlogger import JsonFormatter

from src.environment_config import monitoring_config, is_production

# Get logging configuration
LOG_CONFIG = monitoring_config()
LOG_LEVEL = LOG_CONFIG["log_level"]

# Log directory
LOG_DIR = Path("logs")
if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file path
LOG_FILE = LOG_DIR / "application.log"

# Create custom JSON formatter with additional fields
class CustomJsonFormatter(JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add additional fields to the log record"""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Add log level name
        log_record["level"] = record.levelname
        
        # Add logger name
        log_record["logger"] = record.name
        
        # Add environment information
        log_record["environment"] = os.getenv("ENVIRONMENT", "development")
        
        # Add process and thread information
        log_record["process"] = record.process
        log_record["thread"] = record.thread
        
        # Add file and line information
        log_record["file"] = record.pathname
        log_record["line"] = record.lineno
        
        # Add application version if available
        log_record["version"] = os.getenv("APP_VERSION", "unknown")


# Initialize logging
def initialize_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Initialize logging configuration
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path
    """
    # Determine log level
    level = log_level or LOG_LEVEL
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Determine log file
    log_path = log_file or LOG_FILE
    
    # Create formatters
    json_formatter = CustomJsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s",
        json_ensure_ascii=False
    )
    
    console_formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(numeric_level)
    
    # Create file handler for JSON logs
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(numeric_level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configure library loggers
    for module in ["uvicorn", "sqlalchemy", "asyncio", "httpx"]:
        logging.getLogger(module).setLevel(
            logging.WARNING if is_production() else logging.INFO
        )
    
    # Log initialization
    logging.info(
        "Logging initialized",
        extra={
            "log_level": level,
            "log_file": str(log_path)
        }
    )


# Exception logging utility
def log_exception(exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an exception with additional context
    
    Args:
        exception: The exception to log
        context: Additional context information
    """
    context = context or {}
    
    # Add exception information to context
    context.update({
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "exception_traceback": bool(exception.__traceback__)
    })
    
    # Log the exception
    logging.error(
        f"Exception: {str(exception)}",
        exc_info=True,
        extra=context
    )


# If run directly, initialize logging
if __name__ == "__main__":
    initialize_logging()
    
    # Test logging
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    
    try:
        # Generate an exception
        result = 1 / 0
    except Exception as e:
        log_exception(e, {"operation": "division_test"})
