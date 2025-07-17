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

# JSON logging is handled by our custom formatter
JSON_LOGGING_AVAILABLE = True

# JSON is part of Python standard library, so this should always be available
# But we keep the pattern for consistency with other optional imports
try:
    import json
    JSON_LOGGING_AVAILABLE = True
except ImportError:
    JSON_LOGGING_AVAILABLE = False

try:
    from src.environment_config import monitoring_config, is_production
    # Get logging configuration
    LOG_CONFIG = monitoring_config()
    LOG_LEVEL = LOG_CONFIG["log_level"]
except ImportError:
    # Fallback configuration if environment_config is not available
    LOG_LEVEL = "INFO"
    LOG_CONFIG = {"log_level": "INFO"}

# Fallback is_production function if not imported
if 'is_production' not in globals():
    def is_production():
        return os.getenv("ENVIRONMENT", "development") == "production"

# Log directory
LOG_DIR = Path("logs")
if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file path
LOG_FILE = LOG_DIR / "application.log"

# Create simple JSON formatter without external dependencies
class SimpleJsonFormatter(logging.Formatter):
    """Simple JSON formatter without external dependencies"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "process": record.process,
            "thread": record.thread,
            "file": record.pathname,
            "line": record.lineno,
            "version": os.getenv("APP_VERSION", "unknown")
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


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
    json_formatter = SimpleJsonFormatter()
    
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
