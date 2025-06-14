#!/usr/bin/env python3
"""
🌍 Environment Configuration
=========================

Centralized configuration management for different deployment environments.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

# Constants
ENV_DEV = "development"
ENV_STAGING = "staging"
ENV_PROD = "production"

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

# Determine the environment
def get_environment() -> str:
    """Get the current environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env in [ENV_DEV, "dev"]:
        return ENV_DEV
    elif env in [ENV_STAGING, "stage", "test"]:
        return ENV_STAGING
    elif env in [ENV_PROD, "prod"]:
        return ENV_PROD
    else:
        logger.warning(f"Unknown environment '{env}', defaulting to development")
        return ENV_DEV

# Current environment
ENVIRONMENT = get_environment()

# Default configuration values
DEFAULT_CONFIG = {
    # API settings
    "api": {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "debug": False,
        "allowed_origins": os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
        "token_expiration": 86400,  # 24 hours
        "max_upload_size": 100 * 1024 * 1024,  # 100 MB
    },
    
    # Database settings
    "database": {
        "url": os.getenv("DATABASE_URL", "sqlite:///./tournament.db"),
        "pool_size": 5,
        "max_overflow": 10,
        "echo": False,
    },
    
    # Redis settings
    "redis": {
        "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        "timeout": 10,
    },
    
    # Model settings
    "models": {
        "directory": os.getenv("MODELS_DIR", "../models/deployment"),
        "default_model": "enhanced_cnn",
        "preload_models": True,
        "max_parallel_inferences": 2,
    },
    
    # Audio processing settings
    "audio": {
        "sample_rate": 44100,
        "formats": ["wav", "mp3", "flac", "ogg"],
        "max_duration": 600,  # 10 minutes
        "output_directory": "../mixed_outputs",
    },
    
    # Task system settings
    "tasks": {
        "max_retries": 3,
        "default_timeout": 300,  # 5 minutes
        "max_concurrent_tasks": 4,
    },
    
    # Monitoring settings
    "monitoring": {
        "enabled": True,
        "metrics_port": 9090,
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "request_logging": True,
    },
    
    # Storage settings
    "storage": {
        "type": "local",  # local, s3, azure
        "bucket": os.getenv("STORAGE_BUCKET", ""),
        "key_prefix": os.getenv("STORAGE_KEY_PREFIX", ""),
        "base_url": os.getenv("STORAGE_BASE_URL", ""),
    }
}

# Environment-specific configurations
ENV_CONFIGS = {
    ENV_DEV: {
        "api": {
            "debug": True,
            "allowed_origins": ["*"],
        },
        "database": {
            "echo": True,
        },
        "models": {
            "preload_models": False,
        },
        "monitoring": {
            "log_level": "DEBUG",
        },
    },
    
    ENV_STAGING: {
        "api": {
            "debug": False,
        },
        "database": {
            "pool_size": 10,
        },
        "tasks": {
            "max_concurrent_tasks": 8,
        },
    },
    
    ENV_PROD: {
        "api": {
            "debug": False,
        },
        "database": {
            "pool_size": 20,
            "max_overflow": 20,
        },
        "models": {
            "max_parallel_inferences": 4,
        },
        "tasks": {
            "max_concurrent_tasks": 16,
        },
        "monitoring": {
            "log_level": "INFO",
        },
    }
}

# Get merged configuration
def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration for the current environment
    
    Returns:
        Dict containing the merged configuration
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Apply environment-specific overrides
    env_config = ENV_CONFIGS.get(ENVIRONMENT, {})
    
    # Deep merge the environment config
    for key, value in env_config.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            # Merge dictionaries
            config[key] = {**config[key], **value}
        else:
            # Replace value
            config[key] = value
    
    # Log the configuration
    logger.info(f"Loaded configuration for environment: {ENVIRONMENT}")
    
    return config

# Current configuration
CONFIG = get_config()

# Helper functions to access config sections
def api_config() -> Dict[str, Any]:
    """Get API configuration"""
    return CONFIG["api"]

def db_config() -> Dict[str, Any]:
    """Get database configuration"""
    return CONFIG["database"]

def redis_config() -> Dict[str, Any]:
    """Get Redis configuration"""
    return CONFIG["redis"]

def models_config() -> Dict[str, Any]:
    """Get models configuration"""
    return CONFIG["models"]

def audio_config() -> Dict[str, Any]:
    """Get audio processing configuration"""
    return CONFIG["audio"]

def tasks_config() -> Dict[str, Any]:
    """Get task system configuration"""
    return CONFIG["tasks"]

def monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration"""
    return CONFIG["monitoring"]

def storage_config() -> Dict[str, Any]:
    """Get storage configuration"""
    return CONFIG["storage"]

# Function to get an individual config value
def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a specific configuration value using dot notation
    
    Args:
        key_path: Path to the config value (e.g., "api.port")
        default: Default value if the key is not found
        
    Returns:
        The configuration value or the default
    """
    keys = key_path.split(".")
    value = CONFIG
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

# Is this a production environment?
def is_production() -> bool:
    """Check if we're running in production"""
    return ENVIRONMENT == ENV_PROD

# Is this a development environment?
def is_development() -> bool:
    """Check if we're running in development"""
    return ENVIRONMENT == ENV_DEV

# Is this a staging environment?
def is_staging() -> bool:
    """Check if we're running in staging"""
    return ENVIRONMENT == ENV_STAGING


# Save the current configuration to a file (for debugging)
def save_config_to_file(file_path: str) -> None:
    """
    Save the current configuration to a file
    
    Args:
        file_path: Path to the file
    """
    try:
        with open(file_path, "w") as f:
            json.dump(CONFIG, f, indent=2)
        logger.info(f"Saved configuration to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}")


# If run directly, print the current configuration
if __name__ == "__main__":
    import sys
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Print the current configuration
    print(f"Environment: {ENVIRONMENT}")
    print(json.dumps(CONFIG, indent=2))
    
    # Save the configuration if a file path is provided
    if len(sys.argv) > 1:
        save_config_to_file(sys.argv[1])
