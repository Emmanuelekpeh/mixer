"""
Health check and system verification utilities.
"""

import os
import sys
import logging
from pathlib import Path
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("health_check")

def verify_system():
    """
    Verify system components before starting the application.
    This helps catch configuration issues early.
    """
    logger.info("Starting system verification...")
    
    # Check environment
    env = os.environ.get("ENVIRONMENT", "development")
    logger.info(f"Environment: {env}")
    
    # Check Python path
    logger.info(f"Python path: {sys.path}")
    
    # Check working directory
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check models directory
    models_dir = os.environ.get("MODELS_DIR", "models/deployment")
    models_path = Path(models_dir)
    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_path}")
        # Create directory to prevent errors
        try:
            models_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created models directory: {models_path}")
        except Exception as e:
            logger.error(f"Failed to create models directory: {e}")
    else:
        logger.info(f"Models directory exists: {models_path}")
        # Count model files
        model_files = list(models_path.glob("**/*.pth"))
        logger.info(f"Found {len(model_files)} model files")
    
    # Check database directory
    db_path = Path("tournament_webapp/backend/tournament.db")
    if not db_path.parent.exists():
        logger.warning(f"Database directory not found: {db_path.parent}")
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created database directory: {db_path.parent}")
        except Exception as e:
            logger.error(f"Failed to create database directory: {e}")
    
    # Check for required directories
    required_dirs = [
        "logs",
        "data/mixed_outputs",
        "tournament_webapp/uploads"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            logger.warning(f"Required directory not found: {path}")
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Failed to create directory {path}: {e}")
    
    logger.info("System verification complete")
    return True

def check_health():
    """
    Perform a health check of the system.
    Returns a dictionary with health status information.
    """
    status = {
        "status": "ok",
        "components": {}
    }
    
    # Check models
    try:
        models_dir = os.environ.get("MODELS_DIR", "models/deployment")
        models_path = Path(models_dir)
        model_files = list(models_path.glob("**/*.pth"))
        status["components"]["models"] = {
            "status": "ok" if model_files else "warning",
            "count": len(model_files)
        }
    except Exception as e:
        status["components"]["models"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check database
    try:
        db_path = Path("tournament_webapp/backend/tournament.db")
        status["components"]["database"] = {
            "status": "ok" if db_path.exists() else "warning",
            "path": str(db_path)
        }
    except Exception as e:
        status["components"]["database"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Overall status
    component_statuses = [c["status"] for c in status["components"].values()]
    if "error" in component_statuses:
        status["status"] = "degraded"
    elif "warning" in component_statuses:
        status["status"] = "warning"
    
    return status

if __name__ == "__main__":
    # Can be run directly for testing
    verify_system()
    health_status = check_health()
    print(health_status)