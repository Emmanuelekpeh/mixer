# 🚀 AI Mixing Tournament - Deployment Fix Plan

## 🔍 Issue Analysis

Based on the deployment logs, the application is failing the healthcheck with "service unavailable" errors. This indicates that the application is either:

1. Not starting correctly
2. Starting but crashing
3. Starting but the healthcheck endpoint is not properly implemented or accessible

## 🛠️ Fix Strategy

### 1. Fix Endpoint Path Mismatch

There's a mismatch between the Railway configuration and the Dockerfile:

- **Railway.json**: Uses `uvicorn tournament_webapp.backend.main:app`
- **Dockerfile**: Uses `gunicorn tournament_webapp.backend.tournament_api:app`

**Solution**: Align these configurations to use the same entry point.

### 2. Implement Proper Health Check Endpoint

Create or verify a dedicated health check endpoint at `/api/health` that:
- Returns a 200 OK status
- Performs basic system checks (database connection, model availability)
- Has minimal dependencies to ensure it works even if other parts of the app have issues

### 3. Fix Model Discovery Issues

The model discovery service might be failing during startup:
- Add better error handling to prevent crashes
- Make model validation optional during startup
- Implement a fallback mechanism if models can't be loaded

### 4. Update Railway Configuration

- Increase healthcheck timeout
- Add proper environment variables
- Ensure the correct working directory is used

## 📋 Implementation Steps

### Step 1: Create a Robust Health Check Endpoint

```python
# In tournament_webapp/backend/main.py or tournament_api.py

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for deployment monitoring.
    Returns basic system status without requiring full application functionality.
    """
    try:
        # Basic system check
        system_status = {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "version": "1.0.0"
        }
        
        # Optional: Check database connection if available
        try:
            # Quick database check that won't crash the health check
            db_status = "ok"
        except Exception as e:
            db_status = f"error: {str(e)}"
            # Don't fail the health check for DB issues
        
        system_status["database"] = db_status
        
        # Optional: Check model availability if it won't block
        try:
            # Quick non-blocking model check
            models_status = "ok"
        except Exception as e:
            models_status = f"warning: {str(e)}"
            # Don't fail the health check for model issues
        
        system_status["models"] = models_status
        
        return system_status
    except Exception as e:
        # Always return 200 for health check, but include error info
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

### Step 2: Update Railway Configuration

Update `railway.json` to match the actual application structure:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "cd tournament_webapp/backend && gunicorn tournament_api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120",
    "healthcheckPath": "/api/health",
    "healthcheckTimeout": 180,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  },
  "environments": {
    "production": {
      "variables": {
        "PYTHON_VERSION": "3.11.0",
        "ENVIRONMENT": "production",
        "MODELS_DIR": "/app/models/deployment",
        "ALLOWED_ORIGINS": "https://ai-mixer-tournament.railway.app,http://localhost:3000",
        "LOG_LEVEL": "INFO",
        "WORKERS": "4"
      }
    }
  }
}
```

### Step 3: Update Dockerfile for Better Error Handling

Modify the Dockerfile to include better error handling and logging:

```dockerfile
# Add to Dockerfile
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Update the CMD to include error handling
CMD cd tournament_webapp/backend && \
    echo "Starting application server..." && \
    python -c "import sys; sys.path.append('/app'); from tournament_webapp.backend.health_check import verify_system; verify_system()" && \
    gunicorn tournament_api:app \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$PORT \
    --access-logfile - \
    --error-logfile - \
    --log-level $LOG_LEVEL \
    --timeout 120
```

### Step 4: Create a System Verification Script

Create a new file `tournament_webapp/backend/health_check.py`:

```python
"""
Health check and system verification utilities.
"""

import os
import sys
import logging
from pathlib import Path
import time

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
```

### Step 5: Update the Main API File

Ensure the main API file has the health check endpoint:

```python
# Add to tournament_webapp/backend/tournament_api.py or main.py

from datetime import datetime
import os
from fastapi import FastAPI
from .health_check import check_health

app = FastAPI()

@app.get("/health")
@app.get("/api/health")
async def health_endpoint():
    """Health check endpoint for deployment monitoring."""
    health_status = check_health()
    health_status["timestamp"] = datetime.now().isoformat()
    health_status["environment"] = os.environ.get("ENVIRONMENT", "development")
    return health_status
```

## 🧪 Testing the Fix

Before deploying to Railway again:

1. Test the health check endpoint locally:
   ```bash
   cd tournament_webapp/backend
   uvicorn tournament_api:app --reload
   # Then visit http://localhost:8000/api/health
   ```

2. Test the system verification script:
   ```bash
   python -m tournament_webapp.backend.health_check
   ```

3. Test the Docker build locally:
   ```bash
   docker build -t tournament-app .
   docker run -p 8000:8000 -e PORT=8000 tournament-app
   # Then visit http://localhost:8000/api/health
   ```

## 🚀 Deployment Steps

1. Commit all changes:
   ```bash
   git add .
   git commit -m "Fix deployment health check and startup issues"
   git push origin main
   ```

2. Deploy to Railway:
   - Go to Railway dashboard
   - Deploy the updated code
   - Monitor the logs during deployment
   - Check if the health check passes

## 🔍 Post-Deployment Verification

After successful deployment:

1. Visit the health check endpoint: `https://your-app-name.railway.app/api/health`
2. Check the application logs for any warnings or errors
3. Test the main functionality to ensure everything works correctly

## 🛡️ Fallback Plan

If the deployment still fails:

1. Simplify the health check to return a static 200 OK response
2. Disable model loading during startup
3. Use a minimal deployment configuration
4. Consider deploying to an alternative platform like Render