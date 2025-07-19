#!/bin/bash
set -e

echo "🛠️  Starting Model Manager Service..."
export PYTHONPATH=/root/.local/lib/python3.10/site-packages:/app:$PYTHONPATH
export REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}

exec python -m uvicorn tournament_webapp.model_manager:app --host 0.0.0.0 --port ${MODEL_MANAGER_PORT:-8090} --workers 1 