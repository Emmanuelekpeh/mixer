#!/bin/bash
set -e

echo "🎧 Starting AI Mixer Inference Worker..."

# Ensure Python path includes project directories
export PYTHONPATH=/root/.local/lib/python3.10/site-packages:/app:/app/tournament_webapp/backend:$PYTHONPATH
export PATH=$PATH:/home/appuser/.local/bin:/root/.local/bin

# Default to redis:6379 if not provided
export REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}

# Log Redis URL host (mask password if any)
masked="${REDIS_URL%@*}" && echo "🔗 Connecting to Redis at ${masked##*://}"

exec python tournament_webapp/backend/inference_worker.py 