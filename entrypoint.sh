#!/bin/bash
set -e

echo "🚀 Starting AI Mixer Tournament Application..."

# Set Python path to include the app directory and backend directory
export PYTHONPATH=/app:/app/tournament_webapp/backend:/app/tournament_webapp

# Add local bin to PATH for user-installed packages
export PATH=$PATH:/home/appuser/.local/bin:/root/.local/bin

# Set the port from environment variable or default to 8000
export PORT=${PORT:-8000}

# First, try to start the simple health check to ensure the service is responsive
echo "🏥 Starting health check service..."
python -c "
import sys
import os
sys.path.append('/app')
sys.path.append('/app/tournament_webapp/backend')

# Test if we can import basic modules
try:
    from fastapi import FastAPI
    from datetime import datetime
    print('✅ Basic imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
" &

# Give it a moment to start
sleep 2

# Try to initialize the full application
echo "🔍 Attempting full application startup..."

# Initialize database (non-blocking)
echo "🗄️  Initializing database..."
python -c "
import sys
sys.path.append('/app')
sys.path.append('/app/tournament_webapp/backend')
try:
    from tournament_webapp.backend.database import init_database, get_database_stats
    init_database()
    stats = get_database_stats()
    print(f'📊 Database initialized: {stats}')
except Exception as e:
    print(f'⚠️  Database initialization warning: {e}')
    print('🔄 Continuing without database...')
" || echo "⚠️  Database setup failed, continuing..."

# Verify models directory exists
echo "📁 Checking models directory..."
mkdir -p /app/models

# List available model files for debugging
echo "🤖 Available model files:"
find /app/models -name "*.pth" -type f 2>/dev/null | head -5 || echo "No .pth files found"

# Try to start the full application, fall back to simple health check if it fails
echo "🌐 Starting uvicorn server on port $PORT..."

# Try the full application first
if python -c "
import sys
sys.path.append('/app')
sys.path.append('/app/tournament_webapp/backend')
try:
    from tournament_webapp.backend.tournament_api import app
    print('✅ Full application imports successful')
    exit(0)
except Exception as e:
    print(f'❌ Full application import failed: {e}')
    exit(1)
"; then
    echo "🎯 Starting full tournament application..."
    exec python -m uvicorn tournament_webapp.backend.tournament_api:app \
        --host 0.0.0.0 \
        --port $PORT \
        --workers 1 \
        --log-level info \
        --access-log
else
    echo "⚠️  Full application failed, starting simple health check..."
    exec python -m uvicorn tournament_webapp.backend.simple_health_check:app \
        --host 0.0.0.0 \
        --port $PORT \
        --workers 1 \
        --log-level info \
        --access-log
fi