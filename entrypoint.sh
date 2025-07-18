#!/bin/bash
set -e

echo "🚀 Starting AI Mixer Tournament Application..."

# Set Python path to include the app directory and backend directory
export PYTHONPATH=/app:/app/tournament_webapp/backend:/app/tournament_webapp

# Add local bin to PATH for user-installed packages
export PATH=$PATH:/home/appuser/.local/bin:/root/.local/bin

# Verify system health
echo "🔍 Verifying system health..."
python -c "
import sys
sys.path.append('/app')
sys.path.append('/app/tournament_webapp/backend')
try:
    from tournament_webapp.backend.health_check import verify_system
    verify_system()
    print('✅ System verification passed')
except Exception as e:
    print(f'⚠️  System verification warning: {e}')
    print('🔄 Continuing with startup...')
"

# Initialize database
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
    print(f'❌ Database initialization failed: {e}')
    exit(1)
"

# Verify models directory exists
echo "📁 Checking models directory..."
if [ ! -d "/app/models" ]; then
    echo "⚠️  Models directory not found, creating..."
    mkdir -p /app/models
fi

# List available model files for debugging
echo "🤖 Available model files:"
find /app/models -name "*.pth" -type f 2>/dev/null | head -10 || echo "No .pth files found"

# Set the port from environment variable or default to 8000
export PORT=${PORT:-8000}

# Run the application using uvicorn
echo "🌐 Starting uvicorn server on port $PORT..."
exec python -m uvicorn tournament_webapp.backend.tournament_api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --log-level info \
    --access-log