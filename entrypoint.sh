#!/bin/bash
set -e

echo "Starting application server..."
python -c "import sys; sys.path.append('/app'); from tournament_webapp.backend.health_check import verify_system; verify_system()"

# Install required packages if not already installed
pip install uvicorn fastapi sqlalchemy python-dotenv python-multipart numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install librosa matplotlib

# Add local bin to PATH
export PATH=$PATH:/home/appuser/.local/bin

# Set Python path to include the app directory and backend directory
export PYTHONPATH=/app:/app/tournament_webapp/backend

# Run the application using uvicorn (without changing directory)
echo "Starting uvicorn server..."
python -m uvicorn tournament_webapp.backend.tournament_api:app --host 0.0.0.0 --port ${PORT:-8000}