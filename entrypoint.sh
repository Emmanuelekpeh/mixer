#!/bin/bash
set -e

echo "Starting application server..."
python -c "import sys; sys.path.append('/app'); from tournament_webapp.backend.health_check import verify_system; verify_system()"

# Install required packages if not already installed
pip install uvicorn fastapi

# Run the application using python directly
cd /app
python -c "
import uvicorn
from tournament_webapp.backend.tournament_api import app
uvicorn.run(app, host='0.0.0.0', port=int('${PORT:-8000}'))
"