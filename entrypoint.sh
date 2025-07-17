#!/bin/bash
set -e

echo "Starting application server..."
python -c "import sys; sys.path.append('/app'); from tournament_webapp.backend.health_check import verify_system; verify_system()"

# Run the application
python -m uvicorn tournament_webapp.backend.tournament_api:app --host 0.0.0.0 --port $PORT