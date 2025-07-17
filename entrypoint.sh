#!/bin/bash
set -e

echo "Starting application server..."
python -c "import sys; sys.path.append('/app'); from tournament_webapp.backend.health_check import verify_system; verify_system()"

# Install required packages if not already installed
pip install uvicorn fastapi

# Add local bin to PATH
export PATH=$PATH:/home/appuser/.local/bin

# Set Python path to include the app directory and backend directory
export PYTHONPATH=/app:/app/tournament_webapp/backend

# Run a simple web server for testing
cd /app
python -c "
import http.server
import socketserver

PORT = int('${PORT:-8000}')
Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(('0.0.0.0', PORT), Handler)
print('Serving at port', PORT)
httpd.serve_forever()
"