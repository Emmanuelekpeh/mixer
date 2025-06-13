FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Debug directory structure
RUN ls -la
RUN ls -la tournament_webapp
RUN ls -la tournament_webapp/frontend || echo "Frontend directory not found"
RUN ls -la tournament_webapp/backend || echo "Backend directory not found"

# Set Python path to include the app directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose the port the app will run on
EXPOSE $PORT

# Set environment variables
ENV PRODUCTION=true
ENV MODELS_DIR=../models/deployment
ENV ALLOWED_ORIGINS=https://ai-mixer-tournament.onrender.com,http://localhost:3000

# Simple direct command that skips the dev_server.py script
CMD cd tournament_webapp/backend && uvicorn tournament_api:app --host 0.0.0.0 --port $PORT
