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

# Expose the port the app will run on
EXPOSE $PORT

# Set environment variables
ENV PRODUCTION=true
ENV MODELS_DIR=../models/deployment
ENV ALLOWED_ORIGINS=https://ai-mixer-tournament.onrender.com,http://localhost:3000

# Command to run the application - primary approach
CMD cd tournament_webapp && python dev_server.py --production || cd tournament_webapp/backend && uvicorn tournament_api:app --host 0.0.0.0 --port $PORT
