FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user gunicorn uvicorn

# Create a clean production image with only runtime dependencies
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy the application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data/mixed_outputs /app/tournament_webapp/uploads && \
    chmod -R 755 /app/logs /app/data/mixed_outputs /app/tournament_webapp/uploads

# Create non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set Python path to include the app directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose the port the app will run on
EXPOSE $PORT

# Set environment variables
ENV PRODUCTION=true
ENV MODELS_DIR=/app/models/deployment
ENV ALLOWED_ORIGINS=https://ai-mixer-tournament.onrender.com,https://ai-mixer-tournament.railway.app,http://localhost:3000
ENV LOG_LEVEL=INFO
ENV WORKERS=4
ENV PYTHONUNBUFFERED=1

# Configure Gunicorn for production with improved error handling
CMD echo "Starting application server..." && \
    python -c "import sys; sys.path.append('/app'); from tournament_webapp.backend.health_check import verify_system; verify_system()" && \
    gunicorn "tournament_webapp.backend.tournament_api:app" \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$PORT \
    --access-logfile - \
    --error-logfile - \
    --log-level $LOG_LEVEL \
    --timeout 120
