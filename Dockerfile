FROM node:16-slim AS frontend-builder

WORKDIR /app/frontend

# Copy frontend files
COPY tournament_webapp/frontend/package*.json ./
RUN npm install

COPY tournament_webapp/frontend/ ./
RUN npm run build

FROM python:3.10-slim AS backend-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Remove any dev SQLite database to start fresh
RUN rm -f tournament_webapp/backend/data/tournament.db || true

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user gunicorn uvicorn[standard] && \
    pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user librosa matplotlib

# Create a clean production image with only runtime dependencies
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from backend-builder
COPY --from=backend-builder /root/.local /root/.local
# Duplicate packages to appuser's home to avoid permission issues
RUN mkdir -p /home/appuser/.local && cp -r /root/.local/* /home/appuser/.local/ && chmod -R 755 /home/appuser/.local

# Update environment for appuser
ENV PATH=/home/appuser/.local/bin:/root/.local/bin:$PATH
ENV PYTHONPATH="/home/appuser/.local/lib/python3.10/site-packages:/root/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Copy frontend build from frontend-builder
COPY --from=frontend-builder /app/frontend/build /app/tournament_webapp/frontend/build
# Create a symlink for easier access
RUN mkdir -p /app/tournament_webapp/frontend && \
    ln -sf /app/tournament_webapp/frontend/build /app/frontend-build

# Copy the application code and entrypoint script
COPY . .
COPY entrypoint.sh /app/entrypoint.sh

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data/mixed_outputs /app/tournament_webapp/uploads /app/models && \
    chmod -R 755 /app/logs /app/data/mixed_outputs /app/tournament_webapp/uploads /app/models && \
    # Install gunicorn and uvicorn system-wide for all users
    pip install --no-cache-dir gunicorn uvicorn[standard] && \
    # Make entrypoint executable
    chmod +x /app/entrypoint.sh

# Create non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set Python path to include the app directory
ENV PYTHONPATH="/root/.local/lib/python3.10/site-packages:${PYTHONPATH}"

# Expose the port the app will run on
EXPOSE $PORT

# Set environment variables
ENV PRODUCTION=true
ENV MODELS_DIR=/app/models
ENV ALLOWED_ORIGINS=https://ai-mixer-tournament.onrender.com,https://ai-mixer-tournament.railway.app,http://localhost:3000
ENV LOG_LEVEL=INFO
ENV WORKERS=4
ENV PYTHONUNBUFFERED=1
ENV MODEL_ROOT=/app/models/deployment

# Configure entrypoint
CMD ["/app/entrypoint.sh"]
