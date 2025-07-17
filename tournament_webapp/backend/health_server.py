"""
Standalone health check server for deployment monitoring.
This file can be run directly to provide a minimal health check endpoint.
"""

import os
import sys
from fastapi import FastAPI
from datetime import datetime
import uvicorn

# Create a minimal FastAPI app for health checks
app = FastAPI(
    title="Health Check Server",
    description="Minimal server for deployment health checks",
    version="1.0.0"
)

@app.get("/")
@app.get("/health")
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for deployment monitoring.
    Returns basic system status without requiring full application functionality.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    print(f"Starting health check server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)