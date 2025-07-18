#!/usr/bin/env python3
"""
Simple Health Check API - Minimal FastAPI app for health checks
"""

from fastapi import FastAPI
from datetime import datetime
import os

# Create a minimal FastAPI app just for health checks
app = FastAPI(title="Health Check API")

@app.get("/health")
@app.get("/api/health")
@app.get("/")
async def health_check():
    """Simple health check that always works"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "message": "API is running"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)