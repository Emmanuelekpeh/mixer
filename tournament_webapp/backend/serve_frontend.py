"""
Frontend serving utilities for the tournament web application.
This module provides functions to serve the React frontend from the FastAPI backend.
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

def setup_frontend_serving(app: FastAPI):
    """
    Configure the FastAPI app to serve the React frontend.
    
    Args:
        app: The FastAPI application instance
    """
    # Define the path to the frontend build directory
    frontend_build_dir = Path(__file__).parent.parent / "frontend" / "build"
    
    # Check if the frontend build directory exists
    if frontend_build_dir.exists():
        # Mount the static files directory
        app.mount("/static-frontend", StaticFiles(directory=str(frontend_build_dir / "static")), name="static-frontend")
        
        # Serve the index.html for any unmatched routes (SPA routing)
        @app.get("/app/{full_path:path}")
        async def serve_frontend(full_path: str):
            # Serve index.html for frontend routes
            index_path = frontend_build_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            else:
                return {"detail": "Frontend not built"}
                
        # Serve the root path as well
        @app.get("/app")
        async def serve_frontend_root():
            index_path = frontend_build_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            else:
                return {"detail": "Frontend not built"}
    else:
        # Log a warning if the frontend build directory doesn't exist
        print(f"Warning: Frontend build directory not found at {frontend_build_dir}")