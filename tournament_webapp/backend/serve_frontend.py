"""
Frontend serving utilities for the tournament web application.
This module provides functions to serve the React frontend from the FastAPI backend.
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger(__name__)

def setup_frontend_serving(app: FastAPI):
    """
    Configure the FastAPI app to serve the React frontend.
    
    Args:
        app: The FastAPI application instance
    """
    # Define the path to the frontend build directory
    frontend_build_dir = Path(__file__).parent.parent / "frontend" / "build"
    
    # Also check the symlinked directory from Docker
    alt_frontend_dir = Path("/app/frontend-build")
    
    # Determine which directory to use
    if frontend_build_dir.exists():
        build_dir = frontend_build_dir
        logger.info(f"Using frontend build directory: {frontend_build_dir}")
    elif alt_frontend_dir.exists():
        build_dir = alt_frontend_dir
        logger.info(f"Using alternative frontend build directory: {alt_frontend_dir}")
    else:
        logger.warning(f"Frontend build directory not found at {frontend_build_dir} or {alt_frontend_dir}")
        return
    
    # Mount the static files directory if it exists
    static_dir = build_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Mounted static frontend files from {static_dir}")
    
    # Check for index.html
    index_path = build_dir / "index.html"
    if not index_path.exists():
        logger.warning(f"index.html not found at {index_path}")
        return
        
    # Read the index.html content
    try:
        with open(index_path, "r") as f:
            index_html = f.read()
            logger.info(f"Successfully read index.html ({len(index_html)} bytes)")
    except Exception as e:
        logger.error(f"Failed to read index.html: {str(e)}")
        return
    
    # Serve the frontend at multiple paths
    @app.get("/", response_class=HTMLResponse)
    async def serve_root(request: Request):
        # Always serve the frontend HTML
        return HTMLResponse(content=index_html)
        
    @app.get("/app", response_class=HTMLResponse)
    async def serve_app_root():
        return HTMLResponse(content=index_html)
        
    @app.get("/app/{full_path:path}", response_class=HTMLResponse)
    async def serve_app_path(full_path: str):
        # For SPA routing, always return index.html
        return HTMLResponse(content=index_html)
    
    logger.info("Frontend serving configured successfully")