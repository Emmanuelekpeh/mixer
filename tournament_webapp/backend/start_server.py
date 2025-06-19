#!/usr/bin/env python3
"""
Tournament Webapp Startup Script
Simple launcher to start the server with proper imports
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent))

# Import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    
    # Import the app after fixing the path
    from tournament_api import app
    
    print("🚀 Starting Tournament Webapp Server...")
    print("📊 Database initialized and ready")
    print("🎮 Tournament engine loaded")
    print("🔗 API endpoints active")
    print("")
    print("✅ Server running at: http://localhost:8000")
    print("📱 Frontend will be at: http://localhost:3000")
    print("🔧 API docs available at: http://localhost:8000/docs")
    print("")
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)]
    )
