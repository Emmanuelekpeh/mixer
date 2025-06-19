#!/usr/bin/env python3
"""
Tournament Web API Server

Main entry point for the Tournament Web API server.
Starts the FastAPI server and initializes all required components.
"""

import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from tournament_api import app, API_PREFIX, HOST, PORT, ALLOWED_ORIGINS
from simplified_tournament_engine import EnhancedTournamentEngine
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the API server"""
    logger.info("Starting Tournament API Server...")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Ensure required directories exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    tournaments_dir = data_dir / "tournaments"
    tournaments_dir.mkdir(exist_ok=True)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Initialize the tournament engine
    engine = EnhancedTournamentEngine()
    
    # Set the tournament engine as a global property of the app
    app.state.tournament_engine = engine
    
    # Start the server
    logger.info(f"Server ready at http://{HOST}:{PORT}{API_PREFIX}")
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
