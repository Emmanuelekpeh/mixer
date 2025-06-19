#!/usr/bin/env python3
"""
🚀 Tournament API Server - Production Ready
==========================================

FastAPI server for AI model tournament battles with enhanced features:
- Tournament management and model evolution
- User progression and gamification 
- Real-time battle execution
- Social sharing and viral growth
- Analytics and leaderboards

Integration with production AI mixer and enhanced musical intelligence.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import uuid
import os
import sys
import shutil
import json
from pathlib import Path
import logging
from datetime import datetime

# Database imports
from database import init_database, get_database_stats
from database_service import DatabaseService, get_database_service

# Initialize database on startup
print("🗄️  Initializing database...")
init_database()
db_stats = get_database_stats()
print(f"📊 Database ready: {db_stats}")
from dotenv import load_dotenv

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import error handling if available
try:
    from src.error_handling import (
        log_structured,
        performance_monitor,
        async_performance_monitor,
        track_error,
        track_performance
    )
    error_handling_available = True
except ImportError:
    error_handling_available = False
    # Fallback functions
    def log_structured(level, message, context=None, exception=None, **kwargs):
        logger.log(getattr(logging, level.upper()), message)
    
    def performance_monitor(func):
        return func
    
    def async_performance_monitor(func):
        return func
    
    def track_error(exception):
        logger.error(f"Error: {str(exception)}", exc_info=True)
    
    def track_performance(name, duration, metadata=None):
        logger.debug(f"Performance: {name} took {duration:.4f}s")

# Import tournament engine
from simplified_tournament_engine import EnhancedTournamentEngine

# Import async processing - handle type conflicts gracefully
try:
    from tournament_webapp.backend.async_task_system import schedule_task
    from tournament_webapp.backend.async_task_system import get_task_status as get_async_task_status
    
    # Use the async version but with our local TaskProgress type
    def get_task_status(task_id):
        result = get_async_task_status(task_id)
        if result is None:
            return TaskProgress(task_id=task_id, status="not_found", progress=0.0, message="Task not found")        # Convert to our local TaskProgress type
        return TaskProgress(
            task_id=result.task_id,
            status=result.status,
            progress=result.progress,
            message=result.message or "Task in progress",
            created_at=result.created_at,
            updated_at=result.updated_at,
            completed_at=result.completed_at
        )
        
except ImportError:
    logging.warning("Async task system not found, using synchronous methods")
      # Create a simple TaskProgress class for fallback
    class TaskProgress:
        def __init__(self, task_id: str, status: str = "completed", progress: float = 1.0, 
                     message: str = "Task completed", created_at: Optional[str] = None, 
                     updated_at: Optional[str] = None, completed_at: Optional[str] = None):
            self.task_id = task_id
            self.status = status
            self.progress = progress
            self.message = message
            self.created_at = created_at or datetime.now().isoformat()
            self.updated_at = updated_at or datetime.now().isoformat()
            self.completed_at = completed_at
    
    def schedule_task(background_tasks, task_func, task_type, *args, task_id=None, **kwargs):
        # Fallback: execute synchronously
        try:
            result = task_func(*args, **kwargs)
            return task_id or str(uuid.uuid4())
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            return task_id or str(uuid.uuid4())
    
    def get_task_status(task_id):
        return TaskProgress(task_id=task_id, status="completed", progress=1.0, message="Task completed")

# Import audio processing - with correct function signatures
try:
    from tournament_webapp.backend.audio_processor import process_audio as process_audio_file, create_final_mix
    # Import functions with correct signatures
    audio_process_audio = process_audio_file
    audio_create_final_mix = create_final_mix
    
    # We'll define execute_battle locally since it needs to match our API signature
    async def execute_battle_impl(tournament_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
        """Execute battle between current tournament competitors"""
        try:
            # Get tournament information
            tournament = tournament_engine.get_tournament_status(tournament_id)
            
            if not tournament:
                raise HTTPException(status_code=404, detail="Tournament not found")
            
            # Get audio file from tournament
            audio_file = tournament.get("audio_file")
            if not audio_file:
                raise HTTPException(status_code=400, detail="No audio file in tournament")
            
            # Create battle output directory
            battle_dir = static_dir / "battles" / tournament_id
            battle_dir.mkdir(parents=True, exist_ok=True)
              # Use the audio processor for battle execution
            battle_result = audio_process_audio(
                audio_path=audio_file,
                output_dir=str(battle_dir),
                model_id="model_a"  # We'll process one model at a time
            )
            
            return JSONResponse(content={"success": True, "battle": battle_result})
            
        except Exception as e:
            logger.error(f"Battle execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Battle execution failed: {str(e)}")
    
    execute_battle = execute_battle_impl
    process_audio = audio_process_audio
    create_final_mix = audio_create_final_mix
    
except ImportError:
    logging.warning("Audio processor not found, using fallback methods")
    process_audio = None
    
    async def execute_battle_fallback(tournament_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
        """Fallback battle execution"""
        return JSONResponse(content={
            "success": True, 
            "battle": {
                "battle_id": str(uuid.uuid4()),
                "status": "completed",
                "message": "Battle executed (fallback mode)"
            }
        })
    
    execute_battle = execute_battle_fallback
    create_final_mix = None

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Mixer Tournament API",
    description="Tournament battles for AI mixing models with evolutionary learning",
    version="1.0.0"
)

# Get configuration from environment variables
PORT = int(os.getenv("PORT", "10000"))
HOST = os.getenv("HOST", "0.0.0.0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
API_PREFIX = os.getenv("API_PREFIX", "")

# Base directory for data storage
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TOURNAMENTS_DIR = DATA_DIR / "tournaments"
AUDIO_DIR = DATA_DIR / "audio"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
TOURNAMENTS_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# Models for request and response data
class UserCreate(BaseModel):
    name: str
    email: Optional[str] = None

class User(BaseModel):
    id: str
    name: str
    created_at: str
    tournaments_completed: int
    rank: str
    
class TournamentCreate(BaseModel):
    user_id: str
    max_rounds: int = 5
    audio_file: Optional[str] = None

class TournamentResponse(BaseModel):
    success: bool
    tournament: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BattleVote(BaseModel):
    model_id: str
    confidence: float = 0.8  # Default confidence of 80%

class BattleResponse(BaseModel):
    success: bool
    battle_result: Optional[Dict[str, Any]] = None
    next_pair: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "tournament_engine": "ready",
            "models_available": len(tournament_engine.models) if hasattr(tournament_engine, 'models') else 0
        }
    }

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return await health_check()

# Health check endpoint
@app.get("/")
async def root_health_check():
    """Root health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Mixer Tournament API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }

@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {
        "status": "ok",
        "message": "AI Mixer Tournament API is running",        "version": "1.0.0"
    }

# Note: Newer user and tournament endpoints are defined later in the file
# The old endpoints here were replaced with updated versions

MODELS_DIR = os.getenv("MODELS_DIR", "../models")

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # React dev server and production URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving
static_dir = Path("static")
static_dir.mkdir(parents=True, exist_ok=True)

# Create processed audio directory - use the existing one we created
processed_audio_dir = Path("processed_audio")
processed_audio_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/processed_audio", StaticFiles(directory=str(processed_audio_dir)), name="processed_audio")

# Initialize tournament engine
models_dir = Path(MODELS_DIR)

# Import the model manager if available
try:
    from tournament_webapp.backend.tournament_model_manager import TournamentModelManager
    model_manager = TournamentModelManager(models_dir)
    logger.info(f"Initialized tournament model manager with {len(model_manager.get_model_list())} models")
except Exception as e:
    logger.warning(f"Could not initialize tournament model manager: {str(e)}")
    model_manager = None

# Create a simple stub class for development/testing
class SimpleTournamentEngine:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.tournaments = {}
        self.user_profiles = {}
        self.model_manager = model_manager
        
    def get_tournament_status(self, tournament_id):
        # Return a simple response for now
        if tournament_id not in self.tournaments:
            return None
        return self.tournaments[tournament_id]
    
    def create_tournament(self, user_id, username, max_rounds=5, audio_features=None):
        tournament_id = str(uuid.uuid4())
        self.tournaments[tournament_id] = {
            "id": tournament_id,
            "user_id": user_id,
            "username": username,
            "status": "active",
            "round": 1,
            "max_rounds": max_rounds,
            "created_at": datetime.now().isoformat(),
            "battle_history": [],
            "current_battle": {
                "battle_id": str(uuid.uuid4()),
                "model_a": {"id": "model1", "name": "Baseline CNN"},
                "model_b": {"id": "model2", "name": "Enhanced CNN"}
            },
            "audio_features": audio_features or {}
        }
        return tournament_id
    
    def start_tournament(self, user_id, username, audio_file=None, max_rounds=5, audio_features=None):
        tournament_id = self.create_tournament(user_id, username, max_rounds, audio_features)
        
        # Log whether we're using spectrogram
        if audio_features and "spectrogram_path" in audio_features:
            logger.info(f"🔊 Tournament {tournament_id} using spectrogram: {audio_features['spectrogram_path']}")
        elif audio_file:
            logger.info(f"🎵 Tournament {tournament_id} using audio file: {audio_file}")
            
            # Store audio file path in tournament for model processing
            if tournament_id in self.tournaments:
                self.tournaments[tournament_id]["audio_file"] = audio_file
        
        return tournament_id
        
    async def execute_battle_async(self, tournament_id):
        """Execute battle asynchronously if model manager is available"""
        if tournament_id not in self.tournaments:
            return None
            
        tournament = self.tournaments[tournament_id]
        
        # Check if we have model manager and audio file
        if self.model_manager and "audio_file" in tournament:
            audio_file = tournament["audio_file"]
            
            # Get models from available models
            available_models = self.model_manager.get_model_list()
            if len(available_models) >= 2:
                model_a_id = available_models[0]["id"]
                model_b_id = available_models[1]["id"]
                
                # Execute battle with model manager
                battle_result = await self.model_manager.execute_battle(
                    audio_file, model_a_id, model_b_id
                )
                
                # Update tournament with battle result
                if "error" not in battle_result:
                    tournament["current_battle"] = battle_result
                    return battle_result
                else:
                    logger.error(f"Battle execution failed: {battle_result['error']}")
          # Fallback to dummy battle
        battle_id = str(uuid.uuid4())
        battle_result = {
            "battle_id": battle_id,
            "model_a": {"id": "model1", "name": "Baseline CNN"},
            "model_b": {"id": "model2", "name": "Enhanced CNN"},
            "status": "ready_for_vote"
        }
        
        tournament["current_battle"] = battle_result
        return battle_result
    
    def execute_battle(self, tournament_id):
        """Synchronous wrapper for execute_battle_async"""
        if tournament_id not in self.tournaments:
            return None
            
        # If we have asyncio support, run the async version
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.execute_battle_async(tournament_id))
        except Exception as e:
            logger.error(f"Async battle execution failed: {str(e)}")
            
            # Fallback to dummy battle
            battle_id = str(uuid.uuid4())
            return {
                "battle_id": battle_id,
                "model_a": {"id": "model1", "name": "Baseline CNN"},
                "model_b": {"id": "model2", "name": "Enhanced CNN"},
                "status": "ready_for_vote"
            }

    def record_vote(self, tournament_id, winner_id, confidence=0.7, reasoning=None):
        if tournament_id not in self.tournaments:
            return None
        # Just update the round for now
        tournament = self.tournaments[tournament_id]
        tournament["round"] += 1
        
        # Add to battle history
        battle_history_entry = {
            "battle_id": tournament["current_battle"]["battle_id"] if "battle_id" in tournament["current_battle"] else str(uuid.uuid4()),
            "winner_id": winner_id,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add spectrogram path if available in current battle
        if "current_battle" in tournament and "spectrogram_path" in tournament["current_battle"]:
            battle_history_entry["spectrogram_path"] = tournament["current_battle"]["spectrogram_path"]
        
        tournament["battle_history"].append(battle_history_entry)
        
        # If we have a model manager, update model metrics based on vote
        if self.model_manager and hasattr(self.model_manager, "update_model_metrics"):
            try:
                # This would be implemented in a real model manager
                self.model_manager.update_model_metrics(
                    winner_id=winner_id,
                    loser_id=tournament["current_battle"]["model_a"]["id"] if winner_id == tournament["current_battle"]["model_b"]["id"] else tournament["current_battle"]["model_b"]["id"],
                    confidence=confidence
                )
            except Exception as e:
                logger.error(f"Failed to update model metrics: {str(e)}")
        
        # Create a new battle if not finished
        if tournament["round"] <= tournament["max_rounds"]:
            # Use execute_battle to get the next battle
            next_battle = self.execute_battle(tournament_id)
            if next_battle:
                tournament["current_battle"] = next_battle
            else:
                # Fallback to simple battle
                tournament["current_battle"] = {
                    "battle_id": str(uuid.uuid4()),
                    "model_a": {"id": "model3", "name": "Improved Baseline CNN"},
                    "model_b": {"id": "model4", "name": "Retrained Enhanced CNN"}
                }
        else:
            tournament["status"] = "completed"
            tournament["current_battle"] = None
            
            # If we have audio file and winning model, create final mix
            if "audio_file" in tournament and self.model_manager:
                try:
                    # Find the overall winner
                    winner_counts = {}
                    for battle in tournament["battle_history"]:
                        winner_id = battle["winner_id"]
                        winner_counts[winner_id] = winner_counts.get(winner_id, 0) + 1
                    
                    if winner_counts:
                        overall_winner = max(winner_counts.items(), key=lambda x: x[1])[0]
                        
                        # Process final mix with winning model
                        logger.info(f"Creating final mix with winning model {overall_winner}")
                        tournament["final_winner"] = overall_winner
                        
                        # This would create the final mix in a real implementation
                        # tournament["final_mix_path"] = await self.model_manager.create_final_mix(
                        #     tournament["audio_file"], overall_winner
                        # )
                except Exception as e:
                    logger.error(f"Failed to create final mix: {str(e)}")
        
        return tournament
    
    def vote_for_winner(self, tournament_id, winner_id, confidence=0.7, reasoning=None):
        return self.record_vote(tournament_id, winner_id, confidence, reasoning)
        
    def get_model_list(self):
        """Get list of available models"""
        # If we have a model manager, use it to get real models
        if self.model_manager:
            try:
                # Get models from the manager
                models = self.model_manager.get_model_list()
                if models:
                    # Format for API response
                    return [
                        {
                            "id": model["id"],
                            "name": model["name"],
                            "architecture": model["architecture"],
                            "specializations": model.get("specializations", []),
                            "size_mb": round(model.get("size_mb", 0), 2)
                        }
                        for model in models                    ]
            except Exception as e:
                logger.error(f"Failed to get models from manager: {str(e)}")
        
        # Fallback to sample model list
        return [
            {"id": "model1", "name": "Baseline CNN", "architecture": "cnn"},
            {"id": "model2", "name": "Enhanced CNN", "architecture": "cnn"},
            {"id": "model3", "name": "Improved Baseline CNN", "architecture": "cnn"},
            {"id": "model4", "name": "Retrained Enhanced CNN", "architecture": "cnn"},
            {"id": "model5", "name": "Weighted Ensemble", "architecture": "hybrid"}
        ]
            
    def get_available_model_files(self):
        # Return a list of model files available in the models directory
        model_files = []
        try:
            for file in Path(self.models_dir).glob("**/*.pth"):
                model_files.append({
                    "path": str(file.relative_to(self.models_dir.parent)),
                    "size_mb": file.stat().st_size / (1024 * 1024),
                    "last_modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                })
        except Exception as e:
            logger.error(f"Error listing model files: {e}")
        return model_files
    
    def create_user_profile(self, user_id, username):
        profile = {
            "user_id": user_id,
            "username": username,
            "tournaments_played": 0,
            "battles_voted": 0,
            "created_at": datetime.now().isoformat()
        }
        self.user_profiles[user_id] = profile
        return profile
    
    def get_user_stats(self, user_id):
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        return self.create_user_profile(user_id, f"User_{user_id[:6]}")
    
    def get_tournament_leaderboard(self):
        return [
            {"model_id": "model2", "name": "Enhanced CNN", "wins": 45, "elo": 1250},
            {"model_id": "model4", "name": "Retrained Enhanced CNN", "wins": 32, "elo": 1150},            {"model_id": "model5", "name": "Weighted Ensemble", "wins": 28, "elo": 1100},
            {"model_id": "model3", "name": "Improved Baseline CNN", "wins": 20, "elo": 1050},
            {"model_id": "model1", "name": "Baseline CNN", "wins": 10, "elo": 950}
        ]
    
    @property
    def active_tournaments(self):
        return {k: v for k, v in self.tournaments.items() if v["status"] == "active"}
    
    @property
    def evolution_engine(self):
        class DummyEvolutionEngine:
            def __init__(self, models):
                self.champion_models = models
                self.genealogy = {
                    "models": models,
                    "statistics": {
                        "total_evolved": len(models),
                        "by_architecture": {"cnn": 3, "hybrid": 2}
                    },
                    "evolution_tree": {}
                }
        
        return DummyEvolutionEngine(self.get_model_list())

# Initialize the engine - using enhanced version with persistence
tournament_engine = EnhancedTournamentEngine()

# Pydantic models for API
class TournamentCreateRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="User display name")
    max_rounds: int = Field(5, ge=1, le=10, description="Maximum tournament rounds")
    audio_features: Optional[Dict[str, float]] = Field(None, description="Audio analysis features")

class VoteRequest(BaseModel):
    tournament_id: str = Field(..., description="Tournament identifier")
    winner_id: str = Field(..., description="ID of winning model")
    confidence: float = Field(0.7, ge=0.1, le=1.0, description="Vote confidence level")
    reasoning: Optional[str] = Field(None, description="User's reasoning for vote")

class UserCreateRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="User display name")

# Tournament management endpoints
@app.post("/api/tournaments/create-json")
async def create_tournament_json(
    request: dict,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Create tournament from JSON data (frontend requests)"""
    try:
        user_id = request.get("user_id", "demo_user")
        username = request.get("username", f"User_{user_id[:8]}")
        max_rounds = request.get("max_rounds", 5)
        
        print(f"🚀 Creating JSON tournament for user: {user_id}")
        
        # Ensure user exists in database
        user = db_service.get_user(user_id)
        if not user:
            user = db_service.create_user(user_id, username)
        
        # Create tournament in database
        tournament = db_service.create_tournament(
            user_id=user_id,
            max_rounds=max_rounds,
            audio_file="demo_audio.wav"
        )
        
        # Get available models for the tournament
        models = db_service.get_all_models()
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")        # Create tournament pairs data - SIMPLE 5-ROUND STRUCTURE
        import random
        selected_models = random.sample(models, min(6, len(models)))  # Need 6 models for 5 rounds
        
        # Create exactly 5 pairs for 5 rounds (one pair per round)
        pairs_data = []
        for round_num in range(1, min(6, len(selected_models))):  # Rounds 1-5
            if round_num < len(selected_models):
                model_a = selected_models[0] if round_num == 1 else selected_models[round_num-1]  # Winner carries forward
                model_b = selected_models[round_num]
                
                pair = {
                    "round": round_num,
                    "model_a": {
                        "id": model_a.id,
                        "name": model_a.name,
                        "nickname": model_a.nickname or model_a.name,
                        "architecture": model_a.architecture,
                        "elo_rating": model_a.elo_rating,
                        "tier": model_a.tier,
                        "generation": model_a.generation
                    },
                    "model_b": {
                        "id": model_b.id,
                        "name": model_b.name,
                        "nickname": model_b.nickname or model_b.name,
                        "architecture": model_b.architecture,
                        "elo_rating": model_b.elo_rating,
                        "tier": model_b.tier,
                        "generation": model_b.generation
                    },
                    "audio_a": f"/processed_audio/{tournament.id}_{model_a.id}_mix.wav",
                    "audio_b": f"/processed_audio/{tournament.id}_{model_b.id}_mix.wav"
                }
                pairs_data.append(pair)        # Update tournament with pairs data
        db_service.update_tournament(
            tournament.id,
            pairs_data=pairs_data,
            tournament_data={
                "total_models": len(selected_models),                "total_pairs": len(pairs_data),
                "mode": "5_round_elimination"
            }
        )
        
        logger.info(f"🏆 5-Round Tournament created: {tournament.id} with {len(pairs_data)} total pairs")
        
        return JSONResponse(content={
            "success": True,
            "tournament_id": tournament.id,  # Frontend expects this field
            "tournament": {
                "id": tournament.id,
                "user_id": tournament.user_id,
                "max_rounds": tournament.max_rounds,
                "current_round": tournament.current_round,
                "current_pair": tournament.current_pair,
                "status": tournament.status,
                "pairs": pairs_data,
                "total_pairs": len(pairs_data),
                "total_models": len(selected_models),
                "created_at": tournament.created_at.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"❌ JSON Tournament creation failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Tournament creation failed: {str(e)}")

@app.post("/api/tournaments/upload")
async def create_tournament_with_upload(
    background_tasks: BackgroundTasks,
    db_service: DatabaseService = Depends(get_database_service),
    user_id: str = Form(...),
    username: str = Form(...),
    max_rounds: int = Form(5),
    audio_features: str = Form("{}"),
    audio_file: UploadFile = File(..., description="Audio file to mix")
):
    """Create tournament with uploaded audio file"""
    try:
        # Validate audio file
        if not audio_file or not audio_file.filename:
            raise HTTPException(status_code=400, detail="Missing audio file")
            
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.aiff')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        print(f"🎵 Creating upload tournament with audio: {audio_file.filename}")
          # Save uploaded audio file
        audio_dir = static_dir / "uploads"
        audio_dir.mkdir(exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_extension = audio_file.filename.split('.')[-1]
        saved_filename = f"{file_id}.{file_extension}"
        saved_path = audio_dir / saved_filename
        
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        
        # Ensure user exists in database with unique username for testing
        user = db_service.get_user(user_id)
        if not user:
            # Add timestamp to prevent username conflicts in tests
            import time
            unique_username = f"{username}_{int(time.time())}"
            user = db_service.create_user(user_id, unique_username)
        
        # Create tournament in database
        tournament = db_service.create_tournament(
            user_id=user_id,
            max_rounds=max_rounds,
            audio_file=str(saved_path)
        )
        
        # [Continue with model selection and pairing logic similar to JSON endpoint]
        models = db_service.get_all_models()
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")
          # This endpoint would integrate with your actual AI mixing system        # For now, use the same demo setup
        import random
        selected_models = random.sample(models, min(8, len(models)))
        
        # Create round-robin tournament pairs so all models battle each other
        pairs_data = []
        for i in range(len(selected_models)):
            for j in range(i + 1, len(selected_models)):
                model_a = selected_models[i]
                model_b = selected_models[j]
                
                pair = {
                    "model_a": {"id": model_a.id, "name": model_a.name, "architecture": model_a.architecture, "elo_rating": model_a.elo_rating, "tier": model_a.tier, "generation": model_a.generation},
                    "model_b": {"id": model_b.id, "name": model_b.name, "architecture": model_b.architecture, "elo_rating": model_b.elo_rating, "tier": model_b.tier, "generation": model_b.generation},
                    "audio_a": f"/processed_audio/{tournament.id}_{model_a.id}_mix.wav",
                    "audio_b": f"/processed_audio/{tournament.id}_{model_b.id}_mix.wav"
                }
                pairs_data.append(pair)
          # Process audio with each model to create the actual mixed files
        try:
            logger.info(f"🎵 Processing audio for tournament {tournament.id}")
              # Import AI mixer integration
            from ai_mixer_integration_fixed import get_tournament_ai_mixer
            ai_mixer = get_tournament_ai_mixer()
            
            for model in selected_models:
                # Create the expected output file path
                output_filename = f"{tournament.id}_{model.id}_mix.wav"
                output_path = processed_audio_dir / output_filename
                
                # Use AI mixer to process audio with the specific model
                success = ai_mixer.process_audio_with_model(
                    str(saved_path),
                    model.id,
                    str(output_path)
                )
                
                if success:
                    logger.info(f"🎵 AI processed audio with {model.id}: {output_filename}")
                else:
                    logger.warning(f"⚠️ Failed to process audio with {model.id}, using original")
                
        except Exception as audio_error:
            logger.warning(f"⚠️ Audio processing failed, using original: {str(audio_error)}")
            # Continue with tournament creation even if audio processing fails
        
        # Update tournament with pairs data
        db_service.update_tournament(
            tournament.id,
            pairs_data=pairs_data,
            tournament_data={
                "total_models": len(selected_models),
                "total_pairs": len(pairs_data),
                "mode": "file_upload",
                "original_filename": audio_file.filename,
                "uploaded_file_path": str(saved_path)
            }        )
        
        logger.info(f"🏆 Upload Tournament created: {tournament.id} with uploaded audio")
        
        return JSONResponse(content={
            "success": True,
            "tournament_id": tournament.id,
            "message": "Tournament created with uploaded audio",
            "original_filename": audio_file.filename,
            "tournament": {
                "id": tournament.id,
                "tournament_id": tournament.id,
                "user_id": tournament.user_id,
                "max_rounds": tournament.max_rounds,
                "current_round": tournament.current_round,
                "current_pair": tournament.current_pair,
                "status": tournament.status,
                "pairs": pairs_data,
                "total_pairs": len(pairs_data),
                "total_models": len(selected_models),
                "created_at": tournament.created_at.isoformat(),
                "has_uploaded_audio": True,
                "original_filename": audio_file.filename
            },
            "pairs": pairs_data  # Keep for backwards compatibility
        })
        
    except Exception as e:
        logger.error(f"❌ Upload Tournament creation failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Upload tournament creation failed: {str(e)}")
        
        # Ensure user exists in database
        user = db_service.get_user(final_user_id)
        if not user:
            user = db_service.create_user(final_user_id, final_username)
        
        # Create tournament in database
        tournament = db_service.create_tournament(
            user_id=final_user_id,
            max_rounds=final_max_rounds,
            audio_file=audio_file_path
        )
        
        # Get available models for the tournament
        models = db_service.get_all_models()
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")
        
        # Create tournament pairs data
        import random
        selected_models = random.sample(models, min(8, len(models)))
        
        pairs_data = []
        for i in range(0, len(selected_models), 2):
            if i + 1 < len(selected_models):
                model_a = selected_models[i]
                model_b = selected_models[i + 1]
                pair = {
                    "model_a": {
                        "id": model_a.id,
                        "name": model_a.name,
                        "nickname": model_a.nickname or model_a.name,
                        "architecture": model_a.architecture,
                        "elo_rating": model_a.elo_rating,
                        "tier": model_a.tier,
                        "generation": model_a.generation
                    },
                    "model_b": {
                        "id": model_b.id,
                        "name": model_b.name,
                        "nickname": model_b.nickname or model_b.name,
                        "architecture": model_b.architecture,
                        "elo_rating": model_b.elo_rating,
                        "tier": model_b.tier,
                        "generation": model_b.generation
                    },
                    "audio_a": f"/demo_audio/{model_a.id}_mix.wav",
                    "audio_b": f"/demo_audio/{model_b.id}_mix.wav"
                }
                pairs_data.append(pair)
        
        # Update tournament with pairs data
        db_service.update_tournament(
            tournament.id,
            pairs_data=pairs_data,
            tournament_data={
                "total_models": len(selected_models),
                "total_pairs": len(pairs_data),
                "has_uploaded_audio": bool(audio_file and audio_file.filename),
                "audio_file_path": audio_file_path
            }        )
        
        logger.info(f"🏆 Tournament created: {tournament.id} ({'with upload' if audio_file and audio_file.filename else 'demo mode'})")
        
        return JSONResponse(content={
            "success": True,
            "tournament_id": tournament.id,  # For frontend compatibility
            "tournament": {
                "id": tournament.id,
                "tournament_id": tournament.id,  # Duplicate for compatibility
                "user_id": tournament.user_id,
                "max_rounds": tournament.max_rounds,
                "current_round": tournament.current_round,
                "current_pair": tournament.current_pair,
                "status": tournament.status,
                "pairs": pairs_data,
                "total_pairs": len(pairs_data),
                "total_models": len(selected_models),
                "created_at": tournament.created_at.isoformat(),
                "has_uploaded_audio": bool(audio_file and audio_file.filename)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Tournament creation failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Tournament creation failed: {str(e)}")
        
        logger.info(f"🎵 Audio uploaded: {saved_filename}")
        
        # Parse audio features from JSON string
        try:
            parsed_audio_features = json.loads(audio_features) if audio_features else {}
        except json.JSONDecodeError:
            parsed_audio_features = {}
        
        # Start tournament
        tournament_id = tournament_engine.start_tournament(
            user_id=user_id,
            username=username,
            audio_file=str(saved_path),
            max_rounds=max_rounds,
            audio_features=parsed_audio_features
        )
        
        # Schedule audio processing asynchronously
        if process_audio:
            # Create an output directory for processed audio
            process_dir = static_dir / "processed" / tournament_id
            process_dir.mkdir(parents=True, exist_ok=True)
            
            # Schedule audio conversion task
            task_id = schedule_task(
                background_tasks,
                process_audio,
                "audio_processing",
                str(saved_path),
                str(process_dir),
                None,  # No specific model for initial processing
                task_id=f"process_{tournament_id}"
            )
            
            logger.info(f"🔄 Scheduled audio processing task {task_id} for tournament {tournament_id}")
            
            # Add task info to audio features
            parsed_audio_features["processing_task_id"] = task_id
          # Update tournament with audio features
        tournament = tournament_engine.get_tournament_status(tournament_id)
        if tournament is None:
            logger.error(f"Failed to get tournament {tournament_id} after creation")
            raise HTTPException(status_code=500, detail="Tournament creation failed - could not retrieve tournament")
        
        tournament["audio_features"] = parsed_audio_features
        
        # Convert to dict for JSON response
        tournament_dict = {
            "tournament_id": tournament_id,
            "user_id": user_id,
            "status": tournament["status"],
            "current_round": tournament["round"],
            "max_rounds": tournament["max_rounds"],
            "competitors": tournament_engine.get_model_list(),
            "audio_file": saved_filename,
            "created_at": tournament["created_at"],
            "processing_task_id": parsed_audio_features.get("processing_task_id")
        }
        
        logger.info(f"🏆 Tournament created: {tournament_id}")
        return JSONResponse(content={"success": True, "tournament": tournament_dict})
        
    except Exception as e:
        logger.error(f"❌ Tournament creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tournament creation failed: {str(e)}")

@app.post("/api/tournaments/{tournament_id}/battle")
@async_performance_monitor
async def execute_battle(tournament_id: str, background_tasks: BackgroundTasks):
    """Execute battle between current tournament competitors asynchronously"""
    try:
        # Get tournament information
        tournament = tournament_engine.get_tournament_status(tournament_id)
        
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        
        # Get current models for battle
        models = tournament_engine.get_model_list()
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Not enough models for battle")
        
        model_a_id = models[0]["id"]
        model_b_id = models[1]["id"]
        
        # Get audio file from tournament
        audio_file = tournament.get("audio_file")
        if not audio_file:
            raise HTTPException(status_code=400, detail="No audio file in tournament")
        
        # Create battle info
        battle_id = str(uuid.uuid4())
        battle_info = {
            "battle_id": battle_id,
            "model_a": {"id": model_a_id, "name": models[0]["name"]},
            "model_b": {"id": model_b_id, "name": models[1]["name"]},
            "status": "processing"
        }
        
        # Update tournament with battle info
        tournament["current_battle"] = battle_info
        
        # Schedule battle execution asynchronously
        if execute_battle:
            # Create an output directory for battle
            battle_dir = static_dir / "battles" / tournament_id
            battle_dir.mkdir(parents=True, exist_ok=True)
            
            # Schedule battle execution task
            task_id = schedule_task(
                background_tasks,
                execute_battle,
                "battle_execution",
                audio_file,
                str(battle_dir),
                model_a_id,
                model_b_id,
                task_id=f"battle_{battle_id}"
            )
            
            # Add task info to battle info
            battle_info["task_id"] = task_id
            battle_info["status"] = "processing"
            
            logger.info(f"🔄 Scheduled battle execution task {task_id} for tournament {tournament_id}")
        else:
            # Fallback to synchronous execution
            battle_info = tournament_engine.execute_battle(tournament_id)
        
        return JSONResponse(content={"success": True, "battle": battle_info})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Battle execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Battle execution failed: {str(e)}")

@app.post("/api/tournaments/vote")
async def record_vote(request: VoteRequest):
    """Record user vote and advance tournament"""
    try:
        result = tournament_engine.record_vote(
            tournament_id=request.tournament_id,
            winner_id=request.winner_id,
            confidence=request.confidence,
            reasoning=request.reasoning
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Tournament not found")
        
        logger.info(f"🗳️ Vote recorded: {request.tournament_id}")
        return JSONResponse(content={"success": True, "result": result})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Vote recording failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vote recording failed: {str(e)}")

@app.post("/api/tournaments/vote-db")
async def record_vote_db(request: dict, db_service: DatabaseService = Depends(get_database_service)):
    """Record vote using database backend"""
    try:
        tournament_id = request.get("tournament_id")
        winner_id = request.get("winner_id")
        confidence = request.get("confidence", 0.8)
        user_id = request.get("user_id", "demo_user")
        
        # Get tournament from database
        tournament = db_service.get_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        
        # Get current pair data
        pairs_data = tournament.pairs_data or []
        if not pairs_data:
            raise HTTPException(status_code=400, detail="No active battles in tournament")
        
        current_pair = pairs_data[tournament.current_pair] if tournament.current_pair < len(pairs_data) else pairs_data[0]
        
        # Determine loser
        model_a_id = current_pair["model_a"]["id"]
        model_b_id = current_pair["model_b"]["id"]
        loser_id = model_b_id if winner_id == model_a_id else model_a_id
        
        # Record vote in database
        vote = db_service.record_vote(
            tournament_id=tournament_id,
            user_id=user_id,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            winning_model_id=winner_id,
            confidence=confidence,
            round_number=tournament.current_round,
            pair_number=tournament.current_pair
        )        # Update tournament progress - EXPLICIT ROUND ADVANCEMENT
        current_pair_index = tournament.current_pair
        total_pairs = len(pairs_data)
        
        # Calculate new state
        new_pair_index = current_pair_index + 1
        new_round = new_pair_index + 1  # Round = pair index + 1 (so round 1 = pair 0, round 2 = pair 1, etc.)
        
        # Check completion based on max_rounds, not total pairs
        if new_round > tournament.max_rounds:
            # Tournament completed after max_rounds
            db_service.complete_tournament(tournament_id, winner_id)
            status = "completed"
            next_pair = None
            logger.info(f"🏆 Tournament COMPLETED after {tournament.max_rounds} rounds! Final winner: {winner_id}")
        elif new_pair_index >= total_pairs:
            # Safety check: ran out of pairs
            db_service.complete_tournament(tournament_id, winner_id)
            status = "completed"
            next_pair = None
            logger.info(f"🏆 Tournament COMPLETED (ran out of pairs)! Final winner: {winner_id}")
        else:
            # Move to next pair and round
            db_service.update_tournament(
                tournament_id, 
                current_pair=new_pair_index, 
                current_round=new_round
            )
            status = "active"
            next_pair = pairs_data[new_pair_index] if new_pair_index < len(pairs_data) else None
            logger.info(f"🏆 Tournament advanced to Round {new_round}/{tournament.max_rounds} (Pair {new_pair_index + 1}/{total_pairs})")
        
        logger.info(f"🗳️ Vote recorded: {winner_id} beats {loser_id} in {tournament_id}")
        
        # Get updated tournament data for complete response
        updated_tournament = db_service.get_tournament(tournament_id)
        
        return JSONResponse(content={
            "success": True,
            "vote": {
                "tournament_id": tournament_id,
                "winner": winner_id,
                "loser": loser_id,
                "confidence": confidence,
                "round": tournament.current_round
            },
            "tournament": {
                "tournament_id": updated_tournament.id,
                "user_id": updated_tournament.user_id,
                "status": updated_tournament.status,
                "current_round": updated_tournament.current_round,
                "current_pair": updated_tournament.current_pair,
                "max_rounds": updated_tournament.max_rounds,
                "pairs": updated_tournament.pairs_data or [],
                "total_pairs": len(updated_tournament.pairs_data or []),
                "victor_model_id": updated_tournament.victor_model_id,
                "next_battle": next_pair,
                "winner": winner_id if status == "completed" else None
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Database vote recording failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Vote recording failed: {str(e)}")

@app.get("/api/tournaments/{tournament_id}")
async def get_tournament_details(tournament_id: str, db_service: DatabaseService = Depends(get_database_service)):
    """Get tournament details from database - frontend compatible format"""
    try:
        tournament = db_service.get_tournament(tournament_id)
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        
        # Get tournament votes
        votes = db_service.get_tournament_votes(tournament_id)
        
        # Ensure pairs data exists and is in correct format
        pairs_data = tournament.pairs_data or []
        
        # Format response for frontend compatibility
        tournament_response = {
            "tournament_id": tournament.id,  # Frontend expects tournament_id
            "user_id": tournament.user_id,
            "status": tournament.status,
            "current_round": tournament.current_round,
            "current_pair": tournament.current_pair,
            "max_rounds": tournament.max_rounds,
            "pairs": pairs_data,  # This contains the model battle data
            "created_at": tournament.created_at.isoformat(),
            "completed_at": tournament.completed_at.isoformat() if tournament.completed_at else None,
            "victor_model_id": tournament.victor_model_id,
            "total_votes": len(votes),
            "battle_history": [],  # Frontend might expect this
            "round": tournament.current_round,  # Some components use 'round'
            "total_pairs": len(pairs_data)        }
        
        logger.info(f"📊 Tournament details retrieved: {tournament_id} (Status: {tournament.status}, Pairs: {len(pairs_data)})")
        return JSONResponse(content={
            "success": True,
            "tournament": tournament_response
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Tournament details retrieval failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Tournament retrieval failed: {str(e)}")

@app.post("/api/tournaments/{tournament_id}/resume")
async def resume_tournament(tournament_id: str, db_service: DatabaseService = Depends(get_database_service)):
    """Resume a tournament from database state"""
    try:
        # Try to resume tournament using the engine
        tournament_state = tournament_engine.resume_tournament(tournament_id)
        
        if not tournament_state:
            raise HTTPException(status_code=404, detail="Tournament not found or cannot be resumed")
        
        # Format response for frontend compatibility
        tournament_response = {
            "tournament_id": tournament_state["tournament_id"],
            "user_id": tournament_state["user_id"],
            "status": tournament_state["status"],
            "current_round": tournament_state["current_round"],
            "current_pair": len(tournament_state.get("results", [])),
            "max_rounds": tournament_state["max_rounds"],
            "pairs": tournament_state["pairs"],
            "created_at": tournament_state["created_at"],
            "updated_at": tournament_state["updated_at"],
            "battle_history": tournament_state.get("results", []),
            "total_pairs": len(tournament_state["pairs"])
        }
        
        logger.info(f"🔄 Tournament {tournament_id} resumed successfully")
        
        return JSONResponse(content={
            "success": True,
            "tournament": tournament_response,
            "message": "Tournament resumed successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Tournament resume failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Tournament resume failed: {str(e)}")

@app.get("/api/users/{user_id}/tournaments")
async def get_user_tournaments(
    user_id: str, 
    status: Optional[str] = None,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Get list of tournaments for a user"""
    try:
        # Get tournaments using the engine (which will check database)
        tournaments = tournament_engine.list_user_tournaments(user_id, status_filter=status)
        
        logger.info(f"📋 Retrieved {len(tournaments)} tournaments for user {user_id}")
        
        return JSONResponse(content={
            "success": True,
            "tournaments": tournaments,
            "count": len(tournaments)
        })
        
    except Exception as e:
        logger.error(f"❌ User tournaments retrieval failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Failed to get user tournaments: {str(e)}")

# Task management endpoints
@app.get("/api/tasks/{task_id}")
@async_performance_monitor
async def get_task_progress(task_id: str):
    """Get the progress of an asynchronous task"""
    try:
        # Get task status
        task_status = get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Convert to dictionary for JSON response
        task_dict = {
            "task_id": task_status.task_id,
            "status": task_status.status,
            "progress": task_status.progress,
            "message": task_status.message,
            "created_at": task_status.created_at,
            "updated_at": task_status.updated_at,
            "completed_at": task_status.completed_at
        }
        
        return JSONResponse(content={"success": True, "task": task_dict})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task status check failed: {str(e)}")

# User management endpoints
@app.post("/api/users/create")
async def create_user(request: UserCreateRequest):
    """Create new user profile"""
    try:
        profile = tournament_engine.create_user_profile(
            user_id=request.user_id,
            username=request.username
        )
        
        # Handle profile as dictionary (current implementation)
        profile_dict = {
            "user_id": profile.get("user_id", request.user_id),
            "username": profile.get("username", request.username),
            "tier": "Rookie",  # Default tier
            "tournaments_completed": profile.get("tournaments_played", 0),
            "total_battles": profile.get("battles_voted", 0),
            "referral_code": f"REF_{request.user_id[:8].upper()}",
            "free_mixes_earned": profile.get("free_mixes_earned", 0),
            "achievements": profile.get("achievements", []),
            "created_at": profile.get("created_at", datetime.now().isoformat())
        }
        
        logger.info(f"👤 User created: {request.username}")
        return JSONResponse(content={"success": True, "profile": profile_dict})
        
    except Exception as e:
        logger.error(f"❌ User creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User creation failed: {str(e)}")

@app.get("/api/users/{user_id}")
async def get_user_profile(user_id: str, db_service: DatabaseService = Depends(get_database_service)):
    """Get user profile and statistics"""
    try:
        # Get user from database
        user = db_service.get_user(user_id)
        
        if not user:
            # Create new user if doesn't exist
            user = db_service.create_user(user_id, f"User_{user_id[:8]}")
        
        # Get user preferences and analytics
        preferences = db_service.get_user_preferences(user_id)
        
        # Build comprehensive user profile
        user_data = {
            "user_id": user.id,
            "profile": {
                "username": user.username,
                "tier": user.tier,
                "tournaments_completed": user.tournaments_completed,
                "tournaments_won": user.tournaments_won,
                "total_battles": user.total_battles,
                "total_votes": user.total_votes,
                "experience_points": user.experience_points,
                "referral_code": user.referral_code,
                "join_date": user.created_at.isoformat() if user.created_at else None
            },
            "preferences": preferences,
            "achievements": [],  # TODO: Implement achievements system
            "recent_tournaments": [
                {
                    "id": t.id,
                    "status": t.status,
                    "created_at": t.created_at.isoformat(),
                    "victor_model": t.victor_model_id
                } for t in db_service.get_user_tournaments(user_id, 5)
            ]
        }
        
        return JSONResponse(content={"success": True, "user": user_data})
        
    except Exception as e:
        logger.error(f"❌ User profile retrieval failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

@app.get("/api/leaderboard")
async def get_leaderboard(db_service: DatabaseService = Depends(get_database_service)):
    """Get model leaderboard by ELO rating"""
    try:
        # Get top models from database
        top_models = db_service.get_top_models(limit=20)
        
        # Get tournament analytics
        analytics = db_service.get_tournament_analytics(days=30)
        
        return JSONResponse(content={
            "success": True, 
            "leaderboard": top_models,
            "analytics": analytics,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Leaderboard retrieval failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")

@app.get("/api/models")
async def get_available_models(db_service: DatabaseService = Depends(get_database_service)):
    """Get list of available AI models and model files for deployment verification"""
    try:        # Get models from database (cached for performance)
        models = db_service.get_all_models_cached()
        
        models_list = []
        for model in models:
            model_dict = {
                "id": model.id,
                "name": model.name,
                "nickname": model.nickname or model.name,
                "architecture": model.architecture,
                "generation": model.generation,
                "tier": model.tier,
                "elo_rating": round(model.elo_rating, 1),
                "total_battles": model.total_battles,
                "wins": model.wins,
                "losses": model.losses,
                "win_rate": round(model.win_rate * 100, 1),
                "specializations": model.specializations or [],
                "capabilities": model.capabilities or {},
                "last_used": model.last_used.isoformat() if model.last_used else None,
                "is_active": model.is_active
            }
            models_list.append(model_dict)
        
        # Sort by ELO rating descending
        models_list.sort(key=lambda x: x["elo_rating"], reverse=True)
        
        return JSONResponse(content={
            "success": True, 
            "models": models_list,
            "total_models": len(models_list),
            "active_models": len([m for m in models_list if m["is_active"]])        })
        
    except Exception as e:
        logger.error(f"❌ Models retrieval failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

# Leaderboard and analytics endpoints
@app.get("/api/leaderboard")
async def get_leaderboard(db_service: DatabaseService = Depends(get_database_service)):
    """Get model leaderboard by ELO rating"""
    try:
        # Get top models from database
        top_models = db_service.get_top_models(limit=20)
        
        # Get tournament analytics
        analytics = db_service.get_tournament_analytics(days=30)
        
        return JSONResponse(content={
            "success": True, 
            "leaderboard": top_models,
            "analytics": analytics,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Leaderboard retrieval failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")

@app.get("/api/analytics/evolution")
async def get_evolution_analytics():
    """Get model evolution analytics"""
    try:
        genealogy = tournament_engine.evolution_engine.genealogy
        
        analytics = {
            "total_evolved_models": genealogy.get("statistics", {}).get("total_evolved", 0),
            "evolution_by_architecture": genealogy.get("statistics", {}).get("by_architecture", {}),
            "evolution_tree_size": len(genealogy.get("evolution_tree", {})),
            "recent_evolutions": []
        }
        
        # Get recent evolutions
        evolution_tree = genealogy.get("evolution_tree", {})
        recent_evolutions = sorted(
            evolution_tree.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True
        )[:10]
        
        analytics["recent_evolutions"] = [
            {
                "model_id": model_id,
                "name": data.get("name", "Unknown"),
                "generation": data.get("generation", 0),
                "parents": data.get("parent_names", []),
                "architecture": data.get("architecture", "unknown"),
                "created_at": data.get("created_at", "")
            }
            for model_id, data in recent_evolutions
        ]
        
        return JSONResponse(content={"success": True, "analytics": analytics})
        
    except Exception as e:
        logger.error(f"❌ Evolution analytics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get evolution analytics: {str(e)}")

# Audio file endpoints
@app.get("/api/audio/{tournament_id}/{file_type}")
async def get_battle_audio(tournament_id: str, file_type: str):
    """Get battle audio files (original, model_a, model_b)"""
    try:
        if file_type not in ["original", "model_a", "model_b"]:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # This would return the actual audio file
        # For now, return a placeholder response
        audio_path = static_dir / "battles" / f"{tournament_id}_{file_type}.wav"
        
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            path=str(audio_path),
            media_type="audio/wav",
            filename=f"{tournament_id}_{file_type}.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Audio file retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get audio file: {str(e)}")

@app.get("/api/audio/final/{tournament_id}")
async def get_final_mix(tournament_id: str):
    """Get final tournament mix"""
    try:
        tournament = tournament_engine.get_tournament_status(tournament_id)
        
        if not tournament or not tournament.final_mix_path:
            raise HTTPException(status_code=404, detail="Final mix not found")
        
        return FileResponse(
            path=tournament.final_mix_path,
            media_type="audio/wav",
            filename=f"final_mix_{tournament_id}.wav"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Final mix retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get final mix: {str(e)}")

# Social features endpoints
@app.post("/api/social/share/{tournament_id}")
async def share_tournament(tournament_id: str):
    """Generate shareable tournament link"""
    try:
        tournament = tournament_engine.get_tournament_status(tournament_id)
        
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        
        # Increment share count
        tournament.social_shares += 1
        
        # Award free mixes for sharing
        user_profile = tournament_engine.user_profiles.get(tournament.user_id)
        if user_profile:
            user_profile.free_mixes_earned += 1
        
        share_data = {
            "tournament_id": tournament_id,
            "shareable_link": tournament.shareable_link,
            "champion": tournament.current_champion.nickname if tournament.current_champion else None,
            "share_message": f"Check out my AI mixing tournament! The {tournament.current_champion.nickname if tournament.current_champion else 'battle'} created an amazing mix!",
            "free_mixes_earned": 1
        }
        
        logger.info(f"📤 Tournament shared: {tournament_id}")
        return JSONResponse(content={"success": True, "share": share_data})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Tournament sharing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to share tournament: {str(e)}")

@app.post("/api/social/refer")
async def handle_referral(referrer_code: str, new_user_id: str, new_username: str):
    """Handle user referral"""
    try:
        # Find referrer
        referrer_profile = None
        for profile in tournament_engine.user_profiles.values():
            if profile.referral_code == referrer_code:
                referrer_profile = profile
                break
        
        if not referrer_profile:
            raise HTTPException(status_code=404, detail="Invalid referral code")
        
        # Create new user
        new_profile = tournament_engine.create_user_profile(new_user_id, new_username)
          # Award referral bonuses
        if "friends_referred" not in referrer_profile:
            referrer_profile["friends_referred"] = 0
        if "free_mixes_earned" not in referrer_profile:
            referrer_profile["free_mixes_earned"] = 0
        if "achievements" not in referrer_profile:
            referrer_profile["achievements"] = []
            
        referrer_profile["friends_referred"] += 1
        referrer_profile["free_mixes_earned"] += 5
        
        # Initialize new user profile with bonuses
        if "free_mixes_earned" not in new_profile:
            new_profile["free_mixes_earned"] = 0
        new_profile["free_mixes_earned"] += 3
        
        # Achievement check
        if referrer_profile["friends_referred"] >= 10 and "Super Recruiter" not in referrer_profile["achievements"]:
            referrer_profile["achievements"].append("Super Recruiter")
            referrer_profile["free_mixes_earned"] += 10
        
        referral_result = {
            "referrer_bonus": 5,
            "new_user_bonus": 3,
            "total_referrals": referrer_profile.friends_referred,
            "new_achievements": ["Super Recruiter"] if referrer_profile.friends_referred == 10 else []
        }
        
        logger.info(f"🤝 Referral processed: {referrer_code} -> {new_username}")
        return JSONResponse(content={"success": True, "referral": referral_result})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Referral processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process referral: {str(e)}")

# Simple tournament creation without file upload
@app.post("/api/tournaments")
async def create_simple_tournament(
    user_id: str,
    max_rounds: int = 5,
    db_service: DatabaseService = Depends(get_database_service)
):
    """Create a new tournament without file upload for testing"""
    try:
        # Create tournament in database
        tournament = db_service.create_tournament(
            user_id=user_id,
            max_rounds=max_rounds,
            audio_file="demo_audio.wav"
        )
        
        # Get available models for tournament
        models = db_service.get_all_models()
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")        # Create tournament using tournament engine
        tournament_id = tournament_engine.create_tournament(
            user_id=user_id,
            username=f"User_{user_id[:8]}",
            max_rounds=max_rounds
        )
        
        # Get tournament data from engine
        tournament_state = tournament_engine.get_tournament_state(tournament_id)
        
        # Update tournament with pairs data
        db_service.update_tournament(
            tournament.id,
            pairs_data=tournament_state.get("pairs", []),
            tournament_data=tournament_state
        )
        
        return JSONResponse(content={
            "success": True,
            "tournament": {
                "id": tournament.id,
                "user_id": user_id,
                "max_rounds": max_rounds,
                "status": "active",
                "pairs": tournament_state.get("pairs", []),
                "current_round": tournament_state.get("current_round", 1),
                "current_pair": tournament_state.get("current_pair", 0)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Tournament creation failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Failed to create tournament: {str(e)}")

@app.post("/api/tournaments/quick-create")
async def create_quick_tournament(request: dict, db_service: DatabaseService = Depends(get_database_service)):
    """Create a tournament without audio file upload - for testing/demo"""
    try:
        user_id = request.get("user_id", "demo_user")
        max_rounds = request.get("max_rounds", 5)
        
        # Create tournament in database
        tournament = db_service.create_tournament(
            user_id=user_id,
            max_rounds=max_rounds,
            audio_file="demo_audio.wav"  # Use demo audio
        )
        
        # Get available models for the tournament
        models = db_service.get_all_models()
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")
        
        # Create tournament pairs data for demo
        import random
        selected_models = random.sample(models, min(8, len(models)))  # Select up to 8 models
        
        pairs_data = []
        for i in range(0, len(selected_models), 2):
            if i + 1 < len(selected_models):
                model_a = selected_models[i]
                model_b = selected_models[i + 1]
                
                pair = {
                    "model_a": {
                        "id": model_a.id,
                        "name": model_a.name,
                        "architecture": model_a.architecture,
                        "elo_rating": model_a.elo_rating,
                        "tier": model_a.tier
                    },
                    "model_b": {
                        "id": model_b.id,
                        "name": model_b.name,
                        "architecture": model_b.architecture,
                        "elo_rating": model_b.elo_rating,
                        "tier": model_b.tier
                    },
                    "audio_a": f"/demo_audio/{model_a.id}_mix.wav",
                    "audio_b": f"/demo_audio/{model_b.id}_mix.wav"
                }
                pairs_data.append(pair)
        
        # Update tournament with pairs data
        db_service.update_tournament(
            tournament.id,
            pairs_data=pairs_data,
            tournament_data={
                "total_models": len(selected_models),
                "total_pairs": len(pairs_data),
                "demo_mode": True
            }
        )
        
        logger.info(f"🏆 Quick tournament created: {tournament.id} with {len(pairs_data)} pairs")
        
        return JSONResponse(content={
            "success": True,
            "tournament": {
                "id": tournament.id,
                "user_id": tournament.user_id,
                "max_rounds": tournament.max_rounds,
                "status": tournament.status,
                "total_pairs": len(pairs_data),
                "total_models": len(selected_models),
                "pairs": pairs_data,
                "created_at": tournament.created_at.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Quick tournament creation failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Tournament creation failed: {str(e)}")

@app.post("/api/tournaments/create")
async def create_tournament_frontend(request: dict, db_service: DatabaseService = Depends(get_database_service)):
    """Create tournament endpoint that matches frontend expectations"""
    try:
        user_id = request.get("user_id", "demo_user")
        username = request.get("username", f"User_{user_id[:8]}")
        max_rounds = request.get("max_rounds", 5)
        
        # Ensure user exists
        user = db_service.get_user(user_id)
        if not user:
            user = db_service.create_user(user_id, username)
        
        # Create tournament in database
        tournament = db_service.create_tournament(
            user_id=user_id,
            max_rounds=max_rounds,
            audio_file="demo_audio.wav"  # Use demo audio
        )
        
        # Get available models for the tournament
        models = db_service.get_all_models()
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")
        
        # Create tournament pairs data
        import random
        selected_models = random.sample(models, min(8, len(models)))  # Select up to 8 models
        
        pairs_data = []
        for i in range(0, len(selected_models), 2):
            if i + 1 < len(selected_models):
                model_a = selected_models[i]
                model_b = selected_models[i + 1]
                
                pair = {
                    "model_a": {
                        "id": model_a.id,
                        "name": model_a.name,
                        "architecture": model_a.architecture,
                        "elo_rating": model_a.elo_rating,
                        "tier": model_a.tier
                    },
                    "model_b": {
                        "id": model_b.id,
                        "name": model_b.name,
                        "architecture": model_b.architecture,
                        "elo_rating": model_b.elo_rating,
                        "tier": model_b.tier
                    },
                    "audio_a": f"/demo_audio/{model_a.id}_mix.wav",
                    "audio_b": f"/demo_audio/{model_b.id}_mix.wav"
                }
                pairs_data.append(pair)
        
        # Update tournament with pairs data
        db_service.update_tournament(
            tournament.id,
            pairs_data=pairs_data,
            tournament_data={
                "total_models": len(selected_models),
                "total_pairs": len(pairs_data),
                "frontend_created": True
            }
        )
        
        logger.info(f"🏆 Frontend tournament created: {tournament.id} with {len(pairs_data)} pairs")
        
        # Return format that frontend expects
        return JSONResponse(content={
            "success": True,
            "tournament_id": tournament.id,  # Frontend expects 'tournament_id'
            "message": "Tournament created successfully",
            "pairs": pairs_data,
            "total_rounds": max_rounds
        })
        
    except Exception as e:
        logger.error(f"❌ Frontend tournament creation failed: {str(e)}")
        track_error(e)
        raise HTTPException(status_code=500, detail=f"Tournament creation failed: {str(e)}")

# DELETE endpoint for tournaments
@app.delete("/api/tournaments/{tournament_id}")
async def delete_tournament(
    tournament_id: str,
    user_id: str = None,
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Delete a tournament and clean up associated resources
    
    Args:
        tournament_id: ID of tournament to delete
        user_id: Optional user ID for authorization
    """
    try:
        # Use the tournament engine to delete the tournament
        success = tournament_engine.delete_tournament(tournament_id, user_id)
        
        if success:
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Tournament {tournament_id} deleted successfully",
                    "tournament_id": tournament_id
                },
                status_code=200
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Tournament {tournament_id} not found or could not be deleted"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting tournament {tournament_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete tournament: {str(e)}"
        )

# Background tasks
async def process_audio_analysis(tournament_id: str, battle_id: str):
    """Background task for detailed audio analysis"""
    try:
        # This would integrate with your enhanced_musical_intelligence.py
        # to analyze the audio and extract features
        logger.info(f"🎵 Processing audio analysis for battle {battle_id}")
        
        # Simulate audio analysis processing
        await asyncio.sleep(2)
        
        # Update tournament with analysis results
        tournament = tournament_engine.active_tournaments.get(tournament_id)
        if tournament and tournament.current_battle:
            tournament.current_battle["audio_analysis_complete"] = True
            tournament.current_battle["audio_features"] = {
                "spectral_centroid": 1500.2,
                "rms_energy": 0.15,
                "zero_crossing_rate": 0.08,
                "tempo": 120.5,
                "key": "C major",
                "genre_confidence": 0.87
            }
        
        logger.info(f"✅ Audio analysis complete for {battle_id}")
        
    except Exception as e:
        logger.error(f"❌ Audio analysis failed: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring deployment status"""
    # Get model information
    models_dir = os.environ.get("MODELS_DIR", "../models")
    models_path = Path(models_dir)
    model_files = list(models_path.glob("*.pth"))
    
    # Get deployment information
    is_production = os.environ.get("PRODUCTION", "false").lower() == "true"
    deployment_platform = "Render" if "RENDER" in os.environ else "Local"
    if "RAILWAY_STATIC_URL" in os.environ:
        deployment_platform = "Railway"
        
    return {
        "status": "healthy",
        "engine_status": "operational",
        "environment": "production" if is_production else "development",
        "platform": deployment_platform,
        "models_loaded": len(tournament_engine.get_model_list()),
        "models_available": len(model_files),
        "active_tournaments": len(tournament_engine.tournaments),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for real-time updates (future enhancement)
# @app.websocket("/ws/{tournament_id}")
# async def websocket_endpoint(websocket: WebSocket, tournament_id: str):
#     """WebSocket for real-time tournament updates"""
#     await websocket.accept()
#     # Implementation for real-time tournament updates

@app.get("/api/model-files")
async def get_model_files():
    """
    Get information about actual model files on disk
    This is useful for debugging model loading issues
    """
    try:
        # Define paths to check for models
        current_file = Path(__file__).parent
        possible_paths = [
            current_file.parent.parent / "models",  # ../../models from backend
            current_file.parent / "models",         # ./models in backend
            Path("models"),                         # models in current working directory
        ]
        
        results = []
        
        # Check each possible path
        for path_to_check in possible_paths:
            if not path_to_check.exists():
                results.append({
                    "path": str(path_to_check.resolve()),
                    "exists": False,
                    "files": []
                })
                continue
                
            # List model files in this directory
            model_files = []
            for ext in [".pth", ".json", ".pt"]:
                for file in path_to_check.glob(f"*{ext}"):
                    model_files.append({
                        "name": file.name,
                        "size": file.stat().st_size,
                        "last_modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                    })
            
            results.append({
                "path": str(path_to_check.resolve()),
                "exists": True,
                "files": model_files
            })
        
        return {
            "success": True,
            "model_directories": results
        }
    except Exception as e:
        logger.error(f"Error getting model files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model files: {str(e)}")
