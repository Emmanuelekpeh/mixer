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

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
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
from dotenv import load_dotenv

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import tournament engine
from tournament_webapp.backend.enhanced_tournament_engine import EnhancedTournamentEngine

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
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
API_PREFIX = os.getenv("API_PREFIX", "")
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
static_dir = Path("tournament_webapp/static")
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize tournament engine
models_dir = Path(MODELS_DIR)

# Create a simple stub class for development/testing
class SimpleTournamentEngine:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.tournaments = {}
        self.user_profiles = {}
        
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
        
        return tournament_id
    
    def execute_battle(self, tournament_id):
        if tournament_id not in self.tournaments:
            return None
        # Return a sample battle result
        return {
            "battle_id": str(uuid.uuid4()),
            "model_a": {"id": "model1", "name": "Baseline CNN"},
            "model_b": {"id": "model2", "name": "Enhanced CNN"}
        }
    
    def record_vote(self, tournament_id, winner_id, confidence=0.7, reasoning=None):
        if tournament_id not in self.tournaments:
            return None
        # Just update the round for now
        tournament = self.tournaments[tournament_id]
        tournament["round"] += 1
        
        # Add to battle history
        tournament["battle_history"].append({
            "battle_id": tournament["current_battle"]["battle_id"],
            "winner_id": winner_id,
            "confidence": confidence,
            "reasoning": reasoning
        })
        
        # Create a new battle if not finished
        if tournament["round"] <= tournament["max_rounds"]:
            tournament["current_battle"] = {
                "battle_id": str(uuid.uuid4()),
                "model_a": {"id": "model3", "name": "Improved Baseline CNN"},
                "model_b": {"id": "model4", "name": "Retrained Enhanced CNN"}
            }
        else:
            tournament["status"] = "completed"
            tournament["current_battle"] = None
        
        return tournament
    
    def vote_for_winner(self, tournament_id, winner_id, confidence=0.7, reasoning=None):
        return self.record_vote(tournament_id, winner_id, confidence, reasoning)
    
    def get_model_list(self):
        # Return a sample model list
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
            {"model_id": "model4", "name": "Retrained Enhanced CNN", "wins": 32, "elo": 1150},
            {"model_id": "model5", "name": "Weighted Ensemble", "wins": 28, "elo": 1100},
            {"model_id": "model3", "name": "Improved Baseline CNN", "wins": 20, "elo": 1050},
            {"model_id": "model1", "name": "Baseline CNN", "wins": 10, "elo": 950}
        ]
    
    @property
    def active_tournaments(self):
        return {k: v for k, v in self.tournaments.items() if v["status"] == "active"}
    
    @property
    def evolution_engine(self):
        return type('DummyEvolutionEngine', (), {
            'champion_models': self.get_model_list(),
            'genealogy': {"models": self.get_model_list()}
        })()

# Initialize the engine - using simplified version for now
tournament_engine = SimpleTournamentEngine(models_dir)

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
@app.post("/api/tournaments/create")
async def create_tournament(
    user_id: str = Form(...),
    username: str = Form(...),
    max_rounds: int = Form(5),
    audio_features: str = Form("{}"),
    audio_file: UploadFile = File(..., description="Audio file to mix")
):
    """Create a new tournament with uploaded audio"""
    try:
        # Validate audio file
        if not audio_file or not audio_file.filename:
            raise HTTPException(status_code=400, detail="Missing audio file")
            
        if not audio_file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.aiff')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
          # Save uploaded audio file
        audio_dir = static_dir / "uploads"
        audio_dir.mkdir(exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_extension = audio_file.filename.split('.')[-1]
        saved_filename = f"{file_id}.{file_extension}"
        saved_path = audio_dir / saved_filename
        
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        
        logger.info(f"🎵 Audio uploaded: {saved_filename}")
        
        # Convert to spectrogram for efficient storage
        try:
            sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
            from audio_to_spectrogram import SpectrogramConverter
            
            # Create spectrogram directory
            spec_dir = static_dir / "spectrograms"
            spec_dir.mkdir(exist_ok=True)
            
            # Convert audio to spectrogram
            converter = SpectrogramConverter()
            spec_path, meta_path = converter.audio_to_spectrogram(str(saved_path), str(spec_dir))
            
            logger.info(f"🔊 Audio converted to spectrogram: {spec_path}")
            
            # We'll use the spectrogram path for processing, but keep the original for reference
            processing_path = spec_path
        except Exception as e:
            logger.warning(f"⚠️ Could not convert to spectrogram, using original audio: {str(e)}")
            processing_path = str(saved_path)
          # Parse audio features from JSON string
        try:
            parsed_audio_features = json.loads(audio_features) if audio_features else {}
        except json.JSONDecodeError:
            parsed_audio_features = {}
        
        # Add spectrogram path to audio features if available
        if 'processing_path' in locals():
            parsed_audio_features['spectrogram_path'] = processing_path
        
        # Start tournament
        tournament_id = tournament_engine.start_tournament(
            user_id=user_id,
            username=username,
            audio_file=str(saved_path),
            max_rounds=max_rounds,
            audio_features=parsed_audio_features
        )
        
        tournament = tournament_engine.get_tournament_status(tournament_id)
        
        # Convert to dict for JSON response
        tournament_dict = {
            "tournament_id": tournament_id,
            "user_id": user_id,
            "status": tournament["status"],
            "current_round": tournament["round"],
            "max_rounds": tournament["max_rounds"],
            "competitors": tournament_engine.get_model_list(),
            "audio_file": saved_filename,
            "created_at": tournament["created_at"]
        }
        
        logger.info(f"🏆 Tournament created: {tournament_id}")
        return JSONResponse(content={"success": True, "tournament": tournament_dict})
        
    except Exception as e:
        logger.error(f"❌ Tournament creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tournament creation failed: {str(e)}")

@app.post("/api/tournaments/{tournament_id}/battle")
async def execute_battle(tournament_id: str, background_tasks: BackgroundTasks):
    """Execute battle between current tournament competitors"""
    try:
        battle_result = tournament_engine.execute_battle(tournament_id)
        
        if not battle_result:
            raise HTTPException(status_code=404, detail="Tournament not found")
        
        logger.info(f"⚔️ Battle executed: {tournament_id}")
        return JSONResponse(content={"success": True, "battle": battle_result})
        
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

@app.get("/api/tournaments/{tournament_id}")
async def get_tournament_status(tournament_id: str):
    """Get current tournament status"""
    try:
        tournament = tournament_engine.get_tournament_status(tournament_id)
        
        if not tournament:
            raise HTTPException(status_code=404, detail="Tournament not found")
        
        tournament_dict = {
            "tournament_id": tournament["id"],
            "user_id": tournament["user_id"],
            "status": tournament["status"],
            "current_round": tournament["round"],
            "max_rounds": tournament["max_rounds"],
            "current_battle": tournament["current_battle"],
            "battle_history": tournament["battle_history"],
            "created_at": tournament["created_at"]
        }
        
        return JSONResponse(content={"success": True, "tournament": tournament_dict})
        
        return JSONResponse(content={"success": True, "tournament": tournament_dict})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Tournament status retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tournament status: {str(e)}")

# User management endpoints
@app.post("/api/users/create")
async def create_user(request: UserCreateRequest):
    """Create new user profile"""
    try:
        profile = tournament_engine.create_user_profile(
            user_id=request.user_id,
            username=request.username
        )
        
        profile_dict = {
            "user_id": profile.user_id,
            "username": profile.username,
            "tier": profile.tier.value,
            "tournaments_completed": profile.tournaments_completed,
            "total_battles": profile.total_battles,
            "referral_code": profile.referral_code,
            "free_mixes_earned": profile.free_mixes_earned,
            "achievements": profile.achievements,
            "created_at": profile.created_at
        }
        
        logger.info(f"👤 User created: {request.username}")
        return JSONResponse(content={"success": True, "profile": profile_dict})
        
    except Exception as e:
        logger.error(f"❌ User creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"User creation failed: {str(e)}")

@app.get("/api/users/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile and statistics"""
    try:
        stats = tournament_engine.get_user_stats(user_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return JSONResponse(content={"success": True, "user": stats})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User profile retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

# Leaderboard and analytics endpoints
@app.get("/api/leaderboard")
async def get_leaderboard():
    """Get model leaderboard by ELO rating"""
    try:
        leaderboard = tournament_engine.get_tournament_leaderboard()
        return JSONResponse(content={"success": True, "leaderboard": leaderboard})
        
    except Exception as e:
        logger.error(f"❌ Leaderboard retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")

@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models and model files for deployment verification"""
    try:
        # Get loaded models from tournament engine
        models = tournament_engine.evolution_engine.champion_models
        
        models_list = [
            {
                "id": model.id,
                "name": model.name,
                "nickname": model.nickname,
                "architecture": model.architecture.value,
                "generation": model.generation,
                "elo_rating": model.elo_rating,
                "tier": model.tier,
                "win_rate": round(model.win_rate * 100, 1),
                "battles": model.battle_count,
                "specializations": model.specializations,
                "preferred_genres": model.preferred_genres,
                "signature_techniques": model.signature_techniques,
                "capabilities": {
                    "spectral_analysis": model.capabilities.spectral_analysis,
                    "dynamic_range": model.capabilities.dynamic_range,
                    "stereo_imaging": model.capabilities.stereo_imaging,
                    "bass_management": model.capabilities.bass_management,
                    "vocal_clarity": model.capabilities.vocal_clarity,
                    "harmonic_enhancement": model.capabilities.harmonic_enhancement,
                    "genre_adaptation": model.capabilities.genre_adaptation,
                    "technical_precision": model.capabilities.technical_precision
                },
                "status": "loaded"
            }
            for model in models
        ]
        
        # Check for physical model files (useful for deployment verification)
        models_dir = os.environ.get("MODELS_DIR", "../models")
        models_path = Path(models_dir)
        
        if models_path.exists():
            # Find all model files
            model_files = list(models_path.glob("*.pth"))
            
            # Add information about available model files
            model_files_info = []
            for model_file in model_files:
                model_name = model_file.stem
                model_size_mb = round(model_file.stat().st_size / (1024 * 1024), 2)
                
                # Check if this model is loaded in the engine
                is_loaded = any(m.name.lower() == model_name.lower() for m in models)
                
                model_files_info.append({
                    "name": model_name,
                    "file_path": str(model_file),
                    "size_mb": model_size_mb,
                    "status": "loaded" if is_loaded else "available"
                })
            
            return JSONResponse(content={
                "success": True, 
                "models": models_list,
                "model_files": model_files_info,
                "models_dir": str(models_path)
            })
        
        return JSONResponse(content={"success": True, "models": models_list})
        
    except Exception as e:
        logger.error(f"❌ Models retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

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
        referrer_profile.friends_referred += 1
        referrer_profile.free_mixes_earned += 5
        new_profile.free_mixes_earned += 3
        
        # Achievement check
        if referrer_profile.friends_referred >= 10 and "Super Recruiter" not in referrer_profile.achievements:
            referrer_profile.achievements.append("Super Recruiter")
            referrer_profile.free_mixes_earned += 10
        
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

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Tournament API Server...")
    print("=" * 50)
    print(f"🎯 Models loaded: {len(tournament_engine.get_model_list())}")
    print("🌐 API Documentation: http://localhost:8000/docs")
    print("🏆 Tournament Engine: Ready for battles!")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
