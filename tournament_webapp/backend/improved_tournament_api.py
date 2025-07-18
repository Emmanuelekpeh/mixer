#!/usr/bin/env python3
"""
Improved Tournament API
======================

Enhanced tournament creation with better formats and user experience.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from improved_tournament_structure import ImprovedTournamentStructure, suggest_tournament_format

router = APIRouter()

class TournamentRequest(BaseModel):
    name: str
    format: str = "balanced"  # quick, balanced, comprehensive, competitive, architecture
    max_models: Optional[int] = 8
    description: Optional[str] = ""

class TournamentResponse(BaseModel):
    id: str
    name: str
    format: str
    total_battles: int
    total_models: int
    current_battle: Dict[str, Any]
    progress: Dict[str, Any]
    description: str

def get_available_models() -> List[Dict[str, Any]]:
    """Get available models from the database"""
    try:
        from database_service import DatabaseService
        
        with DatabaseService() as db_service:
            models = db_service.get_all_models()
            return [
                {
                    "id": model.id,
                    "name": model.name,
                    "architecture": model.architecture,
                    "elo_rating": model.elo_rating,
                    "tier": model.tier,
                    "is_active": model.is_active
                }
                for model in models if model.is_active
            ]
    except Exception as e:
        # Fallback to sample data for testing
        return [
            {"id": "ast_transformer", "name": "AST Transformer", "architecture": "transformer", "elo_rating": 1493},
            {"id": "advanced_transformer", "name": "Advanced Transformer", "architecture": "transformer", "elo_rating": 1465},
            {"id": "vae_mixer", "name": "VAE Mixer", "architecture": "vae", "elo_rating": 1444},
            {"id": "resnet_mixer", "name": "ResNet Mixer", "architecture": "resnet", "elo_rating": 1443},
            {"id": "lstm_mixer", "name": "LSTM Mixer", "architecture": "lstm", "elo_rating": 1411},
            {"id": "audio_gan", "name": "Audio GAN", "architecture": "gan", "elo_rating": 1360},
            {"id": "baseline_cnn", "name": "Baseline CNN", "architecture": "cnn", "elo_rating": 1319},
            {"id": "enhanced_cnn", "name": "Enhanced CNN", "architecture": "cnn", "elo_rating": 1287}
        ]

@router.get("/tournament-formats")
async def get_tournament_formats():
    """Get available tournament formats with descriptions"""
    return {
        "formats": {
            "quick": {
                "name": "Quick Battle",
                "description": "3 random battles between models - perfect for testing",
                "battles": "3",
                "duration": "5-10 minutes"
            },
            "balanced": {
                "name": "Bracket Tournament", 
                "description": "Single elimination bracket - classic tournament format",
                "battles": "7 (for 8 models)",
                "duration": "15-20 minutes"
            },
            "comprehensive": {
                "name": "Round Robin",
                "description": "Every model plays every other model - most thorough",
                "battles": "15 (for 6 models)",
                "duration": "30-45 minutes"
            },
            "competitive": {
                "name": "Swiss System",
                "description": "Everyone plays same number of rounds - fair and competitive",
                "battles": "12 (4 rounds)",
                "duration": "20-30 minutes"
            },
            "architecture": {
                "name": "Architecture Showdown",
                "description": "Best model from each architecture competes",
                "battles": "Variable",
                "duration": "10-15 minutes"
            }
        }
    }

@router.post("/tournaments/create-improved")
async def create_improved_tournament(request: TournamentRequest):
    """Create a tournament with improved format"""
    try:
        # Get available models
        models = get_available_models()
        
        if len(models) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 models for tournament")
        
        # Create tournament structure
        tournament_structure = suggest_tournament_format(models, request.format)
        
        # Generate tournament ID
        import time
        tournament_id = f"tournament_{int(time.time() * 1000)}"
        
        # Get first battle
        current_battle = tournament_structure['pairs'][0] if tournament_structure['pairs'] else None
        
        if not current_battle:
            raise HTTPException(status_code=400, detail="No battles could be created")
        
        # Calculate progress
        progress = {
            "current_round": tournament_structure['current_round'],
            "total_rounds": tournament_structure['total_rounds'],
            "battles_completed": 0,
            "total_battles": tournament_structure['total_battles'],
            "percentage": 0
        }
        
        # Store tournament (in a real implementation, this would go to database)
        tournament_data = {
            "id": tournament_id,
            "name": request.name,
            "format": tournament_structure['format'],
            "structure": tournament_structure,
            "current_battle_index": 0,
            "battles_completed": 0,
            "created_at": time.time()
        }
        
        return TournamentResponse(
            id=tournament_id,
            name=request.name,
            format=tournament_structure['format'],
            total_battles=tournament_structure['total_battles'],
            total_models=tournament_structure['total_models'],
            current_battle={
                "battle_id": current_battle['pair_id'],
                "round": current_battle['round'],
                "model_a": current_battle['model_a'],
                "model_b": current_battle['model_b'],
                "battle_type": current_battle.get('battle_type', 'Standard Battle')
            },
            progress=progress,
            description=tournament_structure.get('description', f"{request.name} - {tournament_structure['format']} format")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tournament: {str(e)}")

@router.get("/tournaments/{tournament_id}/next-battle")
async def get_next_battle(tournament_id: str):
    """Get the next battle in the tournament"""
    # In a real implementation, this would fetch from database
    # For now, return a sample next battle
    return {
        "has_next_battle": True,
        "battle": {
            "battle_id": "R1P2",
            "round": 1,
            "model_a": {"id": "lstm_mixer", "name": "LSTM Mixer"},
            "model_b": {"id": "audio_gan", "name": "Audio GAN"},
            "battle_type": "Standard Battle"
        },
        "progress": {
            "current_round": 1,
            "total_rounds": 3,
            "battles_completed": 1,
            "total_battles": 7,
            "percentage": 14.3
        }
    }

@router.post("/tournaments/{tournament_id}/vote-improved")
async def vote_improved(tournament_id: str, winner_id: str, loser_id: str):
    """Record a vote with improved tournament progression"""
    try:
        # In a real implementation, this would:
        # 1. Record the vote in database
        # 2. Update ELO ratings
        # 3. Advance tournament to next battle
        # 4. Check if tournament is complete
        
        return {
            "success": True,
            "message": f"Vote recorded: {winner_id} beats {loser_id}",
            "tournament_status": "in_progress",
            "next_battle_available": True,
            "elo_changes": {
                winner_id: "+15",
                loser_id: "-15"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record vote: {str(e)}")

# Test endpoint
@router.get("/test-improved-tournaments")
async def test_improved_tournaments():
    """Test endpoint to show all tournament formats"""
    models = get_available_models()
    tournament = ImprovedTournamentStructure(models)
    
    formats = {
        "quick_battle": tournament.create_quick_battle(3),
        "bracket_tournament": tournament.create_bracket_tournament(8),
        "architecture_showdown": tournament.create_architecture_showdown(),
        "swiss_tournament": tournament.create_swiss_tournament(3),
        "round_robin": tournament.create_round_robin(6)
    }
    
    return {
        "available_models": len(models),
        "tournament_formats": {
            name: {
                "format": structure['format'],
                "total_battles": structure['total_battles'],
                "total_models": structure['total_models'],
                "total_rounds": structure['total_rounds'],
                "description": structure.get('description', ''),
                "first_battle": structure['pairs'][0] if structure['pairs'] else None
            }
            for name, structure in formats.items()
        }
    }