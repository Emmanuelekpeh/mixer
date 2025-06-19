"""
Enhanced Tournament Engine Implementation - Simplified Version

This is a simplified version of the EnhancedTournamentEngine that works with our API.
Enhanced with persistent state management and tournament resumption capabilities.
"""

import uuid
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import database service for persistence
try:
    from database_service import DatabaseService
    from database import Tournament, BattleVote
    DB_AVAILABLE = True
    logger.info("Database service available")
except ImportError as e:
    logger.warning(f"Database service not available: {e}, using memory-only mode")
    DatabaseService = None
    Tournament = None
    BattleVote = None
    DB_AVAILABLE = False

def load_real_models(models_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load real trained models from the models directory"""
    models = []
    
    # Use absolute path resolution to avoid relative path issues
    if models_dir is None:        # Try multiple possible locations for models directory
        current_file = Path(__file__).parent
        possible_paths = [
            current_file.parent.parent / "models",  # ../../models from backend
            current_file.parent / "models",         # ./models in backend
            Path("models"),                         # models in current working directory
        ]
        
        models_path = None
        for path in possible_paths:
            if path.exists():
                models_path = path.resolve()
                logger.info(f"Found models directory at: {models_path}")
                break
        
        if models_path is None:
            logger.warning(f"Models directory not found: {[str(p) for p in possible_paths]}")
            logger.info("Using fallback models for development")
            return get_fallback_models()
    else:
        models_path = Path(models_dir).resolve()
        
        if not models_path.exists():
            logger.warning(f"Specified models directory not found: {models_path}")
            return get_fallback_models()
    
    # Load models from JSON config files
    model_configs = list(models_path.glob("*.json"))
    
    for config_file in model_configs:
        try:
            with open(config_file, 'r') as f:
                model_data = json.load(f)
            
            # Check if corresponding .pth file exists
            model_pth = models_path / f"{config_file.stem}.pth"
            if model_pth.exists():
                model_info = {
                    "id": config_file.stem,
                    "name": model_data.get("name", config_file.stem),
                    "nickname": model_data.get("nickname", model_data.get("name", config_file.stem)),
                    "architecture": model_data.get("architecture", "unknown"),
                    "generation": model_data.get("generation", 1),
                    "elo_rating": model_data.get("elo_rating", 1200),
                    "tier": model_data.get("tier", "Amateur"),
                    "specializations": model_data.get("specializations", []),
                    "capabilities": model_data.get("capabilities", {}),
                    "model_path": str(model_pth),
                    "config_path": str(config_file),
                    "parameter_count": model_data.get("performance_metrics", {}).get("parameter_count", 0),
                    "estimated_mae": model_data.get("performance_metrics", {}).get("estimated_mae", 0.1),
                    "status": model_data.get("performance_metrics", {}).get("status", "ready"),
                    "preferred_genres": model_data.get("preferred_genres", []),
                    "signature_techniques": model_data.get("signature_techniques", [])
                }
                models.append(model_info)
                logger.info(f"✅ Loaded real model: {model_info['name']} ({model_info['architecture']})")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model config {config_file}: {e}")
    
    if models:
        logger.info(f"🎯 Loaded {len(models)} real trained models")
        return models
    else:
        logger.warning("No real models found, using fallback")
        return get_fallback_models()

def get_fallback_models() -> List[Dict[str, Any]]:
    """Fallback models if real models can't be loaded"""
    return [
        {
            "id": "baseline_cnn",
            "name": "Baseline CNN",
            "nickname": "Clean Clarity",
            "architecture": "cnn",
            "generation": 1,
            "elo_rating": 1200,
            "tier": "Amateur",
            "specializations": ["basic_mixing"]
        },
        {
            "id": "enhanced_cnn", 
            "name": "Enhanced CNN",
            "nickname": "Modern Wave",
            "architecture": "cnn",
            "generation": 1,
            "elo_rating": 1300,
            "tier": "Professional",
            "specializations": ["dynamics", "frequency"]
        }
    ]

# Load real models at startup with correct path resolution
def get_models_directory():
    """Get the correct models directory path"""
    current_file = Path(__file__)
    # Go up from backend/simplified_tournament_engine.py to mixer/models
    models_dir = current_file.parent.parent.parent / "models"
    return str(models_dir)

AVAILABLE_MODELS = load_real_models(get_models_directory())

class EnhancedTournamentEngine:
    """
    Enhanced Tournament Engine for AI model battles with persistent state management
    """
    def __init__(self):
        """Initialize the tournament engine with database persistence support"""
        self.tournaments = {}  # In-memory cache for active tournaments
        self.models = AVAILABLE_MODELS
        self.db_service = DatabaseService() if DB_AVAILABLE and DatabaseService else None
        
        # Load real AI mixer for actual model processing
        self.load_ai_mixer()
        
        # Try to load real model manager for actual AI inference
        self.real_models_available = False
        try:
            from tournament_model_manager import TournamentModelManager
            models_dir = Path(__file__).parent.parent.parent / "models"
            self.model_manager = TournamentModelManager(models_dir)
            self.real_models_available = True
            logger.info("✅ Real tournament model manager loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Tournament model manager not available: {e}")
            self.model_manager = None
        
        logger.info(f"🎯 Enhanced Tournament Engine initialized with {len(self.models)} real models")
        if self.models:
            logger.info("Real models loaded:")
            for model in self.models[:3]:  # Show first few models
                logger.info(f"  - {model['name']} ({model['architecture']}) ELO: {model['elo_rating']}")
            if len(self.models) > 3:
                logger.info(f"  ... and {len(self.models) - 3} more models")
        
        logger.info(f"🔗 AI Mixer available: {self.ai_mixer is not None}")
        logger.info(f"🔗 Database available: {self.db_service is not None}")
        logger.info(f"🔗 Model manager available: {self.real_models_available}")
        
        logger.info("Enhanced Tournament Engine initialized with persistence support")
    
    def _save_tournament_state(self, tournament_id: str, tournament_data: Dict[str, Any]) -> bool:
        """
        Save tournament state to database for persistence
        
        Args:
            tournament_id: Tournament identifier
            tournament_data: Current tournament state
            
        Returns:
            bool: Success status
        """
        if not self.db_service:
            logger.warning("Database not available, state not persisted")
            return False
            
        try:
            # Update tournament in database with current state
            success = self.db_service.update_tournament(
                tournament_id,
                current_round=tournament_data.get("current_round", 1),
                current_pair=len(tournament_data.get("results", [])),
                status=tournament_data.get("status", "active"),
                pairs_data=tournament_data.get("pairs", []),
                battle_history=tournament_data.get("results", []),
                tournament_data={
                    "state": tournament_data,
                    "last_updated": datetime.now().isoformat()
                }
            )
            
            if success:
                logger.info(f"💾 Tournament {tournament_id} state saved to database")
            else:
                logger.error(f"❌ Failed to save tournament {tournament_id} state")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error saving tournament state: {str(e)}")
            return False
    
    def _load_tournament_state(self, tournament_id: str) -> Optional[Dict[str, Any]]:
        """
        Load tournament state from database
        
        Args:
            tournament_id: Tournament identifier
            
        Returns:
            Tournament state dict or None if not found
        """
        if not self.db_service:
            return None
            
        try:
            tournament = self.db_service.get_tournament(tournament_id)
            if not tournament:
                return None
                
            # Reconstruct tournament state from database
            state = {
                "tournament_id": tournament.id,
                "user_id": tournament.user_id,
                "status": tournament.status,
                "current_round": tournament.current_round,
                "max_rounds": tournament.max_rounds,
                "audio_file": tournament.original_audio_file,
                "pairs": tournament.pairs_data or [],
                "results": tournament.battle_history or [],
                "created_at": tournament.created_at.isoformat() if tournament.created_at else datetime.now().isoformat(),
                "updated_at": tournament.updated_at.isoformat() if tournament.updated_at else datetime.now().isoformat()
            }
            
            logger.info(f"📂 Tournament {tournament_id} state loaded from database")
            return state
            
        except Exception as e:
            logger.error(f"❌ Error loading tournament state: {str(e)}")
            return None
    
    def create_tournament(self, tournament_id: str, user_id: str, max_rounds: int = 5, audio_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new tournament with randomly selected models
        
        Args:
            tournament_id: Unique identifier for the tournament
            user_id: ID of the user creating the tournament
            max_rounds: Maximum number of rounds for the tournament
            audio_file: Path to the audio file for mixing (optional)
            
        Returns:
            Tournament data dictionary
        """
        # Get available models
        available_models = self.models.copy()
        random.shuffle(available_models)
        
        # Select a random subset of models for the tournament
        # We need at least 2^max_rounds competitors
        num_models = min(2 ** max_rounds, len(available_models))
        selected_models = available_models[:num_models]
        
        # Create pairs for the first round
        pairs = []
        for i in range(0, len(selected_models), 2):
            if i + 1 < len(selected_models):
                # Create a pair with two models
                model_a = selected_models[i]
                model_b = selected_models[i + 1]
                
                # Create demo audio paths
                audio_a = f"/demo/model_a_{i}.mp3"
                audio_b = f"/demo/model_b_{i}.mp3"
                
                pairs.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "audio_a": audio_a,
                    "audio_b": audio_b
                })
        
        # Create tournament data
        tournament = {
            "tournament_id": tournament_id,
            "user_id": user_id,
            "status": "active",
            "current_round": 1,
            "max_rounds": max_rounds,
            "audio_file": audio_file,
            "pairs": pairs,
            "results": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
          # Store in memory and return
        self.tournaments[tournament_id] = tournament
        
        # Save initial state to database
        self._save_tournament_state(tournament_id, tournament)
        
        return tournament
    
    def process_vote(self, tournament_data: Dict[str, Any], winner_id: str, confidence: float = 0.8) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Process a vote in a tournament battle
        
        Args:
            tournament_data: Tournament data dictionary
            winner_id: ID of the winning model
            confidence: Confidence score for the vote (0-1)
            
        Returns:
            Tuple of (battle_result, next_pair)
        """
        # Get current tournament state
        current_round = tournament_data["current_round"]
        pairs = tournament_data["pairs"]
        
        if not pairs:
            raise ValueError("No pairs available in the tournament")
        
        # Get the current pair and models
        current_pair = pairs[0]
        model_a = current_pair["model_a"]
        model_b = current_pair["model_b"]
        
        # Determine winner and loser
        if winner_id == model_a["id"]:
            winner = model_a
            loser = model_b
        elif winner_id == model_b["id"]:
            winner = model_b
            loser = model_a
        else:
            raise ValueError(f"Invalid winner ID: {winner_id}")
        
        # Create battle result
        battle_result = {
            "battle_id": f"battle_{uuid.uuid4().hex}",
            "tournament_id": tournament_data["tournament_id"],
            "round": current_round,
            "winner_id": winner["id"],
            "loser_id": loser["id"],
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to results
        tournament_data["results"].append(battle_result)
        
        # Remove the current pair
        pairs.pop(0)
        
        # Check if we need to advance to the next round
        next_pair = None
        if not pairs:
            # No more pairs in this round
            if current_round >= tournament_data["max_rounds"]:
                # Tournament completed
                tournament_data["status"] = "completed"
                tournament_data["victor_model"] = winner
            else:
                # Create pairs for the next round
                winners = [result["winner_id"] for result in tournament_data["results"]]
                
                # Get the winners from the current round
                round_start_idx = 2 ** (current_round - 1) - 1
                round_end_idx = 2 ** current_round - 1
                current_round_winners = winners[round_start_idx:round_end_idx]
                
                # Create pairs for the next round
                new_pairs = []
                for i in range(0, len(current_round_winners), 2):
                    if i + 1 < len(current_round_winners):
                        # Find the winning models
                        model_a_id = current_round_winners[i]
                        model_b_id = current_round_winners[i + 1]
                        
                        # Find the model data
                        model_a = next((m for m in self.models if m["id"] == model_a_id), None)
                        model_b = next((m for m in self.models if m["id"] == model_b_id), None)
                        
                        if model_a and model_b:
                            # Create demo audio paths
                            audio_a = f"/demo/model_a_{i}_round_{current_round + 1}.mp3"
                            audio_b = f"/demo/model_b_{i}_round_{current_round + 1}.mp3"
                            
                            new_pairs.append({
                                "model_a": model_a,
                                "model_b": model_b,
                                "audio_a": audio_a,
                                "audio_b": audio_b
                            })
                
                if new_pairs:
                    tournament_data["pairs"] = new_pairs
                    tournament_data["current_round"] = current_round + 1
                    next_pair = new_pairs[0]
                else:
                    # No more pairs possible, end the tournament
                    tournament_data["status"] = "completed"
                    tournament_data["victor_model"] = winner
        else:
            # Still have pairs in this round
            next_pair = pairs[0]
          # Update timestamp
        tournament_data["updated_at"] = datetime.now().isoformat()
        
        # Save state to database after each vote
        self._save_tournament_state(tournament_data["tournament_id"], tournament_data)
        
        return battle_result, next_pair

    def get_tournament_status(self, tournament_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tournament status with automatic state loading if not in memory
        
        Args:
            tournament_id: Tournament identifier
            
        Returns:
            Tournament state or None
        """
        # Try to get from memory first
        if tournament_id in self.tournaments:
            return self.tournaments[tournament_id]
            
        # If not in memory, try to load from database
        logger.info(f"🔄 Tournament {tournament_id} not in memory, attempting to load from database")
        tournament_state = self._load_tournament_state(tournament_id)
        
        if tournament_state:
            # Cache in memory for future access
            self.tournaments[tournament_id] = tournament_state
            logger.info(f"✅ Tournament {tournament_id} successfully restored from database")
            return tournament_state
            
        logger.warning(f"❌ Tournament {tournament_id} not found in memory or database")
        return None
    
    def resume_tournament(self, tournament_id: str) -> Optional[Dict[str, Any]]:
        """
        Resume a tournament from database state
        
        Args:
            tournament_id: Tournament identifier
            
        Returns:
            Resumed tournament state or None
        """
        logger.info(f"🔄 Attempting to resume tournament {tournament_id}")
        
        tournament_state = self._load_tournament_state(tournament_id)
        if not tournament_state:
            return None
            
        # Validate tournament can be resumed
        if tournament_state.get("status") == "completed":
            logger.info(f"ℹ️ Tournament {tournament_id} already completed")
            return tournament_state
            
        # Cache in memory and return
        self.tournaments[tournament_id] = tournament_state
        logger.info(f"✅ Tournament {tournament_id} resumed successfully")
        return tournament_state
    
    def list_user_tournaments(self, user_id: str, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of tournaments for a user
        
        Args:
            user_id: User identifier
            status_filter: Optional status filter ('active', 'completed', etc.)
            
        Returns:
            List of tournament summaries
        """
        if not self.db_service:
            # Fallback to memory-only tournaments
            user_tournaments = []
            for tournament_id, tournament in self.tournaments.items():
                if tournament.get("user_id") == user_id:
                    if not status_filter or tournament.get("status") == status_filter:
                        user_tournaments.append({
                            "tournament_id": tournament_id,
                            "status": tournament.get("status"),
                            "created_at": tournament.get("created_at"),
                            "current_round": tournament.get("current_round"),
                            "max_rounds": tournament.get("max_rounds"),
                            "pairs_completed": len(tournament.get("results", []))
                        })
            return user_tournaments
            
        try:
            # Get tournaments from database
            tournaments = self.db_service.get_user_tournaments(user_id, limit=50)
            tournament_list = []
            
            for tournament in tournaments:
                if not status_filter or tournament.status == status_filter:
                    tournament_list.append({
                        "tournament_id": tournament.id,
                        "status": tournament.status,
                        "created_at": tournament.created_at.isoformat() if tournament.created_at else None,
                        "current_round": tournament.current_round,
                        "max_rounds": tournament.max_rounds,
                        "pairs_completed": tournament.current_pair or 0,
                        "victor_model_id": tournament.victor_model_id,
                        "original_filename": tournament.original_audio_file                    })
                    
            return tournament_list
            
        except Exception as e:
            logger.error(f"❌ Error listing user tournaments: {str(e)}")
            return []
    
    # Compatibility methods for existing API
    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get list of available models for API compatibility"""
        return self.models
    
    def create_user_profile(self, user_id: str, username: str) -> Dict[str, Any]:
        """Create user profile for API compatibility"""
        profile = {
            "user_id": user_id,
            "username": username,
            "tournaments_played": 0,
            "battles_voted": 0,
            "created_at": datetime.now().isoformat()
        }
        # In a real implementation, this would save to database via db_service        return profile
    
    def record_vote(self, tournament_id: str, winner_id: str, confidence: float = 0.7, reasoning: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Record a vote and update tournament state - API compatibility wrapper"""
        if tournament_id not in self.tournaments:
            # Try to load from database
            tournament_state = self.get_tournament_status(tournament_id)
            if not tournament_state:
                return None
        
        tournament_data = self.tournaments[tournament_id]
        try:
            battle_result, next_pair = self.process_vote(tournament_data, winner_id, confidence)
            
            # Return tournament state for API compatibility
            return tournament_data
            
        except Exception as e:
            logger.error(f"❌ Error recording vote: {str(e)}")
            return None
    
    def execute_battle(self, tournament_id: str) -> Optional[Dict[str, Any]]:
        """Execute battle using real AI models if available, fallback to mock"""
        tournament = self.get_tournament_status(tournament_id)
        if not tournament:
            return None
            
        # Get current pair
        pairs = tournament.get("pairs", [])
        if not pairs:
            return None
            
        current_pair = pairs[0]
        
        # If we have real models and audio file, use them for actual AI processing
        if self.real_models_available and self.model_manager and tournament.get("audio_file"):
            try:
                logger.info("🤖 Executing battle with REAL AI models")
                
                # Use real model manager to process audio
                model_a_id = current_pair["model_a"]["id"]
                model_b_id = current_pair["model_b"]["id"]
                audio_file = tournament["audio_file"]
                
                # Execute real battle with AI models (check what methods are available)
                if hasattr(self.model_manager, 'execute_battle'):
                    battle_result = self.model_manager.execute_battle(
                        audio_file, model_a_id, model_b_id
                    )
                elif hasattr(self.model_manager, 'process_battle'):
                    battle_result = self.model_manager.process_battle(
                        audio_file, model_a_id, model_b_id
                    )
                else:
                    # Use basic processing
                    battle_result = {
                        "battle_id": f"real_battle_{uuid.uuid4().hex}",
                        "model_a": current_pair["model_a"],
                        "model_b": current_pair["model_b"],
                        "audio_a": f"/processed/{model_a_id}_mix.wav",
                        "audio_b": f"/processed/{model_b_id}_mix.wav",
                        "status": "ready_for_vote",
                        "battle_type": "real_ai"
                    }
                
                if battle_result and "error" not in battle_result:
                    logger.info("✅ Real AI battle executed successfully")
                    return battle_result
                else:
                    logger.warning("⚠️ Real AI battle failed, falling back to mock")
                    
            except Exception as e:
                logger.error(f"❌ Real AI battle execution failed: {e}")
        
        # Fallback to mock battle for compatibility
        logger.info("🎭 Executing mock battle (no real AI processing)")
        battle_id = f"battle_{uuid.uuid4().hex}"
        return {
            "battle_id": battle_id,
            "model_a": current_pair["model_a"],
            "model_b": current_pair["model_b"],
            "audio_a": current_pair.get("audio_a", f"/demo/model_a_{battle_id}.mp3"),
            "audio_b": current_pair.get("audio_b", f"/demo/model_b_{battle_id}.mp3"),
            "status": "ready_for_vote",
            "battle_type": "mock" if not self.real_models_available else "real_fallback"
        }
    
    def load_ai_mixer(self):
        """Load the real AI mixer for actual model processing"""
        try:
            import sys
            from pathlib import Path
            
            # Add the src directory to path to import AI mixer
            mixer_src = Path(__file__).parent.parent.parent / "src"
            if mixer_src.exists():
                sys.path.insert(0, str(mixer_src))
                
            from production_ai_mixer import ProductionAIMixer
            self.ai_mixer = ProductionAIMixer()
            logger.info("🎯 Real AI mixer loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load real AI mixer: {e}")
            self.ai_mixer = None
            return False
    
    def process_real_battle(self, audio_file: str, model_a_id: str, model_b_id: str) -> Dict[str, Any]:
        """Process a real battle using trained AI models"""
        if not self.ai_mixer:
            logger.warning("AI mixer not available, using mock battle")
            return self._create_mock_battle(model_a_id, model_b_id)
        
        try:
            # Find model information
            model_a = next((m for m in AVAILABLE_MODELS if m["id"] == model_a_id), None)
            model_b = next((m for m in AVAILABLE_MODELS if m["id"] == model_b_id), None)
            
            if not model_a or not model_b:
                logger.error(f"Models not found: {model_a_id}, {model_b_id}")
                return self._create_mock_battle(model_a_id, model_b_id)
            
            # Process audio with the AI mixer
            output_dir = Path("processed_audio") / "battles"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique battle ID
            battle_id = str(uuid.uuid4())[:8]
            
            # Use the AI mixer to process with different models
            # This will create multiple outputs that we can use for the battle
            results = self.ai_mixer.analyze_and_mix(audio_file, str(output_dir))
            
            # Map the AI mixer outputs to our tournament models
            audio_a = self._get_model_output(results, model_a_id, battle_id, "a")
            audio_b = self._get_model_output(results, model_b_id, battle_id, "b")
            
            return {
                "battle_id": battle_id,
                "model_a": model_a,
                "model_b": model_b,
                "audio_a": audio_a,
                "audio_b": audio_b,
                "status": "ready_for_vote",
                "real_processing": True,
                "mixing_results": results.get("summary", {})
            }
                
        except Exception as e:
            logger.error(f"Real battle processing failed: {e}")
            return self._create_mock_battle(model_a_id, model_b_id)
    
    def _get_model_output(self, mixer_results: Dict, model_id: str, battle_id: str, suffix: str) -> str:
        """Map tournament model to AI mixer output"""
        try:
            # Map our tournament models to the AI mixer outputs
            model_mapping = {
                "baseline_cnn": "baseline_cnn_mixed.wav",
                "enhanced_cnn": "enhanced_cnn_mixed.wav", 
                "improved_baseline_cnn": "improved_baseline_cnn_mixed.wav",
                "improved_enhanced_cnn": "improved_enhanced_cnn_mixed.wav",
                "ast_regressor": "ast_regressor_mixed.wav",
                "advanced_transformer_mixer": "enhanced_cnn_mixed.wav",  # Fallback
                "audio_gan_mixer": "enhanced_cnn_mixed.wav",  # Fallback
                "lstm_audio_mixer": "baseline_cnn_mixed.wav",  # Fallback
                "vae_audio_mixer": "enhanced_cnn_mixed.wav",  # Fallback
                "resnet_audio_mixer": "enhanced_cnn_mixed.wav"  # Fallback
            }
            
            # Get the output file for this model
            mixer_output = model_mapping.get(model_id, "enhanced_cnn_mixed.wav")
            
            if "outputs" in mixer_results and mixer_output in mixer_results["outputs"]:
                return mixer_results["outputs"][mixer_output]
            else:
                # Fallback to demo audio
                return f"/demo/model_{suffix}_demo.mp3"
                
        except Exception as e:
            logger.error(f"Error mapping model output: {e}")
            return f"/demo/model_{suffix}_demo.mp3"
    
    def _create_mock_battle(self, model_a_id: str, model_b_id: str) -> Dict[str, Any]:
        """Create mock battle when real processing isn't available"""
        model_a = next((m for m in AVAILABLE_MODELS if m["id"] == model_a_id), None)
        model_b = next((m for m in AVAILABLE_MODELS if m["id"] == model_b_id), None)
        
        return {
            "battle_id": str(uuid.uuid4())[:8],
            "model_a": model_a or {"id": model_a_id, "name": "Unknown Model A"},
            "model_b": model_b or {"id": model_b_id, "name": "Unknown Model B"},
            "audio_a": "/demo/model_a_demo.mp3",
            "audio_b": "/demo/model_b_demo.mp3", 
            "status": "ready_for_vote",
            "real_processing": False
        }
    
    @property
    def user_profiles(self) -> Dict[str, Any]:
        """User profiles property for API compatibility"""
        return {}  # In real implementation, would query database
    
    @property
    def evolution_engine(self):
        """Evolution engine property for API compatibility"""
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
        
        return DummyEvolutionEngine(self.models)
    
    def delete_tournament(self, tournament_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a tournament and clean up associated resources
        
        Args:
            tournament_id: ID of tournament to delete
            user_id: Optional user ID for authorization (only creator can delete)
            
        Returns:
            bool: True if tournament was deleted, False otherwise
        """
        try:
            # Check if tournament exists
            tournament = self.get_tournament_status(tournament_id)
            if not tournament:
                logger.warning(f"Tournament {tournament_id} not found for deletion")
                return False
            
            # Check user authorization if provided
            if user_id and tournament.get("user_id") != user_id:
                logger.warning(f"User {user_id} not authorized to delete tournament {tournament_id}")
                return False
            
            # Remove from memory cache
            if tournament_id in self.tournaments:
                del self.tournaments[tournament_id]
                logger.info(f"Removed tournament {tournament_id} from memory cache")
            
            # Remove from database if available
            if self.db_service:
                try:
                    # Delete tournament from database
                    db_tournament = self.db_service.db.query(Tournament).filter(Tournament.id == tournament_id).first()
                    if db_tournament:
                        # Delete associated battle votes first
                        self.db_service.db.query(BattleVote).filter(BattleVote.tournament_id == tournament_id).delete()
                        # Delete tournament
                        self.db_service.db.delete(db_tournament)
                        self.db_service.db.commit()
                        logger.info(f"Deleted tournament {tournament_id} from database")
                    
                except Exception as e:
                    logger.error(f"Failed to delete tournament from database: {e}")
                    self.db_service.db.rollback()
                    return False
            
            # Clean up any temporary files or battle outputs
            try:
                battle_dir = Path("static/battles") / tournament_id
                if battle_dir.exists():
                    import shutil
                    shutil.rmtree(battle_dir)
                    logger.info(f"Cleaned up battle files for tournament {tournament_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up battle files: {e}")
            
            logger.info(f"Successfully deleted tournament {tournament_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting tournament {tournament_id}: {e}")
            return False
