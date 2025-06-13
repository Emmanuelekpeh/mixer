#!/usr/bin/env python3
"""
🏆 AI Model Tournament Engine
============================

Core tournament logic for AI model battles with evolutionary learning.
Designed for current CNN models but extensible to future architectures:
- CNNs vs Transformers vs Diffusion Models
- Multi-modal vs Single-modal approaches
- Reinforcement Learning integration
- Collective intelligence emergence
"""

import torch
import torch.nn as nn
import numpy as np
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import copy

@dataclass
class ModelInfo:
    """Information about an AI model competitor"""
    id: str
    name: str
    architecture: str  # 'cnn', 'transformer', 'diffusion', 'hybrid'
    generation: int    # 0 = original, 1+ = evolved
    parent_ids: List[str]  # For tracking evolution lineage
    performance_score: float
    specializations: List[str]  # ['bass', 'vocals', 'dynamics', 'spatial']
    strengths: Dict[str, float]  # Measured capabilities
    created_at: str
    model_path: str
    metadata: Dict[str, Any]

@dataclass
class BattleResult:
    """Result of a model battle"""
    battle_id: str
    tournament_id: str
    round_number: int
    model_a: ModelInfo
    model_b: ModelInfo
    winner_id: str
    vote_confidence: float  # How sure was the user (0-1)
    audio_features: Dict[str, float]  # Genre, tempo, etc.
    timestamp: str

@dataclass
class TournamentState:
    """Current state of a tournament"""
    tournament_id: str
    user_id: str
    audio_file: str
    current_round: int
    max_rounds: int
    competitors: List[ModelInfo]
    battle_history: List[BattleResult]
    current_champion: Optional[ModelInfo]
    status: str  # 'active', 'completed', 'paused'
    created_at: str

class ModelEvolutionEngine:
    """Handles AI model evolution and architecture management"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.tournament_models_dir = Path(__file__).parent.parent / "tournament_models"
        self.tournament_models_dir.mkdir(exist_ok=True)
        (self.tournament_models_dir / "evolved").mkdir(exist_ok=True)
        
        # Architecture-specific evolution strategies
        self.evolution_strategies = {
            'cnn': self._evolve_cnn_weights,
            'transformer': self._evolve_transformer_weights,
            'diffusion': self._evolve_diffusion_weights,
            'hybrid': self._evolve_hybrid_architecture
        }
        
        # Load base champion models
        self.base_champions = self._load_champion_models()
        
        # Track model genealogy for research insights
        self.model_genealogy = self._load_genealogy()
    
    def _load_champion_models(self) -> List[ModelInfo]:
        """Load existing trained models as starting champions"""
        champions = []
        
        # Your current CNN champions
        cnn_models = {
            'baseline_cnn.pth': 'Classic Fighter',
            'enhanced_cnn.pth': 'Modern Warrior', 
            'improved_baseline_cnn.pth': 'Veteran Champion',
            'improved_enhanced_cnn.pth': 'Elite Warrior',
            'weighted_ensemble.pth': 'Boss Fighter'
        }
        
        for filename, nickname in cnn_models.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                champion = ModelInfo(
                    id=str(uuid.uuid4()),
                    name=nickname,
                    architecture='cnn',
                    generation=0,
                    parent_ids=[],
                    performance_score=0.8,  # Will be updated from battles
                    specializations=self._infer_specializations(filename),
                    strengths={},
                    created_at=datetime.now().isoformat(),
                    model_path=str(model_path),
                    metadata={'original_champion': True, 'filename': filename}
                )
                champions.append(champion)
        
        return champions
    
    def _infer_specializations(self, filename: str) -> List[str]:
        """Infer model specializations from filename/type"""
        if 'baseline' in filename:
            return ['conservative_mixing', 'stability']
        elif 'enhanced' in filename:
            return ['modern_sound', 'dynamics']
        elif 'ensemble' in filename:
            return ['balanced', 'all_genres', 'boss_level']
        else:
            return ['general_mixing']
    
    def evolve_model(self, winner: ModelInfo, loser: ModelInfo, 
                    battle_context: Dict[str, Any]) -> ModelInfo:
        """Create evolved model by combining winner and loser characteristics"""
        
        # Determine evolution strategy based on architectures
        if winner.architecture == loser.architecture:
            # Same architecture - blend weights
            evolved = self._evolve_same_architecture(winner, loser, battle_context)
        else:
            # Different architectures - create hybrid
            evolved = self._evolve_cross_architecture(winner, loser, battle_context)
          # Update genealogy
        self._update_genealogy(evolved, winner, loser, battle_context)
        
        return evolved
    
    def _evolve_same_architecture(self, winner: ModelInfo, loser: ModelInfo, 
                                context: Dict[str, Any]) -> ModelInfo:
        """Evolve models of the same architecture"""
        architecture = winner.architecture
        evolution_func = self.evolution_strategies.get(architecture, self._evolve_cnn_weights)
        return evolution_func(winner, loser, context)
    
    def _evolve_cnn_weights(self, winner: ModelInfo, loser: ModelInfo, 
                           context: Dict[str, Any]) -> ModelInfo:
        """Evolve CNN models by blending weights"""
        try:
            # Load model weights
            winner_state = torch.load(winner.model_path, map_location='cpu')
            loser_state = torch.load(loser.model_path, map_location='cpu')
            
            # Check if models are compatible (same architecture)
            compatible = True
            for key in winner_state.keys():
                if key in loser_state:
                    if winner_state[key].shape != loser_state[key].shape:
                        compatible = False
                        break
                else:
                    compatible = False
                    break
            
            if not compatible:
                print(f"⚠️ Models have incompatible architectures, creating mutated copy instead")
                return self._create_mutated_copy(winner)
            
            # Adaptive blending based on performance difference
            winner_confidence = context.get('vote_confidence', 0.7)
            winner_ratio = 0.6 + (winner_confidence * 0.3)  # 0.6-0.9 range
            loser_ratio = 1.0 - winner_ratio
            
            # Blend weights
            evolved_state = {}
            for key in winner_state.keys():
                if key in loser_state:
                    evolved_state[key] = (
                        winner_ratio * winner_state[key] + 
                        loser_ratio * loser_state[key]
                    )
                else:
                    evolved_state[key] = winner_state[key]
            
            # Create evolved model info
            evolved = ModelInfo(
                id=str(uuid.uuid4()),
                name=f"Hybrid-{winner.name[:10]}-{loser.name[:10]}",
                architecture='cnn',
                generation=max(winner.generation, loser.generation) + 1,
                parent_ids=[winner.id, loser.id],
                performance_score=(winner.performance_score + loser.performance_score) / 2,
                specializations=list(set(winner.specializations + loser.specializations)),
                strengths=self._blend_strengths(winner.strengths, loser.strengths),
                created_at=datetime.now().isoformat(),
                model_path="",  # Will be set after saving
                metadata={
                    'evolution_method': 'weight_blending',
                    'winner_ratio': winner_ratio,
                    'context': context
                }
            )
            
            # Save evolved model
            evolved_path = self.tournament_models_dir / "evolved" / f"{evolved.id}.pth"
            evolved_path.parent.mkdir(exist_ok=True)
            torch.save(evolved_state, evolved_path)
            evolved.model_path = str(evolved_path)
            
            return evolved
            
        except Exception as e:
            print(f"⚠️ Evolution failed: {e}")
            # Fallback: return winner with slight mutation
            return self._create_mutated_copy(winner)
    
    def _evolve_cross_architecture(self, winner: ModelInfo, loser: ModelInfo, 
                                  context: Dict[str, Any]) -> ModelInfo:
        """Create hybrid when different architectures compete"""
        # Future: CNN + Transformer hybrid, etc.
        # For now, prefer winner architecture with loser's specializations
        
        evolved = copy.deepcopy(winner)
        evolved.id = str(uuid.uuid4())
        evolved.name = f"CrossHybrid-{winner.architecture}-{loser.architecture}"
        evolved.architecture = 'hybrid'
        evolved.generation = max(winner.generation, loser.generation) + 1
        evolved.parent_ids = [winner.id, loser.id]
        evolved.specializations = list(set(winner.specializations + loser.specializations))
        evolved.created_at = datetime.now().isoformat()
        evolved.metadata = {
            'evolution_method': 'cross_architecture',
            'primary_arch': winner.architecture,
            'secondary_arch': loser.architecture,
            'context': context
        }
        
        # For now, use winner's weights but mark as hybrid
        evolved_path = self.tournament_models_dir / "evolved" / f"{evolved.id}.pth"
        evolved_path.parent.mkdir(exist_ok=True)
        
        # Copy winner's model for hybrid base
        import shutil
        shutil.copy(winner.model_path, evolved_path)
        evolved.model_path = str(evolved_path)
        
        return evolved
    
    def _blend_strengths(self, strengths_a: Dict[str, float], 
                        strengths_b: Dict[str, float]) -> Dict[str, float]:
        """Combine model strengths"""
        all_keys = set(strengths_a.keys()) | set(strengths_b.keys())
        blended = {}
        
        for key in all_keys:
            val_a = strengths_a.get(key, 0.5)
            val_b = strengths_b.get(key, 0.5)
            blended[key] = (val_a + val_b) / 2
        
        return blended
    
    def _create_mutated_copy(self, model: ModelInfo) -> ModelInfo:
        """Create slightly mutated copy as fallback"""
        mutated = copy.deepcopy(model)
        mutated.id = str(uuid.uuid4())
        mutated.name = f"Mutated-{model.name}"
        mutated.generation = model.generation + 1
        mutated.parent_ids = [model.id]
        mutated.created_at = datetime.now().isoformat()
        
        # Copy original model file
        import shutil
        mutated_path = self.tournament_models_dir / "evolved" / f"{mutated.id}.pth"
        mutated_path.parent.mkdir(exist_ok=True)
        shutil.copy(model.model_path, mutated_path)
        mutated.model_path = str(mutated_path)
        
        return mutated
    
    # Placeholder evolution strategies for future architectures
    def _evolve_transformer_weights(self, winner: ModelInfo, loser: ModelInfo, 
                                  context: Dict[str, Any]) -> ModelInfo:
        """Future: Transformer model evolution"""
        # Attention mechanism blending, layer-wise evolution, etc.
        return self._create_mutated_copy(winner)
    
    def _evolve_diffusion_weights(self, winner: ModelInfo, loser: ModelInfo, 
                                context: Dict[str, Any]) -> ModelInfo:
        """Future: Diffusion model evolution"""
        # Noise schedule blending, U-Net architecture mixing, etc.
        return self._create_mutated_copy(winner)
    
    def _evolve_hybrid_architecture(self, winner: ModelInfo, loser: ModelInfo, 
                                  context: Dict[str, Any]) -> ModelInfo:
        """Future: Multi-modal hybrid evolution"""
        # Component-wise evolution, attention reweighting, etc.
        return self._create_mutated_copy(winner)
    
    def _load_genealogy(self) -> Dict[str, Any]:
        """Load model evolution history"""
        genealogy_file = self.tournament_models_dir / "model_genealogy.json"
        if genealogy_file.exists():
            with open(genealogy_file, 'r') as f:
                return json.load(f)
        return {"evolution_tree": {}, "architecture_stats": {}}
    
    def _update_genealogy(self, evolved: ModelInfo, winner: ModelInfo, 
                         loser: ModelInfo, context: Dict[str, Any]):
        """Update model evolution tracking"""
        self.model_genealogy["evolution_tree"][evolved.id] = {
            "parents": [winner.id, loser.id],
            "evolution_method": evolved.metadata.get('evolution_method'),
            "generation": evolved.generation,
            "context": context,
            "timestamp": evolved.created_at
        }
        
        # Save genealogy
        genealogy_file = self.tournament_models_dir / "model_genealogy.json"
        with open(genealogy_file, 'w') as f:
            json.dump(self.model_genealogy, f, indent=2)


class TournamentEngine:
    """Core tournament management system"""
    
    def __init__(self, models_dir: Path):
        self.evolution_engine = ModelEvolutionEngine(models_dir)
        self.active_tournaments: Dict[str, TournamentState] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
    def start_tournament(self, user_id: str, audio_file: str, 
                        max_rounds: int = 5) -> TournamentState:
        """Start a new tournament for a user"""
        tournament_id = str(uuid.uuid4())
          # Select initial competitors (2 random champions)
        if len(self.evolution_engine.base_champions) >= 2:
            indices = np.random.choice(
                len(self.evolution_engine.base_champions), 
                size=2, 
                replace=False
            )
            competitors = [self.evolution_engine.base_champions[i] for i in indices]
        else:
            competitors = self.evolution_engine.base_champions.copy()
        
        tournament = TournamentState(
            tournament_id=tournament_id,
            user_id=user_id,
            audio_file=audio_file,
            current_round=1,
            max_rounds=max_rounds,
            competitors=competitors,
            battle_history=[],
            current_champion=None,
            status='active',
            created_at=datetime.now().isoformat()
        )
        
        self.active_tournaments[tournament_id] = tournament
        return tournament
    
    def battle_models(self, tournament_id: str, model_a: ModelInfo, 
                     model_b: ModelInfo) -> Dict[str, Any]:
        """Execute battle between two models"""        # This will integrate with your production_ai_mixer.py
        from pathlib import Path
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
        
        tournament = self.active_tournaments[tournament_id]
        
        try:
            from production_ai_mixer import ProductionAIMixer
            mixer = ProductionAIMixer()
            
            # Process audio with both models
            # Note: This is a simplified version - will need adaptation
            results_a = {"model": model_a, "audio_path": f"temp_mix_a_{tournament_id}.wav"}
            results_b = {"model": model_b, "audio_path": f"temp_mix_b_{tournament_id}.wav"}
            
            return {
                "battle_id": str(uuid.uuid4()),
                "model_a": results_a,
                "model_b": results_b,
                "ready_for_vote": True
            }
            
        except Exception as e:
            print(f"⚠️ Battle execution failed: {e}")
            return {
                "battle_id": str(uuid.uuid4()),
                "model_a": {"model": model_a, "audio_path": None},
                "model_b": {"model": model_b, "audio_path": None},
                "error": str(e)
            }
    
    def record_vote(self, tournament_id: str, battle_id: str, 
                   winner_id: str, confidence: float = 0.7) -> ModelInfo:
        """Record user vote and evolve models"""
        tournament = self.active_tournaments[tournament_id]
        
        # Find winner and loser
        current_competitors = tournament.competitors[-2:]  # Last 2 competitors
        winner = next(m for m in current_competitors if m.id == winner_id)
        loser = next(m for m in current_competitors if m.id != winner_id)
        
        # Record battle result
        battle_result = BattleResult(
            battle_id=battle_id,
            tournament_id=tournament_id,
            round_number=tournament.current_round,
            model_a=current_competitors[0],
            model_b=current_competitors[1],
            winner_id=winner_id,
            vote_confidence=confidence,
            audio_features={},  # Will be filled from audio analysis
            timestamp=datetime.now().isoformat()
        )
        
        tournament.battle_history.append(battle_result)
        
        # Evolve loser by learning from winner
        if tournament.current_round < tournament.max_rounds:
            evolved_model = self.evolution_engine.evolve_model(
                winner, loser, 
                {'vote_confidence': confidence, 'round': tournament.current_round}
            )
            
            # Next round: winner vs evolved model
            tournament.competitors = [winner, evolved_model]
            tournament.current_round += 1
        else:
            # Tournament complete
            tournament.current_champion = winner
            tournament.status = 'completed'
        
        # Update user preferences
        self._update_user_preferences(tournament.user_id, battle_result)
        
        return winner
    
    def _update_user_preferences(self, user_id: str, battle_result: BattleResult):
        """Learn from user voting patterns"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "preferred_architectures": {},
                "preferred_specializations": {},
                "voting_patterns": []
            }
        
        prefs = self.user_preferences[user_id]
        winner = battle_result.model_a if battle_result.winner_id == battle_result.model_a.id else battle_result.model_b
        
        # Track architecture preferences
        arch = winner.architecture
        prefs["preferred_architectures"][arch] = prefs["preferred_architectures"].get(arch, 0) + 1
        
        # Track specialization preferences
        for spec in winner.specializations:
            prefs["preferred_specializations"][spec] = prefs["preferred_specializations"].get(spec, 0) + 1
        
        prefs["voting_patterns"].append({
            "winner_arch": winner.architecture,
            "winner_specs": winner.specializations,
            "confidence": battle_result.vote_confidence,
            "timestamp": battle_result.timestamp
        })
    
    def get_tournament_status(self, tournament_id: str) -> Optional[TournamentState]:
        """Get current tournament state"""
        return self.active_tournaments.get(tournament_id)
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get learned user preferences"""
        return self.user_preferences.get(user_id, {})


# Future: Reinforcement Learning Integration
class RLEnvironment:
    """Future: RL environment for model self-improvement"""
    
    def __init__(self):
        # Multi-agent RL where models learn to improve themselves
        # Reward functions based on user votes, musical quality metrics
        # Exploration vs exploitation in architecture search
        pass
    
    def collective_learning_step(self):
        """Future: Models learn from the entire community"""
        # Federated learning across all user interactions
        # Meta-learning for rapid adaptation to new musical styles
        # Swarm intelligence for discovering optimal architectures
        pass


if __name__ == "__main__":
    # Test the tournament engine
    models_dir = Path(__file__).parent.parent.parent / "models"
    engine = TournamentEngine(models_dir)
    
    print("🏆 Tournament Engine Initialized!")
    print(f"📊 Base Champions: {len(engine.evolution_engine.base_champions)}")
    
    for champion in engine.evolution_engine.base_champions:
        print(f"   🥊 {champion.name} ({champion.architecture})")
    
    print("\n🚀 Ready for AI model battles!")
    print("   ✅ Model evolution system")
    print("   ✅ User preference learning") 
    print("   ✅ Cross-architecture support")
    print("   ✅ Extensible for future AI paradigms")
