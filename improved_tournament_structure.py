#!/usr/bin/env python3
"""
Improved Tournament Structure
============================

Creates more reasonable tournament formats that are engaging and manageable.
"""

import random
import math
from typing import List, Dict, Any, Tuple

class ImprovedTournamentStructure:
    """Creates better tournament formats for AI model battles"""
    
    def __init__(self, models: List[Dict[str, Any]]):
        self.models = models
        
    def create_bracket_tournament(self, max_models: int = 8) -> Dict[str, Any]:
        """
        Create a traditional single-elimination bracket tournament
        
        Args:
            max_models: Maximum number of models (must be power of 2)
            
        Returns:
            Tournament structure with proper brackets
        """
        # Ensure max_models is a power of 2
        max_models = 2 ** int(math.log2(max_models))
        
        # Select models by ELO rating (top performers)
        selected_models = sorted(self.models, key=lambda x: x.get('elo_rating', 1200), reverse=True)[:max_models]
        
        # Calculate number of rounds
        num_rounds = int(math.log2(max_models))
        
        # Create initial bracket pairs
        pairs = []
        for i in range(0, len(selected_models), 2):
            if i + 1 < len(selected_models):
                pairs.append({
                    "round": 1,
                    "pair_id": f"R1P{i//2 + 1}",
                    "model_a": selected_models[i],
                    "model_b": selected_models[i + 1],
                    "winner_advances_to": f"R2P{i//4 + 1}" if num_rounds > 1 else "FINAL"
                })
        
        return {
            "format": "single_elimination",
            "total_rounds": num_rounds,
            "total_models": len(selected_models),
            "total_battles": len(selected_models) - 1,  # Always n-1 battles in elimination
            "current_round": 1,
            "pairs": pairs,
            "bracket_structure": self._create_bracket_visual(selected_models)
        }
    
    def create_swiss_tournament(self, rounds: int = 4) -> Dict[str, Any]:
        """
        Create a Swiss-system tournament (everyone plays same number of rounds)
        
        Args:
            rounds: Number of rounds to play
            
        Returns:
            Swiss tournament structure
        """
        # Shuffle models for first round
        shuffled_models = self.models.copy()
        random.shuffle(shuffled_models)
        
        # Create first round pairs
        pairs = []
        for i in range(0, len(shuffled_models), 2):
            if i + 1 < len(shuffled_models):
                pairs.append({
                    "round": 1,
                    "pair_id": f"R1P{i//2 + 1}",
                    "model_a": shuffled_models[i],
                    "model_b": shuffled_models[i + 1]
                })
        
        return {
            "format": "swiss_system",
            "total_rounds": rounds,
            "total_models": len(self.models),
            "battles_per_round": len(pairs),
            "total_battles": len(pairs) * rounds,
            "current_round": 1,
            "pairs": pairs,
            "scoring": "1 point for win, 0.5 for draw, 0 for loss"
        }
    
    def create_round_robin(self, group_size: int = 6) -> Dict[str, Any]:
        """
        Create a round-robin tournament (everyone plays everyone)
        
        Args:
            group_size: Maximum models in the group
            
        Returns:
            Round-robin tournament structure
        """
        # Select top models by ELO
        selected_models = sorted(self.models, key=lambda x: x.get('elo_rating', 1200), reverse=True)[:group_size]
        
        # Generate all possible pairs
        pairs = []
        pair_id = 1
        for i in range(len(selected_models)):
            for j in range(i + 1, len(selected_models)):
                pairs.append({
                    "round": ((pair_id - 1) // (group_size // 2)) + 1,
                    "pair_id": f"P{pair_id}",
                    "model_a": selected_models[i],
                    "model_b": selected_models[j]
                })
                pair_id += 1
        
        total_battles = len(selected_models) * (len(selected_models) - 1) // 2
        rounds_needed = math.ceil(total_battles / (group_size // 2))
        
        return {
            "format": "round_robin",
            "total_rounds": rounds_needed,
            "total_models": len(selected_models),
            "total_battles": total_battles,
            "current_round": 1,
            "pairs": pairs[:group_size//2],  # First round pairs only
            "remaining_pairs": pairs[group_size//2:],
            "scoring": "Points accumulated across all matches"
        }
    
    def create_quick_battle(self, num_battles: int = 3) -> Dict[str, Any]:
        """
        Create a quick battle format with random matchups
        
        Args:
            num_battles: Number of battles to create
            
        Returns:
            Quick battle structure
        """
        pairs = []
        used_models = set()
        
        for i in range(num_battles):
            # Select two random models that haven't been used
            available_models = [m for m in self.models if m['id'] not in used_models]
            
            if len(available_models) < 2:
                # Reset if we run out of models
                available_models = self.models.copy()
                used_models.clear()
            
            model_a, model_b = random.sample(available_models, 2)
            used_models.add(model_a['id'])
            used_models.add(model_b['id'])
            
            pairs.append({
                "round": i + 1,
                "pair_id": f"QB{i + 1}",
                "model_a": model_a,
                "model_b": model_b
            })
        
        return {
            "format": "quick_battle",
            "total_rounds": num_battles,
            "total_models": len(set([p['model_a']['id'] for p in pairs] + [p['model_b']['id'] for p in pairs])),
            "total_battles": num_battles,
            "current_round": 1,
            "pairs": [pairs[0]],  # Only first battle
            "remaining_pairs": pairs[1:],
            "description": "Quick random battles between models"
        }
    
    def create_architecture_showdown(self) -> Dict[str, Any]:
        """
        Create battles between different architectures
        
        Returns:
            Architecture-based tournament structure
        """
        # Group models by architecture
        arch_groups = {}
        for model in self.models:
            arch = model.get('architecture', 'unknown').lower()
            if arch not in arch_groups:
                arch_groups[arch] = []
            arch_groups[arch].append(model)
        
        # Select best model from each architecture
        arch_champions = []
        for arch, models in arch_groups.items():
            if models:
                champion = max(models, key=lambda x: x.get('elo_rating', 1200))
                champion['represents_architecture'] = arch
                arch_champions.append(champion)
        
        # Create pairs between different architectures
        pairs = []
        pair_id = 1
        for i in range(len(arch_champions)):
            for j in range(i + 1, len(arch_champions)):
                pairs.append({
                    "round": pair_id,
                    "pair_id": f"ARCH{pair_id}",
                    "model_a": arch_champions[i],
                    "model_b": arch_champions[j],
                    "battle_type": f"{arch_champions[i]['represents_architecture']} vs {arch_champions[j]['represents_architecture']}"
                })
                pair_id += 1
        
        return {
            "format": "architecture_showdown",
            "total_rounds": len(pairs),
            "total_models": len(arch_champions),
            "total_battles": len(pairs),
            "current_round": 1,
            "pairs": [pairs[0]] if pairs else [],
            "remaining_pairs": pairs[1:] if len(pairs) > 1 else [],
            "architectures": list(arch_groups.keys()),
            "description": "Best model from each architecture competes"
        }
    
    def _create_bracket_visual(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a visual representation of the bracket"""
        num_models = len(models)
        rounds = int(math.log2(num_models))
        
        bracket = {}
        for round_num in range(1, rounds + 1):
            bracket[f"round_{round_num}"] = {
                "matches": 2 ** (rounds - round_num),
                "description": f"Round {round_num}" if round_num < rounds else "Final"
            }
        
        return bracket

def suggest_tournament_format(models: List[Dict[str, Any]], user_preference: str = "balanced") -> Dict[str, Any]:
    """
    Suggest the best tournament format based on number of models and user preference
    
    Args:
        models: Available models
        user_preference: "quick", "balanced", "comprehensive", "competitive"
        
    Returns:
        Recommended tournament structure
    """
    tournament = ImprovedTournamentStructure(models)
    num_models = len(models)
    
    if user_preference == "quick" or num_models <= 4:
        return tournament.create_quick_battle(3)
    elif user_preference == "balanced" or num_models <= 8:
        return tournament.create_bracket_tournament(min(8, num_models))
    elif user_preference == "comprehensive":
        return tournament.create_round_robin(min(6, num_models))
    elif user_preference == "competitive":
        return tournament.create_swiss_tournament(4)
    else:
        # Default to architecture showdown for variety
        return tournament.create_architecture_showdown()

# Example usage and testing
if __name__ == "__main__":
    # Sample models for testing
    sample_models = [
        {"id": "ast_transformer", "name": "AST Transformer", "architecture": "transformer", "elo_rating": 1493},
        {"id": "advanced_transformer", "name": "Advanced Transformer", "architecture": "transformer", "elo_rating": 1465},
        {"id": "vae_mixer", "name": "VAE Mixer", "architecture": "vae", "elo_rating": 1444},
        {"id": "resnet_mixer", "name": "ResNet Mixer", "architecture": "resnet", "elo_rating": 1443},
        {"id": "lstm_mixer", "name": "LSTM Mixer", "architecture": "lstm", "elo_rating": 1411},
        {"id": "audio_gan", "name": "Audio GAN", "architecture": "gan", "elo_rating": 1360},
        {"id": "baseline_cnn", "name": "Baseline CNN", "architecture": "cnn", "elo_rating": 1319},
        {"id": "enhanced_cnn", "name": "Enhanced CNN", "architecture": "cnn", "elo_rating": 1287}
    ]
    
    print("🏆 IMPROVED TOURNAMENT FORMATS")
    print("=" * 50)
    
    tournament = ImprovedTournamentStructure(sample_models)
    
    # Test different formats
    formats = [
        ("Quick Battle", tournament.create_quick_battle(3)),
        ("Bracket Tournament", tournament.create_bracket_tournament(8)),
        ("Architecture Showdown", tournament.create_architecture_showdown()),
        ("Swiss Tournament", tournament.create_swiss_tournament(3))
    ]
    
    for name, structure in formats:
        print(f"\n📋 {name}:")
        print(f"   Format: {structure['format']}")
        print(f"   Total Battles: {structure['total_battles']}")
        print(f"   Models: {structure['total_models']}")
        print(f"   Rounds: {structure['total_rounds']}")
        if 'description' in structure:
            print(f"   Description: {structure['description']}")