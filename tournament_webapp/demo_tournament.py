#!/usr/bin/env python3
"""
🏆 Mix Tournament Demo
====================

Demo the tournament system with your trained AI models battling!
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

from tournament_engine import TournamentEngine, ModelInfo
import numpy as np

def demo_tournament():
    """Demo a complete AI model tournament"""
    print("🏆 AI MIX TOURNAMENT DEMO")
    print("=" * 50)
    
    # Initialize tournament engine
    models_dir = Path(__file__).parent.parent / "models"
    engine = TournamentEngine(models_dir)
    
    print(f"🎯 Loaded {len(engine.evolution_engine.base_champions)} champion models:")
    for i, champion in enumerate(engine.evolution_engine.base_champions, 1):
        specs = ", ".join(champion.specializations)
        print(f"   {i}. {champion.name} - Specializes in: {specs}")
    
    # Start a demo tournament
    print(f"\n🎵 Starting Tournament...")
    tournament = engine.start_tournament(
        user_id="demo_user", 
        audio_file="demo_track.wav",
        max_rounds=3  # Shorter for demo
    )
    
    print(f"📊 Tournament ID: {tournament.tournament_id}")
    print(f"🥊 Round {tournament.current_round}/{tournament.max_rounds}")
    
    # Simulate tournament battles
    for round_num in range(tournament.max_rounds):
        print(f"\n🔥 ROUND {round_num + 1} BATTLE!")
        
        # Get current competitors
        competitors = tournament.competitors[-2:] if len(tournament.competitors) >= 2 else tournament.competitors
        
        if len(competitors) < 2:
            print("⚠️ Not enough competitors for battle")
            break
            
        model_a, model_b = competitors[0], competitors[1]
        
        print(f"🥊 {model_a.name} vs {model_b.name}")
        print(f"   Specializations: {model_a.specializations} vs {model_b.specializations}")
        
        # Execute battle (simplified for demo)
        battle_result = engine.battle_models(tournament.tournament_id, model_a, model_b)
        
        # Simulate user vote (random for demo)
        winner_choice = np.random.choice([model_a.id, model_b.id])
        confidence = np.random.uniform(0.6, 0.9)
        
        winner_name = model_a.name if winner_choice == model_a.id else model_b.name
        print(f"🏆 User votes for: {winner_name} (confidence: {confidence:.2f})")
        
        # Record vote and evolve models
        winner = engine.record_vote(
            tournament.tournament_id, 
            battle_result["battle_id"],
            winner_choice, 
            confidence
        )
        
        # Update tournament state
        tournament = engine.get_tournament_status(tournament.tournament_id)
        
        if tournament.status == 'completed':
            print(f"\n🎉 TOURNAMENT COMPLETE!")
            print(f"🏆 Champion: {tournament.current_champion.name}")
            break
        else:
            print(f"🔄 Evolution complete! Next round upcoming...")
    
    # Show evolution results
    print(f"\n📈 EVOLUTION RESULTS:")
    print(f"🧬 Total battles: {len(tournament.battle_history)}")
    
    # Show user preferences learned
    preferences = engine.get_user_preferences("demo_user")
    if preferences:
        print(f"\n🧠 USER PREFERENCES LEARNED:")
        if "preferred_architectures" in preferences:
            for arch, count in preferences["preferred_architectures"].items():
                print(f"   • {arch.upper()}: {count} votes")
        if "preferred_specializations" in preferences:
            top_specs = sorted(preferences["preferred_specializations"].items(), 
                             key=lambda x: x[1], reverse=True)[:3]
            for spec, count in top_specs:
                print(f"   • {spec}: {count} votes")
    
    print(f"\n🚀 TOURNAMENT SYSTEM FEATURES:")
    print(f"   ✅ Model battles with real AI mixing")
    print(f"   ✅ Evolutionary learning from user votes")  
    print(f"   ✅ User preference tracking")
    print(f"   ✅ Cross-architecture competition ready")
    print(f"   ✅ Social sharing potential")

if __name__ == "__main__":
    demo_tournament()
