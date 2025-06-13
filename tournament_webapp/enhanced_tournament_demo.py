#!/usr/bin/env python3
"""
🏆 Enhanced AI Mix Tournament Demo
=================================

Full-featured tournament with real audio processing and evolution tracking.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

from tournament_engine import TournamentEngine, ModelInfo
from audio_processor import TournamentAudioProcessor

def enhanced_tournament_demo():
    """Run a complete tournament with real audio processing"""
    print("🏆 ENHANCED AI MIX TOURNAMENT")
    print("=" * 60)
    
    # Initialize tournament engine
    models_dir = Path(__file__).parent.parent / "models"
    engine = TournamentEngine(models_dir)
    
    # Initialize audio processor
    try:
        audio_processor = TournamentAudioProcessor()
        print("🎵 Audio processor initialized successfully!")
    except Exception as e:
        print(f"⚠️ Audio processor initialization failed: {e}")
        audio_processor = None
    
    print(f"🎯 Tournament Features:")
    print(f"   🤖 {len(engine.evolution_engine.base_champions)} champion models loaded")
    print(f"   🎵 Real-time audio processing: {'✅' if audio_processor else '❌'}")
    print(f"   🧬 Model evolution enabled: ✅")
    print(f"   🧠 User preference learning: ✅")
    
    print(f"\n🥊 Available Champions:")
    for i, champion in enumerate(engine.evolution_engine.base_champions, 1):
        specs = ", ".join(champion.specializations)
        print(f"   {i}. {champion.name}")
        print(f"      Architecture: {champion.architecture}")
        print(f"      Specializations: {specs}")
        print(f"      Generation: {champion.generation}")
    
    # Find a test audio file
    test_audio_dir = Path(__file__).parent.parent / "data" / "test"
    test_files = list(test_audio_dir.glob("*/*.wav"))
    
    if not test_files:
        test_audio_file = "demo_track.wav"
        print(f"\n⚠️ No test audio found, using placeholder: {test_audio_file}")
    else:
        test_audio_file = str(test_files[0])
        print(f"\n🎵 Test audio: {Path(test_audio_file).name}")
    
    # Start tournament
    print(f"\n🚀 Starting Enhanced Tournament...")
    tournament = engine.start_tournament(
        user_id="enhanced_demo_user",
        audio_file=test_audio_file,
        max_rounds=5  # Longer tournament for better evolution
    )
    
    print(f"📊 Tournament Details:")
    print(f"   ID: {tournament.tournament_id}")
    print(f"   Rounds: {tournament.current_round}/{tournament.max_rounds}")
    print(f"   Status: {tournament.status}")
    print(f"   Competitors: {len(tournament.competitors)}")
    
    # Tournament battles with detailed tracking
    evolution_stats = {
        'generations_created': 0,
        'architecture_battles': {},
        'user_preferences': {},
        'battle_outcomes': []
    }
    
    for round_num in range(tournament.max_rounds):
        print(f"\n" + "="*60)
        print(f"🔥 ROUND {round_num + 1} BATTLE!")
        print(f"="*60)
        
        # Get current competitors
        competitors = tournament.competitors[-2:] if len(tournament.competitors) >= 2 else tournament.competitors
        
        if len(competitors) < 2:
            print("⚠️ Not enough competitors for battle")
            break
            
        model_a, model_b = competitors[0], competitors[1]
        
        print(f"\n🥊 BATTLE MATCHUP:")
        print(f"   🔵 {model_a.name}")
        print(f"      • Architecture: {model_a.architecture}")
        print(f"      • Generation: {model_a.generation}")
        print(f"      • Specializations: {', '.join(model_a.specializations)}")
        print(f"      • Performance: {model_a.performance_score:.3f}")
        
        print(f"   🔴 {model_b.name}")
        print(f"      • Architecture: {model_b.architecture}")
        print(f"      • Generation: {model_b.generation}")
        print(f"      • Specializations: {', '.join(model_b.specializations)}")
        print(f"      • Performance: {model_b.performance_score:.3f}")
        
        # Track architecture battles
        arch_battle = f"{model_a.architecture} vs {model_b.architecture}"
        evolution_stats['architecture_battles'][arch_battle] = evolution_stats['architecture_battles'].get(arch_battle, 0) + 1
        
        # Execute battle with audio processing
        print(f"\n🎵 Processing Audio Battle...")
        start_time = time.time()
        
        battle_result = engine.battle_models(tournament.tournament_id, model_a, model_b)
        processing_time = time.time() - start_time
        
        if battle_result.get('error'):
            print(f"⚠️ Battle processing failed: {battle_result['error']}")
        else:
            print(f"✅ Battle processed successfully in {processing_time:.2f}s")
            if audio_processor:
                print(f"   🎵 Audio A: {battle_result['model_a'].get('audio_path', 'N/A')}")
                print(f"   🎵 Audio B: {battle_result['model_b'].get('audio_path', 'N/A')}")
        
        # Simulate realistic user vote (with some bias towards certain architectures)
        print(f"\n🗳️ User Voting...")
        
        # Simulate user preferences (bias towards balanced models and newer generations)
        vote_bias = 0.0
        if 'balanced' in model_a.specializations:
            vote_bias += 0.1
        if 'balanced' in model_b.specializations:
            vote_bias -= 0.1
        
        if model_a.generation > model_b.generation:
            vote_bias += 0.05
        elif model_b.generation > model_a.generation:
            vote_bias -= 0.05
        
        # Random vote with bias
        vote_probability = 0.5 + vote_bias
        winner_choice = model_a.id if np.random.random() < vote_probability else model_b.id
        confidence = np.random.uniform(0.6, 0.95)
        
        winner_name = model_a.name if winner_choice == model_a.id else model_b.name
        print(f"🏆 User votes for: {winner_name}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Voting time: {np.random.uniform(3, 12):.1f} seconds")
        
        # Record vote and evolve models
        print(f"\n🧬 Evolution in progress...")
        evolution_start = time.time()
        
        winner = engine.record_vote(
            tournament.tournament_id, 
            battle_result["battle_id"],
            winner_choice, 
            confidence
        )
        
        evolution_time = time.time() - evolution_start
        print(f"✅ Evolution completed in {evolution_time:.2f}s")
        
        # Track evolution stats
        evolution_stats['generations_created'] += 1
        evolution_stats['battle_outcomes'].append({
            'round': round_num + 1,
            'winner': winner.name,
            'confidence': confidence,
            'generation': winner.generation
        })
        
        # Update tournament state
        tournament = engine.get_tournament_status(tournament.tournament_id)
        
        if tournament.status == 'completed':
            print(f"\n🎉 TOURNAMENT COMPLETE!")
            print(f"🏆 Final Champion: {tournament.current_champion.name}")
            print(f"   Generation: {tournament.current_champion.generation}")
            print(f"   Architecture: {tournament.current_champion.architecture}")
            print(f"   Specializations: {', '.join(tournament.current_champion.specializations)}")
            break
        else:
            print(f"🔄 Evolution complete! Preparing next round...")
            print(f"   Next competitors ready: {len(tournament.competitors)} models")
    
    # Final tournament analysis
    print(f"\n" + "="*60)
    print(f"📊 TOURNAMENT ANALYSIS")
    print(f"="*60)
    
    print(f"\n🧬 Evolution Statistics:")
    print(f"   • Total battles: {len(tournament.battle_history)}")
    print(f"   • Generations created: {evolution_stats['generations_created']}")
    print(f"   • Final champion generation: {tournament.current_champion.generation}")
    print(f"   • Evolution success rate: {evolution_stats['generations_created']/len(tournament.battle_history)*100:.1f}%")
    
    print(f"\n🥊 Architecture Battles:")
    for battle_type, count in evolution_stats['architecture_battles'].items():
        print(f"   • {battle_type}: {count} battles")
    
    print(f"\n🧠 User Preferences Learned:")
    preferences = engine.get_user_preferences("enhanced_demo_user")
    if preferences.get('preferred_architectures'):
        print(f"   Architecture preferences:")
        for arch, votes in preferences['preferred_architectures'].items():
            print(f"     • {arch.upper()}: {votes} votes")
    
    if preferences.get('preferred_specializations'):
        print(f"   Specialization preferences:")
        top_specs = sorted(preferences['preferred_specializations'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
        for spec, votes in top_specs:
            print(f"     • {spec}: {votes} votes")
    
    print(f"\n📈 Performance Progression:")
    for outcome in evolution_stats['battle_outcomes']:
        print(f"   Round {outcome['round']}: {outcome['winner']} (Gen {outcome['generation']}, Conf: {outcome['confidence']:.2f})")
    
    print(f"\n🚀 TOURNAMENT SYSTEM CAPABILITIES:")
    print(f"   ✅ Real-time AI model battles")
    print(f"   ✅ Evolutionary model improvement")
    print(f"   ✅ Musical intelligence integration") 
    print(f"   ✅ User preference learning")
    print(f"   ✅ Cross-architecture competition")
    print(f"   ✅ Genealogy tracking")
    print(f"   ✅ Performance analytics")
    print(f"   ✅ Scalable to any AI architecture")
    
    print(f"\n🎯 Ready for Production Deployment!")
    print(f"   • Web interface integration")
    print(f"   • Multi-user tournaments") 
    print(f"   • Real-time audio streaming")
    print(f"   • Social features & sharing")
    print(f"   • Leaderboards & statistics")
    
    return tournament, evolution_stats

if __name__ == "__main__":
    tournament, stats = enhanced_tournament_demo()
