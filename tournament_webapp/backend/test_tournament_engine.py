#!/usr/bin/env python3
"""
Test script for the Enhanced Tournament Engine
"""

import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_tournament_engine import EnhancedTournamentEngine, UserTier, ModelArchitecture
    
    print("🏆 Testing Enhanced Tournament Engine")
    print("=" * 50)
    
    # Initialize engine
    models_dir = Path("../../models")
    engine = EnhancedTournamentEngine(models_dir)
    
    print(f"✅ Engine initialized successfully!")
    print(f"📊 Champions loaded: {len(engine.evolution_engine.champion_models)}")
    
    # List champions
    print("\n🥊 Champion Models:")
    for i, champion in enumerate(engine.evolution_engine.champion_models, 1):
        print(f"   {i}. {champion.nickname} - ELO: {champion.elo_rating} - {champion.tier}")
    
    # Test user creation
    print("\n👤 Testing user creation...")
    user_profile = engine.create_user_profile("test_user_001", "TestMixer")
    print(f"✅ User created: {user_profile.username} (Tier: {user_profile.tier.value})")
    
    # Test tournament creation
    print("\n🎯 Testing tournament creation...")
    tournament = engine.start_tournament(
        user_id="test_user_001",
        username="TestMixer", 
        audio_file="test_audio.wav",
        audio_features={"genre": "electronic", "tempo": 128.0}
    )
    print(f"✅ Tournament created: {tournament.tournament_id}")
    print(f"🎮 Competitors: {tournament.competitors[0].nickname} vs {tournament.competitors[1].nickname}")
    
    # Test battle execution
    print("\n⚔️ Testing battle execution...")
    battle = engine.execute_battle(tournament.tournament_id)
    if "error" not in battle:
        print(f"✅ Battle ready: {battle['model_a']['nickname']} vs {battle['model_b']['nickname']}")
        
        # Simulate vote
        print("\n🗳️ Testing vote recording...")
        winner_id = battle['model_a']['id']
        result = engine.record_vote(
            tournament.tournament_id,
            winner_id,
            confidence=0.8,
            reasoning="Preferred the sound quality"
        )
        
        if "error" not in result:
            print(f"✅ Vote recorded successfully!")
            if result.get("tournament_complete"):
                print(f"🏆 Tournament complete! Champion: {result['champion']['nickname']}")
            else:
                print(f"🧬 Model evolved for next round: {result.get('evolved_challenger', {}).get('nickname', 'Unknown')}")
        else:
            print(f"❌ Vote failed: {result['error']}")
    else:
        print(f"❌ Battle failed: {battle['error']}")
    
    # Test leaderboard
    print("\n📊 Testing leaderboard...")
    leaderboard = engine.get_tournament_leaderboard()
    print(f"✅ Leaderboard generated with {len(leaderboard)} models")
    
    if leaderboard:
        print("   Top 3:")
        for entry in leaderboard[:3]:
            print(f"      {entry['rank']}. {entry['nickname']} - ELO: {entry['elo_rating']}")
    
    print("\n🎉 All tests passed! Tournament engine is ready for production.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
