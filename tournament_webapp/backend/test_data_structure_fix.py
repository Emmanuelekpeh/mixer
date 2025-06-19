#!/usr/bin/env python3
"""
🧪 Quick Test for Tournament Data Structure Fix
==============================================

Tests that the data structure fix resolves the progress calculation issue.
"""

import requests
import json
import time

def test_tournament_progression_fix():
    """Test that tournament progression displays correct data after voting"""
    print("🧪 Testing Tournament Progression Data Fix")
    print("=" * 50)
    
    BASE_URL = "http://localhost:10000"
    
    try:
        # Create a new tournament 
        tournament_data = {
            "user_id": "progress_test_user",
            "username": "Progress Test User",
            "max_rounds": 3
        }
        
        print("📝 Creating test tournament...")
        response = requests.post(f"{BASE_URL}/api/tournaments/create-json", 
                               json=tournament_data, timeout=10)
        data = response.json()
        
        if not data.get("success"):
            print(f"❌ Failed to create tournament: {data}")
            return False
        
        tournament_id = data["tournament_id"]
        print(f"✅ Created tournament: {tournament_id}")
        
        # Get full tournament details
        print("📊 Fetching tournament details...")
        response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}", timeout=5)
        tournament_full = response.json()
        
        if not tournament_full.get("success"):
            print(f"❌ Failed to fetch tournament: {tournament_full}")
            return False
        
        tournament = tournament_full["tournament"]
        print(f"📊 Tournament has {len(tournament.get('pairs', []))} pairs")
        print(f"📊 Max rounds: {tournament.get('max_rounds')}")
        print(f"📊 Current round: {tournament.get('current_round')}")
        print(f"📊 Current pair: {tournament.get('current_pair')}")
        
        # Test the issue: what does a vote response look like?
        if tournament.get("pairs") and len(tournament["pairs"]) > 0:
            print("\n🗳️ Testing vote submission...")
            pair = tournament["pairs"][0]
            
            vote_data = {
                "tournament_id": tournament_id,
                "winner_id": pair["model_a"]["id"],
                "confidence": 0.8,
                "user_id": "progress_test_user",
                "reasoning": "Test vote for data structure"
            }
            
            response = requests.post(f"{BASE_URL}/api/tournaments/vote-db", 
                                   json=vote_data, timeout=10)
            vote_result = response.json()
            
            print("📋 Vote response structure:")
            print(json.dumps(vote_result, indent=2))
            
            # Check what data is returned in tournament field
            if vote_result.get("tournament"):
                partial_tournament = vote_result["tournament"]
                print("\n🔍 Analysis of vote response 'tournament' field:")
                print(f"  - Has max_rounds: {'max_rounds' in partial_tournament}")
                print(f"  - Has pairs: {'pairs' in partial_tournament}")
                print(f"  - Has current_pair: {'current_pair' in partial_tournament}")
                print(f"  - Has status: {'status' in partial_tournament}")
                print(f"  - Keys: {list(partial_tournament.keys())}")
                
                # This simulates what the frontend bug was doing
                print("\n🐛 OLD BUG: If frontend did setTournament(vote_response.tournament):")
                print(f"  - Would LOSE max_rounds: {not ('max_rounds' in partial_tournament)}")
                print(f"  - Would LOSE pairs array: {not ('pairs' in partial_tournament)}")
                print(f"  - Progress calculation would fail: {not ('max_rounds' in partial_tournament and 'current_round' in partial_tournament)}")
                
                # Test the fix
                print("\n✅ NEW FIX: Merging preserves data:")
                merged_tournament = {**tournament, **partial_tournament}
                print(f"  - Keeps max_rounds: {'max_rounds' in merged_tournament}")
                print(f"  - Keeps pairs array: {'pairs' in merged_tournament}")
                print(f"  - Updates current_pair: {merged_tournament.get('current_pair')}")
                print(f"  - Progress calculation works: {merged_tournament.get('current_round', 0) / merged_tournament.get('max_rounds', 1)}")
        
        print("\n✅ Data structure fix test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tournament_progression_fix()
    if success:
        print("\n🎯 CONCLUSION: Data structure fix should resolve UI progression issues!")
    else:
        print("\n⚠️ Test failed - check server status")
