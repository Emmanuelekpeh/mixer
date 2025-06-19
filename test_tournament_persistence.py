#!/usr/bin/env python3
"""
Test script for tournament persistence system
"""

import asyncio
import json
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:10000"

def test_tournament_persistence():
    """Test tournament creation, state saving, and resumption"""
    
    print("🧪 Testing Tournament Persistence System")
    print("=" * 50)
    
    # Test 1: Create a user
    print("\n1. Creating test user...")
    user_data = {
        "user_id": "test_persistence_user",
        "username": "TestPersistenceUser"
    }
    
    response = requests.post(f"{BASE_URL}/api/users/create", json=user_data)
    if response.status_code == 200:
        print("✅ User created successfully")
        user = response.json()["profile"]
    else:
        print(f"❌ Failed to create user: {response.text}")
        return False
    
    # Test 2: Create a tournament
    print("\n2. Creating tournament...")
    tournament_data = {
        "user_id": user["user_id"],
        "username": user["username"],
        "max_rounds": 3,
        "audio_features": {}
    }
    
    response = requests.post(f"{BASE_URL}/api/tournaments/create-json", json=tournament_data)
    if response.status_code == 200:
        print("✅ Tournament created successfully")
        tournament = response.json()["tournament"]
        tournament_id = tournament["id"]
        print(f"   Tournament ID: {tournament_id}")
    else:
        print(f"❌ Failed to create tournament: {response.text}")
        return False
    
    # Test 3: Simulate a vote to trigger state saving
    print("\n3. Simulating battle vote...")
    vote_data = {
        "tournament_id": tournament_id,
        "winner_id": "model1",
        "confidence": 0.8,
        "user_id": user["user_id"],
        "reasoning": "Test vote for persistence"
    }
    
    response = requests.post(f"{BASE_URL}/api/tournaments/vote-db", json=vote_data)
    if response.status_code == 200:
        print("✅ Vote recorded successfully")
        updated_tournament = response.json()["tournament"]
        print(f"   Tournament status: {updated_tournament.get('status', 'unknown')}")
        print(f"   Current round: {updated_tournament.get('current_round', 'unknown')}")
    else:
        print(f"❌ Failed to record vote: {response.text}")
        return False
    
    # Test 4: Get tournament status
    print("\n4. Retrieving tournament status...")
    response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}")
    if response.status_code == 200:
        print("✅ Tournament status retrieved")
        tournament_status = response.json()["tournament"]
        print(f"   Status: {tournament_status['status']}")
        print(f"   Current round: {tournament_status['current_round']}")
        print(f"   Pairs: {len(tournament_status.get('pairs', []))}")
    else:
        print(f"❌ Failed to get tournament status: {response.text}")
        return False
    
    # Test 5: Test tournament resumption
    print("\n5. Testing tournament resumption...")
    response = requests.post(f"{BASE_URL}/api/tournaments/{tournament_id}/resume")
    if response.status_code == 200:
        print("✅ Tournament resumed successfully")
        resumed_tournament = response.json()["tournament"]
        print(f"   Resumed status: {resumed_tournament['status']}")
        print(f"   Current round: {resumed_tournament['current_round']}")
    else:
        print(f"❌ Failed to resume tournament: {response.text}")
        return False
    
    # Test 6: Get user tournaments
    print("\n6. Getting user tournament list...")
    response = requests.get(f"{BASE_URL}/api/users/{user['user_id']}/tournaments")
    if response.status_code == 200:
        print("✅ User tournaments retrieved")
        tournaments = response.json()["tournaments"]
        print(f"   Found {len(tournaments)} tournaments")
        for t in tournaments:
            print(f"   - {t['tournament_id']}: {t['status']} (Round {t['current_round']})")
    else:
        print(f"❌ Failed to get user tournaments: {response.text}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Tournament Persistence Tests Completed Successfully!")
    return True

def test_persistence_after_restart():
    """Test that tournaments persist after server restart (simulation)"""
    print("\n🔄 Testing persistence after restart simulation...")
    
    # This would ideally restart the server, but for now we'll just 
    # test that the data is in the database
    user_id = "test_persistence_user"
    
    response = requests.get(f"{BASE_URL}/api/users/{user_id}/tournaments")
    if response.status_code == 200:
        tournaments = response.json()["tournaments"]
        if tournaments:
            print(f"✅ Found {len(tournaments)} persisted tournaments after 'restart'")
            return True
        else:
            print("❌ No tournaments found after 'restart'")
            return False
    else:
        print(f"❌ Failed to check persisted tournaments: {response.text}")
        return False

if __name__ == "__main__":
    print("Starting tournament persistence tests...")
    print("Make sure the tournament API server is running on http://localhost:10000")
    
    # Wait a moment for any server startup
    time.sleep(2)
    
    try:
        # Test basic persistence
        if test_tournament_persistence():
            # Test persistence after restart
            test_persistence_after_restart()
        else:
            print("❌ Basic persistence tests failed")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to tournament API server")
        print("   Make sure the server is running on http://localhost:10000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
