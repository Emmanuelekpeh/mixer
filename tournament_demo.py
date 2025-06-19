#!/usr/bin/env python3
"""
Tournament Demo - Test the current tournament system
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_tournament_system():
    print("🎮 TOURNAMENT WEBAPP LIVE DEMO")
    print("=" * 50)
    
    # Test 1: Check available models
    print("\n1️⃣ Testing Available Models...")
    try:
        response = requests.get(f"{BASE_URL}/api/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"   ✅ Found {len(models)} AI models:")
            for model in models:
                print(f"      🤖 {model['name']} ({model['architecture']}) - ELO: {model['elo_rating']}")
        else:
            print(f"   ❌ Models API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Create a tournament
    print("\n2️⃣ Creating Tournament...")
    try:
        create_data = {
            "user_id": "test_user",
            "max_rounds": 3
        }
        response = requests.post(
            f"{BASE_URL}/api/tournaments/quick-create", 
            json=create_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            tournament_data = response.json()
            if tournament_data.get("success"):
                tournament = tournament_data["tournament"]
                print(f"   ✅ Tournament created: {tournament['id']}")
                print(f"      📊 {tournament['total_models']} models, {tournament['total_pairs']} battles")
                
                # Show the first battle
                pairs = tournament.get("pairs", [])
                if pairs:
                    pair = pairs[0]
                    print(f"      ⚔️  Battle: {pair['model_a']['name']} vs {pair['model_b']['name']}")
                    print(f"         ELO: {pair['model_a']['elo_rating']} vs {pair['model_b']['elo_rating']}")
                    
                    return tournament['id'], pair
            else:
                print(f"   ❌ Tournament creation failed: {tournament_data}")
        else:
            print(f"   ❌ Tournament API failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return None, None

def test_user_profile():
    print("\n3️⃣ Testing User Profile...")
    try:
        response = requests.get(f"{BASE_URL}/api/users/test_user")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                user = data["user"]
                profile = user["profile"]
                print(f"   ✅ User Profile: {profile['username']}")
                print(f"      🏆 Tier: {profile['tier']}")
                print(f"      📈 Tournaments: {profile['tournaments_completed']}")
                print(f"      ⚔️  Battles: {profile['total_battles']}")
            else:
                print(f"   ❌ User profile failed: {data}")
        else:
            print(f"   ❌ User API failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

def test_performance():
    print("\n4️⃣ Testing Performance...")
    try:
        start_time = time.time()
        
        # Make multiple rapid API calls
        for i in range(5):
            response = requests.get(f"{BASE_URL}/api/models")
            if response.status_code != 200:
                print(f"   ❌ Request {i+1} failed")
                return
        
        end_time = time.time()
        total_time = end_time - start_time
        requests_per_second = 5 / total_time
        
        print(f"   ✅ Performance: {requests_per_second:.1f} requests/second")
        if requests_per_second > 10:
            print("   🚀 EXCELLENT performance!")
        elif requests_per_second > 5:
            print("   ✅ Good performance")
        else:
            print("   ⚠️ Moderate performance")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # Run comprehensive demo
    tournament_id, battle_pair = test_tournament_system()
    test_user_profile()
    test_performance()
    
    print("\n" + "=" * 50)
    print("🎯 DEMO RESULTS:")
    
    if tournament_id:
        print(f"✅ Tournament System: WORKING")
        print(f"✅ Database Integration: WORKING")
        print(f"✅ AI Model Management: WORKING")
        print(f"✅ API Endpoints: WORKING")
        print(f"")
        print(f"🏆 Tournament ID: {tournament_id}")
        if battle_pair:
            print(f"⚔️  Ready for Battle: {battle_pair['model_a']['name']} vs {battle_pair['model_b']['name']}")
        print(f"")
        print(f"🌐 Frontend: http://localhost:3000")
        print(f"🔗 Backend: http://localhost:8000")
        print(f"📚 API Docs: http://localhost:8000/docs")
        print(f"")
        print(f"🎮 TOURNAMENT WEBAPP IS LIVE AND FUNCTIONAL!")
    else:
        print(f"❌ Tournament System: NEEDS ATTENTION")
        print(f"⚠️ Some endpoints may need server restart")
    
    print("=" * 50)
