#!/usr/bin/env python3
"""
Quick test of tournament creation and voting flow
"""
import requests
import json
import time

BASE_URL = "http://localhost:10000"

def test_tournament_flow():
    print("🧪 Testing Tournament Flow...")
    
    # 1. Create a user
    print("\n1. Creating user...")
    user_data = {
        "user_id": f"test_user_{int(time.time())}",
        "username": "TestUser"
    }
    
    response = requests.post(f"{BASE_URL}/api/users/create", json=user_data)
    print(f"   User creation response: {response.status_code}")
    if response.status_code == 200:
        user_result = response.json()
        print(f"   User created: {user_result}")
        user_id = user_result["profile"]["user_id"]
    else:
        print(f"   Error: {response.text}")
        return
    
    # 2. Create a tournament
    print("\n2. Creating tournament...")
    tournament_data = {
        "user_id": user_id,
        "username": user_data["username"],
        "max_rounds": 5
    }
    
    response = requests.post(f"{BASE_URL}/api/tournaments/create-json", json=tournament_data)
    print(f"   Tournament creation response: {response.status_code}")
    if response.status_code == 200:
        tournament_result = response.json()
        print(f"   Tournament created: {tournament_result['tournament_id']}")
        print(f"   Total pairs: {tournament_result['tournament']['total_pairs']}")
        print(f"   Current pair: {tournament_result['tournament']['current_pair']}")
        tournament_id = tournament_result["tournament_id"]
        tournament = tournament_result["tournament"]
    else:
        print(f"   Error: {response.text}")
        return
    
    # 3. Get tournament details
    print("\n3. Getting tournament details...")
    response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}")
    print(f"   Get tournament response: {response.status_code}")
    if response.status_code == 200:
        details = response.json()
        print(f"   Tournament details: {details['tournament']['status']}")
        print(f"   Current round: {details['tournament']['current_round']}")
        print(f"   Current pair: {details['tournament']['current_pair']}")
        print(f"   Pairs count: {len(details['tournament']['pairs'])}")
        if details['tournament']['pairs']:
            current_pair = details['tournament']['pairs'][details['tournament']['current_pair']]
            print(f"   Current battle: {current_pair['model_a']['name']} vs {current_pair['model_b']['name']}")
            model_a_id = current_pair['model_a']['id']
            model_b_id = current_pair['model_b']['id']
        else:
            print("   No pairs found!")
            return
    else:
        print(f"   Error: {response.text}")
        return
      # 4. Cast a vote
    print("\n4. Casting vote...")
    vote_data = {
        "tournament_id": tournament_id,
        "winner_id": model_a_id,  # Vote for model A
        "confidence": 0.8,
        "user_id": user_id,
        "reasoning": "Test vote"
    }
    
    response = requests.post(f"{BASE_URL}/api/tournaments/vote-db", json=vote_data)
    print(f"   Vote response: {response.status_code}")
    if response.status_code == 200:
        vote_result = response.json()
        print(f"   Vote success: {vote_result['success']}")
        print(f"   Tournament status: {vote_result['tournament']['status']}")
        print(f"   New current round: {vote_result['tournament']['current_round']}")
        print(f"   New current pair: {vote_result['tournament']['current_pair']}")
        print(f"   Winner: {vote_result['vote']['winner']}")
        
        # Check if tournament progressed
        if vote_result['tournament']['current_pair'] is not None:
            print(f"   ✅ Tournament progressed to Round {vote_result['tournament']['current_round']}, Pair {vote_result['tournament']['current_pair']}")
        else:
            print(f"   🏁 Tournament completed!")
    else:
        print(f"   Error: {response.text}")
        return
    
    # 5. Get updated tournament details
    print("\n5. Getting updated tournament details...")
    response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}")
    print(f"   Get tournament response: {response.status_code}")
    if response.status_code == 200:
        details = response.json()
        print(f"   Updated tournament status: {details['tournament']['status']}")
        print(f"   Updated current round: {details['tournament']['current_round']}")
        print(f"   Updated current pair: {details['tournament']['current_pair']}")
        print(f"   Total pairs: {len(details['tournament']['pairs'])}")
        
        if details['tournament']['current_pair'] is not None and details['tournament']['current_pair'] < len(details['tournament']['pairs']):
            current_pair = details['tournament']['pairs'][details['tournament']['current_pair']]
            print(f"   Next battle: {current_pair['model_a']['name']} vs {current_pair['model_b']['name']}")
        elif details['tournament']['status'] == 'completed':
            print(f"   🏆 Tournament completed! Winner: {details['tournament'].get('victor_model_id', 'Unknown')}")
        else:
            print(f"   ⚠️  Tournament in unexpected state")
    else:
        print(f"   Error: {response.text}")

if __name__ == "__main__":
    test_tournament_flow()
