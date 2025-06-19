#!/usr/bin/env python3
"""
Test tournament completion after 5 rounds
"""
import requests
import json
import time

BASE_URL = "http://localhost:10000"

def test_full_tournament():
    print("🧪 Testing FULL Tournament (5 rounds)...")
    
    # 1. Create a user
    print("\n1. Creating user...")
    user_data = {
        "user_id": f"completion_test_{int(time.time())}",
        "username": "CompletionTest"
    }
    
    response = requests.post(f"{BASE_URL}/api/users/create", json=user_data)
    if response.status_code == 200:
        user_result = response.json()
        user_id = user_result["profile"]["user_id"]
        print(f"   ✅ User created: {user_id}")
    else:
        print(f"   ❌ Error: {response.text}")
        return
    
    # 2. Create a tournament
    print("\n2. Creating tournament...")
    tournament_data = {
        "user_id": user_id,
        "username": user_data["username"],
        "max_rounds": 5
    }
    
    response = requests.post(f"{BASE_URL}/api/tournaments/create-json", json=tournament_data)
    if response.status_code == 200:
        tournament_result = response.json()
        tournament_id = tournament_result["tournament_id"]
        tournament = tournament_result["tournament"]
        print(f"   ✅ Tournament created: {tournament_id}")
        print(f"   📊 Total pairs: {tournament['total_pairs']}")
        print(f"   🎯 Max rounds: {tournament['max_rounds']}")
    else:
        print(f"   ❌ Error: {response.text}")
        return
    
    # 3. Vote through all 5 rounds
    for round_num in range(1, 6):  # Rounds 1-5
        print(f"\n--- ROUND {round_num} ---")
        
        # Get current tournament state
        response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}")
        if response.status_code == 200:
            details = response.json()
            tournament_data = details['tournament']
            
            if tournament_data['status'] == 'completed':
                print(f"   🏆 Tournament already completed!")
                print(f"   🥇 Winner: {tournament_data.get('victor_model_id', 'Unknown')}")
                break
                
            current_pair_data = tournament_data['pairs'][tournament_data['current_pair']]
            model_a_id = current_pair_data['model_a']['id']
            model_b_id = current_pair_data['model_b']['id']
            
            print(f"   🥊 Battle: {current_pair_data['model_a']['name']} vs {current_pair_data['model_b']['name']}")
            print(f"   📍 Current round: {tournament_data['current_round']}")
            print(f"   📍 Current pair: {tournament_data['current_pair']}")
        else:
            print(f"   ❌ Error getting tournament: {response.text}")
            break
        
        # Cast vote (always vote for model A)
        vote_data = {
            "tournament_id": tournament_id,
            "winner_id": model_a_id,
            "confidence": 0.8,
            "user_id": user_id,
            "reasoning": f"Round {round_num} vote"
        }
        
        response = requests.post(f"{BASE_URL}/api/tournaments/vote-db", json=vote_data)
        if response.status_code == 200:
            vote_result = response.json()
            print(f"   ✅ Vote cast: {vote_result['vote']['winner']} wins!")
            print(f"   📊 New status: {vote_result['tournament']['status']}")
            
            if vote_result['tournament']['status'] == 'completed':
                print(f"   🎉 TOURNAMENT COMPLETED!")
                print(f"   🏆 Final Champion: {vote_result['tournament'].get('winner', 'Unknown')}")
                break
            else:
                print(f"   ➡️  Advanced to Round {vote_result['tournament']['current_round']}")
        else:
            print(f"   ❌ Vote error: {response.text}")
            break
        
        time.sleep(0.5)  # Brief pause between rounds
    
    print(f"\n🔚 Tournament test completed!")

if __name__ == "__main__":
    test_full_tournament()
