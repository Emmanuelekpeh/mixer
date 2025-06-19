"""
Simple test to verify the tournament creation endpoint
"""

import requests
import json

BASE_URL = "http://localhost:10000"

def test_tournament_creation():
    print("🧪 Testing Tournament Creation Endpoint")
    print("=" * 50)
    
    # Test JSON request (frontend style)
    print("\n1️⃣ Testing JSON Request...")
    try:
        data = {
            "user_id": "frontend_user",
            "username": "Frontend User", 
            "max_rounds": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/api/tournaments/create",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                tournament_id = result.get("tournament_id") or result.get("tournament", {}).get("id")
                print(f"   ✅ Tournament created: {tournament_id}")
                
                # Test getting tournament details
                print(f"\n2️⃣ Testing Tournament Retrieval...")
                detail_response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}")
                print(f"   Status: {detail_response.status_code}")
                if detail_response.status_code == 200:
                    print(f"   ✅ Tournament retrieved successfully")
                else:
                    print(f"   ❌ Tournament retrieval failed: {detail_response.text}")
                
                return tournament_id
            else:
                print(f"   ❌ Creation failed: {result}")
        else:
            print(f"   ❌ Request failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return None

if __name__ == "__main__":
    tournament_id = test_tournament_creation()
    
    if tournament_id:
        print(f"\n🎯 SUCCESS: Tournament system is working!")
        print(f"   Tournament ID: {tournament_id}")
        print(f"   Frontend can now create and access tournaments")
    else:
        print(f"\n❌ ISSUE: Tournament system needs fixing")
