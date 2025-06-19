#!/usr/bin/env python3
"""
🧪 Tournament System Automated Test Suite
=========================================

Comprehensive automated testing to identify and fix tournament progression issues.
"""

import requests
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:10000"

class TournamentSystemTester:
    
    def __init__(self):
        self.test_results = []
        self.current_tournament = None
        
    def log_test(self, test_name, status, details):
        """Log test results"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {details}")
    
    def test_api_health(self):
        """Test basic API connectivity"""
        try:
            response = requests.get(f"{BASE_URL}/api/health")
            if response.status_code == 200:
                self.log_test("API Health Check", "PASS", "API is responding")
                return True
            else:
                self.log_test("API Health Check", "FAIL", f"API returned {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Health Check", "FAIL", f"Connection failed: {e}")
            return False
    
    def test_tournament_creation(self):
        """Test tournament creation with proper data structure"""
        try:
            # Create a simple WAV file for testing
            test_audio = Path("test_audio.wav")
            with open(test_audio, "wb") as f:
                # WAV header for a 1-second 44.1kHz mono file
                f.write(b'RIFF$\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
                f.write(b'\x00' * 2048)
            
            # Test tournament creation
            with open(test_audio, "rb") as audio_file:
                files = {"audio_file": ("test.wav", audio_file, "audio/wav")}
                data = {
                    "user_id": "test_user_auto",
                    "username": "Automated Test User",
                    "max_rounds": "3"
                }
                
                response = requests.post(f"{BASE_URL}/api/tournaments/upload", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        self.current_tournament = result
                        tournament_id = result.get("tournament_id")
                        
                        # Analyze the tournament structure
                        pairs = result.get("pairs", [])
                        tournament_data = result.get("tournament", {})
                        
                        self.log_test("Tournament Creation", "PASS", 
                                    f"Created tournament {tournament_id} with {len(pairs)} pairs")
                        
                        # Test the tournament structure
                        self.analyze_tournament_structure(tournament_data, pairs)
                        return True
                    else:
                        self.log_test("Tournament Creation", "FAIL", f"API returned success=false: {result}")
                        return False
                else:
                    self.log_test("Tournament Creation", "FAIL", f"HTTP {response.status_code}: {response.text}")
                    return False
                    
            # Cleanup
            test_audio.unlink()
            
        except Exception as e:
            self.log_test("Tournament Creation", "FAIL", f"Exception: {e}")
            return False
    
    def analyze_tournament_structure(self, tournament_data, pairs):
        """Analyze tournament data structure for issues"""
        issues = []
        
        # Check required fields
        required_fields = ["tournament_id", "current_pair", "status", "pairs"]
        for field in required_fields:
            if field not in tournament_data and field != "pairs":
                issues.append(f"Missing field: {field}")
        
        # Check pairs structure
        if not pairs:
            issues.append("No pairs data provided")
        else:
            # Check first pair structure
            first_pair = pairs[0]
            required_pair_fields = ["model_a", "model_b", "audio_a", "audio_b"]
            for field in required_pair_fields:
                if field not in first_pair:
                    issues.append(f"Pair missing field: {field}")
        
        # Check current_pair value
        current_pair = tournament_data.get("current_pair", -1)
        if current_pair >= len(pairs):
            issues.append(f"current_pair ({current_pair}) >= pairs length ({len(pairs)})")
        
        if issues:
            self.log_test("Tournament Structure Analysis", "WARN", f"Issues found: {', '.join(issues)}")
        else:
            self.log_test("Tournament Structure Analysis", "PASS", "Tournament structure is valid")
    
    def test_tournament_fetch(self):
        """Test fetching tournament details"""
        if not self.current_tournament:
            self.log_test("Tournament Fetch", "SKIP", "No tournament to fetch")
            return False
            
        try:
            tournament_id = self.current_tournament.get("tournament_id")
            response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    tournament_data = result.get("tournament", {})
                    
                    # Check data consistency
                    creation_pairs = len(self.current_tournament.get("pairs", []))
                    fetch_pairs = len(tournament_data.get("pairs", []))
                    
                    if creation_pairs == fetch_pairs:
                        self.log_test("Tournament Fetch", "PASS", 
                                    f"Fetched tournament with {fetch_pairs} pairs (consistent)")
                    else:
                        self.log_test("Tournament Fetch", "WARN", 
                                    f"Pair count mismatch: creation={creation_pairs}, fetch={fetch_pairs}")
                    
                    # Store updated tournament data
                    self.current_tournament["fetched_data"] = tournament_data
                    return True
                else:
                    self.log_test("Tournament Fetch", "FAIL", f"API returned success=false")
                    return False
            else:
                self.log_test("Tournament Fetch", "FAIL", f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Tournament Fetch", "FAIL", f"Exception: {e}")
            return False
    
    def test_vote_submission(self):
        """Test voting and progression"""
        if not self.current_tournament:
            self.log_test("Vote Submission", "SKIP", "No tournament available")
            return False
            
        try:
            tournament_id = self.current_tournament.get("tournament_id")
            pairs = self.current_tournament.get("pairs", [])
            
            if not pairs:
                self.log_test("Vote Submission", "FAIL", "No pairs available for voting")
                return False
            
            # Get the first pair for voting
            first_pair = pairs[0]
            model_a_id = first_pair["model_a"]["id"]
            model_b_id = first_pair["model_b"]["id"]
            
            # Submit vote
            vote_data = {
                "tournament_id": tournament_id,
                "winner_id": model_a_id,  # Vote for model A
                "confidence": 0.8,
                "user_id": "test_user_auto",
                "reasoning": "Automated test vote"
            }
            
            response = requests.post(f"{BASE_URL}/api/tournaments/vote-db", json=vote_data)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    vote_info = result.get("vote", {})
                    tournament_info = result.get("tournament", {})
                    
                    # Check progression
                    new_current_pair = tournament_info.get("current_pair", -1)
                    
                    self.log_test("Vote Submission", "PASS", 
                                f"Vote submitted, current_pair advanced to {new_current_pair}")
                    
                    # Test the progression logic
                    self.test_progression_logic(tournament_info, pairs)
                    return True
                else:
                    self.log_test("Vote Submission", "FAIL", f"Vote failed: {result}")
                    return False
            else:
                self.log_test("Vote Submission", "FAIL", f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Vote Submission", "FAIL", f"Exception: {e}")
            return False
    
    def test_progression_logic(self, tournament_info, original_pairs):
        """Test the progression logic matches frontend expectations"""
        current_pair = tournament_info.get("current_pair", -1)
        next_battle = tournament_info.get("next_battle")
        
        # Check if current_pair is valid
        if 0 <= current_pair < len(original_pairs):
            expected_next_pair = original_pairs[current_pair]
            
            if next_battle:
                # Compare structure
                if (next_battle.get("model_a", {}).get("id") == expected_next_pair["model_a"]["id"] and
                    next_battle.get("model_b", {}).get("id") == expected_next_pair["model_b"]["id"]):
                    self.log_test("Progression Logic", "PASS", "Next battle matches expected pair")
                else:
                    self.log_test("Progression Logic", "WARN", "Next battle doesn't match expected pair")
            else:
                self.log_test("Progression Logic", "WARN", "No next_battle provided in response")
        else:
            if current_pair >= len(original_pairs):
                self.log_test("Progression Logic", "PASS", "Tournament completed (current_pair >= total_pairs)")
            else:
                self.log_test("Progression Logic", "FAIL", f"Invalid current_pair: {current_pair}")
    
    def test_frontend_backend_sync(self):
        """Test data format compatibility between frontend and backend"""
        if not self.current_tournament:
            self.log_test("Frontend-Backend Sync", "SKIP", "No tournament data")
            return False
        
        # Check if the response format matches what frontend expects
        tournament_data = self.current_tournament.get("fetched_data", {})
        issues = []
        
        # Frontend expects these fields
        expected_fields = {
            "tournament_id": str,
            "current_pair": int,
            "pairs": list,
            "status": str,
            "current_round": int,
            "max_rounds": int
        }
        
        for field, expected_type in expected_fields.items():
            if field not in tournament_data:
                issues.append(f"Missing {field}")
            elif not isinstance(tournament_data[field], expected_type):
                issues.append(f"{field} is {type(tournament_data[field])}, expected {expected_type}")
        
        # Check pairs structure
        pairs = tournament_data.get("pairs", [])
        if pairs and len(pairs) > 0:
            pair = pairs[0]
            expected_pair_fields = ["model_a", "model_b", "audio_a", "audio_b"]
            for field in expected_pair_fields:
                if field not in pair:
                    issues.append(f"Pair missing {field}")
        
        if issues:
            self.log_test("Frontend-Backend Sync", "FAIL", f"Compatibility issues: {', '.join(issues)}")
            return False
        else:
            self.log_test("Frontend-Backend Sync", "PASS", "Data format is frontend-compatible")
            return True
    
    def run_full_test_suite(self):
        """Run complete test suite"""
        print("🧪 Starting Tournament System Automated Tests")
        print("=" * 50)
        
        # Run tests in order
        tests = [
            self.test_api_health,
            self.test_tournament_creation,
            self.test_tournament_fetch,
            self.test_frontend_backend_sync,
            self.test_vote_submission,
        ]
        
        passed = 0
        failed = 0
        warned = 0
        
        for test in tests:
            try:
                result = test()
                if result is True:
                    passed += 1
                elif result is False:
                    failed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} crashed: {e}")
                failed += 1
        
        # Count warnings
        warned = sum(1 for result in self.test_results if result["status"] == "WARN")
        
        print("\n" + "="*50)
        print(f"🏁 Test Results: {passed} passed, {failed} failed, {warned} warnings")
        
        if failed > 0:
            print("\n❌ CRITICAL ISSUES FOUND:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['details']}")
        
        if warned > 0:
            print("\n⚠️ WARNINGS:")
            for result in self.test_results:
                if result["status"] == "WARN":
                    print(f"  - {result['test']}: {result['details']}")
        
        return self.test_results

if __name__ == "__main__":
    tester = TournamentSystemTester()
    results = tester.run_full_test_suite()
    
    # Write results to file
    with open("tournament_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
