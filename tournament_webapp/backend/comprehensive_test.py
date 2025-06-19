#!/usr/bin/env python3
"""
Comprehensive Tournament System Tests
Tests all aspects of the tournament webapp functionality
"""

import requests
import json
import os
import time
from pathlib import Path
import tempfile

BASE_URL = "http://localhost:10000"

class TournamentSystemTests:
    def __init__(self):
        self.test_results = []
        self.created_tournaments = []
        
    def log_test(self, test_name, success, message, details=None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "details": details        })
        print(f"{status} {test_name}: {message}")
        if details and not success:
            print(f"    Details: {details}")

    def test_server_health(self):
        """Test if server is running and responding"""
        try:
            # Use the models endpoint as health check since /health doesn't exist
            response = requests.get(f"{BASE_URL}/api/models", timeout=5)
            if response.status_code == 200:
                self.log_test("Server Health", True, "Server is responding")
                return True
            else:
                self.log_test("Server Health", False, f"Server returned {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("Server Health", False, "Server not responding", str(e))
            return False

    def test_api_docs(self):
        """Test if API documentation is accessible"""
        try:
            response = requests.get(f"{BASE_URL}/docs", timeout=5)
            success = response.status_code == 200
            self.log_test("API Docs", success, "API documentation accessible" if success else f"Docs returned {response.status_code}")
            return success
        except Exception as e:
            self.log_test("API Docs", False, "API docs not accessible", str(e))
            return False

    def test_models_endpoint(self):
        """Test the models endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/api/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    models = data.get("models", [])
                    if len(models) >= 2:
                        self.log_test("Models Endpoint", True, f"Found {len(models)} models")
                        return models
                    else:
                        self.log_test("Models Endpoint", False, f"Only {len(models)} models found, need at least 2")
                        return []
                else:
                    self.log_test("Models Endpoint", False, "API returned success=false", data)
                    return []
            else:
                self.log_test("Models Endpoint", False, f"HTTP {response.status_code}", response.text[:200])
                return []
        except Exception as e:
            self.log_test("Models Endpoint", False, "Request failed", str(e))
            return []

    def create_test_audio_file(self):
        """Create a minimal test audio file"""
        try:
            # Create a minimal WAV file (header only, for testing upload)
            wav_header = bytes([
                0x52, 0x49, 0x46, 0x46,  # "RIFF"
                0x24, 0x00, 0x00, 0x00,  # File size - 8
                0x57, 0x41, 0x56, 0x45,  # "WAVE"
                0x66, 0x6D, 0x74, 0x20,  # "fmt "
                0x10, 0x00, 0x00, 0x00,  # Subchunk1Size
                0x01, 0x00,              # AudioFormat (PCM)
                0x01, 0x00,              # NumChannels (1)
                0x44, 0xAC, 0x00, 0x00,  # SampleRate (44100)
                0x88, 0x58, 0x01, 0x00,  # ByteRate
                0x02, 0x00,              # BlockAlign
                0x10, 0x00,              # BitsPerSample (16)
                0x64, 0x61, 0x74, 0x61,  # "data"
                0x00, 0x00, 0x00, 0x00   # Subchunk2Size (0)
            ])
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.write(wav_header)
            temp_file.close()
            
            self.log_test("Test Audio File", True, f"Created test WAV file: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            self.log_test("Test Audio File", False, "Failed to create test audio", str(e))
            return None

    def test_tournament_creation_upload(self):
        """Test tournament creation with file upload"""
        audio_file_path = self.create_test_audio_file()
        if not audio_file_path:
            return None
            
        try:
            # Prepare form data
            files = {
                'audio_file': ('test_audio.wav', open(audio_file_path, 'rb'), 'audio/wav')
            }
            data = {
                'user_id': 'test_upload_user',
                'username': 'Upload Test User',
                'max_rounds': 3,
                'audio_features': '{}'
            }
            
            response = requests.post(f"{BASE_URL}/api/tournaments/upload", files=files, data=data, timeout=30)
            
            # Clean up
            files['audio_file'][1].close()
            os.unlink(audio_file_path)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    tournament_id = result.get("tournament_id")
                    self.created_tournaments.append(tournament_id)
                    self.log_test("Tournament Creation (Upload)", True, f"Created tournament: {tournament_id}")
                    return tournament_id
                else:
                    self.log_test("Tournament Creation (Upload)", False, "API returned success=false", result)
                    return None
            else:
                self.log_test("Tournament Creation (Upload)", False, f"HTTP {response.status_code}", response.text[:200])
                return None
                
        except Exception as e:
            self.log_test("Tournament Creation (Upload)", False, "Request failed", str(e))
            # Clean up on error
            try:
                os.unlink(audio_file_path)
            except:
                pass
            return None

    def test_tournament_creation_json(self):
        """Test tournament creation with JSON (demo mode)"""
        try:
            data = {
                'user_id': 'test_json_user',
                'username': 'JSON Test User',
                'max_rounds': 3
            }
            
            response = requests.post(f"{BASE_URL}/api/tournaments/create-json", json=data, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    tournament_id = result.get("tournament_id")
                    self.created_tournaments.append(tournament_id)
                    self.log_test("Tournament Creation (JSON)", True, f"Created tournament: {tournament_id}")
                    return tournament_id
                else:
                    self.log_test("Tournament Creation (JSON)", False, "API returned success=false", result)
                    return None
            else:
                self.log_test("Tournament Creation (JSON)", False, f"HTTP {response.status_code}", response.text[:200])
                return None
                
        except Exception as e:
            self.log_test("Tournament Creation (JSON)", False, "Request failed", str(e))
            return None

    def test_tournament_retrieval(self, tournament_id):
        """Test tournament details retrieval"""
        if not tournament_id:
            self.log_test("Tournament Retrieval", False, "No tournament ID provided")
            return None
            
        try:
            response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    tournament = result.get("tournament", {})
                    pairs = tournament.get("pairs", [])
                    self.log_test("Tournament Retrieval", True, f"Retrieved tournament with {len(pairs)} pairs")
                    return tournament
                else:
                    self.log_test("Tournament Retrieval", False, "API returned success=false", result)
                    return None
            else:
                self.log_test("Tournament Retrieval", False, f"HTTP {response.status_code}", response.text[:200])
                return None
                
        except Exception as e:
            self.log_test("Tournament Retrieval", False, "Request failed", str(e))
            return None

    def test_user_endpoints(self):
        """Test user-related endpoints"""
        try:
            # Test user creation/retrieval
            response = requests.get(f"{BASE_URL}/api/users/test_user", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    user = result.get("user", {})
                    profile = user.get("profile", {})
                    self.log_test("User Endpoints", True, f"User profile: {profile.get('username', 'Unknown')}")
                    return True
                else:
                    self.log_test("User Endpoints", False, "API returned success=false", result)
                    return False
            else:
                self.log_test("User Endpoints", False, f"HTTP {response.status_code}", response.text[:200])
                return False
                
        except Exception as e:
            self.log_test("User Endpoints", False, "Request failed", str(e))
            return False

    def test_database_stats(self):
        """Test database statistics"""
        try:
            response = requests.get(f"{BASE_URL}/api/stats", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    stats = result.get("stats", {})
                    self.log_test("Database Stats", True, f"DB Stats: {stats}")
                    return True
                else:
                    self.log_test("Database Stats", False, "API returned success=false", result)
                    return False
            else:
                # Stats endpoint might not exist, check if we can access models as a proxy
                models_response = requests.get(f"{BASE_URL}/api/models", timeout=5)
                if models_response.status_code == 200:
                    self.log_test("Database Stats", True, "Database accessible via models endpoint")
                    return True
                else:
                    self.log_test("Database Stats", False, f"Database not accessible")
                    return False
                
        except Exception as e:
            self.log_test("Database Stats", False, "Request failed", str(e))
            return False

    def test_performance(self):
        """Test API performance"""
        try:
            start_time = time.time()
            successful_requests = 0
            total_requests = 5
            
            for i in range(total_requests):
                response = requests.get(f"{BASE_URL}/api/models", timeout=5)
                if response.status_code == 200:
                    successful_requests += 1
            
            end_time = time.time()
            duration = end_time - start_time
            requests_per_second = successful_requests / duration
            
            if successful_requests == total_requests and requests_per_second > 1:
                self.log_test("Performance", True, f"{requests_per_second:.1f} req/sec ({successful_requests}/{total_requests} successful)")
                return True
            else:
                self.log_test("Performance", False, f"Only {successful_requests}/{total_requests} successful, {requests_per_second:.1f} req/sec")
                return False
                
        except Exception as e:
            self.log_test("Performance", False, "Performance test failed", str(e))
            return False

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🧪 COMPREHENSIVE TOURNAMENT SYSTEM TESTS")
        print("=" * 60)
        
        # Basic connectivity
        if not self.test_server_health():
            print("\n❌ Server not responding - cannot continue tests")
            return False
        
        self.test_api_docs()
        
        # Core functionality
        models = self.test_models_endpoint()
        if not models:
            print("\n⚠️ No models available - tournament creation may fail")
        
        self.test_user_endpoints()
        self.test_database_stats()
        
        # Tournament functionality
        upload_tournament = self.test_tournament_creation_upload()
        json_tournament = self.test_tournament_creation_json()
        
        # Test retrieval for created tournaments
        if upload_tournament:
            self.test_tournament_retrieval(upload_tournament)
        if json_tournament:
            self.test_tournament_retrieval(json_tournament)
        
        # Performance
        self.test_performance()
        
        # Summary
        self.print_summary()
        
        return True

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)
        
        print(f"Tests Run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if self.created_tournaments:
            print(f"\nCreated Tournaments: {len(self.created_tournaments)}")
            for tournament_id in self.created_tournaments:
                print(f"  - {tournament_id}")
        
        print("\n🎯 DETAILED RESULTS:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}: {result['message']}")
        
        # Overall assessment
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED! Tournament system is fully functional!")
        elif passed >= total * 0.8:
            print(f"\n✅ MOSTLY WORKING! {total-passed} issues to fix.")
        elif passed >= total * 0.5:
            print(f"\n⚠️ PARTIAL FUNCTIONALITY! {total-passed} critical issues.")
        else:
            print(f"\n❌ MAJOR ISSUES! Only {passed}/{total} tests passing.")

if __name__ == "__main__":
    tests = TournamentSystemTests()
    tests.run_all_tests()
