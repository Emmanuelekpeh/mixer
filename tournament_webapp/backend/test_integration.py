#!/usr/bin/env python3
"""
Frontend-Backend Integration Test
Test that the backend responses match frontend expectations
"""

import requests
import tempfile
import os

BASE_URL = "http://localhost:10000"

def create_test_wav():
    """Create a test WAV file"""
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
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.write(wav_header)
    temp_file.close()
    return temp_file.name

def test_frontend_backend_integration():
    """Test that backend responses match frontend expectations"""
    print("🔗 FRONTEND-BACKEND INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Tournament creation response format
    print("\n1️⃣ Testing Tournament Creation Response Format...")
    
    wav_file = create_test_wav()
    try:
        with open(wav_file, 'rb') as f:
            files = {'audio_file': ('test.wav', f, 'audio/wav')}
            data = {
                'user_id': 'integration_test_user',
                'username': 'Integration Test User',
                'max_rounds': 3,
                'audio_features': '{}'
            }
            
            response = requests.post(f"{BASE_URL}/api/tournaments/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Check required fields for frontend
                required_fields = ['success', 'tournament_id', 'tournament']
                missing_fields = []
                
                for field in required_fields:
                    if field not in result:
                        missing_fields.append(field)
                
                if not missing_fields:
                    print("   ✅ All required fields present")
                    
                    # Check tournament object structure
                    tournament = result['tournament']
                    tournament_fields = ['id', 'tournament_id', 'pairs', 'status', 'current_round']
                    missing_tournament_fields = []
                    
                    for field in tournament_fields:
                        if field not in tournament:
                            missing_tournament_fields.append(field)
                    
                    if not missing_tournament_fields:
                        print("   ✅ Tournament object properly structured")
                        tournament_id = result['tournament_id']
                        
                        # Test 2: Tournament retrieval response format
                        print("\n2️⃣ Testing Tournament Retrieval Response Format...")
                        
                        retrieval_response = requests.get(f"{BASE_URL}/api/tournaments/{tournament_id}")
                        
                        if retrieval_response.status_code == 200:
                            retrieval_result = retrieval_response.json()
                            
                            if 'success' in retrieval_result and 'tournament' in retrieval_result:
                                retrieved_tournament = retrieval_result['tournament']
                                
                                # Check if pairs data is present and properly formatted
                                if 'pairs' in retrieved_tournament and isinstance(retrieved_tournament['pairs'], list):
                                    pairs = retrieved_tournament['pairs']
                                    if len(pairs) > 0:
                                        pair = pairs[0]
                                        if 'model_a' in pair and 'model_b' in pair:
                                            print("   ✅ Tournament retrieval working correctly")
                                            print("   ✅ Pairs data properly formatted")
                                            
                                            # Summary
                                            print("\n" + "=" * 50)
                                            print("🎉 INTEGRATION TEST RESULTS")
                                            print("=" * 50)
                                            print(f"✅ Tournament Creation: Working")
                                            print(f"✅ Response Format: Compatible with frontend")
                                            print(f"✅ Tournament Retrieval: Working")
                                            print(f"✅ Data Structure: Matches frontend expectations")
                                            print(f"")
                                            print(f"🎯 FRONTEND-BACKEND INTEGRATION: SUCCESS!")
                                            print(f"📋 Tournament ID: {tournament_id}")
                                            print(f"📊 Available Pairs: {len(pairs)}")
                                            
                                            return True
                                        else:
                                            print("   ❌ Pair structure missing model_a/model_b")
                                    else:
                                        print("   ❌ No pairs found in tournament")
                                else:
                                    print("   ❌ Pairs data missing or invalid")
                            else:
                                print("   ❌ Tournament retrieval response malformed")
                        else:
                            print(f"   ❌ Tournament retrieval failed: {retrieval_response.status_code}")
                    else:
                        print(f"   ❌ Missing tournament fields: {missing_tournament_fields}")
                else:
                    print(f"   ❌ Missing required fields: {missing_fields}")
            else:
                print(f"   ❌ Tournament creation failed: {response.status_code}")
                print(f"   Response: {response.text}")
    
    finally:
        try:
            os.unlink(wav_file)
        except:
            pass
    
    print("\n❌ INTEGRATION TEST FAILED - Check individual components")
    return False

if __name__ == "__main__":
    success = test_frontend_backend_integration()
    
    if not success:
        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("1. Check if backend server is running on port 10000")
        print("2. Verify frontend proxy configuration")
        print("3. Check browser network tab for API call errors")
        print("4. Ensure database has tournament data")
