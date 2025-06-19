#!/usr/bin/env python3
"""
Test script to verify audio processing functionality
"""

import requests
import json
from pathlib import Path

def test_upload_tournament():
    """Test creating a tournament with uploaded audio"""
    
    # Create a dummy audio file for testing
    test_audio_path = Path("test_audio.wav")
    if not test_audio_path.exists():
        # Create a small dummy WAV file
        with open(test_audio_path, "wb") as f:
            # Write minimal WAV header (44 bytes) + some data
            f.write(b'RIFF\x28\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x04\x00\x00\x00\x00\x00\x00\x00')
    
    # Test data
    test_data = {
        "user_id": "test_user_audio",
        "username": "TestUser",
        "max_rounds": 3,
        "audio_features": "{}"
    }
    
    # Upload file
    with open(test_audio_path, "rb") as f:
        files = {"audio_file": ("test_audio.wav", f, "audio/wav")}
        
        try:
            response = requests.post(
                "http://localhost:10000/api/tournaments/upload",
                data=test_data,
                files=files,
                timeout=30
            )
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Body: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                tournament_id = result.get("tournament_id")
                print(f"✅ Tournament created: {tournament_id}")
                
                # Check if processed audio files were created
                processed_dir = Path("processed_audio")
                audio_files = list(processed_dir.glob(f"{tournament_id}_*_mix.wav"))
                print(f"📁 Created audio files: {len(audio_files)}")
                for audio_file in audio_files:
                    print(f"   - {audio_file.name}")
                
                # Test accessing the processed audio URLs
                for audio_file in audio_files:
                    audio_url = f"http://localhost:10000/processed_audio/{audio_file.name}"
                    audio_response = requests.head(audio_url)
                    print(f"🔗 {audio_url}: {audio_response.status_code}")
                
                return tournament_id
            else:
                print(f"❌ Tournament creation failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
    
    # Clean up
    if test_audio_path.exists():
        test_audio_path.unlink()
    
    return None

if __name__ == "__main__":
    print("🧪 Testing upload tournament creation...")
    tournament_id = test_upload_tournament()
    
    if tournament_id:
        print(f"✅ Test completed successfully for tournament: {tournament_id}")
    else:
        print("❌ Test failed")
