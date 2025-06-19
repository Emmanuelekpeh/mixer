#!/usr/bin/env python3
"""
Simple Upload Response Test - Check exact response format
"""

import requests
import tempfile
import os
import json

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

def test_upload_response():
    """Test upload and show exact response"""
    print("🧪 UPLOAD RESPONSE TEST")
    print("=" * 40)
    
    wav_file = create_test_wav()
    
    try:
        with open(wav_file, 'rb') as f:
            files = {'audio_file': ('test.wav', f, 'audio/wav')}
            data = {
                'user_id': 'response_test_user',
                'username': 'Response Test User',
                'max_rounds': 3,
                'audio_features': '{}'
            }
            
            print(f"📤 Posting to: {BASE_URL}/api/tournaments/upload")
            
            response = requests.post(f"{BASE_URL}/api/tournaments/upload", files=files, data=data)
            
            print(f"📊 Status Code: {response.status_code}")
            print(f"📄 Raw Response:")
            print("-" * 40)
            print(response.text)
            print("-" * 40)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"📋 JSON Structure:")
                    print(json.dumps(result, indent=2))
                    
                    print(f"\n🔍 Top Level Keys:")
                    for key in result.keys():
                        print(f"   - {key}: {type(result[key])}")
                    
                    if 'tournament' in result:
                        print(f"\n🏆 Tournament Object Keys:")
                        for key in result['tournament'].keys():
                            print(f"   - {key}: {type(result['tournament'][key])}")
                    else:
                        print(f"\n❌ NO 'tournament' KEY FOUND!")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error: {e}")
            else:
                print(f"❌ Request failed")
                
    finally:
        try:
            os.unlink(wav_file)
        except:
            pass

if __name__ == "__main__":
    test_upload_response()
