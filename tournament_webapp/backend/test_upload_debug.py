#!/usr/bin/env python3
"""
Upload Endpoint Diagnostic Test - Find the exact 500 error
"""

import requests
import tempfile
import os
import traceback

BASE_URL = "http://localhost:10000"

def create_test_wav():
    """Create a proper test WAV file"""
    try:
        # Create a minimal but valid WAV file
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
        
        print(f"✅ Created test WAV: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        print(f"❌ Failed to create WAV: {e}")
        return None

def test_upload_endpoint():
    """Test the upload endpoint and catch exact errors"""
    print("🧪 TESTING UPLOAD ENDPOINT")
    print("=" * 50)
    
    wav_file = create_test_wav()
    if not wav_file:
        return False
    
    try:
        print(f"📤 Testing: {BASE_URL}/api/tournaments/upload")
        
        with open(wav_file, 'rb') as f:
            files = {
                'audio_file': ('test_song.wav', f, 'audio/wav')
            }
            data = {
                'user_id': 'upload_test_user',
                'username': 'Upload Test User',
                'max_rounds': 3,
                'audio_features': '{}'
            }
            
            print(f"📝 Form data: {data}")
            print(f"📁 File size: {os.path.getsize(wav_file)} bytes")
            
            response = requests.post(
                f"{BASE_URL}/api/tournaments/upload", 
                files=files, 
                data=data, 
                timeout=30
            )
            
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ SUCCESS!")
                print(f"   Tournament ID: {result.get('tournament_id')}")
                print(f"   Message: {result.get('message')}")
                return True
            else:
                print(f"❌ FAILED!")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text}")
                
                try:
                    error_data = response.json()
                    print(f"   Error Details: {error_data}")
                except:
                    print(f"   Raw Response: {response.text[:500]}")
                
                return False
                
    except Exception as e:
        print(f"❌ REQUEST EXCEPTION: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        try:
            os.unlink(wav_file)
            print(f"🗑️ Cleaned up test file")
        except:
            pass

if __name__ == "__main__":
    print("🚀 UPLOAD ENDPOINT DIAGNOSTIC TEST")
    print("=" * 60)
    
    success = test_upload_endpoint()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 UPLOAD ENDPOINT IS WORKING!")
    else:
        print("🔧 UPLOAD ENDPOINT FAILED - CHECK SERVER LOGS!")
        print("   Common issues:")
        print("   - File path handling (Windows paths with backslashes)")
        print("   - Database constraint violations")
        print("   - Missing static directory creation")
