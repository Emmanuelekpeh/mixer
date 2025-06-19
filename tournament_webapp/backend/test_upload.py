"""
Test the file upload endpoint
"""
import requests
import io

def test_upload_endpoint():
    print("🧪 Testing File Upload Endpoint")
    print("=" * 40)
    
    # Create a dummy audio file
    dummy_audio = b"FAKE_AUDIO_DATA_FOR_TESTING"
    audio_file = io.BytesIO(dummy_audio)
    
    # Prepare form data
    files = {
        'audio_file': ('test_audio.wav', audio_file, 'audio/wav')
    }
    
    data = {
        'user_id': 'upload_test_user',
        'username': 'Upload Test User',
        'max_rounds': '3',
        'audio_features': '{}'
    }
    
    try:
        response = requests.post(
            'http://localhost:10000/api/tournaments/upload',
            files=files,
            data=data
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Upload endpoint working!")
                print(f"Tournament ID: {result.get('tournament_id')}")
                return True
            else:
                print(f"❌ Upload failed: {result}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    success = test_upload_endpoint()
    if success:
        print("\n🎯 UPLOAD ENDPOINT IS WORKING!")
        print("Frontend can now upload audio files and create tournaments")
    else:
        print("\n❌ Upload endpoint needs fixing")
