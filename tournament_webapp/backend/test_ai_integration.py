#!/usr/bin/env python3
"""
Test AI Mixer Integration with Tournament Webapp
"""

import sys
from pathlib import Path

# Add backend path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_ai_mixer_integration():
    """Test that AI mixer integration works"""
    print("🧪 Testing AI Mixer Integration")
    print("=" * 40)
    
    try:
        from ai_mixer_integration import get_tournament_ai_mixer
        
        mixer = get_tournament_ai_mixer()
        print(f"✅ AI Mixer instance created")
        print(f"📊 Available: {mixer.is_available()}")
        
        # Test model capabilities
        capabilities = mixer.get_model_capabilities("enhanced_cnn")
        print(f"🎛️ Enhanced CNN capabilities: {capabilities}")
        
        # Test with a simple audio file (create a dummy one)
        test_audio = backend_dir / "test_audio.wav"
        output_audio = backend_dir / "test_output.wav"
        
        # Create a dummy WAV file
        with open(test_audio, "wb") as f:
            # WAV header for a 1-second 44.1kHz mono file
            f.write(b'RIFF$\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
            f.write(b'\x00' * 2048)  # Silent audio data
        
        # Test processing
        result = mixer.process_audio_with_model(
            str(test_audio),
            "enhanced_cnn",
            str(output_audio)
        )
        
        print(f"🎵 Audio processing result: {result}")
        
        if output_audio.exists():
            print(f"📁 Output file created: {output_audio}")
            # Clean up
            test_audio.unlink()
            output_audio.unlink()
        
        print("✅ AI Mixer integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ AI Mixer integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_mixer_integration()
