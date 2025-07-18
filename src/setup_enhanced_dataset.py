#!/usr/bin/env python3
"""
🎵 Enhanced Dataset Setup Script
==============================

This script prepares your environment for working with the enhanced
dataset pipeline for AI mixing. It:

1. Creates the necessary directory structure
2. Installs required dependencies
3. Tests that everything is properly configured
4. Provides next steps for dataset creation
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check that Python is at least 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def install_dependencies():
    """Install required packages for enhanced dataset pipeline"""
    print("📦 Installing required packages...")
    
    requirements = [
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "requests>=2.25.0"
    ]
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directory_structure(base_dir=None):
    """Create the enhanced dataset directory structure"""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "data"
    else:
        base_dir = Path(base_dir)
    
    print(f"📁 Creating directory structure in {base_dir}...")
    
    # Main directories
    dirs = [
        # Raw data
        base_dir / "raw" / "music" / "fma",
        base_dir / "raw" / "vocals" / "damp",
        base_dir / "raw" / "acoustics" / "room_impulse_responses",
        
        # Processed data
        base_dir / "processed" / "clean" / "music",
        base_dir / "processed" / "clean" / "vocals",
        base_dir / "processed" / "clean" / "acoustics",
        base_dir / "processed" / "augmented" / "music",
        base_dir / "processed" / "augmented" / "vocals",
        
        # Features
        base_dir / "features" / "spectrograms",
        base_dir / "features" / "ast_features",
        
        # Metadata
        base_dir / "metadata"
    ]
    
    # Create each directory
    created_count = 0
    for dir_path in dirs:
        try:
            dir_path.mkdir(exist_ok=True, parents=True)
            created_count += 1
        except Exception as e:
            print(f"❌ Error creating directory {dir_path}: {e}")
    
    print(f"✅ Created {created_count}/{len(dirs)} directories")
    return created_count == len(dirs)

def test_environment():
    """Test that the environment is properly set up"""
    print("🧪 Testing environment setup...")
    
    try:
        # Test importing required packages
        import numpy
        import pandas
        import librosa
        import soundfile
        import scipy
        import tqdm
        import matplotlib
        import requests
        
        print("✅ All packages imported successfully")
        
        # Test audio processing capabilities
        print("🔊 Testing audio processing capabilities...")
        
        # Create a simple sine wave
        import numpy as np
        sample_rate = 44100
        duration = 1  # seconds
        frequency = 440  # Hz (A4)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Create a test file path
        test_file_path = Path(__file__).resolve().parent.parent / "data" / "test_audio.wav"
        
        # Save and load audio
        import soundfile as sf
        sf.write(test_file_path, audio, sample_rate)
        
        # Load with librosa
        import librosa
        audio_loaded, sr = librosa.load(str(test_file_path), sr=None)
        
        # Extract features
        import librosa.feature
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_loaded, sr=sr)
        
        # Cleanup test file
        test_file_path.unlink()
        
        print("✅ Audio processing test passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_next_steps():
    """Show next steps for creating the enhanced dataset"""
    print("\n🚀 Next Steps")
    print("=" * 50)
    print("1. Run data acquisition to download the datasets:")
    print("   python src/enhanced_data_acquisition.py")
    print("\n2. Process the raw data:")
    print("   python src/enhanced_audio_processor.py")
    print("\n3. Create augmented versions:")
    print("   python src/enhanced_audio_augmentor.py")
    print("\n4. Train your models with the enhanced dataset!")

def main():
    """Main setup function"""
    print("🎵 Enhanced Dataset Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️ Could not install all dependencies. Some features may not work.")
    
    # Create directory structure
    if not create_directory_structure():
        print("⚠️ Could not create all directories. Check permissions.")
    
    # Test environment
    if not test_environment():
        print("⚠️ Environment test failed. Please check error messages.")
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Setup complete!")

if __name__ == "__main__":
    main()
