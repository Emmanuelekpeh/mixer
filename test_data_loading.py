#!/usr/bin/env python3
"""
Test data loading for dual-path hybrid training
"""

import os
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
import random

def test_data_loading():
    print("🔥 Testing Data Loading")
    print("=" * 50)
    
    # Check dataset
    restoration_dir = os.path.join("data", "restoration")
    clean_dir = os.path.join(restoration_dir, "clean")
    distorted_dir = os.path.join(restoration_dir, "distorted")
    
    # Find file pairs
    clean_files = list(Path(clean_dir).glob("*.wav"))
    file_pairs = []
    
    print(f"📊 Scanning {len(clean_files)} clean files...")
    
    for clean_file in clean_files[:10]:  # Test first 10 files
        base_name = clean_file.stem.replace("_clean", "")
        distorted_pattern = f"{base_name}_distorted_*.wav"
        distorted_files = list(Path(distorted_dir).glob(distorted_pattern))
        
        for distorted_file in distorted_files:
            file_pairs.append((clean_file, distorted_file))
    
    print(f"📊 Found {len(file_pairs)} clean/distorted pairs")
    
    if len(file_pairs) == 0:
        print("❌ No file pairs found!")
        return False
    
    # Test loading a single pair
    clean_path, distorted_path = file_pairs[0]
    print(f"🔍 Testing: {clean_path.name} & {distorted_path.name}")
    
    try:
        # Load audio
        clean_audio, sr1 = librosa.load(clean_path, sr=22050, mono=True)
        distorted_audio, sr2 = librosa.load(distorted_path, sr=22050, mono=True)
        
        print(f"✅ Loaded clean: {clean_audio.shape}, sr={sr1}")
        print(f"✅ Loaded distorted: {distorted_audio.shape}, sr={sr2}")
        
        # Test chunk processing
        chunk_samples = int(3.0 * 22050)  # 3 second chunks
        
        if len(clean_audio) > chunk_samples:
            start = random.randint(0, len(clean_audio) - chunk_samples)
            clean_chunk = clean_audio[start:start + chunk_samples]
            distorted_chunk = distorted_audio[start:start + chunk_samples]
        else:
            pad_length = chunk_samples - len(clean_audio)
            clean_chunk = np.pad(clean_audio, (0, pad_length))
            distorted_chunk = np.pad(distorted_audio, (0, pad_length))
        
        print(f"✅ Chunk processing: {clean_chunk.shape}")
        
        # Test spectrogram conversion
        clean_spec = librosa.feature.melspectrogram(
            y=clean_chunk, sr=22050, n_mels=64, hop_length=512
        )
        distorted_spec = librosa.feature.melspectrogram(
            y=distorted_chunk, sr=22050, n_mels=64, hop_length=512
        )
        
        print(f"✅ Spectrograms: {clean_spec.shape}")
        
        print("🎯 Data loading test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

if __name__ == "__main__":
    test_data_loading()
