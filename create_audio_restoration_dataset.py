#!/usr/bin/env python3
"""
🎵 Audio Restoration Dataset Generator
=====================================

This script creates training data for audio restoration by:
1. Taking clean audio as ground truth
2. Applying realistic distortions (reverb, noise, compression artifacts, etc.)
3. Training models to restore distorted audio back to original quality

Distortions include:
- Background noise (white, pink, room tone)
- Reverb and echo
- Low-pass/high-pass filtering
- Compression artifacts
- Clipping and distortion
- Imbalanced mixes
- Phase issues
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import random
from tqdm import tqdm

def apply_background_noise(audio, sr, noise_level=0.1):
    """Add realistic background noise"""
    noise_types = ['white', 'pink', 'brown']
    noise_type = random.choice(noise_types)
    
    if noise_type == 'white':
        noise = np.random.normal(0, noise_level, len(audio))
    elif noise_type == 'pink':
        # Pink noise (1/f noise)
        noise = np.random.normal(0, 1, len(audio))
        noise = np.cumsum(noise) / np.sqrt(len(audio))
        noise = noise * noise_level
    else:  # brown
        # Brown noise (1/f^2 noise)
        noise = np.random.normal(0, 1, len(audio))
        noise = np.cumsum(np.cumsum(noise)) / len(audio)
        noise = noise * noise_level
    
    return audio + noise

def apply_reverb(audio, sr, room_size=0.5, damping=0.5, wet_level=0.3):
    """Apply simple reverb effect"""
    # Simple reverb using delays
    delay_samples = int(sr * 0.05 * room_size)  # 50ms max delay
    decay = 0.3 * (1 - damping)
    
    reverb_audio = np.copy(audio)
    
    # Add multiple delayed copies with decay
    for i in range(3):
        delay = delay_samples * (i + 1)
        if delay < len(audio):
            delayed = np.pad(audio, (delay, 0), mode='constant')[:len(audio)]
            reverb_audio += delayed * (decay ** (i + 1))
    
    return audio * (1 - wet_level) + reverb_audio * wet_level

def apply_lowpass_filter(audio, sr, cutoff_freq=8000):
    """Apply low-pass filter to simulate old recordings or phone quality"""
    from scipy import signal
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    # Ensure cutoff is in valid range (0 < Wn < 1)
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)

def apply_highpass_filter(audio, sr, cutoff_freq=100):
    """Apply high-pass filter to remove low-end"""
    from scipy import signal
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    # Ensure cutoff is in valid range (0 < Wn < 1)
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio)

def apply_compression_artifacts(audio, sr, compression_ratio=0.7):
    """Simulate lossy compression artifacts"""
    # Simple compression simulation by reducing dynamic range
    rms = np.sqrt(np.mean(audio**2))
    compressed = np.sign(audio) * (np.abs(audio) ** compression_ratio)
    # Normalize to original RMS
    compressed_rms = np.sqrt(np.mean(compressed**2))
    if compressed_rms > 0:
        compressed = compressed * (rms / compressed_rms)
    return compressed

def apply_clipping(audio, threshold=0.8):
    """Apply soft clipping distortion"""
    return np.clip(audio, -threshold, threshold)

def apply_eq_imbalance(audio, sr, bass_gain=1.0, mid_gain=1.0, treble_gain=1.0):
    """Apply EQ imbalance to simulate bad mixing"""
    from scipy import signal
    
    try:
        # Bass (80-250 Hz)
        if bass_gain != 1.0 and sr > 500:  # Ensure valid frequency range
            sos_bass = signal.butter(4, [max(80, sr/200), min(250, sr/4)], btype='band', fs=sr, output='sos')
            bass_band = signal.sosfilt(sos_bass, audio)
            audio = audio + bass_band * (bass_gain - 1)
        
        # Mid (250-4000 Hz)
        if mid_gain != 1.0 and sr > 8000:  # Ensure valid frequency range
            sos_mid = signal.butter(4, [max(250, sr/100), min(4000, sr/3)], btype='band', fs=sr, output='sos')
            mid_band = signal.sosfilt(sos_mid, audio)
            audio = audio + mid_band * (mid_gain - 1)
        
        # Treble (4000+ Hz)
        if treble_gain != 1.0 and sr > 8000:  # Ensure valid frequency range
            treble_cutoff = min(4000, sr/3)
            sos_treble = signal.butter(4, treble_cutoff, btype='high', fs=sr, output='sos')
            treble_band = signal.sosfilt(sos_treble, audio)
            audio = audio + treble_band * (treble_gain - 1)
    except Exception as e:
        # If filtering fails, return original audio
        pass
    
    return audio

def apply_random_distortions(audio, sr):
    """Apply random combination of distortions"""
    distorted = np.copy(audio)
    distortion_types = []
    
    # Background noise (50% chance)
    if random.random() < 0.5:
        noise_level = random.uniform(0.02, 0.15)
        distorted = apply_background_noise(distorted, sr, noise_level)
        distortion_types.append(f"noise_{noise_level:.3f}")
    
    # Reverb (30% chance)
    if random.random() < 0.3:
        room_size = random.uniform(0.2, 0.8)
        damping = random.uniform(0.2, 0.8)
        wet_level = random.uniform(0.1, 0.4)
        distorted = apply_reverb(distorted, sr, room_size, damping, wet_level)
        distortion_types.append(f"reverb_{wet_level:.3f}")
    
    # Low-pass filter (25% chance)
    if random.random() < 0.25:
        cutoff = random.uniform(3000, 12000)
        distorted = apply_lowpass_filter(distorted, sr, cutoff)
        distortion_types.append(f"lowpass_{cutoff:.0f}")
    
    # High-pass filter (20% chance)
    if random.random() < 0.2:
        cutoff = random.uniform(50, 300)
        distorted = apply_highpass_filter(distorted, sr, cutoff)
        distortion_types.append(f"highpass_{cutoff:.0f}")
    
    # Compression artifacts (40% chance)
    if random.random() < 0.4:
        ratio = random.uniform(0.5, 0.9)
        distorted = apply_compression_artifacts(distorted, sr, ratio)
        distortion_types.append(f"compression_{ratio:.3f}")
    
    # Clipping (15% chance)
    if random.random() < 0.15:
        threshold = random.uniform(0.6, 0.9)
        distorted = apply_clipping(distorted, threshold)
        distortion_types.append(f"clipping_{threshold:.3f}")
    
    # EQ imbalance (35% chance)
    if random.random() < 0.35:
        bass_gain = random.uniform(0.3, 1.7)
        mid_gain = random.uniform(0.5, 1.5)
        treble_gain = random.uniform(0.4, 1.6)
        distorted = apply_eq_imbalance(distorted, sr, bass_gain, mid_gain, treble_gain)
        distortion_types.append(f"eq_{bass_gain:.2f}_{mid_gain:.2f}_{treble_gain:.2f}")
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(distorted))
    if max_val > 0.95:
        distorted = distorted * (0.95 / max_val)
    
    return distorted, distortion_types

def create_restoration_dataset(base_dir, num_variations=3):
    """Create audio restoration dataset from clean FMA files"""
    
    data_dir = Path(base_dir) / "data"
    fma_dir = data_dir / "raw" / "music" / "fma"
    
    # Output directories
    clean_dir = data_dir / "restoration" / "clean"
    distorted_dir = data_dir / "restoration" / "distorted"
    
    clean_dir.mkdir(parents=True, exist_ok=True)
    distorted_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all MP3 files
    audio_files = list(fma_dir.rglob("*.mp3"))
    
    if len(audio_files) == 0:
        print("❌ No FMA audio files found. Run extraction first.")
        return False
    
    print(f"🎵 Found {len(audio_files)} audio files")
    print(f"📊 Creating {num_variations} distorted versions of each")
    print(f"📁 Clean audio: {clean_dir}")
    print(f"📁 Distorted audio: {distorted_dir}")
    
    metadata = {
        "total_clean_files": 0,
        "total_distorted_files": 0,
        "distortion_types": {},
        "sample_rate": 22050,
        "duration_seconds": 30
    }
    
    successful_pairs = 0
    failed_files = 0
    
    # Process each audio file
    for i, audio_file in enumerate(tqdm(audio_files, desc="Creating restoration dataset")):
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=22050, mono=True, duration=30.0)
            
            # Skip very short files
            if len(audio) < sr * 5:  # Less than 5 seconds
                continue
            
            # Normalize clean audio
            audio = librosa.util.normalize(audio)
            
            # Create filename
            base_name = Path(audio_file).stem
            clean_filename = f"{base_name}_clean.wav"
            
            # Save clean version
            clean_path = clean_dir / clean_filename
            sf.write(clean_path, audio, sr)
            metadata["total_clean_files"] += 1
            
            # Create distorted versions
            for variation in range(num_variations):
                try:
                    distorted_audio, distortion_list = apply_random_distortions(audio, sr)
                    
                    # Save distorted version
                    distorted_filename = f"{base_name}_distorted_{variation}.wav"
                    distorted_path = distorted_dir / distorted_filename
                    sf.write(distorted_path, distorted_audio, sr)
                    
                    # Track distortion types
                    for distortion in distortion_list:
                        if distortion not in metadata["distortion_types"]:
                            metadata["distortion_types"][distortion] = 0
                        metadata["distortion_types"][distortion] += 1
                    
                    metadata["total_distorted_files"] += 1
                    successful_pairs += 1
                    
                except Exception as e:
                    print(f"⚠️ Error creating distortion {variation} for {audio_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ Error processing {audio_file}: {e}")
            failed_files += 1
            continue
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"📊 Processed {i+1}/{len(audio_files)} files, {successful_pairs} successful pairs")
    
    # Save metadata
    metadata_path = data_dir / "restoration_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Audio restoration dataset created!")
    print(f"📊 {metadata['total_clean_files']} clean files")
    print(f"📊 {metadata['total_distorted_files']} distorted files")
    print(f"📊 {successful_pairs} training pairs")
    print(f"❌ {failed_files} failed files")
    print(f"📄 Metadata saved: {metadata_path}")
    
    return True

def main():
    """Main function"""
    base_dir = Path(__file__).parent
    
    print("🎵 Audio Restoration Dataset Generator")
    print("=" * 45)
    
    # Check if we need scipy
    try:
        import scipy
    except ImportError:
        print("❌ scipy not found. Installing...")
        os.system("pip install scipy")
        import scipy
    
    # Check if we need soundfile
    try:
        import soundfile as sf
    except ImportError:
        print("❌ soundfile not found. Installing...")
        os.system("pip install soundfile")
        import soundfile as sf
    
    success = create_restoration_dataset(base_dir, num_variations=3)
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Update training pipeline to use restoration data")
        print("   2. Train models to restore distorted → clean audio")
        print("   3. Measure restoration quality with MSE, STOI, PESQ")
    else:
        print("\n❌ Dataset creation failed")

if __name__ == "__main__":
    main()
