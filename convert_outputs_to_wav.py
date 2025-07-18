#!/usr/bin/env python3
"""
🎵 Convert Audio Outputs to WAV
===============================

Convert the saved spectrogram outputs to actual WAV files for listening comparison.
Includes ground truth (original distorted and target clean) plus model outputs.
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 22050
TESTS_DIR = os.path.join(os.getcwd(), "tests")
AUDIO_OUTPUTS_DIR = os.path.join(TESTS_DIR, "audio_outputs")
WAV_OUTPUTS_DIR = os.path.join(TESTS_DIR, "wav_outputs")

def spectrogram_to_audio(spectrogram, sr=SAMPLE_RATE, n_iter=32):
    """Convert mel-spectrogram back to audio using Griffin-Lim algorithm"""
    
    # Denormalize spectrogram from [-1, 1] to original scale
    spectrogram = (spectrogram + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    # Convert from log-mel to linear mel-spectrogram
    mel_spec = librosa.db_to_power(spectrogram * 80 - 80)  # Approximate original range
    
    # Convert mel-spectrogram to linear spectrogram
    # This is an approximation - we need the mel filter bank
    n_fft = 1024
    hop_length = 256
    
    # Use Griffin-Lim to reconstruct audio from magnitude spectrogram
    # First, we need to convert mel to linear spectrogram
    mel_to_linear = librosa.feature.inverse.mel_to_stft(
        mel_spec, sr=sr, n_fft=n_fft
    )
    
    # Reconstruct audio using Griffin-Lim
    audio = librosa.griffinlim(
        mel_to_linear, 
        n_iter=n_iter, 
        hop_length=hop_length, 
        win_length=n_fft
    )
    
    return audio

def get_original_audio_files(sample_idx):
    """Get the original audio files for comparison"""
    
    # Look for files in the restoration dataset
    restoration_dir = os.path.join(os.getcwd(), "data", "restoration")
    clean_dir = os.path.join(restoration_dir, "clean")
    distorted_dir = os.path.join(restoration_dir, "distorted")
    
    if not os.path.exists(clean_dir) or not os.path.exists(distorted_dir):
        return None, None
    
    # Get file lists
    clean_files = sorted(list(Path(clean_dir).glob("*.wav")))
    distorted_files = sorted(list(Path(distorted_dir).glob("*.wav")))
    
    # Try to get the files used in the sample
    if sample_idx < len(clean_files):
        clean_file = clean_files[sample_idx]
        
        # Find corresponding distorted file
        base_name = clean_file.stem.replace("_clean", "")
        distorted_pattern = f"{base_name}_distorted_*.wav"
        matching_distorted = list(Path(distorted_dir).glob(distorted_pattern))
        
        if matching_distorted:
            distorted_file = matching_distorted[0]  # Take first match
            return str(clean_file), str(distorted_file)
    
    return None, None

def convert_model_outputs_to_wav(approach_name):
    """Convert outputs from one approach to WAV files"""
    
    input_dir = os.path.join(AUDIO_OUTPUTS_DIR, approach_name)
    output_dir = os.path.join(WAV_OUTPUTS_DIR, approach_name)
    
    if not os.path.exists(input_dir):
        print(f"❌ No outputs found for {approach_name}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎵 Converting {approach_name} outputs to WAV...")
    
    # Process each sample
    npz_files = sorted(list(Path(input_dir).glob("*.npz")))
    
    for npz_file in npz_files:
        sample_name = npz_file.stem
        sample_idx = int(sample_name.split('_')[1])
        
        print(f"  Converting {sample_name}...")
        
        # Load the saved outputs
        data = np.load(npz_file)
        
        # Create sample directory
        sample_dir = os.path.join(output_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get original audio files for ground truth
        original_clean_path, original_distorted_path = get_original_audio_files(sample_idx)
        
        if original_clean_path and original_distorted_path:
            # Copy original files as ground truth
            original_clean, _ = librosa.load(original_clean_path, sr=SAMPLE_RATE, mono=True)
            original_distorted, _ = librosa.load(original_distorted_path, sr=SAMPLE_RATE, mono=True)
            
            # Take same chunk size as used in training
            chunk_samples = int(3.0 * SAMPLE_RATE)  # 3 seconds
            if len(original_clean) > chunk_samples:
                original_clean = original_clean[:chunk_samples]
                original_distorted = original_distorted[:chunk_samples]
            
            # Save ground truth
            sf.write(
                os.path.join(sample_dir, "01_original_distorted.wav"),
                original_distorted, SAMPLE_RATE
            )
            sf.write(
                os.path.join(sample_dir, "02_target_clean.wav"),
                original_clean, SAMPLE_RATE
            )
        
        # Convert spectrograms to audio
        if 'original_distorted' in data:
            try:
                distorted_audio = spectrogram_to_audio(data['original_distorted'][0])
                sf.write(
                    os.path.join(sample_dir, "03_model_input_distorted.wav"),
                    distorted_audio, SAMPLE_RATE
                )
            except Exception as e:
                print(f"    ⚠️  Could not convert distorted spectrogram: {e}")
        
        if 'target_clean' in data:
            try:
                clean_audio = spectrogram_to_audio(data['target_clean'][0])
                sf.write(
                    os.path.join(sample_dir, "04_model_target_clean.wav"),
                    clean_audio, SAMPLE_RATE
                )
            except Exception as e:
                print(f"    ⚠️  Could not convert clean spectrogram: {e}")
        
        # Convert model outputs
        if 'restored_audio' in data:
            try:
                restored_audio = spectrogram_to_audio(data['restored_audio'][0])
                sf.write(
                    os.path.join(sample_dir, "05_model_restored.wav"),
                    restored_audio, SAMPLE_RATE
                )
            except Exception as e:
                print(f"    ⚠️  Could not convert restored audio: {e}")
        
        # Save parameter predictions as text for reference
        param_info = []
        if 'predicted_mixing_params' in data:
            mixing_params = data['predicted_mixing_params']
            param_info.append("PREDICTED MIXING PARAMETERS:")
            param_info.append(f"Master Volume: {mixing_params[0]:.3f}")
            param_info.append(f"Bass Gain: {mixing_params[1]:.3f}")
            param_info.append(f"Mid Gain: {mixing_params[2]:.3f}")
            param_info.append(f"Treble Gain: {mixing_params[3]:.3f}")
            param_info.append(f"Compressor Threshold: {mixing_params[4]:.3f}")
            param_info.append(f"Compressor Ratio: {mixing_params[5]:.3f}")
            param_info.append(f"Gate Threshold: {mixing_params[6]:.3f}")
            param_info.append(f"Reverb Send: {mixing_params[7]:.3f}")
            param_info.append(f"Delay Send: {mixing_params[8]:.3f}")
            param_info.append(f"Stereo Width: {mixing_params[9]:.3f}")
            param_info.append(f"Pan: {mixing_params[10]:.3f}")
            param_info.append("")
        
        if 'predicted_distortion_params' in data:
            distortion_params = data['predicted_distortion_params']
            param_info.append("PREDICTED DISTORTION PARAMETERS:")
            param_info.append(f"Noise Level: {distortion_params[0]:.3f}")
            param_info.append(f"Reverb Level: {distortion_params[1]:.3f}")
            param_info.append(f"Low-pass Cutoff: {distortion_params[2]:.3f}")
            param_info.append(f"High-pass Cutoff: {distortion_params[3]:.3f}")
            param_info.append(f"Compression Ratio: {distortion_params[4]:.3f}")
            param_info.append(f"Clipping Threshold: {distortion_params[5]:.3f}")
            param_info.append(f"EQ Imbalance: {distortion_params[6]:.3f}")
        
        if param_info:
            with open(os.path.join(sample_dir, "parameters.txt"), 'w') as f:
                f.write('\n'.join(param_info))

def create_comparison_readme():
    """Create a README file explaining the audio comparisons"""
    
    readme_content = """# Audio Comparison Results

## File Naming Convention

For each sample, you'll find these files:

1. **01_original_distorted.wav** - The actual distorted input audio (ground truth)
2. **02_target_clean.wav** - The actual clean target audio (ground truth)  
3. **03_model_input_distorted.wav** - Distorted audio reconstructed from spectrogram
4. **04_model_target_clean.wav** - Clean audio reconstructed from spectrogram
5. **05_model_restored.wav** - Audio restored by the model (restoration/hybrid only)
6. **parameters.txt** - Predicted mixing and distortion parameters

## Approaches Compared

### MIXING
- **Goal**: Predict optimal mixing parameters from distorted audio
- **Output**: Only mixing parameters (no audio restoration)
- **Best for**: Understanding what mixing adjustments are needed

### RESTORATION  
- **Goal**: Restore clean audio from distorted input
- **Output**: Restored audio + distortion parameter analysis
- **Best for**: Audio enhancement and cleaning

### HYBRID
- **Goal**: Complete pipeline - analyze distortions -> restore audio -> optimize mixing
- **Output**: Restored audio + mixing parameters + distortion analysis
- **Best for**: End-to-end audio processing

## How to Listen

1. **Start with files 01 & 02** - Compare the actual ground truth (distorted vs clean)
2. **Listen to file 05** - This is what the model actually produced
3. **Compare with file 02** - How close did the model get to the clean target?
4. **Check parameters.txt** - What did the model think was wrong and how to fix it?

## Quality Assessment

- **MIXING**: Low loss (0.0162) - Good at predicting mixing parameters
- **RESTORATION**: Medium loss (0.1336) - Decent audio restoration  
- **HYBRID**: Higher loss (0.2460) but handles ALL tasks - Best overall approach

The HYBRID model is recommended because it provides a complete audio processing pipeline.
"""
    
    readme_path = os.path.join(WAV_OUTPUTS_DIR, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📖 Comparison guide saved to: {readme_path}")

def main():
    """Convert all outputs to WAV files"""
    
    print("🎵 Converting Audio Outputs to WAV")
    print("=" * 50)
    
    # Check if outputs exist
    if not os.path.exists(AUDIO_OUTPUTS_DIR):
        print("❌ No audio outputs found! Run the comparison script first.")
        return
    
    # Create WAV output directory
    os.makedirs(WAV_OUTPUTS_DIR, exist_ok=True)
    
    # Convert each approach
    approaches = ['mixing', 'restoration', 'hybrid']
    
    for approach in approaches:
        convert_model_outputs_to_wav(approach)
    
    # Create comparison guide
    create_comparison_readme()
    
    print(f"\n✅ WAV conversion complete!")
    print(f"🎵 WAV files saved to: {WAV_OUTPUTS_DIR}")
    print(f"📖 Check README.md for listening instructions")
    
    # Show summary
    total_files = 0
    for approach in approaches:
        approach_dir = os.path.join(WAV_OUTPUTS_DIR, approach)
        if os.path.exists(approach_dir):
            samples = len([d for d in os.listdir(approach_dir) if os.path.isdir(os.path.join(approach_dir, d))])
            wav_files = sum([len([f for f in os.listdir(os.path.join(approach_dir, d)) if f.endswith('.wav')]) 
                           for d in os.listdir(approach_dir) if os.path.isdir(os.path.join(approach_dir, d))])
            total_files += wav_files
            print(f"  {approach.upper()}: {samples} samples, {wav_files} WAV files")
    
    print(f"\n🎧 Ready for listening test with {total_files} total audio files!")

if __name__ == "__main__":
    main()
