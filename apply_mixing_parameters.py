#!/usr/bin/env python3
"""
🎛️ Apply Mixing Parameters to Audio
===================================

Takes the mixing parameters predicted by the mixing model and actually applies them
to the original distorted audio to generate the final mixed result.
This completes the comparison by showing what each approach actually produces.
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import warnings
from scipy import signal

warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 22050
TESTS_DIR = os.path.join(os.getcwd(), "tests")
WAV_OUTPUTS_DIR = os.path.join(TESTS_DIR, "wav_outputs")

def apply_eq_filter(audio, sr, freq_bands, gains):
    """Apply EQ using IIR filters for bass, mid, treble"""
    
    # Define frequency bands (Hz)
    bass_freq = 250    # Bass cutoff
    treble_freq = 4000 # Treble cutoff
    
    bass_gain, mid_gain, treble_gain = gains
    
    # Convert gains from [0,1] to dB (-12 to +12 dB)
    bass_db = (bass_gain - 0.5) * 24
    mid_db = (mid_gain - 0.5) * 24  
    treble_db = (treble_gain - 0.5) * 24
    
    # Apply bass filter (low shelf)
    if abs(bass_db) > 0.1:
        sos_bass = signal.iirfilter(2, bass_freq, btype='lowpass', fs=sr, output='sos')
        bass_response = signal.sosfilt(sos_bass, audio)
        audio = audio + bass_response * (10**(bass_db/20) - 1)
    
    # Apply treble filter (high shelf)  
    if abs(treble_db) > 0.1:
        sos_treble = signal.iirfilter(2, treble_freq, btype='highpass', fs=sr, output='sos')
        treble_response = signal.sosfilt(sos_treble, audio)
        audio = audio + treble_response * (10**(treble_db/20) - 1)
    
    # Apply mid boost/cut (bandpass around 1kHz)
    if abs(mid_db) > 0.1:
        sos_mid = signal.iirfilter(2, [500, 2000], btype='bandpass', fs=sr, output='sos')
        mid_response = signal.sosfilt(sos_mid, audio)
        audio = audio + mid_response * (10**(mid_db/20) - 1)
    
    return audio

def apply_compressor(audio, threshold, ratio):
    """Apply simple compressor"""
    
    # Convert threshold and ratio from [0,1] to meaningful values
    threshold_db = -40 + threshold * 30  # -40dB to -10dB
    comp_ratio = 1 + ratio * 9  # 1:1 to 10:1
    
    # Simple peak detection and gain reduction
    envelope = np.abs(audio)
    
    # Smooth the envelope
    envelope = signal.lfilter([0.1], [1, -0.9], envelope)
    
    # Convert to dB
    envelope_db = 20 * np.log10(envelope + 1e-8)
    
    # Calculate gain reduction
    gain_reduction = np.where(
        envelope_db > threshold_db,
        (envelope_db - threshold_db) * (1 - 1/comp_ratio),
        0
    )
    
    # Apply gain reduction
    gain_linear = 10 ** (-gain_reduction / 20)
    compressed = audio * gain_linear
    
    return compressed

def apply_gate(audio, threshold):
    """Apply noise gate"""
    
    # Convert threshold from [0,1] to meaningful value
    gate_threshold = -60 + threshold * 40  # -60dB to -20dB
    
    # Calculate envelope
    envelope = np.abs(audio)
    envelope = signal.lfilter([0.01], [1, -0.99], envelope)
    
    # Convert to dB
    envelope_db = 20 * np.log10(envelope + 1e-8)
    
    # Create gate mask
    gate_open = envelope_db > gate_threshold
    
    # Smooth gate transitions
    gate_smooth = signal.lfilter([0.1], [1, -0.9], gate_open.astype(float))
    
    return audio * gate_smooth

def apply_reverb(audio, send_level):
    """Apply simple reverb using delay lines"""
    
    if send_level < 0.01:
        return audio
    
    # Simple reverb using multiple delays
    delay_samples = [int(0.03 * SAMPLE_RATE), int(0.05 * SAMPLE_RATE), int(0.08 * SAMPLE_RATE)]
    decay_factors = [0.3, 0.2, 0.15]
    
    reverb_signal = np.zeros_like(audio)
    
    for delay, decay in zip(delay_samples, decay_factors):
        if delay < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay:] = audio[:-delay] * decay
            reverb_signal += delayed
    
    # Mix dry and wet signal
    wet_gain = send_level
    dry_gain = 1.0 - send_level * 0.5  # Don't completely remove dry signal
    
    return audio * dry_gain + reverb_signal * wet_gain

def apply_delay(audio, send_level):
    """Apply simple delay effect"""
    
    if send_level < 0.01:
        return audio
    
    # Simple delay (around 200ms)
    delay_samples = int(0.2 * SAMPLE_RATE)
    
    if delay_samples < len(audio):
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples] * 0.4 * send_level
        return audio + delayed
    
    return audio

def apply_mixing_parameters(audio, mixing_params):
    """Apply all mixing parameters to audio"""
    
    # Unpack parameters
    master_volume = mixing_params[0]
    bass_gain = mixing_params[1] 
    mid_gain = mixing_params[2]
    treble_gain = mixing_params[3]
    compressor_threshold = mixing_params[4]
    compressor_ratio = mixing_params[5]
    gate_threshold = mixing_params[6]
    reverb_send = mixing_params[7]
    delay_send = mixing_params[8]
    stereo_width = mixing_params[9]  # Not used for mono
    pan = mixing_params[10]  # Not used for mono
    
    print(f"    Applying: Vol={master_volume:.2f}, EQ=[{bass_gain:.2f},{mid_gain:.2f},{treble_gain:.2f}]")
    print(f"              Comp=[{compressor_threshold:.2f},{compressor_ratio:.2f}], Gate={gate_threshold:.2f}")
    print(f"              FX=[Rev:{reverb_send:.2f}, Del:{delay_send:.2f}]")
    
    # Start with original audio
    processed = audio.copy()
    
    # 1. Apply noise gate (before other processing)
    processed = apply_gate(processed, gate_threshold)
    
    # 2. Apply EQ
    processed = apply_eq_filter(processed, SAMPLE_RATE, None, [bass_gain, mid_gain, treble_gain])
    
    # 3. Apply compressor
    processed = apply_compressor(processed, compressor_threshold, compressor_ratio)
    
    # 4. Apply reverb
    processed = apply_reverb(processed, reverb_send)
    
    # 5. Apply delay
    processed = apply_delay(processed, delay_send)
    
    # 6. Apply master volume
    processed = processed * master_volume
    
    # 7. Normalize to prevent clipping
    if np.max(np.abs(processed)) > 0.95:
        processed = processed / np.max(np.abs(processed)) * 0.95
    
    return processed

def apply_mixing_to_samples():
    """Apply mixing parameters to all mixing model samples"""
    
    mixing_dir = os.path.join(WAV_OUTPUTS_DIR, "mixing")
    
    if not os.path.exists(mixing_dir):
        print("❌ No mixing outputs found!")
        return
    
    print("🎛️ Applying Mixing Parameters to Audio")
    print("=" * 50)
    
    # Process each sample
    sample_dirs = [d for d in os.listdir(mixing_dir) if os.path.isdir(os.path.join(mixing_dir, d))]
    
    for sample_dir in sorted(sample_dirs):
        sample_path = os.path.join(mixing_dir, sample_dir)
        
        print(f"🎵 Processing {sample_dir}...")
        
        # Load original distorted audio
        distorted_file = os.path.join(sample_path, "01_original_distorted.wav")
        parameters_file = os.path.join(sample_path, "parameters.txt")
        
        if not os.path.exists(distorted_file) or not os.path.exists(parameters_file):
            print(f"  ⚠️  Missing files for {sample_dir}")
            continue
        
        # Load audio
        audio, _ = librosa.load(distorted_file, sr=SAMPLE_RATE, mono=True)
        
        # Parse mixing parameters from file
        mixing_params = []
        try:
            with open(parameters_file, 'r') as f:
                lines = f.readlines()
            
            # Extract parameter values
            for line in lines:
                if "Master Volume:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Bass Gain:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Mid Gain:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Treble Gain:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Compressor Threshold:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Compressor Ratio:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Gate Threshold:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Reverb Send:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Delay Send:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Stereo Width:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
                elif "Pan:" in line:
                    mixing_params.append(float(line.split(":")[1].strip()))
            
            if len(mixing_params) != 11:
                print(f"  ⚠️  Could not parse all parameters for {sample_dir}")
                continue
            
        except Exception as e:
            print(f"  ⚠️  Error parsing parameters: {e}")
            continue
        
        # Apply mixing parameters
        try:
            mixed_audio = apply_mixing_parameters(audio, mixing_params)
            
            # Save the mixed result
            output_file = os.path.join(sample_path, "05_model_mixed_result.wav")
            sf.write(output_file, mixed_audio, SAMPLE_RATE)
            
            print(f"  ✅ Mixed audio saved: 05_model_mixed_result.wav")
            
        except Exception as e:
            print(f"  ⚠️  Error applying mixing: {e}")

def update_readme():
    """Update the README to explain the new mixed files"""
    
    readme_path = os.path.join(WAV_OUTPUTS_DIR, "README.md")
    
    if not os.path.exists(readme_path):
        return
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update the file naming section
    updated_content = content.replace(
        "5. **05_model_restored.wav** - Audio restored by the model (restoration/hybrid only)",
        """5. **05_model_restored.wav** - Audio restored by the model (restoration/hybrid only)
5. **05_model_mixed_result.wav** - Audio with mixing parameters applied (mixing approach only)"""
    )
    
    # Update the mixing section
    updated_content = updated_content.replace(
        """### MIXING
- **Goal**: Predict optimal mixing parameters from distorted audio
- **Output**: Only mixing parameters (no audio restoration)
- **Best for**: Understanding what mixing adjustments are needed""",
        """### MIXING
- **Goal**: Predict optimal mixing parameters from distorted audio
- **Output**: Mixing parameters + Applied result audio (05_model_mixed_result.wav)
- **Best for**: Understanding and hearing what mixing adjustments improve the audio"""
    )
    
    # Update the listening instructions
    updated_content = updated_content.replace(
        """2. **Listen to file 05** - This is what the model actually produced""",
        """2. **Listen to file 05** - This is what the model actually produced:
   - **restoration/hybrid**: 05_model_restored.wav (restored audio)
   - **mixing**: 05_model_mixed_result.wav (original + applied mixing parameters)"""
    )
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"📖 README updated with mixing results information")

def main():
    """Apply mixing parameters and update documentation"""
    
    # Apply mixing parameters to create actual audio outputs
    apply_mixing_to_samples()
    
    # Update README
    update_readme()
    
    print(f"\n✅ Mixing parameters applied!")
    print(f"🎧 Now all three approaches produce actual audio outputs:")
    print(f"   MIXING: 05_model_mixed_result.wav (original + mixing parameters)")
    print(f"   RESTORATION: 05_model_restored.wav (restored audio)")
    print(f"   HYBRID: 05_model_restored.wav (restored + mixed audio)")
    print(f"\n🎵 Ready for complete audio comparison!")

if __name__ == "__main__":
    main()
