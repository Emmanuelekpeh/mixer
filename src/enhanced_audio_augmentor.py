#!/usr/bin/env python3
"""
🎵 Enhanced Audio Augmentation
============================

This script implements advanced audio augmentation techniques for AI mixing:
- Noise injection from real-world environments
- Room simulation using impulse responses
- Dynamic range compression with different settings
- EQ variations for frequency response diversity
- Pitch and tempo variations

These augmentations create a more diverse and robust dataset for
training AI mixing models that can handle real-world audio conditions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import scipy.signal
import warnings
warnings.filterwarnings('ignore')

class EnhancedAudioAugmentor:
    """Production-grade audio augmentation for mixing parameter prediction"""
    
    def __init__(self, base_dir=None):
        # Set up base directories
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            self.base_dir = Path(base_dir)
        
        # Main directories
        self.processed_dir = self.base_dir / "processed"
        self.clean_dir = self.processed_dir / "clean"
        self.augmented_dir = self.processed_dir / "augmented"
        
        # Specific directories
        self.clean_music_dir = self.clean_dir / "music"
        self.clean_vocals_dir = self.clean_dir / "vocals"
        self.clean_acoustics_dir = self.clean_dir / "acoustics"
        
        self.aug_music_dir = self.augmented_dir / "music"
        self.aug_vocals_dir = self.augmented_dir / "vocals"
        
        # Metadata directory
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create output directories
        for directory in [self.aug_music_dir, self.aug_vocals_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Standard processing parameters
        self.sr = 44100  # Standard sample rate
        
        # Target parameters to adjust (based on the mixing model parameters)
        self.param_names = [
            "Input Gain", "Compression Ratio", "High-Freq EQ", "Mid-Freq EQ", 
            "Low-Freq EQ", "Presence/Air", "Reverb Send", "Delay Send", 
            "Stereo Width", "Output Level"
        ]
        
        # Initialize augmentation metadata
        self.augmentation_metadata = {
            "augmented_tracks": {
                "music": [],
                "vocals": []
            },
            "augmentation_stats": {
                "total_original_tracks": 0,
                "total_augmented_tracks": 0,
                "augmentation_factor": 0,
                "methods_used": {}
            }
        }
        
        # Load room impulse responses if available
        self.ir_files = list(self.clean_acoustics_dir.glob("**/*.wav"))
        if not self.ir_files:
            print("⚠️ No room impulse responses found. Room simulation will be limited.")
        
        # Generate noise profiles
        self.noise_profiles = self._generate_noise_profiles()
    
    def _generate_noise_profiles(self):
        """Generate various noise profiles for augmentation."""
        profiles = {}
        
        # White noise
        profiles["white"] = lambda duration_samples: np.random.normal(0, 0.01, duration_samples)
        
        # Pink noise (1/f spectrum)
        def pink_noise(duration_samples):
            white = np.random.normal(0, 1, duration_samples)
            # Simple approximation of pink noise spectrum
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            return scipy.signal.lfilter(b, a, white) * 0.01
        
        profiles["pink"] = pink_noise
        
        # Brown noise (1/f^2 spectrum)
        def brown_noise(duration_samples):
            white = np.random.normal(0, 1, duration_samples)
            # Simple approximation of brown noise spectrum
            b = [0.00198, 0.00198]
            a = [1, -0.996]
            return scipy.signal.lfilter(b, a, white) * 0.15
        
        profiles["brown"] = brown_noise
        
        # Fan/HVAC noise (filtered noise)
        def hvac_noise(duration_samples):
            white = np.random.normal(0, 1, duration_samples)
            # Bandpass filter to simulate HVAC sound
            sos = scipy.signal.butter(3, [50, 500], 'bandpass', fs=self.sr, output='sos')
            filtered = scipy.signal.sosfilt(sos, white)
            # Add low frequency rumble
            sos_low = scipy.signal.butter(2, 100, 'lowpass', fs=self.sr, output='sos')
            rumble = scipy.signal.sosfilt(sos_low, np.random.normal(0, 0.5, duration_samples))
            return (filtered * 0.007 + rumble * 0.003)
        
        profiles["hvac"] = hvac_noise
        
        # Crowd/talking noise approximation
        def crowd_noise(duration_samples):
            white = np.random.normal(0, 1, duration_samples)
            # Bandpass filter to simulate human speech frequencies
            sos = scipy.signal.butter(3, [200, 3000], 'bandpass', fs=self.sr, output='sos')
            filtered = scipy.signal.sosfilt(sos, white)
            # Modulate amplitude to simulate speech patterns
            t = np.linspace(0, duration_samples/self.sr, duration_samples)
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t) * np.sin(2 * np.pi * 0.5 * t)
            return filtered * mod * 0.01
        
        profiles["crowd"] = crowd_noise
        
        return profiles
    
    def add_noise(self, audio, noise_level, noise_type="random"):
        """
        Add environmental noise to audio.
        
        Args:
            audio: Audio array (stereo)
            noise_level: Noise amplitude (0.0-1.0)
            noise_type: Type of noise to add
            
        Returns:
            Augmented audio
        """
        # Make a copy to avoid modifying the original
        audio = np.copy(audio)
        
        # Select noise type
        if noise_type == "random":
            noise_type = random.choice(list(self.noise_profiles.keys()))
        
        if noise_type not in self.noise_profiles:
            noise_type = "white"  # Default to white noise
        
        # Generate noise for each channel
        for i in range(audio.shape[0]):
            # Generate noise of the same length as the audio
            noise = self.noise_profiles[noise_type](audio.shape[1])
            
            # Scale noise by the desired level
            noise = noise * noise_level
            
            # Add noise to the audio
            audio[i] = audio[i] + noise
        
        # Normalize if needed
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        return audio
    
    def apply_room_simulation(self, audio, ir_strength=0.5, ir_file=None):
        """
        Simulate room acoustics using impulse responses.
        
        Args:
            audio: Audio array (stereo)
            ir_strength: Strength of the room effect (0.0-1.0)
            ir_file: Specific impulse response file to use (None = random)
            
        Returns:
            Augmented audio
        """
        # Make a copy to avoid modifying the original
        audio = np.copy(audio)
        
        # If no IR files available, return original
        if not self.ir_files and ir_file is None:
            return audio
        
        # Select impulse response file
        if ir_file is None:
            ir_file = random.choice(self.ir_files)
        
        try:
            # Load impulse response
            ir_audio, ir_sr = librosa.load(ir_file, sr=self.sr, mono=False)
            
            # Ensure stereo IR
            if ir_audio.ndim == 1:
                ir_audio = np.stack([ir_audio, ir_audio])
            
            # Apply convolution for each channel
            wet_audio = np.zeros_like(audio)
            for i in range(audio.shape[0]):
                # Convolve audio with impulse response
                wet_channel = scipy.signal.fftconvolve(audio[i], ir_audio[i])
                
                # Trim to original length
                wet_channel = wet_channel[:audio.shape[1]]
                
                # Normalize wet signal
                if np.max(np.abs(wet_channel)) > 0:
                    wet_channel = wet_channel / np.max(np.abs(wet_channel)) * np.max(np.abs(audio[i]))
                
                wet_audio[i] = wet_channel
            
            # Mix dry and wet signals based on strength
            augmented = (1 - ir_strength) * audio + ir_strength * wet_audio
            
            # Normalize if needed
            max_val = np.max(np.abs(augmented))
            if max_val > 0.95:
                augmented = augmented * (0.95 / max_val)
            
            return augmented
            
        except Exception as e:
            print(f"⚠️ Error applying room simulation: {e}")
            return audio
    
    def apply_dynamic_range_compression(self, audio, ratio=0.5, threshold=-20.0):
        """
        Apply dynamic range compression with varying parameters.
        
        Args:
            audio: Audio array (stereo)
            ratio: Compression ratio (0.0-1.0, higher = more compression)
            threshold: Threshold in dB
            
        Returns:
            Augmented audio
        """
        # Make a copy to avoid modifying the original
        audio = np.copy(audio)
        
        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold / 20.0)
        
        # Apply compression to each channel
        for i in range(audio.shape[0]):
            # Detect samples above threshold
            mask = np.abs(audio[i]) > threshold_linear
            
            # Apply compression
            compressed = np.zeros_like(audio[i])
            compressed[mask] = np.sign(audio[i][mask]) * (
                threshold_linear + (np.abs(audio[i][mask]) - threshold_linear) * (1.0 - ratio)
            )
            compressed[~mask] = audio[i][~mask]
            
            # Apply makeup gain
            if np.max(np.abs(compressed)) > 0:
                makeup_gain = np.max(np.abs(audio[i])) / np.max(np.abs(compressed))
                compressed = compressed * makeup_gain * 0.95
            
            audio[i] = compressed
        
        return audio
    
    def apply_eq_variation(self, audio, bands=None):
        """
        Apply random EQ variations for frequency response diversity.
        
        Args:
            audio: Audio array (stereo)
            bands: Dict with band settings (None = random)
            
        Returns:
            Augmented audio
        """
        # Make a copy to avoid modifying the original
        audio = np.copy(audio)
        
        # If no bands specified, generate random settings
        if bands is None:
            bands = {
                "low": random.uniform(-6, 6),    # Low shelf (100Hz)
                "mid": random.uniform(-4, 4),    # Mid peak (1kHz)
                "high": random.uniform(-6, 6)    # High shelf (5kHz)
            }
        
        # Apply EQ to each channel
        for i in range(audio.shape[0]):
            # Convert to frequency domain
            n_fft = 2048
            stft = librosa.stft(audio[i], n_fft=n_fft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            
            # Create EQ curve
            eq_curve = np.ones_like(freqs)
            
            # Apply low shelf
            low_shelf = 1.0 + np.tanh((100 - freqs) / 50) * bands["low"] / 20.0
            
            # Apply mid peak
            mid_peak = 1.0 + np.exp(-((freqs - 1000) ** 2) / (2 * 500 ** 2)) * bands["mid"] / 20.0
            
            # Apply high shelf
            high_shelf = 1.0 + np.tanh((freqs - 5000) / 1000) * bands["high"] / 20.0
            
            # Combine EQ curves
            eq_curve = eq_curve * low_shelf * mid_peak * high_shelf
            
            # Apply EQ
            stft_eq = stft * eq_curve[:, np.newaxis]
            
            # Convert back to time domain
            audio[i] = librosa.istft(stft_eq, length=len(audio[i]))
        
        # Normalize if needed
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        return audio
    
    def apply_pitch_shift(self, audio, n_steps):
        """
        Apply pitch shifting.
        
        Args:
            audio: Audio array (stereo)
            n_steps: Number of semitones to shift
            
        Returns:
            Augmented audio
        """
        # Make a copy to avoid modifying the original
        audio = np.copy(audio)
        
        # Apply pitch shift to each channel
        for i in range(audio.shape[0]):
            audio[i] = librosa.effects.pitch_shift(audio[i], sr=self.sr, n_steps=n_steps)
        
        return audio
    
    def apply_time_stretch(self, audio, rate):
        """
        Apply time stretching.
        
        Args:
            audio: Audio array (stereo)
            rate: Stretch factor (1.0 = original, >1 = slower, <1 = faster)
            
        Returns:
            Augmented audio
        """
        # Make a copy to avoid modifying the original
        audio = np.copy(audio)
        
        # Apply time stretch to each channel
        for i in range(audio.shape[0]):
            audio[i] = librosa.effects.time_stretch(audio[i], rate=rate)
        
        # Ensure consistent length (trim or pad)
        target_length = int(len(audio[0]) / rate)
        for i in range(audio.shape[0]):
            if len(audio[i]) > target_length:
                audio[i] = audio[i][:target_length]
            elif len(audio[i]) < target_length:
                padding = np.zeros(target_length - len(audio[i]))
                audio[i] = np.concatenate([audio[i], padding])
        
        return audio
    
    def adjust_targets_for_noise(self, targets, noise_level, noise_type):
        """
        Adjust mixing targets based on added noise and its type.
        
        Args:
            targets: Original mixing targets array
            noise_level: Noise amplitude (0.0-1.0)
            noise_type: Type of noise added
            
        Returns:
            Adjusted targets
        """
        adjusted = np.copy(targets)
        
        # Scale factor based on noise level
        scale = noise_level * 10.0  # 0.0-1.0 -> 0.0-10.0
        
        # Different adjustments based on noise type
        if noise_type in ["white", "pink"]:
            # For broadband noise, increase high-freq EQ cut, more compression
            adjusted[1] = min(1.0, adjusted[1] + scale * 0.2)  # More compression
            adjusted[2] = max(0.0, adjusted[2] - scale * 0.1)  # Less high freq
            adjusted[5] = max(0.0, adjusted[5] - scale * 0.15)  # Less presence
            adjusted[6] = max(0.0, adjusted[6] - scale * 0.2)  # Less reverb
            
        elif noise_type == "brown":
            # For low-frequency noise, reduce low end, more mid presence
            adjusted[1] = min(1.0, adjusted[1] + scale * 0.15)  # More compression
            adjusted[3] = min(1.0, adjusted[3] + scale * 0.1)  # More mid freq
            adjusted[4] = max(0.0, adjusted[4] - scale * 0.15)  # Less low freq
            
        elif noise_type in ["hvac", "crowd"]:
            # For specific noise types, targeted adjustments
            adjusted[1] = min(1.0, adjusted[1] + scale * 0.25)  # More compression
            adjusted[5] = min(1.0, adjusted[5] + scale * 0.1)  # More presence
            adjusted[8] = min(1.0, adjusted[8] + scale * 0.05)  # More stereo width
        
        return adjusted
    
    def adjust_targets_for_room(self, targets, ir_strength):
        """
        Adjust mixing targets based on room acoustics.
        
        Args:
            targets: Original mixing targets array
            ir_strength: Strength of the room effect (0.0-1.0)
            
        Returns:
            Adjusted targets
        """
        adjusted = np.copy(targets)
        
        # Scale factor based on IR strength
        scale = ir_strength * 10.0  # 0.0-1.0 -> 0.0-10.0
        
        # Room simulation adds reverb, so reduce reverb send
        adjusted[6] = max(0.0, adjusted[6] - scale * 0.3)  # Less reverb
        
        # Room may add darkness, increase presence
        adjusted[5] = min(1.0, adjusted[5] + scale * 0.1)  # More presence
        
        # Room adds natural compression
        adjusted[1] = max(0.0, adjusted[1] - scale * 0.1)  # Less compression
        
        return adjusted
    
    def adjust_targets_for_compression(self, targets, ratio):
        """
        Adjust mixing targets based on dynamic range compression.
        
        Args:
            targets: Original mixing targets array
            ratio: Compression ratio applied (0.0-1.0)
            
        Returns:
            Adjusted targets
        """
        adjusted = np.copy(targets)
        
        # Scale factor based on compression ratio
        scale = ratio * 10.0  # 0.0-1.0 -> 0.0-10.0
        
        # We already applied compression, so reduce compression in mixing
        adjusted[1] = max(0.0, adjusted[1] - scale * 0.4)  # Less compression
        
        # Compression often requires less output gain
        adjusted[9] = max(0.0, adjusted[9] - scale * 0.05)  # Less output gain
        
        return adjusted
    
    def adjust_targets_for_eq(self, targets, bands):
        """
        Adjust mixing targets based on EQ changes.
        
        Args:
            targets: Original mixing targets array
            bands: Dict with EQ band settings
            
        Returns:
            Adjusted targets
        """
        adjusted = np.copy(targets)
        
        # Adjust based on low frequency changes
        if bands["low"] > 0:
            adjusted[4] = max(0.0, adjusted[4] - bands["low"] * 0.02)  # Less low boost
        else:
            adjusted[4] = min(1.0, adjusted[4] - bands["low"] * 0.02)  # More low boost
        
        # Adjust based on mid frequency changes
        if bands["mid"] > 0:
            adjusted[3] = max(0.0, adjusted[3] - bands["mid"] * 0.02)  # Less mid boost
        else:
            adjusted[3] = min(1.0, adjusted[3] - bands["mid"] * 0.02)  # More mid boost
        
        # Adjust based on high frequency changes
        if bands["high"] > 0:
            adjusted[2] = max(0.0, adjusted[2] - bands["high"] * 0.02)  # Less high boost
            adjusted[5] = max(0.0, adjusted[5] - bands["high"] * 0.015)  # Less presence
        else:
            adjusted[2] = min(1.0, adjusted[2] - bands["high"] * 0.02)  # More high boost
            adjusted[5] = min(1.0, adjusted[5] - bands["high"] * 0.015)  # More presence
        
        return adjusted
    
    def adjust_targets_for_pitch(self, targets, n_steps):
        """
        Adjust mixing targets based on pitch shifting.
        
        Args:
            targets: Original mixing targets array
            n_steps: Number of semitones shifted
            
        Returns:
            Adjusted targets
        """
        adjusted = np.copy(targets)
        
        # Scale factor based on pitch shift amount
        scale = abs(n_steps) * 0.05
        
        if n_steps > 0:
            # Pitched up, less low end needed
            adjusted[4] = max(0.0, adjusted[4] - scale)  # Less low boost
            adjusted[5] = min(1.0, adjusted[5] + scale)  # More presence
        else:
            # Pitched down, more low end management needed
            adjusted[4] = min(1.0, adjusted[4] + scale)  # More low boost
            adjusted[5] = max(0.0, adjusted[5] - scale)  # Less presence
        
        return adjusted
    
    def adjust_targets_for_time_stretch(self, targets, rate):
        """
        Adjust mixing targets based on time stretching.
        
        Args:
            targets: Original mixing targets array
            rate: Stretch factor
            
        Returns:
            Adjusted targets
        """
        adjusted = np.copy(targets)
        
        # Scale factor based on stretch amount
        scale = abs(1.0 - rate) * 2.0
        
        if rate > 1.0:
            # Slowed down
            adjusted[1] = max(0.0, adjusted[1] - scale * 0.1)  # Less compression
            adjusted[6] = min(1.0, adjusted[6] + scale * 0.1)  # More reverb
        else:
            # Sped up
            adjusted[1] = min(1.0, adjusted[1] + scale * 0.1)  # More compression
            adjusted[6] = max(0.0, adjusted[6] - scale * 0.1)  # Less reverb
        
        return adjusted
    
    def augment_audio_file(self, file_path, original_targets=None, augmentation_types=None):
        """
        Create multiple augmented versions with different techniques.
        
        Args:
            file_path: Path to the audio file
            original_targets: Original mixing target parameters
            augmentation_types: List of augmentation types to apply
            
        Returns:
            List of tuples (name, audio, targets)
        """
        file_path = Path(file_path)
        file_stem = file_path.stem
        
        # Default augmentation types
        if augmentation_types is None:
            augmentation_types = [
                "noise", "room", "compression", "eq", "pitch", "time"
            ]
        
        # Default targets if not provided
        if original_targets is None:
            # Use placeholder default targets (balanced settings)
            original_targets = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sr, mono=False)
            
            # Convert to stereo if mono
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            # List to store augmentations
            augmentations = []
            
            # 1. Noise injection
            if "noise" in augmentation_types:
                for noise_type in ["white", "pink", "brown", "hvac", "crowd"]:
                    for noise_level in [0.005, 0.01, 0.02]:
                        aug_audio = self.add_noise(audio, noise_level, noise_type)
                        aug_targets = self.adjust_targets_for_noise(original_targets, noise_level, noise_type)
                        aug_name = f"{file_stem}_noise_{noise_type}_{int(noise_level*1000)}"
                        augmentations.append((aug_name, aug_audio, aug_targets))
            
            # 2. Room simulation
            if "room" in augmentation_types and self.ir_files:
                for ir_strength in [0.3, 0.5, 0.7]:
                    ir_file = random.choice(self.ir_files)
                    ir_name = Path(ir_file).stem
                    aug_audio = self.apply_room_simulation(audio, ir_strength, ir_file)
                    aug_targets = self.adjust_targets_for_room(original_targets, ir_strength)
                    aug_name = f"{file_stem}_room_{ir_name}_{int(ir_strength*100)}"
                    augmentations.append((aug_name, aug_audio, aug_targets))
            
            # 3. Compression
            if "compression" in augmentation_types:
                for ratio in [0.3, 0.5, 0.7]:
                    for threshold in [-30, -20, -15]:
                        aug_audio = self.apply_dynamic_range_compression(audio, ratio, threshold)
                        aug_targets = self.adjust_targets_for_compression(original_targets, ratio)
                        aug_name = f"{file_stem}_comp_{int(ratio*100)}_{abs(threshold)}"
                        augmentations.append((aug_name, aug_audio, aug_targets))
            
            # 4. EQ variations
            if "eq" in augmentation_types:
                eq_variations = [
                    {"low": +4, "mid": 0, "high": 0},     # Bass boost
                    {"low": -4, "mid": 0, "high": 0},     # Bass cut
                    {"low": 0, "mid": +3, "high": 0},     # Mid boost
                    {"low": 0, "mid": -3, "high": 0},     # Mid cut
                    {"low": 0, "mid": 0, "high": +4},     # Treble boost
                    {"low": 0, "mid": 0, "high": -4},     # Treble cut
                    {"low": +3, "mid": -2, "high": +3},   # Smiley curve
                    {"low": -2, "mid": +3, "high": -2}    # Telephone effect
                ]
                
                for i, eq_bands in enumerate(eq_variations):
                    aug_audio = self.apply_eq_variation(audio, eq_bands)
                    aug_targets = self.adjust_targets_for_eq(original_targets, eq_bands)
                    aug_name = f"{file_stem}_eq_{i+1}"
                    augmentations.append((aug_name, aug_audio, aug_targets))
            
            # 5. Pitch shifting
            if "pitch" in augmentation_types:
                for n_steps in [-2, -1, +1, +2]:
                    aug_audio = self.apply_pitch_shift(audio, n_steps)
                    aug_targets = self.adjust_targets_for_pitch(original_targets, n_steps)
                    aug_name = f"{file_stem}_pitch_{n_steps:+d}"
                    augmentations.append((aug_name, aug_audio, aug_targets))
            
            # 6. Time stretching
            if "time" in augmentation_types:
                for rate in [0.9, 1.1]:
                    aug_audio = self.apply_time_stretch(audio, rate)
                    aug_targets = self.adjust_targets_for_time_stretch(original_targets, rate)
                    aug_name = f"{file_stem}_tempo_{rate:.1f}"
                    augmentations.append((aug_name, aug_audio, aug_targets))
            
            return augmentations
            
        except Exception as e:
            print(f"❌ Error augmenting {file_path.name}: {e}")
            return []
    
    def process_file(self, file_info, category, original_targets_dict=None):
        """
        Process a single file with augmentation.
        
        Args:
            file_info: Dict with file information
            category: Category (music, vocals)
            original_targets_dict: Dict mapping file IDs to targets
            
        Returns:
            Dict with processing results
        """
        file_path = Path(file_info.get('output', file_info.get('file')))
        
        if not file_path.exists():
            return {
                "file": str(file_path),
                "status": "skipped",
                "reason": "File not found"
            }
        
        # Get original targets if available
        original_targets = None
        if original_targets_dict is not None:
            file_id = file_path.stem
            if file_id in original_targets_dict:
                original_targets = np.array(original_targets_dict[file_id])
        
        # Determine output directory
        if category == "music":
            output_dir = self.aug_music_dir
        elif category == "vocals":
            output_dir = self.aug_vocals_dir
        else:
            return {
                "file": str(file_path),
                "status": "skipped",
                "reason": f"Invalid category: {category}"
            }
        
        # Create augmentations
        augmentations = self.augment_audio_file(file_path, original_targets)
        
        # Save augmented files and update targets
        saved_files = []
        
        for aug_name, aug_audio, aug_targets in augmentations:
            # Create output path
            output_path = output_dir / f"{aug_name}.wav"
            
            # Save audio
            sf.write(output_path, aug_audio.T, self.sr)
            
            # Store information
            saved_files.append({
                "original_file": str(file_path),
                "augmented_file": str(output_path),
                "augmentation_type": aug_name.split('_')[1],  # Extract type from name
                "targets": aug_targets.tolist(),
                "category": category
            })
        
        return {
            "file": str(file_path),
            "status": "success",
            "augmentations": len(saved_files),
            "saved_files": saved_files
        }
    
    def load_targets(self):
        """
        Load existing mixing target parameters.
        
        Returns:
            Dict mapping file IDs to target parameters
        """
        targets_path = self.metadata_dir / "mixing_parameters.json"
        
        if targets_path.exists():
            try:
                with open(targets_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading targets: {e}")
        
        return {}
    
    def save_augmentation_targets(self, augmented_files):
        """
        Save augmentation targets to a JSON file.
        
        Args:
            augmented_files: List of dicts with augmentation info
            
        Returns:
            Path to saved file
        """
        targets_path = self.metadata_dir / "augmented_mixing_parameters.json"
        
        print(f"💾 Saving augmented mixing parameters to {targets_path}...")
        
        # Create targets dict
        targets = {}
        
        for result in augmented_files:
            if result.get("status") == "success":
                for file_info in result.get("saved_files", []):
                    file_id = Path(file_info["augmented_file"]).stem
                    targets[file_id] = file_info["targets"]
        
        with open(targets_path, 'w') as f:
            json.dump(targets, f, indent=2)
        
        print(f"✅ Saved {len(targets)} augmented target parameters")
        return targets_path
    
    def save_augmentation_metadata(self):
        """Save augmentation metadata to a JSON file."""
        metadata_path = self.metadata_dir / "augmentation_metadata.json"
        
        print(f"💾 Saving augmentation metadata to {metadata_path}...")
        
        # Calculate augmentation factor
        if self.augmentation_metadata["augmentation_stats"]["total_original_tracks"] > 0:
            factor = (
                self.augmentation_metadata["augmentation_stats"]["total_augmented_tracks"] /
                self.augmentation_metadata["augmentation_stats"]["total_original_tracks"]
            )
            self.augmentation_metadata["augmentation_stats"]["augmentation_factor"] = factor
        
        with open(metadata_path, 'w') as f:
            json.dump(self.augmentation_metadata, f, indent=2)
        
        print(f"✅ Augmentation metadata saved")
        return metadata_path
    
    def run_augmentation(self, categories=None, subset_size=None):
        """
        Run the entire augmentation pipeline.
        
        Args:
            categories: List of categories to augment. If None, augments all categories.
                       Options: ["music", "vocals"]
            subset_size: Max number of tracks to augment per category (None = all)
            
        Returns:
            Dict with augmentation results
        """
        if categories is None:
            categories = ["music", "vocals"]
        
        print(f"🚀 Starting enhanced audio augmentation pipeline...")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Categories to augment: {', '.join(categories)}")
        
        # Load original processing metadata
        processing_metadata_path = self.metadata_dir / "processing_metadata.json"
        if processing_metadata_path.exists():
            with open(processing_metadata_path, 'r') as f:
                processing_metadata = json.load(f)
        else:
            processing_metadata = {"processed_tracks": {}}
        
        # Load original mixing targets
        original_targets = self.load_targets()
        
        results = {}
        all_results = []
        
        # Process each category
        for category in categories:
            if category not in processing_metadata["processed_tracks"]:
                print(f"⚠️ No processed tracks found for category: {category}")
                continue
            
            # Get track list
            tracks = processing_metadata["processed_tracks"][category]
            
            # Limit subset size if specified
            if subset_size is not None and subset_size < len(tracks):
                print(f"⚠️ Limiting to {subset_size} tracks for {category} (total: {len(tracks)})")
                tracks = tracks[:subset_size]
            
            # Update stats
            self.augmentation_metadata["augmentation_stats"]["total_original_tracks"] += len(tracks)
            
            # Process each track
            print(f"🔄 Augmenting {len(tracks)} {category} tracks...")
            
            category_results = []
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                process_func = partial(self.process_file, category=category, original_targets_dict=original_targets)
                for result in tqdm(executor.map(process_func, tracks), total=len(tracks)):
                    category_results.append(result)
            
            # Count augmentations
            success_count = sum(1 for r in category_results if r.get('status') == 'success')
            skip_count = sum(1 for r in category_results if r.get('status') == 'skipped')
            total_augmentations = sum(r.get('augmentations', 0) for r in category_results)
            
            print(f"✅ Augmented {category} tracks: {success_count} successes, {skip_count} skipped")
            print(f"🎯 Created {total_augmentations} augmented versions ({total_augmentations/len(tracks):.1f}x)")
            
            # Update metadata
            self.augmentation_metadata["augmented_tracks"][category].extend(
                [r for r in category_results if r.get('status') == 'success']
            )
            self.augmentation_metadata["augmentation_stats"]["total_augmented_tracks"] += total_augmentations
            
            # Count augmentation methods
            for result in category_results:
                if result.get('status') == 'success':
                    for file_info in result.get('saved_files', []):
                        aug_type = file_info.get('augmentation_type')
                        if aug_type:
                            self.augmentation_metadata["augmentation_stats"]["methods_used"][aug_type] = (
                                self.augmentation_metadata["augmentation_stats"]["methods_used"].get(aug_type, 0) + 1
                            )
            
            results[category] = category_results
            all_results.extend(category_results)
        
        # Save augmentation targets
        self.save_augmentation_targets(all_results)
        
        # Save metadata
        self.save_augmentation_metadata()
        
        # Summary
        print("\n📋 Augmentation Summary:")
        print("=" * 50)
        orig_count = self.augmentation_metadata["augmentation_stats"]["total_original_tracks"]
        aug_count = self.augmentation_metadata["augmentation_stats"]["total_augmented_tracks"]
        aug_factor = self.augmentation_metadata["augmentation_stats"]["augmentation_factor"]
        
        print(f"Original tracks: {orig_count}")
        print(f"Augmented tracks: {aug_count}")
        print(f"Augmentation factor: {aug_factor:.1f}x")
        print("\nAugmentation methods:")
        
        for method, count in self.augmentation_metadata["augmentation_stats"]["methods_used"].items():
            print(f"  {method}: {count} tracks")
        
        return results

# When run directly
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Audio Augmentation for AI Mixing")
    parser.add_argument('--categories', nargs='+', choices=['music', 'vocals', 'all'],
                        default=['all'], help="Categories to augment")
    parser.add_argument('--subset-size', type=int, default=None,
                        help="Max number of tracks to augment per category")
    
    args = parser.parse_args()
    
    # Process categories argument
    if 'all' in args.categories:
        categories_to_augment = ['music', 'vocals']
    else:
        categories_to_augment = args.categories
    
    # Create augmentor instance
    augmentor = EnhancedAudioAugmentor()
    
    # Run augmentation
    augmentor.run_augmentation(
        categories=categories_to_augment,
        subset_size=args.subset_size
    )
