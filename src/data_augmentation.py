#!/usr/bin/env python3
"""
🎵 Audio Data Augmentation for AI Mixing
========================================

This script expands the training dataset using audio augmentation techniques:
- Pitch shifting: ±2 semitones
- Time stretching: 0.8x to 1.2x speed
- Noise injection: Low-level background noise
- Dynamic range compression: Simulate different recording conditions
- EQ variations: Slight frequency response changes

Goal: Increase dataset size 3-5x while maintaining target validity
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioAugmentor:
    """Advanced audio augmentation for mixing parameter prediction."""
    
    def __init__(self, input_dir, output_dir, targets_file):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load existing targets
        with open(targets_file, 'r') as f:
            self.targets = json.load(f)
        
        self.augmented_targets = {}
        self.sr = 22050
        
    def pitch_shift(self, audio, n_steps):
        """Shift pitch by n_steps semitones."""
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def time_stretch(self, audio, rate):
        """Change tempo by rate factor."""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def add_noise(self, audio, noise_level=0.005):
        """Add low-level Gaussian noise."""
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise
    
    def dynamic_range_compress(self, audio, ratio=0.3):
        """Apply gentle compression to simulate different recording conditions."""
        # Simple soft compression
        threshold = 0.1
        compressed = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) * (1 - ratio)),
            audio
        )
        return compressed
    
    def eq_variation(self, audio, freq_range, gain_db):
        """Apply subtle EQ changes."""
        # Simple frequency-domain EQ simulation
        stft = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        # Find frequency bin range
        start_bin = np.argmin(np.abs(freqs - freq_range[0]))
        end_bin = np.argmin(np.abs(freqs - freq_range[1]))
        
        # Apply gain
        gain_linear = 10 ** (gain_db / 20)
        stft[start_bin:end_bin] *= gain_linear
        
        # Convert back to time domain
        return librosa.istft(stft)
    
    def augment_audio_file(self, file_path, original_targets):
        """Create multiple augmented versions of an audio file."""
        audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
        file_stem = file_path.stem
        
        augmentations = []
        
        # 1. Pitch shift variations
        for pitch_shift in [-2, -1, 1, 2]:  # ±2 semitones
            aug_audio = self.pitch_shift(audio, pitch_shift)
            aug_targets = self.adjust_targets_for_pitch(original_targets, pitch_shift)
            augmentations.append((f"{file_stem}_pitch{pitch_shift:+d}", aug_audio, aug_targets))
        
        # 2. Time stretch variations
        for rate in [0.85, 0.95, 1.05, 1.15]:  # ±15% tempo change
            aug_audio = self.time_stretch(audio, rate)
            aug_targets = self.adjust_targets_for_tempo(original_targets, rate)
            augmentations.append((f"{file_stem}_tempo{rate:.2f}", aug_audio, aug_targets))
        
        # 3. Noise injection
        for noise_level in [0.003, 0.007]:
            aug_audio = self.add_noise(audio, noise_level)
            aug_targets = self.adjust_targets_for_noise(original_targets, noise_level)
            augmentations.append((f"{file_stem}_noise{noise_level:.3f}", aug_audio, aug_targets))
        
        # 4. Dynamic range compression
        for ratio in [0.2, 0.4]:
            aug_audio = self.dynamic_range_compress(audio, ratio)
            aug_targets = self.adjust_targets_for_compression(original_targets, ratio)
            augmentations.append((f"{file_stem}_comp{ratio:.1f}", aug_audio, aug_targets))
        
        # 5. EQ variations
        eq_variations = [
            ((1000, 3000), 1.5),   # Mid boost
            ((200, 800), -1.0),    # Low-mid cut
            ((5000, 10000), 2.0),  # High boost
        ]
        
        for i, (freq_range, gain_db) in enumerate(eq_variations):
            aug_audio = self.eq_variation(audio, freq_range, gain_db)
            aug_targets = self.adjust_targets_for_eq(original_targets, freq_range, gain_db)
            augmentations.append((f"{file_stem}_eq{i}", aug_audio, aug_targets))
        
        return augmentations
    
    def adjust_targets_for_pitch(self, targets, pitch_shift):
        """Adjust mixing targets based on pitch shift."""
        adjusted = targets.copy()
        
        # Higher pitch might need less low-end EQ, more high-end
        if pitch_shift > 0:  # Higher pitch
            adjusted[2] = min(1.0, adjusted[2] + 0.1)    # Increase high-freq EQ
            adjusted[4] = max(0.0, adjusted[4] - 0.05)   # Decrease low-freq EQ
        else:  # Lower pitch
            adjusted[2] = max(0.0, adjusted[2] - 0.1)    # Decrease high-freq EQ
            adjusted[4] = min(1.0, adjusted[4] + 0.05)   # Increase low-freq EQ
        
        return adjusted
    
    def adjust_targets_for_tempo(self, targets, rate):
        """Adjust mixing targets based on tempo change."""
        adjusted = targets.copy()
        
        # Faster tempo might need less reverb, more compression
        if rate > 1.0:  # Faster
            adjusted[6] = max(0.0, adjusted[6] - 0.1)    # Less reverb
            adjusted[1] = min(1.0, adjusted[1] + 0.05)   # More compression
        else:  # Slower
            adjusted[6] = min(1.0, adjusted[6] + 0.1)    # More reverb
            adjusted[1] = max(0.0, adjusted[1] - 0.05)   # Less compression
        
        return adjusted
    
    def adjust_targets_for_noise(self, targets, noise_level):
        """Adjust mixing targets based on added noise."""
        adjusted = targets.copy()
        
        # More noise might need more compression and less reverb
        noise_factor = noise_level * 100  # Scale factor
        adjusted[1] = min(1.0, adjusted[1] + noise_factor * 2)    # More compression
        adjusted[6] = max(0.0, adjusted[6] - noise_factor)        # Less reverb
        
        return adjusted
    
    def adjust_targets_for_compression(self, targets, ratio):
        """Adjust mixing targets based on pre-compression."""
        adjusted = targets.copy()
        
        # Pre-compressed audio needs less compression
        adjusted[1] = max(0.0, adjusted[1] - ratio * 0.5)
        
        return adjusted
    
    def adjust_targets_for_eq(self, targets, freq_range, gain_db):
        """Adjust mixing targets based on EQ changes."""
        adjusted = targets.copy()
        
        # Compensate for pre-applied EQ
        if freq_range[1] <= 1000:  # Low frequency boost/cut
            adjusted[4] = max(0.0, min(1.0, adjusted[4] - gain_db * 0.05))
        elif freq_range[0] >= 5000:  # High frequency boost/cut
            adjusted[2] = max(0.0, min(1.0, adjusted[2] - gain_db * 0.05))
        else:  # Mid frequency boost/cut
            adjusted[3] = max(0.0, min(1.0, adjusted[3] - gain_db * 0.05))
        
        return adjusted
    
    def process_dataset(self):
        """Process entire dataset with augmentation."""
        print(f"🎵 Starting audio data augmentation...")
        print(f"📁 Input: {self.input_dir}")
        print(f"📁 Output: {self.output_dir}")
        
        audio_files = list(self.input_dir.glob("*.mp4"))
        total_augmentations = 0
        
        for file_path in tqdm(audio_files, desc="Processing files"):
            file_key = file_path.stem
            
            if file_key not in self.targets:
                print(f"⚠️ No targets found for {file_key}, skipping...")
                continue
            
            try:
                # Get original targets
                original_targets = self.targets[file_key]
                
                # Create augmented versions
                augmentations = self.augment_audio_file(file_path, original_targets)
                
                # Save augmented audio files and update targets
                for aug_name, aug_audio, aug_targets in augmentations:
                    # Save audio
                    output_path = self.output_dir / f"{aug_name}.wav"
                    sf.write(output_path, aug_audio, self.sr)
                    
                    # Store targets
                    self.augmented_targets[aug_name] = aug_targets
                    total_augmentations += 1
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                continue
        
        # Save augmented targets
        targets_output = self.output_dir.parent / "targets_augmented.json"
        
        # Combine original and augmented targets
        combined_targets = {**self.targets, **self.augmented_targets}
        
        with open(targets_output, 'w') as f:
            json.dump(combined_targets, f, indent=2)
        
        print(f"\\n✅ Augmentation complete!")
        print(f"📊 Original files: {len(audio_files)}")
        print(f"🎯 Augmented files: {total_augmentations}")
        print(f"📈 Dataset expansion: {total_augmentations / len(audio_files):.1f}x")
        print(f"💾 Targets saved to: {targets_output}")
        
        return total_augmentations

def main():
    """Run data augmentation on training dataset."""
    print("🎵 AI Mixing Data Augmentation")
    print("=" * 40)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "train"
    output_dir = base_dir / "data" / "train_augmented"
    targets_file = base_dir / "data" / "targets_generated.json"
    
    if not targets_file.exists():
        print(f"❌ Targets file not found: {targets_file}")
        print("💡 Run generate_targets.py first to create targets")
        return
    
    # Create augmentor and process
    augmentor = AudioAugmentor(input_dir, output_dir, targets_file)
    total_augmented = augmentor.process_dataset()
    
    print(f"\\n🚀 Ready for enhanced training!")
    print(f"   Use the augmented dataset: {output_dir}")
    print(f"   With targets: {output_dir.parent / 'targets_augmented.json'}")
    print(f"\\n💡 Next steps:")
    print("1. Update training scripts to use augmented data")
    print("2. Run hyperparameter optimization")
    print("3. Train models with expanded dataset")
    print("4. Compare performance improvements")

if __name__ == "__main__":
    main()
