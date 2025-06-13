#!/usr/bin/env python3
"""
🎵 Spectrogram Data Augmentation for AI Mixing
=============================================

This script expands the training dataset by augmenting existing spectrograms:
- SpecAugment: Time and frequency masking
- Noise injection: Add spectral noise
- Intensity variations: Adjust magnitude
- Frequency shifting: Simulate pitch changes
- Time stretching: Simulate tempo changes

Goal: Increase dataset size from ~54 to 300+ training samples
"""

import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SpectrogramAugmentor:
    """Advanced spectrogram augmentation for mixing parameter prediction."""
    
    def __init__(self, input_dir, output_dir, targets_file):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load existing targets
        with open(targets_file, 'r') as f:
            self.targets = json.load(f)
        
        self.augmented_targets = {}
        
    def time_mask(self, spectrogram, max_mask_pct=0.15, num_masks=2):
        """Apply time masking to spectrogram."""
        spec = spectrogram.copy()
        _, time_steps = spec.shape
        
        for _ in range(num_masks):
            mask_length = int(np.random.uniform(0, max_mask_pct) * time_steps)
            mask_start = np.random.randint(0, time_steps - mask_length)
            spec[:, mask_start:mask_start + mask_length] = 0
        
        return spec
    
    def freq_mask(self, spectrogram, max_mask_pct=0.15, num_masks=2):
        """Apply frequency masking to spectrogram."""
        spec = spectrogram.copy()
        freq_bins, _ = spec.shape
        
        for _ in range(num_masks):
            mask_length = int(np.random.uniform(0, max_mask_pct) * freq_bins)
            mask_start = np.random.randint(0, freq_bins - mask_length)
            spec[mask_start:mask_start + mask_length, :] = 0
        
        return spec
    
    def add_spectral_noise(self, spectrogram, noise_level=0.01):
        """Add noise to spectrogram."""
        noise = np.random.normal(0, noise_level, spectrogram.shape)
        return np.clip(spectrogram + noise, 0, None)
    
    def intensity_variation(self, spectrogram, factor_range=(0.8, 1.2)):
        """Vary overall intensity of spectrogram."""
        factor = np.random.uniform(*factor_range)
        return spectrogram * factor
    
    def freq_shift(self, spectrogram, max_shift_bins=5):
        """Simulate pitch shift by shifting frequency bins."""
        spec = spectrogram.copy()
        shift = np.random.randint(-max_shift_bins, max_shift_bins + 1)
        
        if shift > 0:
            spec[shift:, :] = spec[:-shift, :]
            spec[:shift, :] = 0
        elif shift < 0:
            spec[:shift, :] = spec[-shift:, :]
            spec[shift:, :] = 0
        
        return spec
    
    def time_stretch_sim(self, spectrogram, stretch_range=(0.9, 1.1)):
        """Simulate time stretching by interpolating."""
        factor = np.random.uniform(*stretch_range)
        
        if factor == 1.0:
            return spectrogram
        
        # Simple interpolation for time stretching effect
        original_length = spectrogram.shape[1]
        new_length = int(original_length / factor)
        
        if new_length <= 0:
            return spectrogram
        
        # Create new time indices
        old_indices = np.linspace(0, original_length - 1, new_length)
        old_indices = np.clip(old_indices, 0, original_length - 1).astype(int)
        
        # Extract columns and pad/truncate to match original length
        stretched = spectrogram[:, old_indices]
        
        if stretched.shape[1] < original_length:
            # Pad with zeros
            padding = np.zeros((spectrogram.shape[0], original_length - stretched.shape[1]))
            stretched = np.concatenate([stretched, padding], axis=1)
        elif stretched.shape[1] > original_length:
            # Truncate
            stretched = stretched[:, :original_length]
        
        return stretched
    
    def augment_spectrogram(self, spec_path, original_targets):
        """Create multiple augmented versions of a spectrogram."""
        spectrogram = np.load(spec_path)
        artist_name = spec_path.parent.name
        track_name = spec_path.stem.replace(' - ', '_').replace('.npy', '')
        
        augmentations = []
        
        # 1. SpecAugment variations
        for i in range(3):
            aug_spec = self.time_mask(spectrogram)
            aug_spec = self.freq_mask(aug_spec)
            aug_targets = self.adjust_targets_for_masking(original_targets)
            augmentations.append((f"{artist_name}_{track_name}_specaug{i}", aug_spec, aug_targets))
        
        # 2. Noise injection
        for noise_level in [0.005, 0.01, 0.02]:
            aug_spec = self.add_spectral_noise(spectrogram, noise_level)
            aug_targets = self.adjust_targets_for_noise(original_targets, noise_level)
            augmentations.append((f"{artist_name}_{track_name}_noise{noise_level:.3f}", aug_spec, aug_targets))
        
        # 3. Intensity variations
        for factor in [0.8, 0.9, 1.1, 1.2]:
            aug_spec = self.intensity_variation(spectrogram, (factor, factor))
            aug_targets = self.adjust_targets_for_intensity(original_targets, factor)
            augmentations.append((f"{artist_name}_{track_name}_int{factor:.1f}", aug_spec, aug_targets))
        
        # 4. Frequency shifting (pitch simulation)
        for shift in [-3, -1, 1, 3]:
            aug_spec = self.freq_shift(spectrogram, abs(shift))
            aug_targets = self.adjust_targets_for_freq_shift(original_targets, shift)
            augmentations.append((f"{artist_name}_{track_name}_fshift{shift:+d}", aug_spec, aug_targets))
        
        # 5. Time stretching simulation
        for stretch in [0.9, 0.95, 1.05, 1.1]:
            aug_spec = self.time_stretch_sim(spectrogram, (stretch, stretch))
            aug_targets = self.adjust_targets_for_time_stretch(original_targets, stretch)
            augmentations.append((f"{artist_name}_{track_name}_stretch{stretch:.2f}", aug_spec, aug_targets))
        
        return augmentations
    
    def adjust_targets_for_masking(self, targets):
        """Adjust for SpecAugment masking."""
        adjusted = np.array(targets).copy()
        # Masking might require slight compression increase
        adjusted[1] = np.clip(adjusted[1] + 0.03, 0, 1)  # Compression
        return adjusted.tolist()
    
    def adjust_targets_for_noise(self, targets, noise_level):
        """Adjust for added noise."""
        adjusted = np.array(targets).copy()
        noise_factor = noise_level * 20
        adjusted[1] = np.clip(adjusted[1] + noise_factor, 0, 1)      # More compression
        adjusted[6] = np.clip(adjusted[6] - noise_factor/2, 0, 1)   # Less reverb
        return adjusted.tolist()
    
    def adjust_targets_for_intensity(self, targets, factor):
        """Adjust for intensity changes."""
        adjusted = np.array(targets).copy()
        if factor < 1.0:  # Quieter
            adjusted[0] = np.clip(adjusted[0] + (1-factor) * 0.5, 0, 1)  # More gain
        else:  # Louder
            adjusted[0] = np.clip(adjusted[0] - (factor-1) * 0.3, 0, 1)  # Less gain
        return adjusted.tolist()
    
    def adjust_targets_for_freq_shift(self, targets, shift):
        """Adjust for frequency shifting."""
        adjusted = np.array(targets).copy()
        if shift > 0:  # Higher frequencies emphasized
            adjusted[2] = np.clip(adjusted[2] + 0.05, 0, 1)    # High-freq EQ
            adjusted[4] = np.clip(adjusted[4] - 0.03, 0, 1)    # Low-freq EQ
        else:  # Lower frequencies emphasized
            adjusted[2] = np.clip(adjusted[2] - 0.05, 0, 1)    # High-freq EQ
            adjusted[4] = np.clip(adjusted[4] + 0.03, 0, 1)    # Low-freq EQ
        return adjusted.tolist()
    
    def adjust_targets_for_time_stretch(self, targets, stretch):
        """Adjust for time stretching."""
        adjusted = np.array(targets).copy()
        if stretch > 1.0:  # Faster/shorter
            adjusted[6] = np.clip(adjusted[6] - 0.05, 0, 1)    # Less reverb
            adjusted[1] = np.clip(adjusted[1] + 0.03, 0, 1)    # More compression
        else:  # Slower/longer
            adjusted[6] = np.clip(adjusted[6] + 0.05, 0, 1)    # More reverb
            adjusted[1] = np.clip(adjusted[1] - 0.03, 0, 1)    # Less compression
        return adjusted.tolist()
    
    def process_dataset(self):
        """Process entire spectrogram dataset with augmentation."""
        print(f"🎵 Starting spectrogram data augmentation...")
        print(f"📁 Input: {self.input_dir}")
        print(f"📁 Output: {self.output_dir}")
        
        # Find all spectrogram files
        spec_files = []
        for artist_dir in self.input_dir.iterdir():
            if artist_dir.is_dir() and artist_dir.name in self.targets:
                for spec_file in artist_dir.glob("*.npy"):
                    spec_files.append(spec_file)
        
        print(f"📊 Found {len(spec_files)} spectrogram files to augment")
        total_augmentations = 0
        
        for spec_path in tqdm(spec_files, desc="Processing spectrograms"):
            artist_name = spec_path.parent.name
            
            if artist_name not in self.targets:
                continue
            
            try:
                # Get original targets
                original_targets = self.targets[artist_name]
                
                # Create augmented versions
                augmentations = self.augment_spectrogram(spec_path, original_targets)
                
                # Save augmented spectrograms and update targets
                for aug_name, aug_spec, aug_targets in augmentations:
                    # Create artist directory in output
                    artist_output_dir = self.output_dir / artist_name
                    artist_output_dir.mkdir(exist_ok=True)
                    
                    # Save spectrogram
                    output_path = artist_output_dir / f"{aug_name}.npy"
                    np.save(output_path, aug_spec)
                    
                    # Store targets (using artist name as key, same as original)
                    if artist_name not in self.augmented_targets:
                        self.augmented_targets[artist_name] = []
                    
                    total_augmentations += 1
                
            except Exception as e:
                print(f"❌ Error processing {spec_path}: {e}")
                continue
        
        # Save augmented targets (combine with original)
        targets_output = self.output_dir.parent / "targets_augmented.json"
        combined_targets = {**self.targets}  # Keep original targets
        
        # Note: We keep the same artist-based target structure since the model
        # learns to map artist style to mixing parameters
        
        with open(targets_output, 'w') as f:
            json.dump(combined_targets, f, indent=2)
        
        print(f"\\n✅ Augmentation complete!")
        print(f"📊 Original files: {len(spec_files)}")
        print(f"🎯 Augmented files: {total_augmentations}")
        print(f"📈 Dataset expansion: {total_augmentations / len(spec_files):.1f}x")
        print(f"💾 Targets saved to: {targets_output}")
        
        return total_augmentations

def main():
    """Run spectrogram augmentation on training dataset."""
    print("🎵 AI Mixing Spectrogram Augmentation")
    print("=" * 40)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "data" / "spectrograms" / "train"
    output_dir = base_dir / "data" / "spectrograms_augmented"
    targets_file = base_dir / "data" / "targets_generated.json"
    
    if not targets_file.exists():
        print(f"❌ Targets file not found: {targets_file}")
        print("💡 Run generate_targets.py first to create targets")
        return
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create augmentor and process
    augmentor = SpectrogramAugmentor(input_dir, output_dir, targets_file)
    total_augmented = augmentor.process_dataset()
    
    print(f"\\n🚀 Ready for enhanced training!")
    print(f"   Original spectrograms: {input_dir}")
    print(f"   Augmented spectrograms: {output_dir}")
    print(f"   Targets: {output_dir.parent / 'targets_augmented.json'}")
    print(f"\\n💡 Next steps:")
    print("1. Update training scripts to use augmented data")
    print("2. Run hyperparameter optimization")
    print("3. Train models with expanded dataset")
    print("4. Compare performance improvements")

if __name__ == "__main__":
    main()
