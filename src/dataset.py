import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random

# Configuration
SPECTROGRAMS_FOLDER = Path(__file__).resolve().parent.parent / "data" / "spectrograms"
TARGETS_FILE = Path(__file__).resolve().parent.parent / "data" / "targets_generated.json"
BATCH_SIZE = 16
N_MELS = 128
N_OUTPUTS = 10
AUGMENT = True
AUG_TIME_MASK = 0.1  # Fraction of time steps to mask
AUG_FREQ_MASK = 0.1  # Fraction of freq bins to mask
AUG_NOISE_STD = 0.01 # Std of Gaussian noise

class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, targets_file, n_outputs=N_OUTPUTS, augment=False):
        self.samples = []
        self.targets = json.load(open(targets_file))
        for track_dir in Path(spectrogram_dir).rglob("*_spec.npy"):
            self.samples.append(track_dir)
        self.n_outputs = n_outputs
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def time_mask(self, spec, mask_frac):
        t = spec.shape[1]
        mask_len = int(t * mask_frac)
        if mask_len > 0:
            start = random.randint(0, t - mask_len)
            spec[:, start:start+mask_len] = 0
        return spec

    def freq_mask(self, spec, mask_frac):
        f = spec.shape[0]
        mask_len = int(f * mask_frac)
        if mask_len > 0:
            start = random.randint(0, f - mask_len)
            spec[start:start+mask_len, :] = 0
        return spec

    def add_noise(self, spec, std):
        return spec + np.random.normal(0, std, spec.shape)

    def __getitem__(self, idx):
        spec_path = self.samples[idx]
        spec = np.load(spec_path, allow_pickle=True) # Allow pickle for legacy formats

        # --- Robust Spectrogram Loading ---
        # Define a standard shape for replacement of corrupted data
        target_shape = (N_MELS, 1000) # Using a common target length

        # 1. Handle non-array data (e.g., from corrupted files)
        if not isinstance(spec, np.ndarray):
            print(f"WARNING: Corrupted data in {spec_path}. Replacing with zeros.")
            spec = np.zeros(target_shape)

        # 2. Squeeze 3D arrays to 2D
        if spec.ndim == 3:
            spec = np.squeeze(spec, axis=0)

        # 3. Handle 1D or other incorrect dimensions
        if spec.ndim != 2:
            print(f"WARNING: Spectrogram at {spec_path} has unexpected shape {spec.shape}. Replacing with zeros.")
            spec = np.zeros(target_shape)
        
        # 4. Pad/Crop Mel dimension to be consistent
        if spec.shape[0] > N_MELS:
            # Crop from center
            start = (spec.shape[0] - N_MELS) // 2
            spec = spec[start:start + N_MELS, :]
        elif spec.shape[0] < N_MELS:
            # Pad with zeros
            pad_width = N_MELS - spec.shape[0]
            pad_top = pad_width // 2
            pad_bottom = pad_width - pad_top
            spec = np.pad(spec, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)


        # Normalize
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        
        # Fixed time dimension (crop or pad to consistent length)
        target_time_steps = 1000  # Fixed length
        if spec.shape[1] > target_time_steps:
            # Crop from center
            start = (spec.shape[1] - target_time_steps) // 2
            spec = spec[:, start:start + target_time_steps]
        elif spec.shape[1] < target_time_steps:
            # Pad with zeros
            pad_width = target_time_steps - spec.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        # Data augmentation
        if self.augment:
            if random.random() < 0.5:
                spec = self.time_mask(spec, AUG_TIME_MASK)
            if random.random() < 0.5:
                spec = self.freq_mask(spec, AUG_FREQ_MASK)
            if random.random() < 0.5:
                spec = self.add_noise(spec, AUG_NOISE_STD)
        
        # Add channel dimension
        spec = np.expand_dims(spec, axis=0)
        
        # Extract track name for target lookup
        track_name = spec_path.stem.replace('_spec', '')
        
        # Use real targets if available, else zeros
        target = self.targets.get(track_name, [0.0]*self.n_outputs)
        target = np.array(target, dtype=np.float32)
        
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def create_data_loaders(spectrograms_folder, targets_file, batch_size=BATCH_SIZE, train_split=0.7, val_split=0.15):
    """Create train/val/test data loaders"""
    
    # Create full dataset
    full_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=False)
    
    # Split indices
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets with augmentation for training
    train_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=AUGMENT)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    
    val_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
    
    test_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=False)
    test_dataset.samples = [full_dataset.samples[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
