#!/usr/bin/env python3
"""
Fixed SpectrogramDataset for New Architecture Training Pipeline
===============================================================

This is a modified version of the dataset class that handles variable-shaped spectrograms
properly for compatibility with all model architectures.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import json
import random

class FixedSpectrogramDataset(Dataset):
    """Fixed dataset class that properly handles spectrogram dimension issues."""
    
    def __init__(self, spectrogram_dir, targets_file, n_outputs=10, augment=False):
        self.samples = []
        self.targets = json.load(open(targets_file))
        for track_dir in Path(spectrogram_dir).rglob("*.npy"):
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
        try:
            spec_path = self.samples[idx]
            spec = np.load(spec_path)
            
            # Normalize
            spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
            
            # Fixed time dimension (crop or pad to consistent length)
            target_time_steps = 1000  # Fixed length
            current_shape = spec.shape
            
            # Ensure spec is 2D with shape (n_mels, time_steps)
            if len(current_shape) != 2:
                # Convert to 2D if it's not already
                spec = spec.reshape(128, -1)  # Assuming first dim is 128 mel bands
            
            if spec.shape[1] > target_time_steps:
                # Crop from center
                start = (spec.shape[1] - target_time_steps) // 2
                spec = spec[:, start:start + target_time_steps]
            elif spec.shape[1] < target_time_steps:
                # Calculate padding
                pad_width = target_time_steps - spec.shape[1]
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                
                # Apply padding safely with explicit tuple shape
                spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
            
            # Data augmentation
            if self.augment:
                if random.random() < 0.5:
                    spec = self.time_mask(spec, 0.1)
                if random.random() < 0.5:
                    spec = self.freq_mask(spec, 0.1)
                if random.random() < 0.5:
                    spec = self.add_noise(spec, 0.01)
            
            # Get target mixing parameters (or zeros if not found)
            track_id = spec_path.stem
            target = self.targets.get(track_id, [0] * self.n_outputs)
            target = target[:self.n_outputs]  # Ensure we only use the requested number of outputs
              # Convert to torch tensors
            # For 1D convolutional models (LSTM), shape should be [batch, channels, features]
            # For 2D convolutional models (CNN, ResNet), shape should be [batch, channels, height, width]
            # We'll make it work for both by returning the proper format
            spec_tensor = torch.tensor(spec, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            return spec_tensor, target_tensor
            
        except Exception as e:
            print(f"Error loading sample {idx} from {self.samples[idx]}: {e}")
            # Return a default sample as fallback
            return torch.zeros((1, 128, 1000), dtype=torch.float32), torch.zeros(self.n_outputs, dtype=torch.float32)
