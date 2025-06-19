#!/usr/bin/env python3
"""
🎵 LSTM Audio Mixer - Temporal Sequence Modeling
===============================================

Recurrent neural network with LSTM cells for sequential audio processing.
Specializes in temporal dynamics, memory retention, and sequential analysis.

Architecture Features:
- Bidirectional LSTM for forward/backward temporal context
- Multi-layer LSTM with attention mechanism
- Temporal pooling for variable-length sequences
- Specialized for dynamic mixing parameters

Capabilities:
- Temporal modeling: 0.95
- Sequence memory: 0.9  
- Dynamic adaptation: 0.85
- Spectral analysis: 0.75
- Harmonic enhancement: 0.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class TemporalAttention(nn.Module):
    """Attention mechanism for temporal sequence processing."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, sequence_length, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch_size, sequence_length, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        # attended_output: (batch_size, hidden_size)
        
        return attended_output, attention_weights

class SpectrogramToSequence(nn.Module):
    """Convert spectrogram to sequence for LSTM processing."""
    
    def __init__(self, n_mels: int = 128, hidden_size: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        
        # Convolutional layers for feature extraction from mel bands
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, spectrogram):
        # spectrogram: (batch_size, n_mels, time_steps)
        # Apply convolutions across mel bands for each time step
        features = self.conv_layers(spectrogram)
        # features: (batch_size, hidden_size, time_steps)
        
        # Transpose for LSTM: (batch_size, time_steps, hidden_size)
        sequence = features.transpose(1, 2)
        
        return sequence

class LSTMAudioMixer(nn.Module):
    """
    LSTM-based audio mixer for temporal sequence modeling.
    
    Processes spectrograms as temporal sequences to capture
    dynamic mixing patterns and temporal dependencies.
    """
    
    def __init__(self, 
                 n_mels: int = 128,
                 hidden_size: int = 256,
                 num_layers: int = 3,
                 n_outputs: int = 10,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_outputs = n_outputs
        self.bidirectional = bidirectional
        
        # Convert spectrogram to sequence
        self.spec_to_sequence = SpectrogramToSequence(n_mels, hidden_size)
        
        # Multi-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism for temporal aggregation
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.temporal_attention = TemporalAttention(lstm_output_size)
        
        # Output layers for mixing parameters
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, n_outputs)
        )
        
        # Parameter constraints for safe mixing
        self.register_buffer('param_mins', torch.tensor([
            0.0,  # Input Gain (0-2)
            1.0,  # Compression Ratio (1-10)  
            -12.0, # High-Freq EQ (-12 to +12 dB)
            -12.0, # Mid-Freq EQ (-12 to +12 dB)
            -12.0, # Low-Freq EQ (-12 to +12 dB)
            0.0,  # Presence/Air (0-1)
            0.0,  # Reverb Send (0-1)
            0.0,  # Delay Send (0-1)
            0.0,  # Stereo Width (0-2)
            0.0   # Output Level (0-1)
        ]))
        
        self.register_buffer('param_maxs', torch.tensor([
            2.0,  # Input Gain
            10.0, # Compression Ratio
            12.0, # High-Freq EQ
            12.0, # Mid-Freq EQ  
            12.0, # Low-Freq EQ
            1.0,  # Presence/Air
            1.0,  # Reverb Send
            1.0,  # Delay Send
            2.0,  # Stereo Width
            1.0   # Output Level
        ]))
        
    def forward(self, spectrogram):
        """
        Forward pass for LSTM audio mixer.
        
        Args:
            spectrogram: (batch_size, n_mels, time_steps)
            
        Returns:
            mixing_params: (batch_size, n_outputs) - constrained mixing parameters
        """
        batch_size = spectrogram.size(0)
        
        # Convert spectrogram to sequence
        sequence = self.spec_to_sequence(spectrogram)
        # sequence: (batch_size, time_steps, hidden_size)
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(sequence)
        # lstm_output: (batch_size, time_steps, hidden_size * num_directions)
        
        # Apply temporal attention
        attended_output, attention_weights = self.temporal_attention(lstm_output)
        # attended_output: (batch_size, hidden_size * num_directions)
        
        # Generate mixing parameters
        raw_params = self.output_layers(attended_output)
        # raw_params: (batch_size, n_outputs)
        
        # Apply parameter constraints using sigmoid and scaling
        normalized_params = torch.sigmoid(raw_params)
        constrained_params = (self.param_mins + 
                            normalized_params * (self.param_maxs - self.param_mins))
        
        return constrained_params
    
    def get_temporal_attention_weights(self, spectrogram):
        """Get attention weights for visualization."""
        with torch.no_grad():
            sequence = self.spec_to_sequence(spectrogram)
            lstm_output, _ = self.lstm(sequence)
            _, attention_weights = self.temporal_attention(lstm_output)
            return attention_weights.squeeze(-1)  # (batch_size, time_steps)

class LSTMSequenceDataset(torch.utils.data.Dataset):
    """Dataset for LSTM sequence training with temporal augmentation."""
    
    def __init__(self, spectrogram_dir, targets_file, 
                 sequence_length: int = 1000,
                 temporal_augment: bool = True):
        self.samples = []
        self.targets = json.load(open(targets_file))
        self.sequence_length = sequence_length
        self.temporal_augment = temporal_augment
        
        # Find all spectrogram files
        for track_dir in Path(spectrogram_dir).rglob("*.npy"):
            self.samples.append(track_dir)
    
    def __len__(self):
        return len(self.samples)
    
    def temporal_shift(self, spec, max_shift_ratio=0.1):
        """Apply temporal shifting for data augmentation."""
        if not self.temporal_augment:
            return spec
            
        time_steps = spec.shape[1]
        max_shift = int(time_steps * max_shift_ratio)
        shift = np.random.randint(-max_shift, max_shift + 1)
        
        if shift > 0:
            # Shift right, pad left
            spec = np.pad(spec, ((0, 0), (shift, 0)), mode='constant')[:, :time_steps]
        elif shift < 0:
            # Shift left, pad right  
            spec = np.pad(spec, ((0, 0), (0, -shift)), mode='constant')[:, -shift:]
            
        return spec
    
    def temporal_mask(self, spec, mask_ratio=0.1):
        """Apply temporal masking for data augmentation."""
        if not self.temporal_augment:
            return spec
            
        time_steps = spec.shape[1]
        mask_length = int(time_steps * mask_ratio)
        
        if mask_length > 0:
            start = np.random.randint(0, time_steps - mask_length)
            spec[:, start:start + mask_length] = 0
            
        return spec
    
    def __getitem__(self, idx):
        spec_path = self.samples[idx]
        spec = np.load(spec_path)
        
        # Normalize
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        
        # Fixed sequence length handling
        if spec.shape[1] > self.sequence_length:
            # Random crop for training diversity
            start = np.random.randint(0, spec.shape[1] - self.sequence_length)
            spec = spec[:, start:start + self.sequence_length]
        elif spec.shape[1] < self.sequence_length:
            # Pad to sequence length
            pad_width = self.sequence_length - spec.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant')
        
        # Apply temporal augmentations
        if self.temporal_augment:
            spec = self.temporal_shift(spec)
            spec = self.temporal_mask(spec)
        
        # Get target from filename
        track_name = spec_path.stem.replace("_mel_spec", "")
        target = self.targets.get(track_name, [0.5] * 10)  # Default neutral parameters
        
        return torch.FloatTensor(spec), torch.FloatTensor(target)

def create_lstm_mixer_model(config: Optional[Dict] = None) -> LSTMAudioMixer:
    """Factory function to create LSTM mixer model."""
    default_config = {
        'n_mels': 128,
        'hidden_size': 256,
        'num_layers': 3,
        'n_outputs': 10,
        'dropout': 0.3,
        'bidirectional': True
    }
    
    if config:
        default_config.update(config)
    
    model = LSTMAudioMixer(**default_config)
    logger.info(f"Created LSTM Audio Mixer with config: {default_config}")
    
    return model

# Model metadata for tournament integration
MODEL_METADATA = {
    "id": "lstm_mixer",
    "name": "LSTM Audio Mixer",
    "architecture": "LSTM/RNN",
    "description": "Recurrent neural network with LSTM cells for sequential audio processing",
    "specializations": ["temporal_processing", "dynamic_mixing", "sequential_analysis"],
    "capabilities": {
        "temporal_modeling": 0.95,
        "sequence_memory": 0.9,
        "dynamic_adaptation": 0.85,
        "spectral_analysis": 0.75,
        "harmonic_enhancement": 0.7
    },
    "preferred_genres": ["electronic", "ambient", "experimental"],
    "signature_techniques": ["temporal_smoothing", "dynamic_gating", "memory_retention"]
}

if __name__ == "__main__":
    # Test model creation
    model = create_lstm_mixer_model()
    
    # Test forward pass
    batch_size, n_mels, time_steps = 4, 128, 1000
    test_input = torch.randn(batch_size, n_mels, time_steps)
    
    with torch.no_grad():
        output = model(test_input)
        attention_weights = model.get_temporal_attention_weights(test_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Sample output: {output[0]}")
    print("✅ LSTM Audio Mixer test passed!")
