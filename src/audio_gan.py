#!/usr/bin/env python3
"""
🎨 Audio GAN - Generative Adversarial Network for Creative Mixing
================================================================

Generative Adversarial Network for creative audio mixing and enhancement.
Specializes in generative mixing, style transfer, and creative synthesis.

Architecture Features:
- Generator network for creative mixing parameter generation
- Discriminator for realistic parameter validation
- Style transfer capabilities for genre adaptation
- Adversarial training for novel mixing approaches

Capabilities:
- Creative generation: 0.9
- Style transfer: 0.85
- Novelty creation: 0.8
- Spectral analysis: 0.7
- Dynamic range: 0.75
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional, List
import random

logger = logging.getLogger(__name__)

class SpectralEncoder(nn.Module):
    """Encoder to extract features from spectrograms."""
    
    def __init__(self, n_mels: int = 128, latent_dim: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        
        # Convolutional encoder for spectrogram features
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, n_mels, time_steps)
            nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate flattened dimension after convolutions
        # This will be computed dynamically in forward pass
        self.fc_layers = None
        
    def forward(self, spectrogram):
        # spectrogram: (batch_size, n_mels, time_steps)
        x = spectrogram.unsqueeze(1)  # Add channel dimension
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Initialize FC layers if not done yet
        if self.fc_layers is None:
            self.fc_layers = nn.Sequential(
                nn.Linear(x.size(1), self.latent_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.latent_dim * 2, self.latent_dim)
            ).to(x.device)
        
        features = self.fc_layers(x)
        return features

class MixingGenerator(nn.Module):
    """Generator network for creative mixing parameter generation."""
    
    def __init__(self, 
                 latent_dim: int = 256,
                 style_dim: int = 64,
                 n_outputs: int = 10,
                 hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.n_outputs = n_outputs
        
        # Style embedding for genre/style conditioning
        self.style_embedding = nn.Embedding(10, style_dim)  # 10 different styles
        
        # Generator network
        input_dim = latent_dim + style_dim
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, n_outputs)
        )
          # Creative noise injection layers
        self.noise_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
    def forward(self, audio_features, style_id=None, creativity_factor=0.5):
        """
        Generate mixing parameters with controllable creativity.
        
        Args:
            audio_features: (batch_size, latent_dim) - encoded audio features
            style_id: Optional style conditioning (0-9)
            creativity_factor: Control randomness/creativity (0.0-1.0)
        """
        batch_size = audio_features.size(0)
        
        # Generate or use provided style
        if style_id is None:
            style_id = torch.randint(0, 10, (batch_size,), device=audio_features.device)
        elif isinstance(style_id, int):
            style_id = torch.full((batch_size,), style_id, device=audio_features.device)
        
        # Get style embeddings
        style_features = self.style_embedding(style_id)
        
        # Combine audio features with style
        combined_features = torch.cat([audio_features, style_features], dim=1)
        
        # Add creative noise based on creativity factor
        if creativity_factor > 0:
            noise_scale = creativity_factor * 0.1
            creative_noise = torch.randn_like(combined_features) * noise_scale
            combined_features = combined_features + creative_noise
        
        # Generate mixing parameters
        raw_params = self.generator(combined_features)        # Apply creative modulation with additional noise layers
        for i, noise_layer in enumerate(self.noise_layers):
            if creativity_factor > 0.3:  # Only apply for higher creativity
                modulation_input = torch.randn(batch_size, 512,  # Use fixed hidden dimension
                                             device=raw_params.device)
                modulation = noise_layer(modulation_input)
                raw_params = raw_params + modulation[:, :raw_params.size(1)] * creativity_factor * 0.05# Apply parameter constraints using individual parameter handling
        constrained_params = torch.zeros_like(raw_params)
        normalized_params = torch.sigmoid(raw_params)
        
        # Input Gain: 0.1-1.5
        constrained_params[:, 0] = normalized_params[:, 0] * 1.4 + 0.1
        
        # Compression Ratio: 1.0-8.0
        constrained_params[:, 1] = normalized_params[:, 1] * 7.0 + 1.0
        
        # EQ parameters: -8.0 to +8.0 dB
        for i in range(2, 5):  # High, Mid, Low
            constrained_params[:, i] = normalized_params[:, i] * 16.0 - 8.0
            
        # Presence/Air: 0.0-0.8
        constrained_params[:, 5] = normalized_params[:, 5] * 0.8
        
        # Reverb Send: 0.0-0.7
        constrained_params[:, 6] = normalized_params[:, 6] * 0.7
        
        # Delay Send: 0.0-0.6
        constrained_params[:, 7] = normalized_params[:, 7] * 0.6
        
        # Stereo Width: 0.5-1.5
        constrained_params[:, 8] = normalized_params[:, 8] * 1.0 + 0.5
        
        # Output Level: 0.3-0.9
        constrained_params[:, 9] = normalized_params[:, 9] * 0.6 + 0.3
        
        return constrained_params

class MixingDiscriminator(nn.Module):
    """Discriminator to validate realistic mixing parameters."""
    
    def __init__(self, n_outputs: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.n_outputs = n_outputs
        
        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(n_outputs, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, mixing_params):
        """
        Discriminate between real and generated mixing parameters.
        
        Returns:
            probability: (batch_size, 1) - probability of being real parameters
        """
        return self.discriminator(mixing_params)

class AudioGANMixer(nn.Module):
    """
    Complete Audio GAN system for creative mixing.
    
    Combines encoder, generator, and discriminator for end-to-end
    creative mixing parameter generation with style control.
    """
    
    def __init__(self,
                 n_mels: int = 128,
                 latent_dim: int = 256,
                 style_dim: int = 64,
                 n_outputs: int = 10):
        super().__init__()
        
        self.encoder = SpectralEncoder(n_mels, latent_dim)
        self.generator = MixingGenerator(latent_dim, style_dim, n_outputs)
        self.discriminator = MixingDiscriminator(n_outputs)
        
        # Style mapping for interpretation
        self.style_names = [
            "Classical", "Jazz", "Rock", "Electronic", "Hip-Hop",
            "Ambient", "Folk", "Metal", "Pop", "Experimental"
        ]
        
    def forward(self, spectrogram, style_id=None, creativity_factor=0.5):
        """
        Forward pass for creative mixing generation.
        
        Args:
            spectrogram: (batch_size, n_mels, time_steps)
            style_id: Optional style conditioning
            creativity_factor: Control creativity level
        """
        # Encode audio features
        audio_features = self.encoder(spectrogram)
        
        # Generate creative mixing parameters
        mixing_params = self.generator(audio_features, style_id, creativity_factor)
        
        return mixing_params
    
    def style_transfer(self, spectrogram, source_style=None, target_style=None):
        """
        Perform style transfer between musical genres.
        
        Args:
            spectrogram: Input audio spectrogram
            source_style: Source genre style (0-9)
            target_style: Target genre style (0-9)
        """
        # Encode audio features
        audio_features = self.encoder(spectrogram)
        
        # Generate with target style
        if target_style is None:
            target_style = random.randint(0, 9)
        
        # Use higher creativity for style transfer
        mixing_params = self.generator(audio_features, target_style, 
                                     creativity_factor=0.8)
        
        return mixing_params, self.style_names[target_style]
    
    def discriminate_quality(self, mixing_params):
        """Evaluate quality/realism of mixing parameters."""
        return self.discriminator(mixing_params)

class GANLoss(nn.Module):
    """Custom loss function for GAN training."""
    
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.criterion = nn.BCELoss()
        
    def generator_loss(self, discriminator_output_fake):
        """Loss for generator (wants discriminator to think fake is real)."""
        labels = torch.full_like(discriminator_output_fake, self.real_label)
        return self.criterion(discriminator_output_fake, labels)
    
    def discriminator_loss(self, discriminator_output_real, discriminator_output_fake):
        """Loss for discriminator (wants to distinguish real from fake)."""
        # Real samples should be classified as real
        real_labels = torch.full_like(discriminator_output_real, self.real_label)
        real_loss = self.criterion(discriminator_output_real, real_labels)
        
        # Fake samples should be classified as fake
        fake_labels = torch.full_like(discriminator_output_fake, self.fake_label)
        fake_loss = self.criterion(discriminator_output_fake, fake_labels)
        
        return real_loss + fake_loss

class GANAudioDataset(torch.utils.data.Dataset):
    """Dataset for GAN training with real mixing parameters."""
    
    def __init__(self, spectrogram_dir, targets_file, 
                 augment_creativity: bool = True):
        self.samples = []
        self.targets = json.load(open(targets_file))
        self.augment_creativity = augment_creativity
        
        # Find all spectrogram files
        for track_dir in Path(spectrogram_dir).rglob("*.npy"):
            self.samples.append(track_dir)
    
    def __len__(self):
        return len(self.samples)
    
    def creative_augment_params(self, params):
        """Apply creative augmentation to mixing parameters."""
        if not self.augment_creativity:
            return params
            
        # Add controlled randomness to parameters
        noise_scale = 0.05  # 5% variation
        creative_noise = np.random.normal(0, noise_scale, len(params))
        augmented_params = params + creative_noise
        
        # Ensure parameters stay in valid ranges
        augmented_params = np.clip(augmented_params, 0.0, 1.0)
        
        return augmented_params
    
    def __getitem__(self, idx):
        spec_path = self.samples[idx]
        spec = np.load(spec_path)
        
        # Normalize spectrogram
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        
        # Handle sequence length
        target_time_steps = 1000
        if spec.shape[1] > target_time_steps:
            start = np.random.randint(0, spec.shape[1] - target_time_steps)
            spec = spec[:, start:start + target_time_steps]
        elif spec.shape[1] < target_time_steps:
            pad_width = target_time_steps - spec.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant')
        
        # Get target parameters
        track_name = spec_path.stem.replace("_mel_spec", "")
        target = self.targets.get(track_name, [0.5] * 10)
        
        # Apply creative augmentation
        if self.augment_creativity:
            target = self.creative_augment_params(np.array(target))
        
        # Random style for training
        style_id = random.randint(0, 9)
        
        return (torch.FloatTensor(spec), 
                torch.FloatTensor(target),
                torch.LongTensor([style_id]))

def create_audio_gan_model(config: Optional[Dict] = None) -> AudioGANMixer:
    """Factory function to create Audio GAN model."""
    default_config = {
        'n_mels': 128,
        'latent_dim': 256,
        'style_dim': 64,
        'n_outputs': 10
    }
    
    if config:
        default_config.update(config)
    
    model = AudioGANMixer(**default_config)
    logger.info(f"Created Audio GAN Mixer with config: {default_config}")
    
    return model

# Model metadata for tournament integration
MODEL_METADATA = {
    "id": "audio_gan",
    "name": "Generative Audio Mixer",
    "architecture": "GAN",
    "description": "Generative Adversarial Network for creative audio mixing and enhancement",
    "specializations": ["generative_mixing", "style_transfer", "creative_enhancement"],
    "capabilities": {
        "creative_generation": 0.9,
        "style_transfer": 0.85,
        "novelty_creation": 0.8,
        "spectral_analysis": 0.7,
        "dynamic_range": 0.75
    },
    "preferred_genres": ["experimental", "electronic", "fusion"],
    "signature_techniques": ["adversarial_training", "style_interpolation", "creative_synthesis"]
}

if __name__ == "__main__":
    # Test model creation
    model = create_audio_gan_model()
    
    # Test forward pass
    batch_size, n_mels, time_steps = 4, 128, 1000
    test_input = torch.randn(batch_size, n_mels, time_steps)
    
    with torch.no_grad():
        # Standard generation
        output = model(test_input, creativity_factor=0.5)
        
        # Style transfer test
        transfer_output, style_name = model.style_transfer(test_input, target_style=3)
        
        # Quality assessment
        quality_score = model.discriminate_quality(output)
    
    print(f"Model output shape: {output.shape}")
    print(f"Style transfer output: {transfer_output.shape}, Style: {style_name}")
    print(f"Quality scores: {quality_score.mean().item():.3f}")
    print(f"Sample output: {output[0]}")
    print("✅ Audio GAN Mixer test passed!")
