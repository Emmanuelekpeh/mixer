#!/usr/bin/env python3
"""
🎛️ VAE Audio Mixer - Variational Autoencoder for Latent Space Manipulation
==========================================================================

Variational Autoencoder for latent space audio manipulation and mixing.
Specializes in latent manipulation, smooth interpolation, and probabilistic mixing.

Architecture Features:
- Encoder-decoder architecture with latent space bottleneck
- Variational inference for smooth latent representations
- Latent space interpolation for style blending
- Probabilistic mixing parameter generation

Capabilities:
- Latent modeling: 0.9
- Smooth interpolation: 0.85
- Probabilistic mixing: 0.8
- Spectral analysis: 0.75
- Dynamic range: 0.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional, List
import math

logger = logging.getLogger(__name__)

class SpectrogramEncoder(nn.Module):
    """Encoder network for spectrogram to latent space mapping."""
    
    def __init__(self, n_mels: int = 128, latent_dim: int = 64):
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        
        # Convolutional encoder
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, n_mels, time_steps)
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Will be computed dynamically
        self.fc_layers = None
        
    def forward(self, spectrogram):
        # spectrogram: (batch_size, n_mels, time_steps)
        x = spectrogram.unsqueeze(1)  # Add channel dimension
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Initialize FC layers if needed
        if self.fc_layers is None:
            self.fc_layers = nn.Sequential(
                nn.Linear(x.size(1), 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3)
            ).to(x.device)
        
        features = self.fc_layers(x)
        return features

class VAELatentSpace(nn.Module):
    """Variational latent space with reparameterization trick."""
    
    def __init__(self, feature_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        # Mean and log variance networks
        self.mu_layer = nn.Linear(feature_dim, latent_dim)
        self.logvar_layer = nn.Linear(feature_dim, latent_dim)
        
        # Prior parameters (learnable)
        self.register_parameter('prior_mu', nn.Parameter(torch.zeros(latent_dim)))
        self.register_parameter('prior_logvar', nn.Parameter(torch.zeros(latent_dim)))
        
    def encode(self, features):
        """Encode features to latent distribution parameters."""
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, mu, logvar):
        """Compute KL divergence with learned prior."""
        # Expand prior to match batch size
        batch_size = mu.size(0)
        prior_mu = self.prior_mu.expand(batch_size, -1)
        prior_logvar = self.prior_logvar.expand(batch_size, -1)
        
        # KL divergence between two multivariate Gaussians
        kl = 0.5 * torch.sum(
            prior_logvar - logvar + 
            (torch.exp(logvar) + (mu - prior_mu)**2) / torch.exp(prior_logvar) - 1,
            dim=1
        )
        return kl.mean()
    
    def forward(self, features):
        """Forward pass through variational latent space."""
        mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)
        kl_loss = self.kl_divergence(mu, logvar)
        return z, mu, logvar, kl_loss

class MixingDecoder(nn.Module):
    """Decoder network for latent space to mixing parameters."""
    
    def __init__(self, latent_dim: int = 64, n_outputs: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_outputs = n_outputs
        
        # Decoder network with multiple paths for different parameter types
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Specialized heads for different parameter categories
        self.gain_head = nn.Linear(hidden_dim, 2)  # Input gain, output level
        self.dynamics_head = nn.Linear(hidden_dim, 1)  # Compression
        self.eq_head = nn.Linear(hidden_dim, 4)  # High, Mid, Low, Presence
        self.effects_head = nn.Linear(hidden_dim, 3)  # Reverb, Delay, Stereo width
        
    def forward(self, latent_code):
        """Decode latent code to mixing parameters."""
        shared_features = self.shared_decoder(latent_code)
        
        # Generate parameter components
        gain_params = self.gain_head(shared_features)
        dynamics_params = self.dynamics_head(shared_features)
        eq_params = self.eq_head(shared_features)
        effects_params = self.effects_head(shared_features)
        
        # Combine all parameters
        raw_params = torch.cat([
            gain_params[:, :1],    # Input gain
            dynamics_params,       # Compression
            eq_params[:, :3],      # High, Mid, Low EQ
            eq_params[:, 3:],      # Presence
            effects_params,        # Reverb, Delay, Stereo
            gain_params[:, 1:]     # Output level
        ], dim=1)
        
        return raw_params

class VAEAudioMixer(nn.Module):
    """
    Complete VAE system for audio mixing parameter generation.
    
    Provides smooth latent space interpolation and probabilistic
    mixing parameter generation with controllable uncertainty.
    """
    
    def __init__(self,
                 n_mels: int = 128,
                 latent_dim: int = 64,
                 n_outputs: int = 10):
        super().__init__()
        
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        self.n_outputs = n_outputs
        
        # VAE components
        self.encoder = SpectrogramEncoder(n_mels, latent_dim * 4)  # Higher dim for VAE
        self.latent_space = VAELatentSpace(256, latent_dim)
        self.decoder = MixingDecoder(latent_dim, n_outputs)
        
        # Interpolation control
        self.interpolation_strength = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, spectrogram, return_latent=False):
        """
        Forward pass through VAE mixer.
        
        Args:
            spectrogram: (batch_size, n_mels, time_steps)
            return_latent: Whether to return latent representations
        """
        # Encode to features
        features = self.encoder(spectrogram)
        
        # Process through latent space
        latent_code, mu, logvar, kl_loss = self.latent_space(features)
        
        # Decode to mixing parameters
        raw_params = self.decoder(latent_code)
        
        # Apply parameter constraints
        constrained_params = self._constrain_parameters(raw_params)
        
        if return_latent:
            return constrained_params, latent_code, mu, logvar, kl_loss
        else:
            return constrained_params, kl_loss
    
    def _constrain_parameters(self, raw_params):
        """Apply safe parameter constraints."""
        constrained = torch.zeros_like(raw_params)
        sigmoid_params = torch.sigmoid(raw_params)
        
        # Input Gain: 0.2-1.2
        constrained[:, 0] = sigmoid_params[:, 0] * 1.0 + 0.2
        
        # Compression Ratio: 1.0-6.0
        constrained[:, 1] = sigmoid_params[:, 1] * 5.0 + 1.0
        
        # EQ parameters: -6.0 to +6.0 dB (conservative)
        for i in range(2, 5):
            constrained[:, i] = sigmoid_params[:, i] * 12.0 - 6.0
            
        # Presence/Air: 0.0-0.6
        constrained[:, 5] = sigmoid_params[:, 5] * 0.6
        
        # Reverb Send: 0.0-0.5
        constrained[:, 6] = sigmoid_params[:, 6] * 0.5
        
        # Delay Send: 0.0-0.4
        constrained[:, 7] = sigmoid_params[:, 7] * 0.4
        
        # Stereo Width: 0.6-1.4
        constrained[:, 8] = sigmoid_params[:, 8] * 0.8 + 0.6
        
        # Output Level: 0.4-0.8
        constrained[:, 9] = sigmoid_params[:, 9] * 0.4 + 0.4
        
        return constrained
    
    def encode_to_latent(self, spectrogram):
        """Encode spectrogram to latent representation."""
        features = self.encoder(spectrogram)
        latent_code, mu, logvar, kl_loss = self.latent_space(features)
        return latent_code, mu, logvar
    
    def decode_from_latent(self, latent_code):
        """Decode latent code to mixing parameters."""
        raw_params = self.decoder(latent_code)
        return self._constrain_parameters(raw_params)
    
    def interpolate_mixing(self, spec_a, spec_b, alpha=0.5):
        """
        Interpolate between two audio spectrograms in latent space.
        
        Args:
            spec_a, spec_b: Input spectrograms
            alpha: Interpolation factor (0=spec_a, 1=spec_b)
        """
        # Encode both spectrograms
        latent_a, mu_a, logvar_a = self.encode_to_latent(spec_a)
        latent_b, mu_b, logvar_b = self.encode_to_latent(spec_b)
        
        # Interpolate in latent space
        interpolated_latent = (1 - alpha) * latent_a + alpha * latent_b
        
        # Decode interpolated representation
        interpolated_params = self.decode_from_latent(interpolated_latent)
        
        return interpolated_params, interpolated_latent
    
    def sample_from_prior(self, batch_size=1):
        """Sample mixing parameters from learned prior distribution."""
        device = next(self.parameters()).device
        
        # Sample from learned prior
        prior_mu = self.latent_space.prior_mu.expand(batch_size, -1)
        prior_logvar = self.latent_space.prior_logvar.expand(batch_size, -1)
        
        # Sample latent code
        std = torch.exp(0.5 * prior_logvar)
        eps = torch.randn_like(std)
        sampled_latent = prior_mu + eps * std
        
        # Decode to parameters
        sampled_params = self.decode_from_latent(sampled_latent)
        
        return sampled_params, sampled_latent
    
    def explore_latent_space(self, center_spec, exploration_radius=0.5, num_samples=5):
        """
        Explore latent space around a center spectrogram.
        
        Args:
            center_spec: Center spectrogram for exploration
            exploration_radius: Radius of exploration in latent space
            num_samples: Number of samples to generate
        """
        # Encode center spectrogram
        center_latent, _, _ = self.encode_to_latent(center_spec)
        
        # Generate exploration samples
        explorations = []
        for _ in range(num_samples):
            # Random direction in latent space
            noise = torch.randn_like(center_latent) * exploration_radius
            explored_latent = center_latent + noise
            
            # Decode explored point
            explored_params = self.decode_from_latent(explored_latent)
            explorations.append((explored_params, explored_latent))
        
        return explorations

class VAELoss(nn.Module):
    """VAE loss combining reconstruction and KL divergence."""
    
    def __init__(self, kl_weight=0.1, reconstruction_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_params, target_params, kl_loss):
        """Compute VAE loss."""
        # Reconstruction loss
        reconstruction_loss = self.mse_loss(predicted_params, target_params)
        
        # Total VAE loss
        total_loss = (self.reconstruction_weight * reconstruction_loss + 
                     self.kl_weight * kl_loss)
        
        return total_loss, reconstruction_loss, kl_loss

class VAEAudioDataset(torch.utils.data.Dataset):
    """Dataset for VAE training with probabilistic augmentation."""
    
    def __init__(self, spectrogram_dir, targets_file, 
                 probabilistic_augment: bool = True):
        self.samples = []
        self.targets = json.load(open(targets_file))
        self.probabilistic_augment = probabilistic_augment
        
        # Find all spectrogram files
        for track_dir in Path(spectrogram_dir).rglob("*.npy"):
            self.samples.append(track_dir)
    
    def __len__(self):
        return len(self.samples)
    
    def probabilistic_param_augment(self, params):
        """Apply probabilistic augmentation to parameters."""
        if not self.probabilistic_augment:
            return params
            
        # Add controlled Gaussian noise
        noise_std = 0.03  # 3% standard deviation
        noise = np.random.normal(0, noise_std, len(params))
        augmented_params = params + noise
        
        # Ensure valid range
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
        
        # Apply probabilistic augmentation
        if self.probabilistic_augment:
            target = self.probabilistic_param_augment(np.array(target))
        
        return torch.FloatTensor(spec), torch.FloatTensor(target)

def create_vae_mixer_model(config: Optional[Dict] = None) -> VAEAudioMixer:
    """Factory function to create VAE mixer model."""
    default_config = {
        'n_mels': 128,
        'latent_dim': 64,
        'n_outputs': 10
    }
    
    if config:
        default_config.update(config)
    
    model = VAEAudioMixer(**default_config)
    logger.info(f"Created VAE Audio Mixer with config: {default_config}")
    
    return model

# Model metadata for tournament integration
MODEL_METADATA = {
    "id": "vae_mixer",
    "name": "Variational Audio Encoder",
    "architecture": "VAE",
    "description": "Variational Autoencoder for latent space audio manipulation",
    "specializations": ["latent_manipulation", "smooth_interpolation", "probabilistic_mixing"],
    "capabilities": {
        "latent_modeling": 0.9,
        "smooth_interpolation": 0.85,
        "probabilistic_mixing": 0.8,
        "spectral_analysis": 0.75,
        "dynamic_range": 0.8
    },
    "preferred_genres": ["ambient", "downtempo", "atmospheric"],
    "signature_techniques": ["latent_interpolation", "probabilistic_sampling", "smooth_transitions"]
}

if __name__ == "__main__":
    # Test model creation
    model = create_vae_mixer_model()
    
    # Test forward pass
    batch_size, n_mels, time_steps = 4, 128, 1000
    test_input = torch.randn(batch_size, n_mels, time_steps)
    
    with torch.no_grad():
        # Standard generation
        output, kl_loss = model(test_input)
        
        # Latent space interpolation test
        interp_output, interp_latent = model.interpolate_mixing(
            test_input[:2], test_input[2:], alpha=0.5
        )
        
        # Prior sampling test
        sampled_output, sampled_latent = model.sample_from_prior(batch_size=2)
        
        # Latent space exploration
        explorations = model.explore_latent_space(test_input[:1], num_samples=3)
    
    print(f"Model output shape: {output.shape}")
    print(f"KL loss: {kl_loss.item():.4f}")
    print(f"Interpolated output shape: {interp_output.shape}")
    print(f"Sampled output shape: {sampled_output.shape}")
    print(f"Explorations: {len(explorations)} samples")
    print(f"Sample output: {output[0]}")
    print("✅ VAE Audio Mixer test passed!")
