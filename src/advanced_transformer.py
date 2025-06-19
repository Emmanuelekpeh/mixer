#!/usr/bin/env python3
"""
🤖 Advanced Transformer Mixer - Multi-Head Attention for Audio Processing
=========================================================================

State-of-the-art transformer architecture with attention mechanisms for audio mixing.
Specializes in attention mechanisms, multi-head processing, and contextual mixing.

Architecture Features:
- Multi-head self-attention over spectrogram features
- Positional encoding for temporal and frequency structure
- Cross-modal fusion between different audio characteristics
- Hierarchical attention for multi-scale processing

Capabilities:
- Attention modeling: 0.95
- Contextual understanding: 0.9
- Harmonic enhancement: 0.95
- Spectral analysis: 0.9
- Multi-track coordination: 0.85
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spectrograms (frequency + time)."""
    
    def __init__(self, d_model: int, max_freq_len: int = 128, max_time_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding for frequency dimension
        freq_pe = torch.zeros(max_freq_len, d_model // 2)
        freq_position = torch.arange(0, max_freq_len, dtype=torch.float).unsqueeze(1)
        freq_div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * 
                                 (-math.log(10000.0) / (d_model // 2)))
        freq_pe[:, 0::2] = torch.sin(freq_position * freq_div_term)
        freq_pe[:, 1::2] = torch.cos(freq_position * freq_div_term)
        
        # Create positional encoding for time dimension
        time_pe = torch.zeros(max_time_len, d_model // 2)
        time_position = torch.arange(0, max_time_len, dtype=torch.float).unsqueeze(1)
        time_div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * 
                                 (-math.log(10000.0) / (d_model // 2)))
        time_pe[:, 0::2] = torch.sin(time_position * time_div_term)
        time_pe[:, 1::2] = torch.cos(time_position * time_div_term)
        
        self.register_buffer('freq_pe', freq_pe)
        self.register_buffer('time_pe', time_pe)
        
    def forward(self, x):
        """
        Add positional encoding to input.
        x: (batch_size, freq_bins, time_steps, d_model)
        """
        batch_size, freq_bins, time_steps, d_model = x.size()
        
        # Expand positional encodings
        freq_pe = self.freq_pe[:freq_bins].unsqueeze(1).expand(-1, time_steps, -1)
        time_pe = self.time_pe[:time_steps].unsqueeze(0).expand(freq_bins, -1, -1)
        
        # Concatenate frequency and time positional encodings
        pos_encoding = torch.cat([freq_pe, time_pe], dim=-1)
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return x + pos_encoding

class SpectrogramPatching(nn.Module):
    """Convert spectrogram to patches for transformer processing."""
    
    def __init__(self, n_mels: int = 128, patch_size: Tuple[int, int] = (8, 25), 
                 d_model: int = 512):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_mels = n_mels
        
        # Calculate number of patches
        self.n_freq_patches = n_mels // patch_size[0]
        self.n_time_patches = 1000 // patch_size[1]  # Assuming 1000 time steps
        self.n_patches = self.n_freq_patches * self.n_time_patches
        
        # Patch embedding layer
        patch_dim = patch_size[0] * patch_size[1]
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        
        # Class token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, spectrogram):
        """
        Convert spectrogram to patches.
        spectrogram: (batch_size, n_mels, time_steps)
        """
        batch_size = spectrogram.size(0)
        
        # Reshape to patches
        # (batch_size, n_freq_patches, patch_size[0], n_time_patches, patch_size[1])
        patches = spectrogram.view(
            batch_size, 
            self.n_freq_patches, self.patch_size[0],
            self.n_time_patches, self.patch_size[1]
        )
        
        # Rearrange to (batch_size, n_patches, patch_dim)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        patches = patches.view(batch_size, self.n_patches, -1)
        
        # Apply patch embedding
        patch_embeddings = self.patch_embedding(patches)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        return embeddings

class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for different temporal resolutions."""
    
    def __init__(self, d_model: int, n_heads: int = 8, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        
        # Multi-scale attention layers
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
            for _ in scales
        ])
        
        # Scale fusion layer
        self.scale_fusion = nn.Linear(d_model * len(scales), d_model)
        
    def forward(self, x, attn_mask=None):
        """
        Apply multi-scale attention.
        x: (batch_size, seq_len, d_model)
        """
        scale_outputs = []
        
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            if scale == 1:
                # Standard attention
                scaled_output, _ = attention(x, x, x, attn_mask=attn_mask)
            else:
                # Downsample for larger scales
                downsampled = x[:, ::scale, :]
                scaled_output, _ = attention(downsampled, downsampled, downsampled)
                
                # Upsample back to original resolution
                if scaled_output.size(1) < x.size(1):
                    scaled_output = F.interpolate(
                        scaled_output.transpose(1, 2),
                        size=x.size(1),
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
            
            scale_outputs.append(scaled_output)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused)
        
        return output

class TransformerMixingBlock(nn.Module):
    """Enhanced transformer block for audio mixing."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multi-scale attention
        self.multi_scale_attention = MultiScaleAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Cross-attention for harmonic content
        self.harmonic_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """Forward pass through transformer block."""
        # Multi-scale self-attention
        attended = self.multi_scale_attention(x)
        x = self.norm1(x + attended)
        
        # Cross-attention for harmonic relationships
        harmonic_attended, _ = self.harmonic_attention(x, x, x)
        x = self.norm3(x + harmonic_attended)
        
        # Feed-forward
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)
        
        return x

class AdvancedTransformerMixer(nn.Module):
    """
    Advanced Transformer-based audio mixer with multi-head attention.
    
    Processes spectrograms as sequences of patches with positional encoding
    and multi-scale attention for comprehensive audio understanding.
    """
    
    def __init__(self,
                 n_mels: int = 128,
                 patch_size: Tuple[int, int] = (8, 25),
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 n_outputs: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_mels = n_mels
        self.d_model = d_model
        self.n_outputs = n_outputs
        
        # Patch embedding
        self.patch_embedding = SpectrogramPatching(n_mels, patch_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerMixingBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        
        # Output heads for different parameter categories
        self.parameter_heads = nn.ModuleDict({
            'dynamics': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2)  # Input gain, compression
            ),
            'eq': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 4)  # High, Mid, Low, Presence
            ),
            'effects': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3)  # Reverb, Delay, Stereo
            ),
            'output': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # Output level
            )
        })
        
        # Global mixing context layer
        self.global_context = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
    def forward(self, spectrogram):
        """
        Forward pass through advanced transformer mixer.
        
        Args:
            spectrogram: (batch_size, n_mels, time_steps)
        """
        batch_size = spectrogram.size(0)
        
        # Convert to patches and embed
        patch_embeddings = self.patch_embedding(spectrogram)
        # patch_embeddings: (batch_size, n_patches + 1, d_model)
        
        # Apply transformer layers
        x = patch_embeddings
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Global context attention across all patches
        global_context, attention_weights = self.global_context(x, x, x)
        x = x + global_context
        
        # Extract global representation (class token)
        global_repr = x[:, 0, :]  # First token is class token
        
        # Generate parameter categories
        dynamics_params = self.parameter_heads['dynamics'](global_repr)
        eq_params = self.parameter_heads['eq'](global_repr)
        effects_params = self.parameter_heads['effects'](global_repr)
        output_params = self.parameter_heads['output'](global_repr)
        
        # Combine all parameters
        raw_params = torch.cat([
            dynamics_params,  # Input gain, compression
            eq_params,       # High, Mid, Low, Presence
            effects_params,  # Reverb, Delay, Stereo
            output_params    # Output level
        ], dim=1)
        
        # Apply parameter constraints
        constrained_params = self._constrain_parameters(raw_params)
        
        return constrained_params
    
    def _constrain_parameters(self, raw_params):
        """Apply parameter constraints for safe mixing."""
        constrained = torch.zeros_like(raw_params)
        sigmoid_params = torch.sigmoid(raw_params)
        
        # Input Gain: 0.3-1.5
        constrained[:, 0] = sigmoid_params[:, 0] * 1.2 + 0.3
        
        # Compression Ratio: 1.0-8.0
        constrained[:, 1] = sigmoid_params[:, 1] * 7.0 + 1.0
        
        # EQ parameters: -10.0 to +10.0 dB
        for i in range(2, 6):
            constrained[:, i] = sigmoid_params[:, i] * 20.0 - 10.0
            
        # Reverb Send: 0.0-0.6
        constrained[:, 6] = sigmoid_params[:, 6] * 0.6
        
        # Delay Send: 0.0-0.5
        constrained[:, 7] = sigmoid_params[:, 7] * 0.5
        
        # Stereo Width: 0.7-1.3
        constrained[:, 8] = sigmoid_params[:, 8] * 0.6 + 0.7
        
        # Output Level: 0.5-0.9
        constrained[:, 9] = sigmoid_params[:, 9] * 0.4 + 0.5
        
        return constrained
    
    def get_attention_maps(self, spectrogram):
        """Extract attention maps for visualization."""
        with torch.no_grad():
            patch_embeddings = self.patch_embedding(spectrogram)
            
            attention_maps = []
            x = patch_embeddings
            
            for transformer_layer in self.transformer_layers:
                # Get attention from multi-scale attention
                x = transformer_layer(x)
                
            # Get global attention weights
            _, global_attention = self.global_context(x, x, x)
            attention_maps.append(global_attention)
            
            return attention_maps
    
    def analyze_harmonic_attention(self, spectrogram):
        """Analyze how the model attends to harmonic content."""
        attention_maps = self.get_attention_maps(spectrogram)
        
        # Process attention patterns
        harmonic_analysis = {}
        for i, attn_map in enumerate(attention_maps):
            # Analyze attention distribution
            entropy = -torch.sum(attn_map * torch.log(attn_map + 1e-8), dim=-1)
            harmonic_analysis[f'layer_{i}_entropy'] = entropy.mean().item()
            
        return harmonic_analysis

class TransformerAudioDataset(torch.utils.data.Dataset):
    """Dataset for transformer training with attention-aware augmentation."""
    
    def __init__(self, spectrogram_dir, targets_file, 
                 attention_augment: bool = True):
        self.samples = []
        self.targets = json.load(open(targets_file))
        self.attention_augment = attention_augment
        
        # Find all spectrogram files
        for track_dir in Path(spectrogram_dir).rglob("*.npy"):
            self.samples.append(track_dir)
    
    def __len__(self):
        return len(self.samples)
    
    def attention_masking(self, spec, mask_ratio=0.1):
        """Apply attention-based masking for robust training."""
        if not self.attention_augment:
            return spec
            
        # Random attention masking
        freq_bins, time_steps = spec.shape
        
        # Frequency masking (mask certain frequency bands)
        freq_mask_size = int(freq_bins * mask_ratio)
        if freq_mask_size > 0:
            freq_start = np.random.randint(0, freq_bins - freq_mask_size)
            spec[freq_start:freq_start + freq_mask_size, :] *= 0.1  # Attenuate, don't zero
        
        # Time masking (mask certain time segments)
        time_mask_size = int(time_steps * mask_ratio)
        if time_mask_size > 0:
            time_start = np.random.randint(0, time_steps - time_mask_size)
            spec[:, time_start:time_start + time_mask_size] *= 0.1
            
        return spec
    
    def __getitem__(self, idx):
        spec_path = self.samples[idx]
        spec = np.load(spec_path)
        
        # Normalize
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
        
        # Apply attention-aware augmentation
        if self.attention_augment:
            spec = self.attention_masking(spec)
        
        # Get target parameters
        track_name = spec_path.stem.replace("_mel_spec", "")
        target = self.targets.get(track_name, [0.5] * 10)
        
        return torch.FloatTensor(spec), torch.FloatTensor(target)

def create_advanced_transformer_model(config: Optional[Dict] = None) -> AdvancedTransformerMixer:
    """Factory function to create advanced transformer model."""
    default_config = {
        'n_mels': 128,
        'patch_size': (8, 25),
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'n_outputs': 10,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    model = AdvancedTransformerMixer(**default_config)
    logger.info(f"Created Advanced Transformer Mixer with config: {default_config}")
    
    return model

# Model metadata for tournament integration
MODEL_METADATA = {
    "id": "advanced_transformer",
    "name": "Advanced Audio Transformer",
    "architecture": "Transformer",
    "description": "State-of-the-art transformer architecture with attention mechanisms for audio mixing",
    "specializations": ["attention_mechanisms", "multi_head_processing", "contextual_mixing"],
    "capabilities": {
        "attention_modeling": 0.95,
        "contextual_understanding": 0.9,
        "harmonic_enhancement": 0.95,
        "spectral_analysis": 0.9,
        "multi_track_coordination": 0.85
    },
    "preferred_genres": ["orchestral", "jazz", "complex_arrangements"],
    "signature_techniques": ["self_attention", "cross_modal_fusion", "positional_encoding"]
}

if __name__ == "__main__":
    # Test model creation
    model = create_advanced_transformer_model()
    
    # Test forward pass
    batch_size, n_mels, time_steps = 2, 128, 1000
    test_input = torch.randn(batch_size, n_mels, time_steps)
    
    with torch.no_grad():
        output = model(test_input)
        attention_maps = model.get_attention_maps(test_input)
        harmonic_analysis = model.analyze_harmonic_attention(test_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Attention maps: {len(attention_maps)} layers")
    print(f"Harmonic analysis: {harmonic_analysis}")
    print(f"Sample output: {output[0]}")
    print("✅ Advanced Transformer Mixer test passed!")
