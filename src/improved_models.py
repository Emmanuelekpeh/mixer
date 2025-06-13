#!/usr/bin/env python3
"""
🚀 Enhanced AI Mixing Models with Advanced Architectures
========================================================

This module contains improved model architectures with:
- Better regularization to prevent over-aggressive predictions
- Advanced feature extraction with attention mechanisms  
- Multi-scale processing for different temporal patterns
- Constrained outputs to ensure safe mixing parameters

The models are designed to address the issues found in the original Enhanced CNN:
- Over-aggressive gain and output levels
- Risk of clipping and over-processing
- Poor generalization (high MAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionModule(nn.Module):
    """Self-attention mechanism for audio feature importance."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SafeResidualBlock(nn.Module):
    """Residual block with better regularization for stable training."""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.attention = AttentionModule(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedEnhancedCNN(nn.Module):
    """
    Improved Enhanced CNN with safe parameter prediction.
    
    Key improvements:
    - Constrained output ranges to prevent over-processing
    - Better regularization with dropout and batch norm
    - Attention mechanisms for important feature selection
    - Multi-head outputs for different parameter types
    """
    
    def __init__(self, n_outputs=10, dropout=0.3):
        super().__init__()
        
        # Initial feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with progressive feature extraction
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
          # Global feature aggregation
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Multi-head classifier for different parameter types
        feature_dim = 256 * 2  # concat of avg and max pooling
        
        # Separate heads for different parameter categories
        self.gain_head = self._make_head(feature_dim, 2, dropout)  # Input gain, Output level
        self.eq_head = self._make_head(feature_dim, 4, dropout)    # High, Mid, Low, Presence
        self.fx_head = self._make_head(feature_dim, 4, dropout)    # Compression, Reverb, Delay, Stereo
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dropout=0.3):
        layers = []
        layers.append(SafeResidualBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(SafeResidualBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    
    def _make_head(self, input_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # Less dropout in final layer
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global feature aggregation
        avg_pool = self.global_avgpool(x).view(x.size(0), -1)
        max_pool = self.global_maxpool(x).view(x.size(0), -1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Multi-head predictions
        gain_params = self.gain_head(features)      # [Input Gain, Output Level]
        eq_params = self.eq_head(features)          # [High EQ, Mid EQ, Low EQ, Presence]
        fx_params = self.fx_head(features)          # [Compression, Reverb, Delay, Stereo]
        
        # Combine all parameters
        output = torch.cat([
            gain_params[:, 0:1],  # Input Gain
            fx_params[:, 0:1],    # Compression Ratio
            eq_params[:, 0:1],    # High-Freq EQ
            eq_params[:, 1:2],    # Mid-Freq EQ
            eq_params[:, 2:3],    # Low-Freq EQ
            eq_params[:, 3:4],    # Presence/Air
            fx_params[:, 1:2],    # Reverb Send
            fx_params[:, 2:3],    # Delay Send
            fx_params[:, 3:4],    # Stereo Width
            gain_params[:, 1:2],  # Output Level
        ], dim=1)
        
        # Apply safe constraints to prevent over-processing
        output = self._apply_safe_constraints(output)
        
        return output
    
    def _apply_safe_constraints(self, x):
        """Apply constraints to ensure safe mixing parameters."""
        # Clamp each parameter to safe ranges
        constrained = torch.zeros_like(x)
        
        # Input Gain: 0.3-1.2 (prevent extreme gain)
        constrained[:, 0] = torch.clamp(torch.sigmoid(x[:, 0]) * 0.9 + 0.3, 0.3, 1.2)
        
        # Compression Ratio: 0.0-0.7 (prevent over-compression)
        constrained[:, 1] = torch.clamp(torch.sigmoid(x[:, 1]) * 0.7, 0.0, 0.7)
        
        # EQ parameters: 0.2-0.8 (moderate EQ changes)
        for i in range(2, 6):  # High, Mid, Low, Presence
            constrained[:, i] = torch.clamp(torch.sigmoid(x[:, i]) * 0.6 + 0.2, 0.2, 0.8)
        
        # Reverb Send: 0.1-0.8 (prevent excessive reverb)
        constrained[:, 6] = torch.clamp(torch.sigmoid(x[:, 6]) * 0.7 + 0.1, 0.1, 0.8)
        
        # Delay Send: 0.0-0.3 (subtle delay)
        constrained[:, 7] = torch.clamp(torch.sigmoid(x[:, 7]) * 0.3, 0.0, 0.3)
        
        # Stereo Width: 0.4-0.8 (safe stereo range)
        constrained[:, 8] = torch.clamp(torch.sigmoid(x[:, 8]) * 0.4 + 0.4, 0.4, 0.8)
        
        # Output Level: 0.5-0.95 (prevent clipping)
        constrained[:, 9] = torch.clamp(torch.sigmoid(x[:, 9]) * 0.45 + 0.5, 0.5, 0.95)
        
        return constrained

class MultiScaleTransformerMixer(nn.Module):
    """
    Advanced transformer-based model for mixing parameter prediction.
    
    Features:
    - Multi-scale temporal analysis
    - Self-attention for important feature selection
    - Positional encoding for temporal patterns
    - Genre-aware conditioning (future extension)
    """
    
    def __init__(self, n_outputs=10, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        # Feature extraction from spectrograms
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Pool frequency dimension
        )
        
        # Positional encoding for temporal patterns
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, n_outputs),
            nn.Sigmoid()  # Ensure outputs are in [0,1] range
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features
        features = self.feature_extractor(x)  # [B, d_model, 1, T]
        features = features.squeeze(2).transpose(1, 2)  # [B, T, d_model]
        
        # Add positional encoding
        seq_len = features.size(1)
        if seq_len <= self.pos_encoding.size(0):
            features = features + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer processing
        features = self.transformer(features)
        
        # Global temporal attention
        attended, _ = self.temporal_attention(features, features, features)
        
        # Global pooling
        global_features = torch.mean(attended, dim=1)
        
        # Final prediction
        output = self.classifier(global_features)
        
        return output

class DataAugmentation:
    """Audio data augmentation for improved model generalization."""
    
    @staticmethod
    def time_stretch(spectrogram, factor_range=(0.8, 1.2)):
        """Simulate time stretching by interpolating spectrogram."""
        factor = np.random.uniform(*factor_range)
        return F.interpolate(
            spectrogram, 
            scale_factor=(1.0, factor), 
            mode='bilinear', 
            align_corners=False
        )
    
    @staticmethod
    def freq_mask(spectrogram, max_mask_pct=0.1):
        """Mask random frequency bands."""
        freq_mask_param = int(max_mask_pct * spectrogram.size(-2))
        return torchaudio.transforms.FrequencyMasking(freq_mask_param)(spectrogram)
    
    @staticmethod
    def time_mask(spectrogram, max_mask_pct=0.05):
        """Mask random time segments."""
        time_mask_param = int(max_mask_pct * spectrogram.size(-1))
        return torchaudio.transforms.TimeMasking(time_mask_param)(spectrogram)
    
    @staticmethod
    def add_noise(spectrogram, noise_level=0.01):
        """Add random noise to spectrogram."""
        noise = torch.randn_like(spectrogram) * noise_level
        return spectrogram + noise

# Export the improved models
__all__ = [
    'ImprovedEnhancedCNN',
    'MultiScaleTransformerMixer', 
    'DataAugmentation',
    'AttentionModule',
    'SafeResidualBlock'
]

if __name__ == "__main__":
    # Test the improved models
    print("🧪 Testing Improved Model Architectures...")
    
    # Test input (batch_size=2, channels=1, height=128, width=1000)
    test_input = torch.randn(2, 1, 128, 1000)
    
    # Test ImprovedEnhancedCNN
    print("\n🔧 Testing ImprovedEnhancedCNN...")
    improved_model = ImprovedEnhancedCNN()
    output = improved_model(test_input)
    print(f"✅ Output shape: {output.shape}")
    print(f"📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test MultiScaleTransformerMixer
    print("\n🤖 Testing MultiScaleTransformerMixer...")
    transformer_model = MultiScaleTransformerMixer()
    output = transformer_model(test_input)
    print(f"✅ Output shape: {output.shape}")
    print(f"📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n🎯 All models working correctly!")
    print("Ready for training with improved architectures! 🚀")
