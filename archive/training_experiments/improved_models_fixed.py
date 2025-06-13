#!/usr/bin/env python3
"""
🚀 Enhanced AI Mixing Models with Advanced Architectures
========================================================

Working improved models with safe constraints and better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedEnhancedCNN(nn.Module):
    """Enhanced CNN with safe parameter constraints and better architecture."""
    
    def __init__(self, n_outputs=10, dropout=0.3):
        super().__init__()
        
        # Feature extraction with residual connections
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, n_outputs)
        )
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        # Global pooling
        avg_pool = self.avgpool(x).view(x.size(0), -1)
        max_pool = self.maxpool(x).view(x.size(0), -1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classification
        output = self.classifier(features)
        
        # Apply safe constraints
        return self._apply_safe_constraints(output)
    
    def _apply_safe_constraints(self, x):
        """Apply constraints to ensure safe mixing parameters."""
        constrained = torch.zeros_like(x)
        
        # Input Gain: 0.4-1.1 (safer range)
        constrained[:, 0] = torch.clamp(torch.sigmoid(x[:, 0]) * 0.7 + 0.4, 0.4, 1.1)
        
        # Compression Ratio: 0.0-0.6 (prevent over-compression)
        constrained[:, 1] = torch.clamp(torch.sigmoid(x[:, 1]) * 0.6, 0.0, 0.6)
        
        # EQ parameters: 0.3-0.7 (moderate EQ changes)
        for i in range(2, 6):  # High, Mid, Low, Presence
            constrained[:, i] = torch.clamp(torch.sigmoid(x[:, i]) * 0.4 + 0.3, 0.3, 0.7)
        
        # Reverb Send: 0.1-0.7 (prevent excessive reverb)
        constrained[:, 6] = torch.clamp(torch.sigmoid(x[:, 6]) * 0.6 + 0.1, 0.1, 0.7)
        
        # Delay Send: 0.0-0.25 (subtle delay)
        constrained[:, 7] = torch.clamp(torch.sigmoid(x[:, 7]) * 0.25, 0.0, 0.25)
        
        # Stereo Width: 0.5-0.8 (safe stereo range)
        constrained[:, 8] = torch.clamp(torch.sigmoid(x[:, 8]) * 0.3 + 0.5, 0.5, 0.8)
        
        # Output Level: 0.6-0.9 (prevent clipping)
        constrained[:, 9] = torch.clamp(torch.sigmoid(x[:, 9]) * 0.3 + 0.6, 0.6, 0.9)
        
        return constrained

class MultiScaleTransformerMixer(nn.Module):
    """Simplified transformer-based mixer."""
    
    def __init__(self, n_outputs=10, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
            nn.Sigmoid()  # Constrain to [0,1]
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)  # [B, d_model, 1, T]
        features = features.squeeze(2).transpose(1, 2)  # [B, T, d_model]
        
        # Transformer processing
        features = self.transformer(features)
        
        # Global pooling
        global_features = torch.mean(features, dim=1)
        
        # Classification
        return self.classifier(global_features)

if __name__ == "__main__":
    # Test the models
    print("🧪 Testing Improved Models...")
    
    test_input = torch.randn(2, 1, 128, 1000)
    
    # Test ImprovedEnhancedCNN
    model1 = ImprovedEnhancedCNN()
    output1 = model1(test_input)
    print(f"✅ ImprovedEnhancedCNN: {output1.shape}, range: [{output1.min():.3f}, {output1.max():.3f}]")
    
    # Test MultiScaleTransformerMixer
    model2 = MultiScaleTransformerMixer()
    output2 = model2(test_input)
    print(f"✅ MultiScaleTransformerMixer: {output2.shape}, range: [{output2.min():.3f}, {output2.max():.3f}]")
    
    print("🎯 Models ready for training!")
