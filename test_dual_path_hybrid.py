#!/usr/bin/env python3
"""
Quick test of dual-path hybrid training
"""

import os
import torch
import torch.nn as nn
from pathlib import Path

# Simple test model
class SimpleDualPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)  # 10 mixing parameters
        
    def forward(self, x):
        # x shape: [batch, 1, time]
        x = self.conv(x)
        x = torch.mean(x, dim=-1)  # Global average pooling
        x = self.fc(x)
        return x

def main():
    print("🔥 Testing Dual-Path Hybrid Components")
    print("=" * 50)
    
    # Check dataset
    restoration_dir = os.path.join("data", "restoration")
    clean_dir = os.path.join(restoration_dir, "clean")
    distorted_dir = os.path.join(restoration_dir, "distorted")
    
    if not os.path.exists(clean_dir) or not os.path.exists(distorted_dir):
        print("❌ Restoration dataset not found!")
        return
    
    # Count files
    clean_files = list(Path(clean_dir).glob("*.wav"))
    print(f"📊 Found {len(clean_files)} clean files")
    
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    model = SimpleDualPath().to(device)
    print(f"🏗️ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    time_steps = 1000
    test_input = torch.randn(batch_size, 1, time_steps).to(device)
    
    output = model(test_input)
    print(f"✅ Forward pass successful: {test_input.shape} -> {output.shape}")
    
    print("🎯 Basic components working correctly!")
    return True

if __name__ == "__main__":
    main()
