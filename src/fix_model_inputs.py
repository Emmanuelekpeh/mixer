#!/usr/bin/env python3
"""
🔧 Fix Model Input Shapes
========================

The models were designed to work with their individual test scripts but need
input shape adaptation for consistent testing. This script fixes the input
handling for all models.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

# Import our models
from lstm_mixer import LSTMAudioMixer
from audio_gan import AudioGANMixer
from vae_mixer import VAEAudioMixer
from advanced_transformer import AdvancedTransformerMixer
from resnet_mixer import ResNetAudioMixer

logger = logging.getLogger(__name__)

def test_individual_models():
    """Test each model with its expected input format."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🔧 Testing Individual Model Input Shapes")
    print("=" * 50)
    
    # Test 1: LSTM Audio Mixer
    print("\n🧪 Testing LSTM Audio Mixer...")
    try:
        model = LSTMAudioMixer().to(device)
        # LSTM expects: (batch_size, channels, time_steps) for 1D conv
        # So input should be (batch_size, n_mels, time_steps)
        test_input = torch.randn(2, 128, 250, device=device)  # Remove channel dimension
        output = model(test_input)
        print(f"   ✅ LSTM success! Input: {test_input.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"   ❌ LSTM failed: {e}")
      # Test 2: Audio GAN
    print("\n🧪 Testing Audio GAN...")
    try:
        model = AudioGANMixer().to(device)
        # GAN expects: (batch_size, n_mels, time_steps) - it adds channel dim internally
        test_input = torch.randn(2, 128, 250, device=device)  # 3D tensor
        output = model(test_input)
        print(f"   ✅ GAN success! Input: {test_input.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"   ❌ GAN failed: {e}")
    
    # Test 3: VAE Audio Mixer
    print("\n🧪 Testing VAE Audio Mixer...")
    try:
        model = VAEAudioMixer().to(device)
        # VAE expects: (batch_size, n_mels, time_steps) - it adds channel dim internally
        test_input = torch.randn(2, 128, 250, device=device)  # 3D tensor
        output = model(test_input)
        print(f"   ✅ VAE success! Input: {test_input.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"   ❌ VAE failed: {e}")
    
    # Test 4: Advanced Transformer
    print("\n🧪 Testing Advanced Transformer...")
    try:
        model = AdvancedTransformerMixer().to(device)
        # Transformer might expect flattened or sequence data
        # Let's try different shapes
        shapes_to_try = [
            (2, 128, 250),      # (batch, height, width)
            (2, 1, 128, 250),   # (batch, channels, height, width)
            (2, 32000),         # (batch, flattened)
            (2, 250, 128),      # (batch, time, features)
        ]
        
        success = False
        for shape in shapes_to_try:
            try:
                test_input = torch.randn(*shape, device=device)
                output = model(test_input)
                print(f"   ✅ Transformer success! Input: {test_input.shape}, Output: {output.shape}")
                success = True
                break
            except Exception as e:
                continue
        
        if not success:
            print(f"   ❌ Transformer failed with all input shapes")
            
    except Exception as e:
        print(f"   ❌ Transformer failed: {e}")
      # Test 5: ResNet Audio Mixer
    print("\n🧪 Testing ResNet Audio Mixer...")
    try:
        model = ResNetAudioMixer().to(device)
        # ResNet expects: (batch_size, n_mels, time_steps) - it adds channel dim internally
        test_input = torch.randn(2, 128, 250, device=device)  # 3D tensor
        output = model(test_input)
        print(f"   ✅ ResNet success! Input: {test_input.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"   ❌ ResNet failed: {e}")

def create_input_adapter():
    """Create a universal input adapter for consistent testing."""
    
    class ModelInputAdapter:
        """Adapter to handle different input requirements for each model."""
        
        def __init__(self, device):
            self.device = device
              def prepare_input_for_model(self, model_name: str, batch_size: int = 2, 
                                  n_mels: int = 128, time_steps: int = 250):
            """Prepare the correct input shape for each model."""
            
            if 'LSTM' in model_name:
                # LSTM expects (batch_size, n_mels, time_steps)
                return torch.randn(batch_size, n_mels, time_steps, device=self.device)
            
            elif 'Transformer' in model_name:
                # Transformer expects (batch_size, time_steps, n_mels) - sequence format
                return torch.randn(batch_size, time_steps, n_mels, device=self.device)
            
            else:
                # GAN, VAE, ResNet expect (batch_size, n_mels, time_steps) - they add channel dim internally
                return torch.randn(batch_size, n_mels, time_steps, device=self.device)
    
    return ModelInputAdapter

def test_with_adapter():
    """Test all models using the input adapter."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter = create_input_adapter()(device)
    
    print("\n🔧 Testing with Input Adapter")
    print("=" * 50)
    
    models_to_test = [
        ('LSTM Audio Mixer', LSTMAudioMixer()),
        ('Audio GAN Mixer', AudioGANMixer()),
        ('VAE Audio Mixer', VAEAudioMixer()),
        ('Advanced Transformer Mixer', AdvancedTransformerMixer()),
        ('ResNet Audio Mixer', ResNetAudioMixer())
    ]
    
    successful_models = []
    
    for model_name, model in models_to_test:
        try:
            print(f"\n🧪 Testing {model_name}...")
            model = model.to(device)
            
            # Get correct input shape
            test_input = adapter.prepare_input_for_model(model_name)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            print(f"   ✅ Success! Input: {test_input.shape}, Output: {output.shape}")
            successful_models.append(model_name)
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Successful models: {len(successful_models)}/{len(models_to_test)}")
    print(f"   Success rate: {len(successful_models)/len(models_to_test)*100:.1f}%")
    
    if successful_models:
        print(f"\n✅ Working models:")
        for model_name in successful_models:
            print(f"   • {model_name}")

if __name__ == "__main__":
    # Test individual models first
    test_individual_models()
    
    # Test with adapter
    test_with_adapter()
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Update the comprehensive test script with correct input shapes")
    print(f"   2. Fix any remaining model issues")
    print(f"   3. Proceed with training pipeline")
