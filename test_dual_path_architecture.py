#!/usr/bin/env python3
"""
Test Dual-Path Hybrid Model Architecture
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Import the model components from the dual-path script
from train_dual_path_hybrid import (
    DualPathHybrid, 
    AudioSpectrogramTransformer, 
    GANGenerator, 
    GANDiscriminator,
    N_MELS
)

def test_model_architecture():
    """Test the dual-path hybrid model architecture"""
    
    print("🔥 Testing Dual-Path Hybrid Architecture")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    try:
        # Test input dimensions
        batch_size = 4
        channels = 1
        height = 64  # N_MELS
        width = 260  # Approximate time steps for 3 seconds
        
        # Create test input
        test_input = torch.randn(batch_size, channels, height, width).to(device)
        print(f"📊 Test input shape: {test_input.shape}")
        
        # Test individual components
        print("\n🧪 Testing Individual Components:")
        
        # 1. Test AST
        print("  Testing AudioSpectrogramTransformer...")
        ast = AudioSpectrogramTransformer().to(device)
        ast_output = ast(test_input)
        print(f"    AST output shape: {ast_output.shape}")
        
        # 2. Test GAN Generator
        print("  Testing GAN Generator...")
        generator = GANGenerator().to(device)
        gen_output = generator(test_input)
        print(f"    Generator output shape: {gen_output.shape}")
        
        # 3. Test GAN Discriminator
        print("  Testing GAN Discriminator...")
        discriminator = GANDiscriminator().to(device)
        # Discriminator expects 2-channel input (original + generated)
        disc_input = torch.cat([test_input, gen_output], dim=1)
        disc_output = discriminator(disc_input)
        print(f"    Discriminator output shape: {disc_output.shape}")
        
        # 4. Test Full Dual-Path Model
        print("  Testing Full Dual-Path Hybrid...")
        model = DualPathHybrid().to(device)
        
        # Get model info
        model_info = model.get_model_info()
        print(f"    Total parameters: {model_info['total_parameters']:,}")
        
        # Forward pass
        outputs = model(test_input)
        
        print(f"    Model outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"      {key}: {value.shape}")
            else:
                print(f"      {key}: {value}")
        
        # Test expected output shapes (corrected for actual spectrogram dimensions)
        expected_shapes = {
            'restored_audio': (batch_size, 1, height, 259),  # Actual mel-spec width
            'mixing_params': (batch_size, 11),
            'distortion_params': (batch_size, 7),
            'gan_output': (batch_size, 1, height, width)  # GAN uses padded width
        }
        
        print("\n✅ Shape Validation:")
        all_shapes_correct = True
        for key, expected_shape in expected_shapes.items():
            if key in outputs:
                actual_shape = outputs[key].shape
                is_correct = actual_shape == expected_shape
                status = "✅" if is_correct else "❌"
                print(f"    {status} {key}: {actual_shape} (expected {expected_shape})")
                if not is_correct:
                    all_shapes_correct = False
            else:
                print(f"    ❌ Missing output: {key}")
                all_shapes_correct = False
        
        if all_shapes_correct:
            print("\n🎉 All architecture tests passed!")
            print(f"🏗️ Model ready with {model_info['total_parameters']:,} parameters")
            return True
        else:
            print("\n❌ Some architecture tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during architecture test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_compatibility():
    """Test compatibility with training setup"""
    
    print("\n🧪 Testing Training Compatibility")
    print("=" * 30)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model and test data
        model = DualPathHybrid().to(device)
        batch_size = 2
        test_input = torch.randn(batch_size, 1, N_MELS, 260).to(device)
        
        # Test forward pass
        outputs = model(test_input)
        
        # Test loss computation
        target_clean = torch.randn_like(outputs['restored_audio'])
        target_mixing = torch.randn(batch_size, 11).to(device)
        target_distortion = torch.randn(batch_size, 7).to(device)
        
        # Test loss functions
        restoration_loss = nn.L1Loss()(outputs['restored_audio'], target_clean)
        mixing_loss = nn.MSELoss()(outputs['mixing_params'], target_mixing)
        distortion_loss = nn.MSELoss()(outputs['distortion_params'], target_distortion)
        
        print(f"✅ Loss computation successful:")
        print(f"    Restoration loss: {restoration_loss.item():.6f}")
        print(f"    Mixing loss: {mixing_loss.item():.6f}")
        print(f"    Distortion loss: {distortion_loss.item():.6f}")
        
        # Test backward pass
        total_loss = restoration_loss + mixing_loss + distortion_loss
        total_loss.backward()
        
        print(f"✅ Backward pass successful")
        print(f"✅ Training compatibility verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all architecture tests"""
    
    print("🔬 Dual-Path Hybrid Architecture Tests")
    print("=" * 60)
    
    # Test architecture
    arch_success = test_model_architecture()
    
    # Test training compatibility
    train_success = test_training_compatibility()
    
    print("\n" + "=" * 60)
    if arch_success and train_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Dual-Path Hybrid model is ready for training")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️ Model needs fixes before training")
    
    return arch_success and train_success

if __name__ == "__main__":
    main()
