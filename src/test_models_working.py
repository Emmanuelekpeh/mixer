#!/usr/bin/env python3
"""
🧪 Working Model Test Suite
==========================

Test all 5 new AI model architectures with correct input shapes.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging

# Import our models
from lstm_mixer import LSTMAudioMixer
from audio_gan import AudioGANMixer
from vae_mixer import VAEAudioMixer
from advanced_transformer import AdvancedTransformerMixer
from resnet_mixer import ResNetAudioMixer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_all_models():
    """Test all model architectures with correct input shapes."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Testing on device: {device}")
    print("=" * 60)
      # Define models and their expected input shapes
    models_to_test = [
        ('LSTM Audio Mixer', LSTMAudioMixer(), (2, 128, 250)),        # 3D
        ('Audio GAN Mixer', AudioGANMixer(), (2, 128, 250)),         # 3D  
        ('VAE Audio Mixer', VAEAudioMixer(), (2, 128, 250)),         # 3D
        ('Advanced Transformer Mixer', AdvancedTransformerMixer(), (2, 128, 1000)), # 3D - needs 1000 time steps
        ('ResNet Audio Mixer', ResNetAudioMixer(), (2, 128, 250))    # 3D
    ]
    
    successful_models = []
    results = []
    
    for model_name, model, input_shape in models_to_test:
        try:
            print(f"\n🧪 Testing {model_name}...")
            
            # Move model to device
            model = model.to(device)
            model.eval()
            
            # Create test input
            test_input = torch.randn(*input_shape, device=device)
              # Time the inference
            start_time = time.time()
            with torch.no_grad():
                model_output = model(test_input)
                
                # Handle different return types
                if isinstance(model_output, tuple):
                    output = model_output[0]  # Take first element (the actual output)
                else:
                    output = model_output
                    
            inference_time = time.time() - start_time
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"   ✅ Success!")
            print(f"   📊 Input shape: {test_input.shape}")
            print(f"   📊 Output shape: {output.shape}")
            print(f"   ⚡ Parameters: {param_count:,}")
            print(f"   ⏱️ Inference time: {inference_time:.4f}s")
            
            # Validate output
            if len(output.shape) == 2 and output.shape[1] == 10:
                print(f"   ✅ Output format valid")
            else:
                print(f"   ⚠️ Unexpected output shape")
            
            successful_models.append(model_name)
            results.append({
                'name': model_name,
                'success': True,
                'input_shape': input_shape,
                'output_shape': list(output.shape),
                'parameter_count': param_count,
                'inference_time': inference_time
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'name': model_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    total_models = len(models_to_test)
    success_count = len(successful_models)
    success_rate = (success_count / total_models) * 100
    
    print(f"\n🎯 Results:")
    print(f"   Total models: {total_models}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {total_models - success_count}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    if successful_models:
        print(f"\n✅ Working Models:")
        # Sort by inference time
        successful_results = [r for r in results if r['success']]
        successful_results.sort(key=lambda x: x['inference_time'])
        
        for i, result in enumerate(successful_results, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            print(f"   {status} {result['name']}:")
            print(f"       • Parameters: {result['parameter_count']:,}")
            print(f"       • Speed: {result['inference_time']:.4f}s")
    
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print(f"\n❌ Failed Models:")
        for result in failed_results:
            print(f"   • {result['name']}: {result['error']}")
    
    print(f"\n🎯 Status:")
    if success_rate >= 80:
        print(f"   ✅ Excellent! Models ready for training")
        print(f"   🚀 Next: Run training pipeline")
    elif success_rate >= 60:
        print(f"   ⚠️ Most models working, some issues")
        print(f"   🔧 Fix remaining issues")
    else:
        print(f"   ❌ Multiple failures detected")
        print(f"   🔧 Major fixes needed")

if __name__ == "__main__":
    test_all_models()
