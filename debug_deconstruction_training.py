#!/usr/bin/env python3
"""
Debug Deconstruction Training - Find and fix dimension issues
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path

# Import the dataset class
import sys
sys.path.append(os.getcwd())
from train_deconstruction_models import DeconstructionRestorationDataset, audio_to_spectrogram

# Audio parameters
SAMPLE_RATE = 22050
N_MELS = 128
CHUNK_DURATION = 5.0

def debug_tensor_shapes():
    """Debug tensor shapes throughout the pipeline"""
    print("🔍 Debugging Tensor Shapes")
    print("=" * 40)
    
    # Create a sample dataset
    clean_dir = "data/restoration/clean"
    distorted_dir = "data/restoration/distorted"
    
    if not os.path.exists(clean_dir) or not os.path.exists(distorted_dir):
        print("❌ Restoration dataset not found!")
        return
    
    # Create dataset
    dataset = DeconstructionRestorationDataset(clean_dir, distorted_dir)
    
    if len(dataset) == 0:
        print("❌ No data found!")
        return
    
    # Get one sample
    distorted_spec, clean_spec, distortion_params = dataset[0]
    
    print(f"📊 Input shapes:")
    print(f"  Distorted spec: {distorted_spec.shape}")
    print(f"  Clean spec: {clean_spec.shape}")
    print(f"  Distortion params: {distortion_params.shape}")
    
    # Test with batch
    batch_size = 2
    distorted_batch = torch.stack([distorted_spec, distorted_spec])
    clean_batch = torch.stack([clean_spec, clean_spec])
    params_batch = torch.stack([distortion_params, distortion_params])
    
    print(f"\n📊 Batch shapes:")
    print(f"  Distorted batch: {distorted_batch.shape}")
    print(f"  Clean batch: {clean_batch.shape}")
    print(f"  Params batch: {params_batch.shape}")
    
    return distorted_batch, clean_batch, params_batch

def create_simple_deconstruction_model():
    """Create a simplified deconstruction model for debugging"""
    
    class SimpleDeconstruction(nn.Module):
        def __init__(self):
            super().__init__()
            # Simple CNN backbone
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((8, 8))
            
            # Distortion parameter prediction
            self.distortion_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 7),  # 7 distortion parameters
                nn.Sigmoid()
            )
            
            # Simple restoration head
            self.restoration_head = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, padding=1),
                nn.Tanh()
            )
            
        def forward(self, x):
            # Input: [B, 1, n_mels, time]
            print(f"Input shape: {x.shape}")
            
            # Feature extraction
            x1 = F.relu(self.conv1(x))
            print(f"After conv1: {x1.shape}")
            
            x2 = F.relu(self.conv2(x1))
            print(f"After conv2: {x2.shape}")
            
            # For distortion prediction
            pooled = self.pool(x2)
            print(f"After pool: {pooled.shape}")
            
            distortion_params = self.distortion_head(pooled)
            print(f"Distortion params: {distortion_params.shape}")
            
            # For restoration - need to match input size
            # Upsample back to original size
            restored = F.interpolate(x2, size=x.shape[2:], mode='bilinear', align_corners=False)
            restored = self.restoration_head(restored)
            print(f"Restored audio: {restored.shape}")
            
            return {
                'distortion_params': distortion_params,
                'restored_audio': restored,
                'components': {'features': x2}  # Simplified components
            }
    
    return SimpleDeconstruction()

def test_model_forward():
    """Test the model forward pass"""
    print("\n🧪 Testing Model Forward Pass")
    print("=" * 40)
    
    # Get sample data
    distorted_batch, clean_batch, params_batch = debug_tensor_shapes()
    
    # Create model
    model = create_simple_deconstruction_model()
    
    # Test forward pass
    try:
        print("\n🔄 Running forward pass...")
        with torch.no_grad():
            outputs = model(distorted_batch)
        
        print("\n✅ Forward pass successful!")
        print("📊 Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: {v.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation"""
    print("\n💡 Testing Loss Computation")
    print("=" * 40)
    
    # Get sample data
    distorted_batch, clean_batch, params_batch = debug_tensor_shapes()
    
    # Create model
    model = create_simple_deconstruction_model()
    
    try:
        with torch.no_grad():
            outputs = model(distorted_batch)
        
        # Test different loss functions
        pred_distortion = outputs['distortion_params']
        restored_audio = outputs['restored_audio']
        
        print(f"Predicted distortion: {pred_distortion.shape}")
        print(f"Target distortion: {params_batch.shape}")
        print(f"Restored audio: {restored_audio.shape}")
        print(f"Target audio: {clean_batch.shape}")
        
        # MSE loss for distortion params
        distortion_loss = F.mse_loss(pred_distortion, params_batch)
        print(f"✅ Distortion loss: {distortion_loss.item():.4f}")
        
        # L1 loss for audio restoration
        restoration_loss = F.l1_loss(restored_audio, clean_batch)
        print(f"✅ Restoration loss: {restoration_loss.item():.4f}")
        
        total_loss = distortion_loss + 10.0 * restoration_loss
        print(f"✅ Total loss: {total_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debugging function"""
    print("🐛 Deconstruction Training Debug")
    print("=" * 50)
    
    # Step 1: Debug tensor shapes
    try:
        debug_tensor_shapes()
    except Exception as e:
        print(f"❌ Shape debugging failed: {e}")
        return
    
    # Step 2: Test model forward pass
    if not test_model_forward():
        print("❌ Model forward pass failed - stopping here")
        return
    
    # Step 3: Test loss computation
    if not test_loss_computation():
        print("❌ Loss computation failed - stopping here")
        return
    
    print("\n🎉 All debugging tests passed!")
    print("✅ Ready to modify the main training script")

if __name__ == "__main__":
    main()
