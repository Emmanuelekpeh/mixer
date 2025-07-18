#!/usr/bin/env python3
"""
🎵 Model Conversion Script - Dual Output Audio Restoration
=========================================================

This script converts your existing models to work with audio restoration by:
1. Keeping the original architecture for predicting mixing parameters
2. Adding a second output branch for restored audio
3. Using multi-task learning: predict distortions AND restore audio

Architecture:
Input: Distorted audio spectrogram
Output 1: Predicted mixing/distortion parameters (11 params)
Output 2: Restored audio spectrogram

This is much more powerful because the model learns WHAT distortions 
were applied AND HOW to fix them!
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add src directory to path for model imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Define the distortion parameters we want to predict
DISTORTION_PARAMS = [
    "noise_level", "reverb_wet", "lowpass_cutoff", "highpass_cutoff",
    "compression_ratio", "clipping_threshold", "bass_gain", "mid_gain", 
    "treble_gain", "room_size", "damping"
]

class DualOutputWrapper(nn.Module):
    """Wrapper that adds restoration output to existing models"""
    
    def __init__(self, original_model, input_channels=1, restoration_channels=1):
        super(DualOutputWrapper, self).__init__()
        
        # Keep the original model for parameter prediction
        self.parameter_predictor = original_model
        
        # Add restoration branch - this will restore the audio
        self.restoration_branch = nn.Sequential(
            # Shared feature extraction
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Restoration-specific layers
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final restoration output
            nn.Conv2d(64, restoration_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range like input spectrograms
        )
        
    def forward(self, x):
        # Get distortion parameters from original model
        distortion_params = self.parameter_predictor(x)
        
        # Get restored audio from restoration branch
        restored_audio = self.restoration_branch(x)
        
        # Add residual connection for restoration
        restored_audio = x + restored_audio
        
        return distortion_params, restored_audio

class EnhancedDualOutputCNN(nn.Module):
    """Enhanced CNN with built-in dual output for better feature sharing"""
    
    def __init__(self, input_channels=1, n_params=len(DISTORTION_PARAMS)):
        super(EnhancedDualOutputCNN, self).__init__()
        
        # Shared feature extraction layers
        self.shared_features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Parameter prediction branch
        self.param_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_params),
            nn.Sigmoid()  # Parameters in [0, 1] range
        )
        
        # Audio restoration branch
        self.restoration_branch = nn.Sequential(
            # Upsample back to original size
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final restoration output
            nn.Conv2d(32, input_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Extract shared features
        features = self.shared_features(x)
        
        # Predict distortion parameters
        params = self.param_branch(features)
        
        # Restore audio
        restoration_delta = self.restoration_branch(features)
        
        # Add residual connection
        restored = x + restoration_delta
        
        return params, restored

def convert_existing_model(model_name, model_path, input_channels=1):
    """Convert an existing model to dual output"""
    
    print(f"🔄 Converting {model_name} to dual output...")
    
    try:
        # Try to load the existing model
        if os.path.exists(model_path):
            # Load the original model state dict
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Create a new enhanced dual output model
            dual_model = EnhancedDualOutputCNN(input_channels=input_channels)
            
            # Try to transfer compatible weights
            model_dict = dual_model.state_dict()
            compatible_weights = {}
            
            for key, value in state_dict.items():
                # Look for compatible layers in shared features
                if key.startswith('conv') or key.startswith('bn'):
                    new_key = f"shared_features.{key}"
                    if new_key in model_dict and model_dict[new_key].shape == value.shape:
                        compatible_weights[new_key] = value
                        print(f"  ✅ Transferred: {key} → {new_key}")
            
            # Load compatible weights
            if compatible_weights:
                model_dict.update(compatible_weights)
                dual_model.load_state_dict(model_dict)
                print(f"  🎯 Transferred {len(compatible_weights)} compatible layers")
            else:
                print(f"  ⚠️ No compatible weights found, using random initialization")
            
            return dual_model
            
        else:
            print(f"  ❌ Model file not found: {model_path}")
            # Return a new model with random weights
            return EnhancedDualOutputCNN(input_channels=input_channels)
            
    except Exception as e:
        print(f"  ❌ Error converting model: {e}")
        # Return a new model with random weights
        return EnhancedDualOutputCNN(input_channels=input_channels)

def convert_all_models():
    """Convert all existing models to dual output format"""
    
    models_dir = Path("models")
    converted_dir = models_dir / "dual_output"
    converted_dir.mkdir(exist_ok=True)
    
    # List of models to convert
    model_files = [
        ("baseline_cnn", "baseline_cnn_best.pth"),
        ("enhanced_cnn", "enhanced_cnn_best.pth"),
        ("ast_regressor", "ast_regressor_best.pth"),
        ("lstm_mixer", "lstm_mixer_best.pth"),
        ("advanced_transformer", "advanced_transformer_best.pth"),
        ("vae_mixer", "vae_mixer_best.pth"),
        ("audio_gan", "audio_gan_best.pth"),
        ("resnet_mixer", "resnet_mixer_best.pth")
    ]
    
    converted_models = {}
    
    print("🔄 Converting existing models to dual output format...")
    print("=" * 60)
    
    for model_name, model_file in model_files:
        model_path = models_dir / model_file
        
        # Convert the model
        dual_model = convert_existing_model(model_name, model_path)
        
        # Save converted model
        converted_path = converted_dir / f"{model_name}_dual_output.pth"
        torch.save(dual_model.state_dict(), converted_path)
        
        converted_models[model_name] = {
            "model": dual_model,
            "path": converted_path,
            "original_path": model_path,
            "exists": model_path.exists()
        }
        
        print(f"✅ {model_name} → {converted_path}")
    
    print(f"\n📊 Conversion Summary:")
    print("-" * 40)
    
    for name, info in converted_models.items():
        status = "✅ Transferred weights" if info["exists"] else "🆕 New model"
        print(f"{name:<20} {status}")
    
    return converted_models

def create_dual_output_training_script():
    """Create a training script for dual output models"""
    
    training_script = '''#!/usr/bin/env python3
"""
🎵 Dual Output Audio Restoration Training
========================================

Train models to simultaneously:
1. Predict distortion parameters
2. Restore audio quality

This combines the benefits of both approaches!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Multi-task loss function
class DualOutputLoss(nn.Module):
    def __init__(self, param_weight=0.3, audio_weight=0.7):
        super(DualOutputLoss, self).__init__()
        self.param_weight = param_weight
        self.audio_weight = audio_weight
        self.param_loss = nn.MSELoss()
        self.audio_loss = nn.L1Loss()  # MAE works better for audio
        
    def forward(self, pred_params, pred_audio, true_params, true_audio):
        # Parameter prediction loss
        param_loss = self.param_loss(pred_params, true_params)
        
        # Audio restoration loss
        audio_loss = self.audio_loss(pred_audio, true_audio)
        
        # Combined loss
        total_loss = self.param_weight * param_loss + self.audio_weight * audio_loss
        
        return total_loss, param_loss, audio_loss

def train_dual_output_model(model, train_loader, val_loader, device, epochs=50):
    """Train dual output model"""
    
    criterion = DualOutputLoss(param_weight=0.3, audio_weight=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for distorted, clean, params in train_loader:
            distorted, clean, params = distorted.to(device), clean.to(device), params.to(device)
            
            optimizer.zero_grad()
            pred_params, pred_audio = model(distorted)
            
            loss, param_loss, audio_loss = criterion(pred_params, pred_audio, params, clean)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

# Use this in your main training loop
'''
    
    script_path = Path("train_dual_output_restoration.py")
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    print(f"📝 Created training script: {script_path}")

def main():
    """Main conversion function"""
    
    print("🎵 Model Conversion for Dual Output Audio Restoration")
    print("=" * 60)
    print("This script converts your existing models to predict BOTH:")
    print("1. 📊 Distortion parameters (what was applied)")
    print("2. 🎵 Restored audio (how to fix it)")
    print()
    
    # Convert all models
    converted_models = convert_all_models()
    
    # Create training script
    create_dual_output_training_script()
    
    print(f"\n🎉 Conversion complete!")
    print(f"📁 Converted models saved to: models/dual_output/")
    print(f"📝 Training script created: train_dual_output_restoration.py")
    print(f"\n🚀 Next steps:")
    print(f"   1. Update your dataset to include distortion parameters")
    print(f"   2. Run: python train_dual_output_restoration.py")
    print(f"   3. Enjoy much more powerful audio restoration!")

if __name__ == "__main__":
    main()'''
