#!/usr/bin/env python3
"""
🎵 Audio Deconstruction & Restoration Model Converter
====================================================

This script converts existing mixer models into sophisticated audio deconstruction
and restoration models that learn to:

1. DECONSTRUCT audio into components:
   - Frequency bands (bass, mid, treble)
   - Dynamic components (transients vs sustain)
   - Harmonic vs noise content
   - Distortion parameters

2. RECONSTRUCT clean audio:
   - Process each component separately
   - Apply inverse distortion
   - Recombine intelligently

Architecture:
Input: Distorted Audio → Multi-Head Deconstruction → Component Processing → Reconstruction → Clean Audio
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add src directory to path for model imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Audio processing parameters
N_MELS = 128
FREQ_BANDS = 3  # Bass, Mid, Treble
DISTORTION_PARAMS = 7  # noise, reverb, lowpass, highpass, compression, clipping, eq

class AudioDeconstructionHead(nn.Module):
    """Multi-head deconstruction of audio into learnable components"""
    
    def __init__(self, input_features, n_freq_bands=FREQ_BANDS):
        super(AudioDeconstructionHead, self).__init__()
        
        self.n_freq_bands = n_freq_bands
        
        # Frequency band separation head
        self.freq_separator = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, n_freq_bands, kernel_size=3, padding=1),
            nn.Sigmoid()  # Soft masks for each frequency band
        )
        
        # Dynamic component analysis head (transients vs sustain)
        self.dynamic_analyzer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3)),  # Temporal analysis
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),  # Transient vs Sustain
            nn.Sigmoid()
        )
        
        # Harmonic vs noise separation head
        self.harmonic_separator = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),  # Harmonic vs Noise
            nn.Sigmoid()
        )
        
        # Distortion parameter detection head
        self.distortion_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, DISTORTION_PARAMS),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Deconstruct audio into multiple components
        
        Args:
            x: Input spectrogram [B, 1, n_mels, time]
            
        Returns:
            components: Dictionary of deconstructed components
        """
        # Frequency band separation
        freq_masks = self.freq_separator(x)  # [B, n_bands, n_mels, time]
        freq_components = []
        for i in range(self.n_freq_bands):
            component = x * freq_masks[:, i:i+1, :, :]
            freq_components.append(component)
        
        # Dynamic component analysis
        dynamic_masks = self.dynamic_analyzer(x)  # [B, 2, n_mels, time]
        transient_component = x * dynamic_masks[:, 0:1, :, :]
        sustain_component = x * dynamic_masks[:, 1:2, :, :]
        
        # Harmonic vs noise separation
        harmonic_masks = self.harmonic_separator(x)  # [B, 2, n_mels, time]
        harmonic_component = x * harmonic_masks[:, 0:1, :, :]
        noise_component = x * harmonic_masks[:, 1:2, :, :]
        
        # Distortion parameter detection
        distortion_params = self.distortion_detector(x)  # [B, distortion_params]
        
        return {
            'freq_components': freq_components,  # List of [B, 1, n_mels, time]
            'transient': transient_component,    # [B, 1, n_mels, time]
            'sustain': sustain_component,        # [B, 1, n_mels, time]
            'harmonic': harmonic_component,      # [B, 1, n_mels, time]
            'noise': noise_component,            # [B, 1, n_mels, time]
            'distortion_params': distortion_params  # [B, distortion_params]
        }

class ComponentProcessor(nn.Module):
    """Process individual audio components for restoration"""
    
    def __init__(self, component_channels=1):
        super(ComponentProcessor, self).__init__()
        
        # Lightweight processing for each component
        self.processor = nn.Sequential(
            nn.Conv2d(component_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, component_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, component):
        """Process a single component"""
        # Add residual connection for stability
        processed = self.processor(component)
        return component + processed * 0.5  # Gentle processing

class AudioReconstructionHead(nn.Module):
    """Intelligently recombine processed components into clean audio"""
    
    def __init__(self, n_freq_bands=FREQ_BANDS):
        super(AudioReconstructionHead, self).__init__()
        
        self.n_freq_bands = n_freq_bands
        
        # Component combination network
        total_components = n_freq_bands + 4  # freq_bands + transient + sustain + harmonic + noise
        
        self.combiner = nn.Sequential(
            nn.Conv2d(total_components, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        
        # Attention mechanism for component weighting
        self.attention = nn.Sequential(
            nn.Conv2d(total_components, total_components, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, processed_components):
        """
        Recombine processed components into clean audio
        
        Args:
            processed_components: Dictionary of processed components
            
        Returns:
            restored_audio: Clean audio spectrogram [B, 1, n_mels, time]
        """
        # Stack all components
        components_list = (
            processed_components['freq_components'] +
            [processed_components['transient']] +
            [processed_components['sustain']] +
            [processed_components['harmonic']] +
            [processed_components['noise']]
        )
        
        stacked_components = torch.cat(components_list, dim=1)  # [B, total_components, n_mels, time]
        
        # Apply attention weighting
        attention_weights = self.attention(stacked_components)
        weighted_components = stacked_components * attention_weights
        
        # Combine into final output
        restored_audio = self.combiner(weighted_components)
        
        return restored_audio

class DeconstructionRestorationWrapper(nn.Module):
    """Wrapper that converts any existing model into a deconstruction+restoration model"""
    
    def __init__(self, base_model, input_shape=(1, N_MELS, 216)):
        super(DeconstructionRestorationWrapper, self).__init__()
        
        self.base_model = base_model
        
        # Add deconstruction head
        self.deconstruction_head = AudioDeconstructionHead(input_shape[0])
        
        # Component processors
        self.freq_processors = nn.ModuleList([
            ComponentProcessor() for _ in range(FREQ_BANDS)
        ])
        self.transient_processor = ComponentProcessor()
        self.sustain_processor = ComponentProcessor()
        self.harmonic_processor = ComponentProcessor()
        self.noise_processor = ComponentProcessor()
        
        # Reconstruction head
        self.reconstruction_head = AudioReconstructionHead()
        
        # Distortion parameter prediction (using base model features)
        self.distortion_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, DISTORTION_PARAMS),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Full deconstruction and restoration pipeline
        
        Args:
            x: Distorted audio spectrogram [B, 1, n_mels, time]
            
        Returns:
            outputs: Dictionary containing:
                - distortion_params: Predicted distortion parameters
                - restored_audio: Restored audio spectrogram
                - components: Intermediate components (for analysis)
        """
        # Step 1: Deconstruct audio into components
        components = self.deconstruction_head(x)
        
        # Step 2: Process each component individually
        processed_components = {}
        
        # Process frequency components
        processed_freq_components = []
        for i, freq_component in enumerate(components['freq_components']):
            processed_freq = self.freq_processors[i](freq_component)
            processed_freq_components.append(processed_freq)
        processed_components['freq_components'] = processed_freq_components
        
        # Process dynamic components
        processed_components['transient'] = self.transient_processor(components['transient'])
        processed_components['sustain'] = self.sustain_processor(components['sustain'])
        
        # Process harmonic vs noise components
        processed_components['harmonic'] = self.harmonic_processor(components['harmonic'])
        processed_components['noise'] = self.noise_processor(components['noise'])
        
        # Step 3: Use base model for additional feature extraction
        base_features = None
        try:
            # Try to get intermediate features from base model
            if hasattr(self.base_model, 'conv1'):
                base_features = self.base_model.conv1(x)
            elif hasattr(self.base_model, 'enc1'):
                base_features = self.base_model.enc1(x)
            else:
                # Fallback: use input
                base_features = x
        except:
            base_features = x
        
        # Step 4: Predict distortion parameters
        distortion_params = self.distortion_predictor(base_features)
        
        # Step 5: Reconstruct clean audio
        restored_audio = self.reconstruction_head(processed_components)
        
        return {
            'distortion_params': distortion_params,
            'restored_audio': restored_audio,
            'components': components,  # For analysis/visualization
            'processed_components': processed_components
        }

def convert_model_to_deconstruction(model, model_name):
    """Convert an existing model to deconstruction+restoration model"""
    
    print(f"🔧 Converting {model_name} to deconstruction+restoration model...")
    
    # Wrap the existing model
    converted_model = DeconstructionRestorationWrapper(model)
    
    # Count parameters
    original_params = sum(p.numel() for p in model.parameters())
    total_params = sum(p.numel() for p in converted_model.parameters())
    added_params = total_params - original_params
    
    print(f"📊 Original model parameters: {original_params:,}")
    print(f"📊 Added parameters: {added_params:,}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📈 Parameter increase: {(added_params/original_params)*100:.1f}%")
    
    return converted_model

def main():
    """Convert all existing models to deconstruction+restoration models"""
    
    print("🎵 Audio Deconstruction & Restoration Model Converter")
    print("=" * 60)
    
    # Import existing models
    models_to_convert = []
    
    try:
        from train_mixer_pipeline_ultra_fixed import (
            BaselineCNN, EnhancedCNN, ASTRegressor
        )
        models_to_convert.extend([
            ("baseline_cnn", BaselineCNN(input_channels=1, output_dim=11)),
            ("enhanced_cnn", EnhancedCNN(input_channels=1, output_dim=11)),
            ("ast_regressor", ASTRegressor(input_channels=1, output_dim=11))
        ])
        print("✅ Successfully imported basic CNN models")
    except Exception as e:
        print(f"⚠️ Could not import basic models: {e}")
    
    # Try to import advanced models
    try:
        from lstm_mixer import LSTMAudioMixer
        models_to_convert.append(
            ("lstm_mixer", LSTMAudioMixer(n_mels=N_MELS, n_outputs=11))
        )
        print("✅ Successfully imported LSTM model")
    except Exception as e:
        print(f"⚠️ Could not import LSTM model: {e}")
    
    try:
        from advanced_transformer import AdvancedTransformerMixer
        models_to_convert.append(
            ("transformer_mixer", AdvancedTransformerMixer(n_mels=N_MELS, n_outputs=11))
        )
        print("✅ Successfully imported Transformer model")
    except Exception as e:
        print(f"⚠️ Could not import Transformer model: {e}")
    
    try:
        from vae_mixer import VAEAudioMixer
        models_to_convert.append(
            ("vae_mixer", VAEAudioMixer(n_mels=N_MELS, n_outputs=11))
        )
        print("✅ Successfully imported VAE model")
    except Exception as e:
        print(f"⚠️ Could not import VAE model: {e}")
    
    try:
        from resnet_mixer import ResNetAudioMixer, SpectralResidualBlock
        if SpectralResidualBlock is not None:
            models_to_convert.append(
                ("resnet_mixer", ResNetAudioMixer(
                    block=SpectralResidualBlock,
                    layers=[2, 2, 2, 2],
                    n_outputs=11,
                    n_mels=N_MELS
                ))
            )
            print("✅ Successfully imported ResNet model")
    except Exception as e:
        print(f"⚠️ Could not import ResNet model: {e}")
    
    if not models_to_convert:
        print("❌ No models found to convert!")
        return
    
    print(f"\n🔧 Converting {len(models_to_convert)} models...")
    
    # Convert each model
    converted_models = {}
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for model_name, model in models_to_convert:
        try:
            # Convert model
            converted_model = convert_model_to_deconstruction(model, model_name)
            converted_models[model_name] = converted_model
            
            # Save converted model architecture (not weights yet)
            save_path = models_dir / f"{model_name}_deconstruction_architecture.pth"
            torch.save(converted_model.state_dict(), save_path)
            print(f"✅ Saved {model_name} architecture to {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to convert {model_name}: {e}")
            continue
    
    print(f"\n🎉 Successfully converted {len(converted_models)} models!")
    print("\n📋 Converted Models:")
    print("-" * 40)
    
    for model_name, model in converted_models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ {model_name}: {param_count:,} parameters")
    
    print("\n🎯 What these models can now do:")
    print("  1. 🔍 Deconstruct audio into frequency bands")
    print("  2. ⚡ Separate transients from sustained sounds")
    print("  3. 🎵 Distinguish harmonic content from noise")
    print("  4. 📊 Predict applied distortion parameters")
    print("  5. 🔧 Process each component separately")
    print("  6. 🎨 Intelligently reconstruct clean audio")
    
    print("\n🚀 Next steps:")
    print("  1. Use these models in the new training pipeline")
    print("  2. Train on the restoration dataset")
    print("  3. Evaluate both parameter prediction AND audio quality")
    
    return converted_models

if __name__ == "__main__":
    main()
