#!/usr/bin/env python3
"""
🏗️ Debug Version - New Architecture Training Pipeline
====================================

Debugging version to identify shape issues when training the 5 models:
- LSTM Audio Mixer
- Audio GAN Mixer
- VAE Audio Mixer
- Advanced Transformer
- ResNet Audio Mixer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
import logging
from datetime import datetime
import sys
import traceback
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our models
from baseline_cnn import DEVICE, N_OUTPUTS
from fixed_dataset import FixedSpectrogramDataset
from lstm_mixer import LSTMAudioMixer
from audio_gan import AudioGANMixer
from vae_mixer import VAEAudioMixer
from advanced_transformer import AdvancedTransformerMixer
from resnet_mixer import ResNetAudioMixer

def train_debug():
    """Debug training to identify shape issues."""
    logger.info("🔍 Starting Debug Training for Shape Issues")
    logger.info("=" * 60)
      # Load datasets with smaller batch size
    data_dir = Path(r"C:\Users\emman\Projects\mixer\data")
    train_dataset = FixedSpectrogramDataset(data_dir / "train", targets_file=data_dir / "targets_example.json", augment=True)
    
    # Create a smaller batch size for debugging
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Get a single batch for testing
    logger.info("Fetching a single batch for shape analysis...")
    for batch_idx, (data, target) in enumerate(train_loader):
        logger.info(f"Batch shape: Input={data.shape}, Target={target.shape}")
        
        # Test each model with this batch
        models_to_test = [
            ('LSTM Audio Mixer', LSTMAudioMixer()),
            ('Audio GAN Mixer', AudioGANMixer()),
            ('VAE Audio Mixer', VAEAudioMixer()),
            ('Advanced Transformer Mixer', AdvancedTransformerMixer()),
            ('ResNet Audio Mixer', ResNetAudioMixer())
        ]
        
        for model_name, model in models_to_test:
            logger.info(f"\nTesting {model_name}...")
            try:                # Move to device
                model = model.to(DEVICE)
                data = data.to(DEVICE)
                target = target.to(DEVICE)
                
                # Reshape input based on model architecture
                if model_name == 'LSTM Audio Mixer':
                    # LSTM expects [batch, channels, features, time]
                    input_data = data.unsqueeze(1)  # Add channel dimension
                    logger.info(f"  LSTM input shape: {input_data.shape}")
                elif model_name in ['Audio GAN Mixer', 'VAE Audio Mixer', 'ResNet Audio Mixer']:
                    # 2D CNNs expect [batch, channels, height, width]
                    input_data = data.unsqueeze(1)  # Add channel dimension
                    logger.info(f"  2D CNN input shape: {input_data.shape}")
                else:  # Transformer
                    # Transformer can handle the original input
                    input_data = data.unsqueeze(1)  # Add channel dimension
                    logger.info(f"  Transformer input shape: {input_data.shape}")
                
                # Get output and check shape
                output = model(input_data)
                logger.info(f"  Model output shape: {output.shape}")
                logger.info(f"  Target shape: {target.shape}")
                
                # Test loss calculation
                criterion = nn.MSELoss()
                loss = criterion(output, target)
                logger.info(f"  Loss calculation successful: {loss.item()}")
                
                logger.info(f"✅ {model_name} passed shape test")
                
            except Exception as e:
                logger.error(f"❌ {model_name} failed: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Only process one batch
        break
    
    logger.info("\n== Shape Analysis Complete ==")

if __name__ == "__main__":
    train_debug()
