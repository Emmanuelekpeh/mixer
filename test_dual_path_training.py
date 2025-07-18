#!/usr/bin/env python3
"""
Quick test for dual-path hybrid training with small dataset
"""

import os
import sys
import json
import time
import random
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import uuid
from datetime import datetime

# Import the dual-path hybrid model
from train_dual_path_hybrid import DualPathHybrid, DualPathDataset, save_tournament_model

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants
SAMPLE_RATE = 22050
N_MELS = 64
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

# Training parameters (reduced for quick test)
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 3
SUBSET_SIZE = 100  # Small subset for quick test

# Directories
DATA_DIR = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(os.getcwd(), "models")
TOURNAMENT_MODELS_DIR = os.path.join(os.getcwd(), "tournament_webapp", "tournament_models", "evolved")
RESULTS_DIR = os.path.join(os.getcwd(), "dual_path_test_results")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TOURNAMENT_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

def quick_test_training():
    """Quick test of dual-path hybrid training"""
    
    print("🔥 Dual-Path Hybrid Quick Test")
    print("=" * 50)
    
    # Check dataset
    restoration_dir = os.path.join(DATA_DIR, "restoration")
    clean_dir = os.path.join(restoration_dir, "clean")
    distorted_dir = os.path.join(restoration_dir, "distorted")
    
    if not os.path.exists(clean_dir) or not os.path.exists(distorted_dir):
        print("❌ Restoration dataset not found!")
        return False
    
    # Create small dataset
    print(f"📊 Creating dataset with {SUBSET_SIZE} samples...")
    dataset = DualPathDataset(clean_dir, distorted_dir, subset_size=SUBSET_SIZE)
    
    if len(dataset) == 0:
        print("❌ No data found!")
        return False
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Training: {train_size}, Validation: {val_size}")
    
    # Create model
    model = DualPathHybrid().to(device)
    discriminator = None  # Skip discriminator for quick test
    
    # Print model info
    model_info = model.get_model_info()
    print(f"🏗️ Model: {model_info['total_parameters']:,} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss functions
    restoration_criterion = nn.L1Loss()
    mixing_criterion = nn.MSELoss()
    distortion_criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"\n🚀 Starting quick training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, (distorted, clean, mixing_params, distortion_params) in enumerate(pbar):
            distorted = distorted.to(device)
            clean = clean.to(device)
            mixing_params = mixing_params.to(device)
            distortion_params = distortion_params.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(distorted)
            
            # Multi-task losses
            restoration_loss = restoration_criterion(outputs['restored_audio'], clean)
            mixing_loss = mixing_criterion(outputs['mixing_params'], mixing_params)
            distortion_loss = distortion_criterion(outputs['distortion_params'], distortion_params)
            
            # Combined loss (no adversarial for quick test)
            total_loss = (
                2.0 * restoration_loss +
                1.0 * mixing_loss +
                0.5 * distortion_loss
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Rest': f"{restoration_loss.item():.4f}",
                'Mix': f"{mixing_loss.item():.4f}"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for distorted, clean, mixing_params, distortion_params in val_loader:
                distorted = distorted.to(device)
                clean = clean.to(device)
                mixing_params = mixing_params.to(device)
                distortion_params = distortion_params.to(device)
                
                outputs = model(distorted)
                
                restoration_loss = restoration_criterion(outputs['restored_audio'], clean)
                mixing_loss = mixing_criterion(outputs['mixing_params'], mixing_params)
                distortion_loss = distortion_criterion(outputs['distortion_params'], distortion_params)
                
                total_loss = 2.0 * restoration_loss + 1.0 * mixing_loss + 0.5 * distortion_loss
                val_loss += total_loss.item()
        
        # Calculate averages
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
    
    # Save model for tournament
    model_id = save_tournament_model(model)
    
    # Save test results
    results = {
        'model_id': model_id,
        'epochs': NUM_EPOCHS,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'model_info': model_info,
        'test_params': {
            'subset_size': SUBSET_SIZE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }
    
    results_path = os.path.join(RESULTS_DIR, "quick_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Quick Training Test Complete!")
    print(f"✅ Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"✅ Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"🏆 Tournament model ID: {model_id}")
    print(f"📊 Results saved to: {results_path}")
    print(f"🎯 Model ready for integration!")
    
    return True

if __name__ == "__main__":
    success = quick_test_training()
    if success:
        print("\n✅ Dual-Path Hybrid model tested successfully!")
        print("📋 Next steps:")
        print("   1. ✅ Model architecture validated")
        print("   2. ✅ Training compatibility confirmed")
        print("   3. ✅ Quick training test passed")
        print("   4. 🔄 Ready for full integration")
    else:
        print("\n❌ Quick test failed!")
