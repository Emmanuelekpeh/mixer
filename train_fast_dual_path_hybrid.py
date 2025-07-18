#!/usr/bin/env python3
"""
🔥 Fast Dual-Path Hybrid Audio Model Training
============================================

Lightweight version for quick training and tournament integration.
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

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants
SAMPLE_RATE = 22050
N_MELS = 32  # Reduced from 64
CHUNK_DURATION = 2.0  # Reduced from 3.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
BATCH_SIZE = 64  # Reduced from 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 22  # Reduced for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = "data"
RESULTS_DIR = "dual_path_results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def audio_to_spectrogram(audio):
    """Convert audio to mel spectrogram"""
    spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=512
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = (spec + 80) / 80  # Normalize to [0, 1]
    return spec.astype(np.float32)

class FastDataset(Dataset):
    """Fast dataset for dual-path hybrid training"""
    
    def __init__(self, clean_dir, distorted_dir, max_pairs=50):
        self.clean_dir = Path(clean_dir)
        self.distorted_dir = Path(distorted_dir)
        self.chunk_samples = CHUNK_SAMPLES
        
        # Find file pairs (limited for speed)
        self.clean_files = list(self.clean_dir.glob("*.wav"))[:max_pairs]
        self.file_pairs = []
        
        print(f"📊 Scanning {len(self.clean_files)} clean files...")
        
        for clean_file in self.clean_files:
            base_name = clean_file.stem.replace("_clean", "")
            distorted_pattern = f"{base_name}_distorted_*.wav"
            distorted_files = list(self.distorted_dir.glob(distorted_pattern))
            
            for distorted_file in distorted_files[:1]:  # Only first distorted variant
                self.file_pairs.append((clean_file, distorted_file))
        
        print(f"📊 Dataset: {len(self.file_pairs)} clean/distorted pairs")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        clean_path, distorted_path = self.file_pairs[idx]
        
        try:
            # Load audio
            clean_audio, _ = librosa.load(clean_path, sr=SAMPLE_RATE, mono=True)
            distorted_audio, _ = librosa.load(distorted_path, sr=SAMPLE_RATE, mono=True)
            
            # Process chunks
            if len(clean_audio) > self.chunk_samples:
                start = random.randint(0, len(clean_audio) - self.chunk_samples)
                clean_audio = clean_audio[start:start + self.chunk_samples]
                distorted_audio = distorted_audio[start:start + self.chunk_samples]
            else:
                pad_length = self.chunk_samples - len(clean_audio)
                clean_audio = np.pad(clean_audio, (0, pad_length))
                distorted_audio = np.pad(distorted_audio, (0, pad_length))
            
            # Convert to spectrograms
            clean_spec = audio_to_spectrogram(clean_audio)
            distorted_spec = audio_to_spectrogram(distorted_audio)
            
            # Generate mixing parameters (simplified)
            mixing_params = np.array([
                0.7,  # master_volume
                0.5,  # bass_gain
                0.5,  # mid_gain
                0.5,  # treble_gain
                0.3,  # compression
                0.1,  # reverb
                0.5,  # low_pass
                0.1,  # high_pass
                0.7,  # comp_ratio
                0.8   # clip_thresh
            ], dtype=np.float32)
            
            return {
                'distorted_spec': torch.FloatTensor(distorted_spec),
                'clean_spec': torch.FloatTensor(clean_spec),
                'distorted_audio': torch.FloatTensor(distorted_audio),
                'clean_audio': torch.FloatTensor(clean_audio),
                'mixing_params': torch.FloatTensor(mixing_params)
            }
            
        except Exception as e:
            print(f"Error loading {clean_path}: {e}")
            # Return dummy data
            dummy_spec = torch.zeros(N_MELS, 87)  # Approximate spec size
            dummy_audio = torch.zeros(CHUNK_SAMPLES)
            dummy_params = torch.zeros(10)
            
            return {
                'distorted_spec': dummy_spec,
                'clean_spec': dummy_spec,
                'distorted_audio': dummy_audio,
                'clean_audio': dummy_audio,
                'mixing_params': dummy_params
            }

class FastDualPathHybrid(nn.Module):
    """Fast dual-path hybrid model"""
    
    def __init__(self):
        super().__init__()
        
        # Spectrogram branch (simplified AST)
        spec_time_dim = 87  # Approximate for 2-second chunks
        self.spec_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64)
        )
        
        # Audio branch (simplified GAN)
        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
            nn.Linear(16 * 64, 64)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-task heads
        self.restoration_head = nn.Linear(64, 32)  # Audio restoration features
        self.mixing_head = nn.Linear(64, 10)      # Mixing parameters
        
    def forward(self, distorted_spec, distorted_audio):
        # Spec branch
        spec_features = self.spec_conv(distorted_spec.unsqueeze(1))
        
        # Audio branch
        audio_features = self.audio_conv(distorted_audio.unsqueeze(1))
        
        # Fusion
        combined = torch.cat([spec_features, audio_features], dim=1)
        fused = self.fusion(combined)
        
        # Multi-task outputs
        restoration_features = self.restoration_head(fused)
        mixing_params = torch.sigmoid(self.mixing_head(fused))
        
        return {
            'restoration_features': restoration_features,
            'mixing_params': mixing_params
        }
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_parameters': total_params,
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

def train_model():
    """Train the fast dual-path hybrid model"""
    
    print("🔥 Fast Dual-Path Hybrid Training")
    print("=" * 50)
    
    # Check dataset
    restoration_dir = os.path.join(DATA_DIR, "restoration")
    clean_dir = os.path.join(restoration_dir, "clean")
    distorted_dir = os.path.join(restoration_dir, "distorted")
    
    if not os.path.exists(clean_dir) or not os.path.exists(distorted_dir):
        print("❌ Restoration dataset not found!")
        return
    
    # Create dataset
    dataset = FastDataset(clean_dir, distorted_dir, max_pairs=20)  # Small for testing
    
    if len(dataset) == 0:
        print("❌ No data found!")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Training: {train_size}, Validation: {val_size}")
    
    # Create model
    model = FastDualPathHybrid().to(DEVICE)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"🏗️ Model: {model_info['total_parameters']:,} parameters")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"🚀 Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            distorted_spec = batch['distorted_spec'].to(DEVICE)
            distorted_audio = batch['distorted_audio'].to(DEVICE)
            clean_spec = batch['clean_spec'].to(DEVICE)
            mixing_params = batch['mixing_params'].to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(distorted_spec, distorted_audio)
            
            # Multi-task loss
            mixing_loss = mse_loss(outputs['mixing_params'], mixing_params)
            # Note: restoration loss would need clean audio reconstruction target
            
            loss = mixing_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                distorted_spec = batch['distorted_spec'].to(DEVICE)
                distorted_audio = batch['distorted_audio'].to(DEVICE)
                mixing_params = batch['mixing_params'].to(DEVICE)
                
                outputs = model(distorted_spec, distorted_audio)
                loss = mse_loss(outputs['mixing_params'], mixing_params)
                val_loss += loss.item()
        
        # Update history
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, "fast_dual_path_hybrid.pth")
    torch.save(model.state_dict(), model_path)
    
    # Save results
    results = {
        'model_info': model_info,
        'training_config': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'dataset_size': len(dataset)
        },
        'history': history,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(RESULTS_DIR, "fast_training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Training completed!")
    print(f"📁 Model saved: {model_path}")
    print(f"📊 Results saved: {results_path}")
    
    return model, results

if __name__ == "__main__":
    train_model()
