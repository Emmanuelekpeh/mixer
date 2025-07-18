#!/usr/bin/env python3
"""
🎵 Complete Audio Mixer Training Pipeline
=======================================

This script handles both preprocessing and training for audio mixing models.
It works around the AST feature extractor compatibility issues by using MFCC features.

Usage:
    python train_mixer_pipeline.py
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import random
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path('data')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('training_results')

# Audio parameters
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SPLIT_RATIO = 0.8

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
PATIENCE = 5
N_OUTPUTS = 17  # Number of mixing parameters to predict

# Setup directory structure
for d in [
    BASE_DIR / 'train',
    BASE_DIR / 'val', 
    BASE_DIR / 'test',
    BASE_DIR / 'spectrograms' / 'train',
    BASE_DIR / 'spectrograms' / 'val',
    BASE_DIR / 'spectrograms' / 'test',
    BASE_DIR / 'features' / 'train',
    BASE_DIR / 'features' / 'val',
    BASE_DIR / 'features' / 'test',
    MODELS_DIR,
    RESULTS_DIR
]:
    d.mkdir(exist_ok=True, parents=True)

# Define MFCC feature dataset
class MFCCFeatureDataset(Dataset):
    def __init__(self, features_dir, targets_file, n_outputs=17):
        self.features_dir = Path(features_dir)
        self.n_outputs = n_outputs
        
        # Load mixing targets
        with open(targets_file, 'r') as f:
            self.targets = json.load(f)
        
        # Find all MFCC feature files
        self.feature_files = list(self.features_dir.glob("*_mfcc.npy"))
        
        # Filter to keep only files with targets
        self.valid_files = []
        for file in self.feature_files:
            track_id = file.stem.split("_")[0]  # Remove _mfcc suffix
            if track_id in self.targets:
                self.valid_files.append(file)
        
        print(f"Loaded {len(self.valid_files)} MFCC feature files with targets")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        feature_file = self.valid_files[idx]
        track_id = feature_file.stem.split("_")[0]  # Remove _mfcc suffix
        
        # Load MFCC features
        mfcc_features = np.load(feature_file)
        
        # Calculate mean over time dimension (temporal average pooling)
        mfcc_mean = np.mean(mfcc_features, axis=1)
        
        # Get target parameters
        target = self.targets[track_id]
        
        # Convert to tensors
        features_tensor = torch.tensor(mfcc_mean, dtype=torch.float32)
        target_tensor = torch.tensor(target[:self.n_outputs], dtype=torch.float32)
        
        return features_tensor, target_tensor

# Define mel spectrogram dataset
class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, targets_file, n_outputs=17, fixed_length=1292):
        self.spectrogram_dir = Path(spectrogram_dir)
        self.n_outputs = n_outputs
        self.fixed_length = fixed_length
        
        # Load mixing targets
        with open(targets_file, 'r') as f:
            self.targets = json.load(f)
        
        # Find all spectrogram files
        self.spec_files = list(self.spectrogram_dir.glob("*.npy"))
        
        # Filter to keep only files with targets
        self.valid_files = []
        for file in self.spec_files:
            track_id = file.stem
            if track_id in self.targets:
                self.valid_files.append(file)
        
        print(f"Loaded {len(self.valid_files)} spectrogram files with targets")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        spec_file = self.valid_files[idx]
        track_id = spec_file.stem
        
        # Load spectrogram
        spec = np.load(spec_file)

        # Pad or truncate spectrogram to a fixed length
        if spec.shape[1] < self.fixed_length:
            pad_width = self.fixed_length - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80) # Pad with silence
        else:
            spec = spec[:, :self.fixed_length]
        
        # Normalize spectrogram (can be done during preprocessing instead)
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        
        # Get target parameters
        target = self.targets[track_id]
        
        # Convert to tensors
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        target_tensor = torch.tensor(target[:self.n_outputs], dtype=torch.float32)
        
        return spec_tensor, target_tensor

# Define model architectures
class BaselineCNN(nn.Module):
    """Simple CNN for audio mixing parameter prediction."""
    def __init__(self, n_outputs=17, n_conv_layers=3, dropout=0.3, spec_length=1292):
        super(BaselineCNN, self).__init__()
        
        # Define convolutional layers
        layers = []
        in_channels = 1
        out_channels = 1 # Initialize out_channels
        
        for i in range(n_conv_layers):
            out_channels = 16 * (2 ** i)  # Double channels each time
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size of conv layers to define linear layer size
        final_height = N_MELS // (2 ** n_conv_layers)
        final_width = spec_length // (2 ** n_conv_layers)
        self.fc_input_size = out_channels * final_height * final_width
        
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_outputs),
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x

class EnhancedCNN(nn.Module):
    """Enhanced CNN with residual connections for audio mixing."""
    def __init__(self, n_outputs=17, dropout=0.3, spec_length=1292):
        super(EnhancedCNN, self).__init__()
        
        # Define convolutional layers with residual connections
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        # Residual block 1
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.res_conv1 = nn.Conv2d(32, 64, kernel_size=1)  # for residual connection
        
        # Residual block 2
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.res_conv2 = nn.Conv2d(64, 128, kernel_size=1)  # for residual connection
        
        # Residual block 3
        self.conv4a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.res_conv3 = nn.Conv2d(128, 256, kernel_size=1)  # for residual connection
        
        final_height = N_MELS // 16
        final_width = spec_length // 16
        self.fc_input_size = 256 * final_height * final_width
        
        # Define fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, n_outputs)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # First block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Residual block 1
        residual = self.res_conv1(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = self.relu(x + residual)  # Add residual connection
        x = self.pool2(x)
        
        # Residual block 2
        residual = self.res_conv2(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = self.relu(x + residual)  # Add residual connection
        x = self.pool3(x)
        
        # Residual block 3
        residual = self.res_conv3(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.bn4b(self.conv4b(x))
        x = self.relu(x + residual)  # Add residual connection
        x = self.pool4(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc_bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.fc_bn2(self.fc2(x))))
        x = self.sigmoid(self.fc3(x))
        
        return x

class MFCCRegressor(nn.Module):
    """Regressor model using MFCC features (fallback for AST)."""
    def __init__(self, input_dim=120, n_outputs=17, hidden_dim=256, dropout=0.3):
        super(MFCCRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs),
            nn.Sigmoid()  # Parameters are in [0, 1]
        )
        
    def forward(self, x):
        return self.model(x)

def preprocess_dataset():
    """Preprocess the FMA dataset."""
    print("🚀 Starting preprocessing...")
    
    # Find FMA audio files
    fma_dir = BASE_DIR / 'raw' / 'music' / 'fma' / 'fma_small'
    print(f'Finding audio files in {fma_dir}...')
    
    if not fma_dir.exists():
        print(f'FMA directory not found at {fma_dir}.')
        print(f'Please download the FMA dataset and extract it to {fma_dir}')
        sys.exit(1)
    
    audio_files = list(fma_dir.glob('**/*.mp3'))
    if len(audio_files) == 0:
        print(f'No audio files found in {fma_dir}')
        sys.exit(1)
        
    print(f'Found {len(audio_files)} audio files')
    
    # Split into train/val/test
    random.seed(42)
    random.shuffle(audio_files)
    train_size = int(len(audio_files) * SPLIT_RATIO)
    val_size = int((len(audio_files) - train_size) / 2)
    train_files = audio_files[:train_size]
    val_files = audio_files[train_size:train_size + val_size]
    test_files = audio_files[train_size + val_size:]
    print(f'Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test')
    
    # Process files in sequence (to avoid multiprocessing issues)
    all_track_info = {}
    total_duration = 0
    
    # Process a single file
    def process_file(audio_file, output_dir, spec_dir, feature_dir):
        try:
            track_id = audio_file.stem
            output_path = output_dir / f'{track_id}.wav'
            spec_path = spec_dir / f'{track_id}.npy'
            mfcc_path = feature_dir / f'{track_id}_mfcc.npy'
            
            # Skip if already processed
            if output_path.exists() and spec_path.exists() and mfcc_path.exists():
                return track_id, {'status': 'already_processed', 'duration': 0}
            
            # Load and normalize audio
            audio, _ = librosa.load(audio_file, sr=SR, mono=True)
            if len(audio) < SR:  # Skip very short clips
                return track_id, {'status': 'too_short', 'duration': 0}
            
            audio = librosa.util.normalize(audio)
            
            # Save preprocessed audio
            sf.write(output_path, audio, SR)
            
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            S_db = librosa.power_to_db(S, ref=np.max)
            np.save(spec_path, S_db)
            
            # Generate MFCC features (good alternative to AST)
            mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            np.save(mfcc_path, mfcc_features)
            
            duration = len(audio) / SR
            return track_id, {
                'duration': duration,
                'sample_rate': SR,
                'n_samples': len(audio),
                'status': 'success'
            }
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return audio_file.stem, {'status': f'error: {e}', 'duration': 0}
    
    # Process each split
    for name, files, out_dir, spec_dir, feat_dir in [
        ('Training', train_files, BASE_DIR / 'train', BASE_DIR / 'spectrograms' / 'train', BASE_DIR / 'features' / 'train'),
        ('Validation', val_files, BASE_DIR / 'val', BASE_DIR / 'spectrograms' / 'val', BASE_DIR / 'features' / 'val'),
        ('Test', test_files, BASE_DIR / 'test', BASE_DIR / 'spectrograms' / 'test', BASE_DIR / 'features' / 'test')
    ]:
        print(f'\nProcessing {name} set...')
        
        results = {}
        errors = []
        
        # Process files with progress bar (sequential)
        for file in tqdm(files, desc=f'Processing {name} files'):
            track_id, info = process_file(file, out_dir, spec_dir, feat_dir)
            if info['status'] == 'success':
                results[track_id] = info
                total_duration += info['duration']
            elif info['status'] == 'already_processed':
                # For already processed files, we need to add them to the results
                # but don't increment duration since we don't know it
                results[track_id] = info
            elif info['status'].startswith('error'):
                errors.append((track_id, info['status']))
        
        print(f'Processed {len(results)} files successfully')
        if errors:
            print(f'Encountered {len(errors)} errors')
        
        all_track_info.update(results)
    
    # Generate fake mixing parameters
    print('\nGenerating synthetic mixing targets...')
    targets = {}
    
    for track_id in all_track_info:
        # Generate 17 parameters with realistic constraints
        params = []
        params.append(0.6 + 0.3 * random.random())  # Input Gain
        params.append(0.1 + 0.6 * random.random())  # Compression Ratio
        params.append(0.2 + 0.6 * random.random())  # Compression Attack
        params.append(0.3 + 0.4 * random.random())  # Compression Release
        for _ in range(5):  # EQ parameters
            params.append(0.2 + 0.6 * random.random())
        params.append(0.3 + 0.4 * random.random())  # Presence
        params.append(0.1 + 0.4 * random.random())  # Reverb Send
        params.append(random.random())             # Reverb Type
        params.append(0.05 + 0.25 * random.random()) # Delay Send
        params.append(random.random())             # Delay Time
        params.append(0.4 + 0.4 * random.random())  # Stereo Width
        params.append(0.6 + 0.4 * random.random())  # Bass Mono
        params.append(0.7 + 0.25 * random.random()) # Output Level
        targets[track_id] = params
    
    # Save targets
    targets_file = BASE_DIR / 'targets_generated.json'
    with open(targets_file, 'w') as f:
        json.dump(targets, f, indent=2)
    
    # Save metadata
    metadata = {
        'preprocessing_info': {
            'date': time.strftime('%Y-%m-%d'),
            'sample_rate': SR,
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'n_mels': N_MELS,
            'split_ratio': SPLIT_RATIO
        },
        'dataset_stats': {
            'total_tracks': len(all_track_info),
            'train_tracks': len(train_files),
            'val_tracks': len(val_files),
            'test_tracks': len(test_files),
            'total_duration_hours': total_duration / 3600
        },
        'track_info': all_track_info
    }
    
    metadata_file = BASE_DIR / 'preprocessing_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print('\n✅ Preprocessing complete!')
    print(f'🎵 Processed {len(all_track_info)} tracks ({total_duration / 3600:.1f} hours)')
    print(f'📊 Dataset splits: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test')
    print(f'📁 Metadata saved to: {metadata_file}')
    print(f'🎚️ Mixing targets saved to: {targets_file}')
    print(f'🧠 MFCC features generated for all tracks')
    
    return targets_file

def train_model(model_name, model, train_loader, val_loader, device='cuda'):
    """Train a model and save checkpoints."""
    print(f"\n🏋️‍♀️ Training {model_name}...")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for features, targets in pbar:
                features, targets = features.to(device), targets.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint if it's the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODELS_DIR / f"{model_name}.pth")
            patience_counter = 0
            print(f"✅ Model improved! Saved checkpoint.")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"⚠️ Early stopping after {epoch+1} epochs")
            break
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training History')
    plt.legend()
    plt.savefig(RESULTS_DIR / f"{model_name}_training.png")
    
    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses)
    }
    
    with open(RESULTS_DIR / f"{model_name}_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ {model_name} training complete. Best validation loss: {best_val_loss:.4f}")
    return model, best_val_loss

def main():
    """Main function to run preprocessing and training."""
    start_time = time.time()
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Preprocess dataset
    targets_file = preprocess_dataset()
    
    print("\n-----------------------------------------------")
    print("    Step 2: Training Models")
    print("-----------------------------------------------")
    
    # Define models to train
    models_to_train = [
        ('baseline_cnn', BaselineCNN(n_outputs=N_OUTPUTS, n_conv_layers=3, dropout=0.3)),
        ('enhanced_cnn', EnhancedCNN(n_outputs=N_OUTPUTS, dropout=0.3)),
        ('mfcc_regressor', MFCCRegressor(input_dim=120, n_outputs=N_OUTPUTS, hidden_dim=256, dropout=0.3))
    ]
    
    # Train all models
    results = {}
    
    # Train Baseline CNN and Enhanced CNN
    for model_name, model in models_to_train[:2]:  # CNN models
        # Create dataset and data loaders
        train_dataset = SpectrogramDataset(
            BASE_DIR / "spectrograms" / "train", 
            targets_file,
            n_outputs=N_OUTPUTS
        )
        
        val_dataset = SpectrogramDataset(
            BASE_DIR / "spectrograms" / "val", 
            targets_file,
            n_outputs=N_OUTPUTS
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train model
        _, best_val_loss = train_model(model_name, model, train_loader, val_loader, device)
        results[model_name] = best_val_loss
    
    # Train MFCC Regressor
    model_name, model = models_to_train[2]  # MFCC Regressor
    
    # Create dataset and data loaders
    train_dataset = MFCCFeatureDataset(
        BASE_DIR / "features" / "train", 
        targets_file,
        n_outputs=N_OUTPUTS
    )
    
    val_dataset = MFCCFeatureDataset(
        BASE_DIR / "features" / "val", 
        targets_file,
        n_outputs=N_OUTPUTS
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train model
    _, best_val_loss = train_model(model_name, model, train_loader, val_loader, device)
    results[model_name] = best_val_loss
    
    # Save overall results
    with open(RESULTS_DIR / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot comparison of models
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.xlabel('Model')
    plt.ylabel('Validation Loss')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n===============================================")
    print("    Training Complete!")
    print("===============================================")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"📊 Models trained: {', '.join(results.keys())}")
    if results:
        best_model_name = min(results, key=lambda k: results[k])
        print(f"🏆 Best model: {best_model_name} (Loss: {results[best_model_name]:.4f})")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print(f"💾 Models saved to: {MODELS_DIR}")
    print("\nTo use the models for mixing:")
    print("  python demo_ai_mixer.py path/to/your/audio.wav")

if __name__ == "__main__":
    main()
