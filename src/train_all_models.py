#!/usr/bin/env python3
"""
🧠 Complete Model Training Pipeline
=================================

Train all AI mixing models in sequence or selectively.
This script handles training for:
1. Baseline CNN
2. Enhanced CNN
3. AST Regressor
4. LSTM Mixer
5. Advanced Transformer
6. VAE Mixer
7. Audio GAN Mixer
8. ResNet Mixer
9. Weighted Ensemble (combines best models)

Usage:
    python train_all_models.py

Options:
    --models=all                Models to train (all/baseline/enhanced/ast/lstm/transformer/vae/gan/resnet/ensemble)
    --epochs=10                 Number of epochs for training
    --batch-size=16             Batch size for training
    --learning-rate=0.001       Learning rate for training
    --device=auto               Device to use (cuda/cpu/auto)
    --save-checkpoints          Save model checkpoints during training
    --patience=5                Early stopping patience
    --augment                   Use data augmentation during training
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our models
sys.path.append(str(Path(__file__).resolve().parent))

# Try to import all models
try:
    from baseline_cnn import BaselineCNN, EnhancedCNN
    from dataset import SpectrogramDataset
    BASELINE_AVAILABLE = True
except ImportError:
    BASELINE_AVAILABLE = False
    print("⚠️ Baseline CNN models not available")

try:
    from ast_regressor import ASTRegressor, ASTFeatureDataset
    AST_AVAILABLE = True
except ImportError:
    AST_AVAILABLE = False
    print("⚠️ AST Regressor model not available")
      # Define a fallback MFCC-based regressor
    class MFCCRegressor(nn.Module):
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
    
    # Define a dataset for MFCC features
    class MFCCFeatureDataset(torch.utils.data.Dataset):
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

try:
    from lstm_mixer import LSTMAudioMixer, LSTMSpectrogramDataset
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️ LSTM Mixer model not available")

try:
    from advanced_transformer import TransformerMixer, TransformerSpectrogramDataset
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("⚠️ Advanced Transformer model not available")

try:
    from vae_mixer import VAEAudioMixer, VAESpectrogramDataset
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    print("⚠️ VAE Mixer model not available")

try:
    from audio_gan import AudioGANMixer, GANSpectrogramDataset
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False
    print("⚠️ Audio GAN model not available")

try:
    from resnet_mixer import ResNetAudioMixer, ResNetSpectrogramDataset
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False
    print("⚠️ ResNet Mixer model not available")

try:
    from ensemble_training import WeightedEnsemble
    ENSEMBLE_AVAILABLE = True
except Exception:
    ENSEMBLE_AVAILABLE = False
    print("⚠️ Ensemble models not available (import failed)")

class ModelTrainer:
    """Unified training pipeline for all model architectures."""
    
    def __init__(self, 
                 base_dir=None,
                 epochs=10,
                 batch_size=16,
                 learning_rate=0.001,
                 device='auto',
                 save_checkpoints=True,
                 patience=5,
                 use_augmentation=True):
        """Initialize the trainer with parameters."""
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_checkpoints = save_checkpoints
        self.patience = patience
        self.use_augmentation = use_augmentation
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set up directories
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "training_results"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Load targets
        self.targets_file = self.data_dir / "targets_generated.json"
        if not self.targets_file.exists():
            raise FileNotFoundError(f"Targets file not found at {self.targets_file}. Run preprocessing first.")
        
        # Training metrics
        self.training_history = {}
        
        # Baseline parameters
        self.n_outputs = 17  # Number of mixing parameters to predict
        
        print(f"🧠 Model Trainer initialized")
        print(f"📊 Device: {self.device}")
        print(f"⚙️ Parameters: epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate}")
    
    def train_baseline_cnn(self):
        """Train the Baseline CNN model."""
        if not BASELINE_AVAILABLE:
            print("⚠️ Skipping Baseline CNN (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training Baseline CNN...")
        
        # Create dataset and data loaders
        train_dataset = SpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=self.use_augmentation
        )
        
        val_dataset = SpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        model = BaselineCNN(n_outputs=self.n_outputs, n_conv_layers=3, dropout=0.3)
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(specs)
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
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    outputs = model(specs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "baseline_cnn.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["baseline_cnn"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Baseline CNN Training History')
        plt.legend()
        plt.savefig(self.results_dir / "baseline_cnn_training.png")
        
        print(f"✅ Baseline CNN training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_enhanced_cnn(self):
        """Train the Enhanced CNN model."""
        if not BASELINE_AVAILABLE:  # Enhanced CNN is in the same file
            print("⚠️ Skipping Enhanced CNN (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training Enhanced CNN...")
        
        # Create dataset and data loaders
        train_dataset = SpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=self.use_augmentation
        )
        
        val_dataset = SpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        model = EnhancedCNN(n_outputs=self.n_outputs, dropout=0.3)
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(specs)
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
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    outputs = model(specs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "enhanced_cnn.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["enhanced_cnn"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Enhanced CNN Training History')
        plt.legend()
        plt.savefig(self.results_dir / "enhanced_cnn_training.png")
        
        print(f"✅ Enhanced CNN training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_ast_regressor(self):
        """Train the AST Regressor model (or MFCC Regressor if AST is not available)."""
        if not AST_AVAILABLE:
            print("⚠️ AST Regressor not available, using MFCC Regressor instead")
            return self.train_mfcc_regressor()
        
        print("\n🏋️‍♀️ Training AST Regressor...")
        
        # Check if AST features are available
        if not (self.data_dir / "ast_features" / "train").exists():
            print("⚠️ AST features not found. Using MFCC features instead.")
            return self.train_mfcc_regressor()
        
        # Create dataset and data loaders
        train_dataset = ASTFeatureDataset(
            self.data_dir / "ast_features" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs
        )
        
        val_dataset = ASTFeatureDataset(
            self.data_dir / "ast_features" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Get input dimensions from first sample
        input_dim = next(iter(train_loader))[0].shape[1]
        
        # Create model
        model = ASTRegressor(input_dim=input_dim, n_outputs=self.n_outputs, hidden_dim=256, dropout=0.3)
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for features, targets in pbar:
                    features, targets = features.to(self.device), targets.to(self.device)
                    
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
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "ast_regressor.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["ast_regressor"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('AST Regressor Training History')
        plt.legend()
        plt.savefig(self.results_dir / "ast_regressor_training.png")
        
        print(f"✅ AST Regressor training complete. Best validation loss: {best_val_loss:.4f}")
        return model
        
    def train_mfcc_regressor(self):
        """Train the MFCC Regressor model (fallback for AST)."""
        print("\n🏋️‍♀️ Training MFCC Regressor...")
        
        # Check if features are available
        if not (self.data_dir / "features" / "train").exists():
            print("⚠️ MFCC features not found. Run preprocessing first.")
            return None
        
        # Create dataset and data loaders
        train_dataset = MFCCFeatureDataset(
            self.data_dir / "features" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs
        )
        
        val_dataset = MFCCFeatureDataset(
            self.data_dir / "features" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Get input dimensions from first sample
        input_dim = next(iter(train_loader))[0].shape[0]  # MFCC feature dimension
        
        # Create model
        model = MFCCRegressor(input_dim=input_dim, n_outputs=self.n_outputs, hidden_dim=256, dropout=0.3)
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for features, targets in pbar:
                    features, targets = features.to(self.device), targets.to(self.device)
                    
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
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "mfcc_regressor.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["mfcc_regressor"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MFCC Regressor Training History')
        plt.legend()
        plt.savefig(self.results_dir / "mfcc_regressor_training.png")
        
        print(f"✅ MFCC Regressor training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_lstm_mixer(self):
        """Train the LSTM Mixer model."""
        if not LSTM_AVAILABLE:
            print("⚠️ Skipping LSTM Mixer (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training LSTM Mixer...")
        
        # Create dataset and data loaders
        train_dataset = LSTMSpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=self.use_augmentation
        )
        
        val_dataset = LSTMSpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model (get input shape from first batch)
        sample_batch, _ = next(iter(train_loader))
        input_size = sample_batch.size(2)  # Number of features (mel bands)
        
        model = LSTMAudioMixer(
            input_size=input_size, 
            hidden_size=128, 
            num_layers=2, 
            n_outputs=self.n_outputs, 
            bidirectional=True,
            dropout=0.3
        )
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(specs)
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
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    outputs = model(specs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "lstm_mixer.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["lstm_mixer"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Mixer Training History')
        plt.legend()
        plt.savefig(self.results_dir / "lstm_mixer_training.png")
        
        print(f"✅ LSTM Mixer training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_transformer_mixer(self):
        """Train the Advanced Transformer Mixer model."""
        if not TRANSFORMER_AVAILABLE:
            print("⚠️ Skipping Transformer Mixer (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training Advanced Transformer Mixer...")
        
        # Create dataset and data loaders
        train_dataset = TransformerSpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=self.use_augmentation
        )
        
        val_dataset = TransformerSpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Get input dimensions from first sample
        sample_batch, _ = next(iter(train_loader))
        input_dim = sample_batch.size(1)  # Number of features (mel bands)
        seq_len = sample_batch.size(2)    # Sequence length (time steps)
        
        # Create model
        model = TransformerMixer(
            input_dim=input_dim,
            seq_len=seq_len,
            n_outputs=self.n_outputs,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.3
        )
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(specs)
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
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    outputs = model(specs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "transformer_mixer.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["transformer_mixer"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Transformer Mixer Training History')
        plt.legend()
        plt.savefig(self.results_dir / "transformer_mixer_training.png")
        
        print(f"✅ Transformer Mixer training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_vae_mixer(self):
        """Train the VAE Mixer model."""
        if not VAE_AVAILABLE:
            print("⚠️ Skipping VAE Mixer (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training VAE Audio Mixer...")
        
        # Create dataset and data loaders
        train_dataset = VAESpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=self.use_augmentation
        )
        
        val_dataset = VAESpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Get input dimensions from first sample
        sample_batch, _ = next(iter(train_loader))
        input_shape = sample_batch.shape[1:]  # (channels, height, width)
        
        # Create model
        model = VAEAudioMixer(
            input_shape=input_shape,
            n_outputs=self.n_outputs,
            latent_dim=64,
            dropout=0.3
        )
        model = model.to(self.device)
        
        # Loss function and optimizer
        # VAE has its own loss function that combines reconstruction and KL divergence
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs, mu, logvar = model(specs)
                    loss = model.loss_function(outputs, targets, mu, logvar)
                    
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
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    outputs, mu, logvar = model(specs)
                    loss = model.loss_function(outputs, targets, mu, logvar)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "vae_mixer.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["vae_mixer"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Mixer Training History')
        plt.legend()
        plt.savefig(self.results_dir / "vae_mixer_training.png")
        
        print(f"✅ VAE Mixer training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_gan_mixer(self):
        """Train the GAN Mixer model."""
        if not GAN_AVAILABLE:
            print("⚠️ Skipping GAN Mixer (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training GAN Audio Mixer...")
        
        # Create dataset and data loaders
        train_dataset = GANSpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=self.use_augmentation
        )
        
        val_dataset = GANSpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Get input dimensions from first sample
        sample_batch, _ = next(iter(train_loader))
        input_shape = sample_batch.shape[1:]  # (channels, height, width)
        
        # Create model
        model = AudioGANMixer(
            input_shape=input_shape,
            n_outputs=self.n_outputs,
            latent_dim=100,
            dropout=0.3
        )
        model = model.to(self.device)
        
        # Optimizers for generator and discriminator
        g_optimizer = optim.Adam(model.generator.parameters(), lr=self.learning_rate)
        d_optimizer = optim.Adam(model.discriminator.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses_g = []  # Generator losses
        train_losses_d = []  # Discriminator losses
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss_g = 0.0
            running_loss_d = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Train discriminator
                    d_optimizer.zero_grad()
                    d_loss = model.train_discriminator(specs, targets)
                    d_loss.backward()
                    d_optimizer.step()
                    
                    # Train generator
                    g_optimizer.zero_grad()
                    g_loss = model.train_generator(specs, targets)
                    g_loss.backward()
                    g_optimizer.step()
                    
                    # Update statistics
                    running_loss_g += g_loss.item()
                    running_loss_d += d_loss.item()
                    pbar.set_postfix({"G_loss": g_loss.item(), "D_loss": d_loss.item()})
            
            avg_train_loss_g = running_loss_g / len(train_loader)
            avg_train_loss_d = running_loss_d / len(train_loader)
            train_losses_g.append(avg_train_loss_g)
            train_losses_d.append(avg_train_loss_d)
            
            # Validation (using MSE loss for parameter prediction)
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    # Generate parameters using the generator
                    predicted_params = model.predict(specs)
                    # Calculate MSE
                    mse_loss = nn.MSELoss()(predicted_params, targets)
                    val_loss += mse_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: G Loss = {avg_train_loss_g:.4f}, D Loss = {avg_train_loss_d:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "gan_mixer.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["gan_mixer"] = {
            "train_losses_g": train_losses_g,
            "train_losses_d": train_losses_d,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses_g)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses_g, label='Generator Loss')
        plt.plot(train_losses_d, label='Discriminator Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Mixer Training History')
        plt.legend()
        plt.savefig(self.results_dir / "gan_mixer_training.png")
        
        print(f"✅ GAN Mixer training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_resnet_mixer(self):
        """Train the ResNet Mixer model."""
        if not RESNET_AVAILABLE:
            print("⚠️ Skipping ResNet Mixer (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training ResNet Audio Mixer...")
        
        # Create dataset and data loaders
        train_dataset = ResNetSpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=self.use_augmentation
        )
        
        val_dataset = ResNetSpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Get input dimensions from first sample
        sample_batch, _ = next(iter(train_loader))
        input_shape = sample_batch.shape[1:]  # (channels, height, width)
        
        # Create model
        model = ResNetAudioMixer(
            input_shape=input_shape,
            n_outputs=self.n_outputs,
            n_blocks=[2, 2, 2, 2],  # Similar to ResNet-18
            dropout=0.3
        )
        model = model.to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Train the model
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(specs)
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
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    outputs = model(specs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.models_dir / "resnet_mixer.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["resnet_mixer"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('ResNet Mixer Training History')
        plt.legend()
        plt.savefig(self.results_dir / "resnet_mixer_training.png")
        
        print(f"✅ ResNet Mixer training complete. Best validation loss: {best_val_loss:.4f}")
        return model
    
    def train_weighted_ensemble(self, models=None):
        """Train a weighted ensemble of models."""
        if not ENSEMBLE_AVAILABLE:
            print("⚠️ Skipping Weighted Ensemble (model not available)")
            return None
        
        print("\n🏋️‍♀️ Training Weighted Ensemble...")
        
        # Load models if not provided
        if models is None:
            models = []
            
            # Load all trained models
            model_files = {
                "baseline_cnn.pth": (BaselineCNN, {"n_outputs": self.n_outputs, "n_conv_layers": 3, "dropout": 0.3}),
                "enhanced_cnn.pth": (EnhancedCNN, {"n_outputs": self.n_outputs, "dropout": 0.3}),
                "resnet_mixer.pth": (ResNetAudioMixer, {"input_shape": (1, 128, 128), "n_outputs": self.n_outputs, "n_blocks": [2, 2, 2, 2], "dropout": 0.3}),
                # Add other models as needed
            }
            
            for model_file, (model_class, params) in model_files.items():
                if (self.models_dir / model_file).exists():
                    try:
                        # Initialize model
                        model = model_class(**params).to(self.device)
                        # Load weights
                        model.load_state_dict(torch.load(self.models_dir / model_file, map_location=self.device))
                        model.eval()  # Set to evaluation mode
                        models.append(model)
                        print(f"✅ Loaded {model_file} for ensemble")
                    except Exception as e:
                        print(f"⚠️ Error loading {model_file}: {e}")
            
            if len(models) < 2:
                print("⚠️ Need at least 2 models for ensemble. Train more models first.")
                return None
        
        # Create ensemble model
        ensemble = WeightedEnsemble(models).to(self.device)
        
        # Create dataset and data loaders (using spectrogram dataset for simplicity)
        train_dataset = SpectrogramDataset(
            self.data_dir / "spectrograms" / "train", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        val_dataset = SpectrogramDataset(
            self.data_dir / "spectrograms" / "val", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss function and optimizer (only optimizing the ensemble weights)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ensemble.parameters(), lr=self.learning_rate)
        
        # Train the ensemble
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            ensemble.train()  # Only trains the weights, not the underlying models
            running_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}") as pbar:
                for specs, targets in pbar:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = ensemble(specs)
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
            ensemble.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for specs, targets in val_loader:
                    specs, targets = specs.to(self.device), targets.to(self.device)
                    outputs = ensemble(specs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Get current weights
            weights = ensemble.get_weights()
            weight_str = ", ".join([f"{w:.3f}" for w in weights])
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            print(f"Ensemble Weights: [{weight_str}]")
            
            # Save checkpoint if it's the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(ensemble.state_dict(), self.models_dir / "weighted_ensemble.pth")
                patience_counter = 0
                print(f"✅ Model improved! Saved checkpoint.")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                print(f"⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Save training history
        self.training_history["weighted_ensemble"] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
            "final_weights": ensemble.get_weights().tolist()
        }
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Weighted Ensemble Training History')
        plt.legend()
        plt.savefig(self.results_dir / "weighted_ensemble_training.png")
        
        print(f"✅ Weighted Ensemble training complete. Best validation loss: {best_val_loss:.4f}")
        print(f"Final ensemble weights: [{weight_str}]")
        
        return ensemble
    
    def evaluate_all_models(self):
        """Evaluate all trained models on the test set and compute MAE."""
        print("\n📊 Evaluating all trained models...")
        
        # Prepare test dataset
        test_dataset = SpectrogramDataset(
            self.data_dir / "spectrograms" / "test", 
            self.targets_file,
            n_outputs=self.n_outputs,
            augment=False
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize results dictionary
        results = {}
        
        # Define models to evaluate
        models_to_evaluate = {
            "baseline_cnn": (BaselineCNN, {"n_outputs": self.n_outputs, "n_conv_layers": 3, "dropout": 0.3}),
            "enhanced_cnn": (EnhancedCNN, {"n_outputs": self.n_outputs, "dropout": 0.3}),
            # Add more models as needed
        }
        
        # Evaluate each model
        for model_name, (model_class, params) in models_to_evaluate.items():
            model_path = self.models_dir / f"{model_name}.pth"
            
            if not model_path.exists():
                print(f"⚠️ Model {model_name} not found, skipping evaluation")
                continue
            
            try:
                # Initialize model
                model = model_class(**params).to(self.device)
                # Load weights
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                
                # Evaluate
                all_targets = []
                all_predictions = []
                
                with torch.no_grad():
                    for specs, targets in test_loader:
                        specs, targets = specs.to(self.device), targets.to(self.device)
                        outputs = model(specs)
                        
                        # Collect for MAE calculation
                        all_targets.append(targets.cpu().numpy())
                        all_predictions.append(outputs.cpu().numpy())
                
                # Concatenate batches
                all_targets = np.vstack(all_targets)
                all_predictions = np.vstack(all_predictions)
                
                # Calculate MAE
                mae = mean_absolute_error(all_targets, all_predictions)
                
                # Store result
                results[model_name] = {
                    "mae": mae,
                    "test_samples": len(test_dataset)
                }
                
                print(f"✅ {model_name}: MAE = {mae:.4f}")
                
            except Exception as e:
                print(f"⚠️ Error evaluating {model_name}: {e}")
        
        # Save evaluation results
        results_file = self.results_dir / "model_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Evaluation complete. Results saved to {results_file}")
        return results
    
    def train_all_models(self, models_to_train="all"):
        """Train all selected models in sequence."""
        start_time = time.time()
        print("🚀 Starting training pipeline for all models...")
        
        # Parse which models to train
        if models_to_train == "all":
            models_list = ["baseline", "enhanced", "ast", "lstm", "transformer", "vae", "gan", "resnet", "ensemble"]
        else:
            models_list = models_to_train.split(",")
        
        # Train models in sequence
        trained_models = {}
        
        if "baseline" in models_list:
            trained_models["baseline_cnn"] = self.train_baseline_cnn()
        
        if "enhanced" in models_list:
            trained_models["enhanced_cnn"] = self.train_enhanced_cnn()
            
        if "ast" in models_list:
            trained_models["ast_regressor"] = self.train_ast_regressor()
            
        if "lstm" in models_list:
            trained_models["lstm_mixer"] = self.train_lstm_mixer()
            
        if "transformer" in models_list:
            trained_models["transformer_mixer"] = self.train_transformer_mixer()
            
        if "vae" in models_list:
            trained_models["vae_mixer"] = self.train_vae_mixer()
            
        if "gan" in models_list:
            trained_models["gan_mixer"] = self.train_gan_mixer()
            
        if "resnet" in models_list:
            trained_models["resnet_mixer"] = self.train_resnet_mixer()
            
        # Train ensemble after all models (if requested)
        if "ensemble" in models_list:
            # Filter out None values
            ensemble_models = [model for model in trained_models.values() if model is not None]
            if len(ensemble_models) >= 2:
                trained_models["weighted_ensemble"] = self.train_weighted_ensemble(ensemble_models)
            else:
                print("⚠️ Not enough models available for ensemble training")
        
        # Save training history
        history_file = self.results_dir / "training_history.json"
        with open(history_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_safe_history = {}
            for model_name, history in self.training_history.items():
                json_safe_history[model_name] = {}
                for key, value in history.items():
                    if isinstance(value, np.ndarray):
                        json_safe_history[model_name][key] = value.tolist()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        json_safe_history[model_name][key] = [v.tolist() for v in value]
                    else:
                        json_safe_history[model_name][key] = value
                        
            json.dump(json_safe_history, f, indent=2)
        
        # Evaluate all models
        evaluation_results = self.evaluate_all_models()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print("\n✅ Training pipeline complete!")
        print(f"⏱️  Total training time: {elapsed_time:.1f} seconds")
        print(f"🧠 Models trained: {len(trained_models)}")
        
        # Print evaluation summary
        print("\n📊 Model Performance Summary:")
        if evaluation_results:
            # Sort by MAE (ascending)
            sorted_results = sorted(
                [(name, info["mae"]) for name, info in evaluation_results.items()],
                key=lambda x: x[1]
            )
            
            for i, (name, mae) in enumerate(sorted_results):
                print(f"{i+1}. {name}: MAE = {mae:.4f}")
        
        print(f"\n📁 Training history saved to: {history_file}")
        print(f"📊 Model weights saved to: {self.models_dir}")
        
        return trained_models, evaluation_results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train all AI mixing models")
    parser.add_argument("--models", type=str, default="all",
                        help="Models to train (all/baseline/enhanced/ast/lstm/transformer/vae/gan/resnet/ensemble)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for training")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable saving checkpoints during training")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation during training")
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = ModelTrainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_checkpoints=not args.no_checkpoints,
        patience=args.patience,
        use_augmentation=not args.no_augment
    )
    
    trainer.train_all_models(models_to_train=args.models)
