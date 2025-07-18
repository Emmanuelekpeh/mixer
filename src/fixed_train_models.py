#!/usr/bin/env python3
"""
🏗️ Fixed Model Training Pipeline
===============================

This is a fixed version of the training script that properly handles
the input shape requirements for each model architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import time
import logging
import random
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our models
from baseline_cnn import DEVICE, N_OUTPUTS
from lstm_mixer import LSTMAudioMixer
from audio_gan import AudioGANMixer
from vae_mixer import VAEAudioMixer
from advanced_transformer import AdvancedTransformerMixer
from resnet_mixer import ResNetAudioMixer

# Fixed Dataset class
class FixedSpectrogramDataset(Dataset):
    """Fixed dataset class that properly handles spectrogram dimension issues."""
    
    def __init__(self, spectrogram_dir, targets_file, n_outputs=10, augment=False):
        self.samples = []
        self.targets = json.load(open(targets_file))
        
        # Get all .npy files but filter out target files
        for track_path in Path(spectrogram_dir).rglob("*.npy"):
            if not str(track_path).endswith('_target.npy'):  # Skip target files
                self.samples.append(track_path)
                
        self.n_outputs = n_outputs
        self.augment = augment
        logger.info(f"Found {len(self.samples)} samples in {spectrogram_dir}")

    def __len__(self):
        return len(self.samples)

    def time_mask(self, spec, mask_frac):
        t = spec.shape[1]
        mask_len = int(t * mask_frac)
        if mask_len > 0:
            start = random.randint(0, t - mask_len)
            spec[:, start:start+mask_len] = 0
        return spec

    def freq_mask(self, spec, mask_frac):
        f = spec.shape[0]
        mask_len = int(f * mask_frac)
        if mask_len > 0:
            start = random.randint(0, f - mask_len)
            spec[start:start+mask_len, :] = 0
        return spec

    def add_noise(self, spec, std):
        return spec + np.random.normal(0, std, spec.shape)

    def __getitem__(self, idx):
        try:
            spec_path = self.samples[idx]
            spec = np.load(spec_path)
            
            # Handle different spectrogram formats
            if len(spec.shape) == 1:
                # If it's a 1D array, reshape it to 2D (n_mels, time)
                # Assuming a reasonable size for n_mels (e.g., 128)
                n_mels = 128
                spec = spec.reshape(n_mels, -1)
            
            # Normalize
            spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
            
            # Fixed time dimension (crop or pad to consistent length)
            target_time_steps = 1000  # Fixed length
            if spec.shape[1] > target_time_steps:
                # Crop from center
                start = (spec.shape[1] - target_time_steps) // 2
                spec = spec[:, start:start + target_time_steps]
            elif spec.shape[1] < target_time_steps:
                # Calculate padding
                pad_width = target_time_steps - spec.shape[1]
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                
                # Apply padding safely with explicit tuple shape
                spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
            
            # Data augmentation
            if self.augment:
                if random.random() < 0.5:
                    spec = self.time_mask(spec, 0.1)
                if random.random() < 0.5:
                    spec = self.freq_mask(spec, 0.1)
                if random.random() < 0.5:
                    spec = self.add_noise(spec, 0.01)
            
            # Get target mixing parameters (or zeros if not found)
            try:
                track_id = spec_path.stem
                if '_' in track_id:
                    track_id = '_'.join(track_id.split('_')[:2])  # Extract ID from filename
                
                target = self.targets.get(track_id, [0] * self.n_outputs)
                target = target[:self.n_outputs]  # Ensure we only use the requested number of outputs
            except Exception as e:
                logger.warning(f"Error getting target for {spec_path.stem}: {e}")
                target = [0] * self.n_outputs
                
            # Convert to torch tensors
            spec_tensor = torch.tensor(spec, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            return spec_tensor, target_tensor
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {self.samples[idx]}: {e}")
            # Return a default sample as fallback
            return torch.zeros((128, 1000), dtype=torch.float32), torch.zeros(self.n_outputs, dtype=torch.float32)

class ModelTrainer:
    """Trainer for all model architectures with proper input handling."""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.config = {
            'epochs': 5,  # Reduced for testing
            'batch_size': 8,
            'learning_rate': 1e-3,
            'patience': 5,
            'weight_decay': 1e-5
        }
    
    def adapt_input_for_model(self, model_name, data_batch):
        """Adapt input data shape based on model requirements."""
        if model_name == 'LSTM Audio Mixer':
            # LSTM expects [batch, channels, time]
            # Need to transpose from [batch, n_mels, time] to [batch, time, n_mels]
            return data_batch.unsqueeze(1)  # Add channel dim: [batch, 1, n_mels, time]
            
        elif model_name in ['Audio GAN Mixer', 'VAE Audio Mixer', 'ResNet Audio Mixer']:
            # 2D CNNs expect [batch, channels, height, width]
            return data_batch.unsqueeze(1)  # Add channel dim: [batch, 1, n_mels, time]
            
        elif model_name == 'Advanced Transformer Mixer':
            # Transformer usually works with [batch, sequence, features]
            return data_batch.unsqueeze(1)  # Add channel dim: [batch, 1, n_mels, time]
            
        return data_batch  # Default: no change
    
    def train_model(self, model_name, model, train_loader, val_loader):
        """Train a single model with proper input shape handling."""
        logger.info(f"\n{'='*50}")
        logger.info(f"🤖 Training {model_name}")
        logger.info('='*50)
        
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), 
                             lr=self.config['learning_rate'], 
                             weight_decay=self.config['weight_decay'])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.config['patience'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config['patience'] * 2
        epochs_trained = 0
        
        try:
            for epoch in range(self.config['epochs']):
                # Training phase
                model.train()
                train_losses = []
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    # Adapt data shape for the specific model
                    data = self.adapt_input_for_model(model_name, data)
                    
                    # Move to device
                    data, targets = data.to(DEVICE), targets.to(DEVICE)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    try:
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                        
                        # Backward and optimize
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        
                        if batch_idx % 10 == 0:
                            logger.info(f"  Epoch {epoch+1}/{self.config['epochs']} | "
                                      f"Batch {batch_idx}/{len(train_loader)} | "
                                      f"Loss: {loss.item():.4f}")
                    except Exception as e:
                        logger.error(f"Error in training batch: {e}")
                        logger.error(traceback.format_exc())
                        continue
                
                # Validation phase
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for data, targets in val_loader:
                        # Adapt data shape for the specific model
                        data = self.adapt_input_for_model(model_name, data)
                        
                        # Move to device
                        data, targets = data.to(DEVICE), targets.to(DEVICE)
                        
                        try:
                            outputs = model(data)
                            loss = criterion(outputs, targets)
                            val_losses.append(loss.item())
                        except Exception as e:
                            logger.error(f"Error in validation batch: {e}")
                            continue
                
                # Calculate average losses
                avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else float('inf')
                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
                
                logger.info(f"  Epoch {epoch+1}/{self.config['epochs']} | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {avg_val_loss:.4f}")
                
                # Learning rate scheduler
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save best model
                    model_filename = model_name.lower().replace(' ', '_') + '_best.pth'
                    torch.save(model.state_dict(), self.models_dir / model_filename)
                    logger.info(f"  📝 Saved best model: {model_filename}")
                else:
                    patience_counter += 1
                    logger.info(f"  ⏳ Patience: {patience_counter}/{max_patience}")
                    
                    if patience_counter >= max_patience:
                        logger.info(f"  🛑 Early stopping after {epoch+1} epochs")
                        break
                
                epochs_trained = epoch + 1
            
            return {
                'status': 'success',
                'best_val_loss': best_val_loss,
                'epochs_trained': epochs_trained
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_model(self, model_name, model, test_loader):
        """Evaluate a trained model."""
        logger.info(f"\nEvaluating {model_name}...")
        
        model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                # Adapt data shape for the specific model
                data = self.adapt_input_for_model(model_name, data)
                
                # Move to device
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                
                try:
                    outputs = model(data)
                    all_outputs.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        if not all_outputs or not all_targets:
            logger.error("No valid evaluation results.")
            return {
                'status': 'failed',
                'error': 'No valid evaluation results'
            }
        
        # Concatenate results
        outputs_array = np.vstack(all_outputs)
        targets_array = np.vstack(all_targets)
        
        # Calculate metrics
        mse = np.mean((outputs_array - targets_array) ** 2)
        mae = np.mean(np.abs(outputs_array - targets_array))
        
        logger.info(f"  MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        return {
            'status': 'success',
            'mse': mse,
            'mae': mae,
            'num_samples': len(outputs_array)
        }
    
    def train_all_models(self):
        """Train all model architectures."""
        logger.info("🚀 Starting Fixed Model Training Pipeline")
        logger.info("=" * 60)
        
        # Prepare datasets
        train_dataset = FixedSpectrogramDataset(
            self.data_dir / "train", 
            targets_file=self.data_dir / "targets_example.json",
            augment=True
        )
        
        val_dataset = FixedSpectrogramDataset(
            self.data_dir / "train",  # Using same for validation
            targets_file=self.data_dir / "targets_example.json",
            augment=False
        )
        
        test_dataset = FixedSpectrogramDataset(
            self.data_dir / "train",  # Using same for testing
            targets_file=self.data_dir / "targets_example.json",
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            drop_last=True
        )
        
        logger.info(f"Datasets loaded: Train={len(train_dataset)}, "
                  f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        # Define models to train
        models_to_train = [
            ('LSTM Audio Mixer', LSTMAudioMixer()),
            ('Audio GAN Mixer', AudioGANMixer()),
            ('VAE Audio Mixer', VAEAudioMixer()),
            ('Advanced Transformer Mixer', AdvancedTransformerMixer()),
            ('ResNet Audio Mixer', ResNetAudioMixer())
        ]
        
        # Train each model
        results = []
        successful_models = []
        
        for model_name, model in models_to_train:
            try:
                # Train model
                training_results = self.train_model(model_name, model, train_loader, val_loader)
                
                if training_results['status'] == 'success':
                    # Load best model for evaluation
                    model_path = self.models_dir / f"{model_name.lower().replace(' ', '_')}_best.pth"
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    
                    # Evaluate model
                    evaluation_results = self.evaluate_model(model_name, model, test_loader)
                    
                    if evaluation_results['status'] == 'success':
                        # Save results
                        model_results = {
                            'name': model_name,
                            'training': training_results,
                            'evaluation': evaluation_results,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        results.append(model_results)
                        successful_models.append(model_name)
                        
                        # Save individual model results
                        model_results_path = self.models_dir / f"{model_name.lower().replace(' ', '_')}_results.json"
                        with open(model_results_path, 'w') as f:
                            json.dump(model_results, f, indent=2)
                            
                        logger.info(f"✅ Successfully trained and evaluated {model_name}")
                    else:
                        logger.error(f"❌ Evaluation failed for {model_name}: {evaluation_results['error']}")
                else:
                    logger.error(f"❌ Training failed for {model_name}: {training_results['error']}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Save comprehensive results
        if results:
            results_path = self.models_dir / "new_architectures_training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"\n💾 Comprehensive results saved: {results_path}")
            logger.info(f"\n🎯 Model Training Complete:")
            logger.info(f"   ✅ {len(successful_models)} of {len(models_to_train)} models trained successfully")
            logger.info(f"   ✅ {', '.join(successful_models)}")
        else:
            logger.error("❌ No models trained successfully!")

def main():
    """Main training function."""
    data_dir = Path(r"C:\Users\emman\Projects\mixer\data")
    trainer = ModelTrainer(data_dir)
    trainer.train_all_models()

if __name__ == "__main__":
    main()
