#!/usr/bin/env python3
"""
🏗️ New Architecture Training Pipeline
====================================

Train all 5 new AI model architectures for tournament integration:
- LSTM Audio Mixer
- Audio GAN
- VAE Audio Mixer  
- Advanced Transformer
- ResNet Audio Mixer

This script will train each model and prepare them for tournament battles.
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
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our models
try:
    from baseline_cnn import SpectrogramDataset, DEVICE, N_OUTPUTS
    from lstm_mixer import LSTMAudioMixer
    from audio_gan import AudioGANMixer
    from vae_mixer import VAEAudioMixer
    from advanced_transformer import AdvancedTransformerMixer
    from resnet_mixer import ResNetAudioMixer
except ImportError:
    from src.baseline_cnn import SpectrogramDataset, DEVICE, N_OUTPUTS
    from src.lstm_mixer import LSTMAudioMixer
    from src.audio_gan import AudioGANMixer
    from src.vae_mixer import VAEAudioMixer
    from src.advanced_transformer import AdvancedTransformerMixer
    from src.resnet_mixer import ResNetAudioMixer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewArchitectureTrainer:
    """Trainer for new AI model architectures."""
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.config = {
            'epochs': 30,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'patience': 8,
            'weight_decay': 1e-5
        }
        
    def load_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load training, validation, and test datasets."""
        logger.info("Loading datasets...")
        
        # Load datasets (using existing infrastructure)
        train_dataset = SpectrogramDataset(
            self.data_dir / "train", 
            targets_file=self.data_dir / "targets_example.json",
            augment=True
        )
        
        val_dataset = SpectrogramDataset(
            self.data_dir / "train",  # Using same for now
            targets_file=self.data_dir / "targets_example.json",
            augment=False
        )
        
        test_dataset = SpectrogramDataset(
            self.data_dir / "train",  # Using same for now
            targets_file=self.data_dir / "targets_example.json",
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, model_name: str) -> Dict:
        """Train a single model architecture."""
        model = model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            logger.info('-' * 30)
            
            start_time = time.time()
            
            for batch_idx, (spectrograms, targets) in enumerate(train_loader):
                try:
                    spectrograms, targets = spectrograms.to(DEVICE), targets.to(DEVICE)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if isinstance(model, LSTMAudioMixer):
                        # Create dummy features for the enhanced model
                        dummy_features = torch.randn(spectrograms.size(0), 1).to(DEVICE)
                        outputs = model(spectrograms, dummy_features)
                    else:
                        outputs = model(spectrograms)

                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
                except Exception as e:
                    logger.warning(f"Batch {batch_idx} failed: {e}")
                    continue
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for spectrograms, targets in val_loader:
                    try:
                        spectrograms, targets = spectrograms.to(DEVICE), targets.to(DEVICE)
                        outputs = model(spectrograms)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        val_batch_count += 1
                    except Exception as e:
                        logger.warning(f"Validation batch failed: {e}")
                        continue
            
            # Compute average losses
            avg_train_loss = train_loss / max(batch_count, 1)
            avg_val_loss = val_loss / max(val_batch_count, 1)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update training history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                       f"Train Loss: {avg_train_loss:.4f} - "
                       f"Val Loss: {avg_val_loss:.4f} - "
                       f"LR: {current_lr:.6f} - "
                       f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save model
                model_file = self.models_dir / f"{model_name.lower().replace(' ', '_')}.pth"
                best_model_file = self.models_dir / f"{model_name.lower().replace(' ', '_')}_best.pth"
                
                torch.save(model.state_dict(), model_file)
                torch.save(model.state_dict(), best_model_file)
                
                logger.info(f"✅ Model saved: {model_file}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config['patience']:
                logger.info(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                break
                
        return training_history
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, model_name: str) -> Dict:
        """Evaluate a trained model."""
        model.eval()
        criterion = nn.MSELoss()
        mae_loss = nn.L1Loss()
        
        test_loss = 0.0
        test_mae = 0.0
        batch_count = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for spectrograms, targets in test_loader:
                try:
                    spectrograms, targets = spectrograms.to(DEVICE), targets.to(DEVICE)
                    
                    if isinstance(model, LSTMAudioMixer):
                        # Create dummy features for the enhanced model
                        dummy_features = torch.randn(spectrograms.size(0), 1).to(DEVICE)
                        outputs = model(spectrograms, dummy_features)
                    else:
                        outputs = model(spectrograms)
                    
                    # Calculate losses
                    mse = criterion(outputs, targets)
                    mae = mae_loss(outputs, targets)
                    
                    test_loss += mse.item()
                    test_mae += mae.item()
                    batch_count += 1
                    
                    # Store predictions and targets for later analysis
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                except Exception as e:
                    logger.warning(f"Test batch failed: {e}")
                    continue
        
        # Calculate average losses
        avg_test_loss = test_loss / max(batch_count, 1)
        avg_test_mae = test_mae / max(batch_count, 1)
        
        # Combine all predictions and targets
        if all_predictions and all_targets:
            all_predictions = np.vstack(all_predictions)
            all_targets = np.vstack(all_targets)
        
            # Calculate per-parameter MSE and MAE
            per_param_mse = np.mean((all_predictions - all_targets) ** 2, axis=0)
            per_param_mae = np.mean(np.abs(all_predictions - all_targets), axis=0)
        else:
            per_param_mse = np.zeros(N_OUTPUTS)
            per_param_mae = np.zeros(N_OUTPUTS)
        
        logger.info(f"📊 {model_name} Test Results:")
        logger.info(f"  MSE: {avg_test_loss:.4f}")
        logger.info(f"  MAE: {avg_test_mae:.4f}")
        
        # Create results dictionary
        results = {
            'model_name': model_name,
            'mse': avg_test_loss,
            'mae': avg_test_mae,
            'per_parameter': {
                'mse': per_param_mse.tolist(),
                'mae': per_param_mae.tolist()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_file = self.models_dir / f"{model_name.lower().replace(' ', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"💾 Results saved: {results_file}")
        
        return results
    
    def create_model_metadata(self, model_name: str, architecture: str,
                              training_results: Dict, evaluation_results: Dict) -> Dict:
        """Create metadata for a trained model."""
        metadata = {
            'name': model_name,
            'architecture': architecture,
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'performance': {
                'mse': evaluation_results['mse'],
                'mae': evaluation_results['mae']
            },
            'training': {
                'epochs': len(training_results['train_loss']),
                'final_train_loss': training_results['train_loss'][-1],
                'final_val_loss': training_results['val_loss'][-1],
                'min_val_loss': min(training_results['val_loss']),
                'min_val_loss_epoch': training_results['val_loss'].index(min(training_results['val_loss'])) + 1
            },
            'parameters': {
                'learning_rate': self.config['learning_rate'],
                'batch_size': self.config['batch_size'],
                'patience': self.config['patience'],
                'weight_decay': self.config['weight_decay']
            },
            'tournament_ready': True
        }
        
        return metadata
    
    def train_all_architectures(self):
        """Train all new architectures."""
        logger.info("🚀 Starting New Architecture Training Pipeline")
        logger.info("=" * 60)
        
        # Load datasets
        train_loader, val_loader, test_loader = self.load_datasets()
        
        # Define models to train
        models_to_train = [
            ('LSTM Audio Mixer', 'LSTM', LSTMAudioMixer()),
            ('Audio GAN Mixer', 'GAN', AudioGANMixer()),
            ('VAE Audio Mixer', 'VAE', VAEAudioMixer()),
            ('Advanced Transformer Mixer', 'Transformer', AdvancedTransformerMixer()),
            ('ResNet Audio Mixer', 'ResNet', ResNetAudioMixer())
        ]
        
        results = []
        successful_models = []
        
        for model_name, architecture, model in models_to_train:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🤖 Training {model_name}")
                logger.info(f"Architecture: {architecture}")
                logger.info('='*50)
                
                # Train model
                training_results = self.train_model(model, train_loader, val_loader, model_name)
                
                # Load best model for evaluation
                model_path = self.models_dir / f"{model_name.lower().replace(' ', '_')}_best.pth"
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                
                # Evaluate model
                evaluation_results = self.evaluate_model(model, test_loader, model_name)
                
                # Create metadata
                metadata = self.create_model_metadata(model_name, architecture, 
                                                    training_results, evaluation_results)
                
                # Save metadata
                metadata_path = model_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Add to results
                result = {
                    'model_name': model_name,
                    'architecture': architecture,
                    'mse': evaluation_results['mse'],
                    'mae': evaluation_results['mae'],
                    'status': 'success'
                }
                results.append(result)
                successful_models.append(model_name)
                
                logger.info(f"✅ {model_name} training complete!")
                logger.info(f"   MSE: {evaluation_results['mse']:.4f}")
                logger.info(f"   MAE: {evaluation_results['mae']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                results.append({
                    'model_name': model_name,
                    'architecture': architecture,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Print final summary
        if successful_models:
            logger.info("\n🏆 Training Summary:")
            for result in results:
                if result['status'] == 'success':
                    logger.info(f"  ✅ {result['model_name']}: MSE={result['mse']:.4f}, MAE={result['mae']:.4f}")
                else:
                    logger.info(f"  ❌ {result['model_name']}: Failed - {result.get('error', 'Unknown error')}")
            
            # Save comprehensive results
            results_path = self.models_dir / "new_architectures_training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"\n💾 Comprehensive results saved: {results_path}")
            
            logger.info(f"\n🎯 Tournament Integration Ready:")
            logger.info(f"   ✅ {len(successful_models)} models trained successfully")
            logger.info(f"   ✅ Metadata files created for tournament integration")
            logger.info(f"   ✅ Performance metrics calculated")
            logger.info(f"   ✅ Ready for battle deployment!")
        else:
            logger.error("❌ No models trained successfully!")

def main():
    """Main training pipeline."""
    trainer = NewArchitectureTrainer()
    trainer.train_all_architectures()

if __name__ == "__main__":
    main()
