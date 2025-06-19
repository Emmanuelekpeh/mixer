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
from baseline_cnn import SpectrogramDataset, DEVICE, N_OUTPUTS
from lstm_mixer import LSTMAudioMixer
from audio_gan import AudioGANMixer
from vae_mixer import VAEAudioMixer
from advanced_transformer import AdvancedTransformerMixer
from resnet_mixer import ResNetAudioMixer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewArchitectureTrainer:
    """Trainer for new AI model architectures."""
      def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("../data")
        self.models_dir = Path("../models")
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
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        logger.info(f"Datasets loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        return train_loader, val_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str) -> Dict:
        """Train a single model."""
        logger.info(f"🚀 Training {model_name}...")
        
        model = model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), 
                             lr=self.config['learning_rate'],
                             weight_decay=self.config['weight_decay'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
          # Training metrics
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        epoch = 0  # Initialize epoch counter
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (spectrograms, targets) in enumerate(train_loader):
                spectrograms, targets = spectrograms.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                  # Handle different model outputs
                try:
                    outputs = model(spectrograms)
                    loss = criterion(outputs, targets)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                except Exception as e:
                    logger.warning(f"Batch {batch_idx} failed: {e}")
                    continue
            
            # Validation phase
            model.eval()
            val_loss = 0.0
              with torch.no_grad():
                for spectrograms, targets in val_loader:
                    spectrograms, targets = spectrograms.to(DEVICE), targets.to(DEVICE)
                    
                    try:
                        outputs = model(spectrograms)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                    except Exception as e:
                        continue
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                model_path = self.models_dir / f"{model_name.lower()}_best.pth"
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
            
            # Progress logging
            if epoch % 5 == 0 or epoch == self.config['epochs'] - 1:
                logger.info(f"Epoch {epoch+1}/{self.config['epochs']}: "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"✅ {model_name} training complete! Best Val Loss: {best_val_loss:.6f}")
        logger.info(f"Training time: {training_time:.1f}s")
        
        return {
            'model_name': model_name,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      model_name: str) -> Dict:
        """Evaluate trained model."""
        logger.info(f"📊 Evaluating {model_name}...")
        
        model.eval()
        test_loss = 0.0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for spectrograms, targets in test_loader:
                spectrograms, targets = spectrograms.to(DEVICE), targets.to(DEVICE)
                  try:
                    outputs = model(spectrograms)
                    
                    loss = nn.MSELoss()(outputs, targets)
                    test_loss += loss.item()
                    
                    predictions.append(outputs.cpu().numpy())
                    targets_list.append(targets.cpu().numpy())
                except Exception as e:
                    continue
        
        # Calculate metrics
        avg_test_loss = test_loss / len(test_loader)
        
        # Combine predictions and targets
        all_predictions = np.concatenate(predictions, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        
        # Calculate MAE per parameter
        mae_per_param = np.mean(np.abs(all_predictions - all_targets), axis=0)
        overall_mae = np.mean(mae_per_param)
        
        logger.info(f"✅ {model_name} evaluation complete!")
        logger.info(f"Test Loss: {avg_test_loss:.6f}, MAE: {overall_mae:.6f}")
        
        return {
            'test_loss': avg_test_loss,
            'mae': overall_mae,
            'mae_per_param': mae_per_param.tolist(),
            'predictions_sample': all_predictions[:5].tolist(),
            'targets_sample': all_targets[:5].tolist()
        }
    
    def create_model_metadata(self, model_name: str, architecture: str, 
                            training_results: Dict, evaluation_results: Dict) -> Dict:
        """Create metadata for tournament integration."""
        
        # Define model capabilities based on architecture
        capabilities_map = {
            'LSTM': {
                'temporal_modeling': 0.95,
                'sequence_memory': 0.9,
                'dynamic_adaptation': 0.85,
                'spectral_analysis': 0.75,
                'harmonic_enhancement': 0.7
            },
            'GAN': {
                'creative_generation': 0.9,
                'style_transfer': 0.85,
                'novelty_creation': 0.8,
                'spectral_analysis': 0.7,
                'dynamic_range': 0.75
            },
            'VAE': {
                'latent_modeling': 0.9,
                'smooth_interpolation': 0.85,
                'probabilistic_mixing': 0.8,
                'spectral_analysis': 0.75,
                'dynamic_range': 0.8
            },
            'Transformer': {
                'attention_modeling': 0.95,
                'contextual_understanding': 0.9,
                'harmonic_enhancement': 0.95,
                'spectral_analysis': 0.9,
                'multi_track_coordination': 0.85
            },
            'ResNet': {
                'deep_feature_extraction': 0.9,
                'robustness': 0.95,
                'frequency_analysis': 0.85,
                'spectral_analysis': 0.9,
                'stability': 0.9
            }
        }
        
        specializations_map = {
            'LSTM': ['temporal_processing', 'dynamic_mixing', 'sequential_analysis'],
            'GAN': ['creative_mixing', 'style_transfer', 'generative_enhancement'],
            'VAE': ['latent_manipulation', 'smooth_blending', 'probabilistic_mixing'],
            'Transformer': ['attention_mechanisms', 'contextual_mixing', 'harmonic_analysis'],
            'ResNet': ['deep_processing', 'robust_mixing', 'frequency_analysis']
        }
        
        preferred_genres_map = {
            'LSTM': ['electronic', 'ambient', 'experimental'],
            'GAN': ['experimental', 'creative', 'fusion'],
            'VAE': ['ambient', 'atmospheric', 'experimental'],
            'Transformer': ['orchestral', 'jazz', 'complex_arrangements'],
            'ResNet': ['rock', 'metal', 'high_energy']
        }
        
        # Calculate tier based on performance
        mae = evaluation_results['mae']
        if mae < 0.03:
            tier = "Expert"
            elo_rating = 1600
        elif mae < 0.05:
            tier = "Professional"
            elo_rating = 1400
        elif mae < 0.08:
            tier = "Intermediate"
            elo_rating = 1200
        else:
            tier = "Amateur"
            elo_rating = 1000
        
        metadata = {
            'name': model_name,
            'architecture': architecture,
            'created_at': datetime.now().isoformat(),
            'description': f'{architecture}-based audio mixer with specialized {", ".join(specializations_map.get(architecture, []))}',
            'generation': 1,
            'tier': tier,
            'elo_rating': elo_rating,
            'capabilities': capabilities_map.get(architecture, {}),
            'specializations': specializations_map.get(architecture, []),
            'preferred_genres': preferred_genres_map.get(architecture, []),
            'performance_metrics': {
                'test_loss': evaluation_results['test_loss'],
                'mae': evaluation_results['mae'],
                'training_time': training_results['training_time'],
                'epochs_trained': training_results['epochs_trained']
            },
            'signature_techniques': self._get_signature_techniques(architecture)
        }
        
        return metadata
    
    def _get_signature_techniques(self, architecture: str) -> List[str]:
        """Get signature techniques for each architecture."""
        techniques_map = {
            'LSTM': ['temporal_smoothing', 'dynamic_gating', 'memory_retention'],
            'GAN': ['adversarial_training', 'style_transfer', 'creative_synthesis'],
            'VAE': ['latent_interpolation', 'probabilistic_encoding', 'smooth_generation'],
            'Transformer': ['self_attention', 'cross_modal_fusion', 'positional_encoding'],
            'ResNet': ['residual_connections', 'deep_feature_maps', 'skip_connections']
        }
        return techniques_map.get(architecture, [])
    
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
                
                logger.info(f"💾 Model and metadata saved:")
                logger.info(f"   Model: {model_path}")
                logger.info(f"   Metadata: {metadata_path}")
                
                # Store results
                combined_results = {**training_results, **evaluation_results, **metadata}
                results.append(combined_results)
                successful_models.append(model_name)
                
            except Exception as e:
                logger.error(f"❌ Training failed for {model_name}: {e}", exc_info=True)
                continue
        
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("🏆 TRAINING PIPELINE COMPLETE")
        logger.info('='*60)
        
        if results:
            # Sort by MAE (lower is better)
            results.sort(key=lambda x: x['mae'])
            
            logger.info(f"\n📊 Model Performance Ranking:")
            for i, result in enumerate(results, 1):
                status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                logger.info(f"{status} {result['name']}: MAE = {result['mae']:.6f}, "
                          f"Tier: {result['tier']}, ELO: {result['elo_rating']}")
            
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
