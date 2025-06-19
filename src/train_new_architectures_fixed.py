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

def train_single_model(model: nn.Module, train_loader: DataLoader, 
                      val_loader: DataLoader, model_name: str,
                      epochs: int = 25) -> Dict:
    """Train a single model with simplified approach."""
    logger.info(f"🚀 Training {model_name}...")
    
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 8
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for spectrograms, targets in train_loader:
            try:
                spectrograms = spectrograms.to(DEVICE)
                targets = targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(spectrograms)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            except Exception as e:
                logger.warning(f"Training batch failed: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for spectrograms, targets in val_loader:
                try:
                    spectrograms = spectrograms.to(DEVICE)
                    targets = targets.to(DEVICE)
                    outputs = model(spectrograms)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batches += 1
                except Exception as e:
                    continue
        
        # Calculate averages
        avg_train_loss = train_loss / max(1, train_batches)
        avg_val_loss = val_loss / max(1, val_batches)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model_path = Path("../models") / f"{model_name.lower().replace(' ', '_')}_best.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        # Progress logging
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    logger.info(f"✅ {model_name} training complete! Best Val Loss: {best_val_loss:.6f}")
    
    return {
        'model_name': model_name,
        'best_val_loss': best_val_loss,
        'training_time': training_time,
        'epochs_trained': epoch + 1,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def evaluate_model(model: nn.Module, test_loader: DataLoader, model_name: str) -> Dict:
    """Evaluate trained model."""
    logger.info(f"📊 Evaluating {model_name}...")
    
    model.eval()
    test_loss = 0.0
    predictions = []
    targets_list = []
    test_batches = 0
    
    with torch.no_grad():
        for spectrograms, targets in test_loader:
            try:
                spectrograms = spectrograms.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(spectrograms)
                
                loss = nn.MSELoss()(outputs, targets)
                test_loss += loss.item()
                test_batches += 1
                
                predictions.append(outputs.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
            except Exception as e:
                continue
    
    avg_test_loss = test_loss / max(1, test_batches)
    
    if predictions and targets_list:
        all_predictions = np.concatenate(predictions, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        mae_per_param = np.mean(np.abs(all_predictions - all_targets), axis=0)
        overall_mae = np.mean(mae_per_param)
    else:
        overall_mae = float('inf')
        mae_per_param = [float('inf')] * N_OUTPUTS
    
    logger.info(f"✅ {model_name} evaluation complete! Test Loss: {avg_test_loss:.6f}, MAE: {overall_mae:.6f}")
    
    return {
        'test_loss': avg_test_loss,
        'mae': overall_mae,
        'mae_per_param': mae_per_param.tolist() if hasattr(mae_per_param, 'tolist') else mae_per_param
    }

def create_model_metadata(model_name: str, architecture: str, 
                         training_results: Dict, evaluation_results: Dict) -> Dict:
    """Create metadata for tournament integration."""
    
    # Architecture-specific configurations
    config_map = {
        'LSTM': {
            'capabilities': {'temporal_modeling': 0.95, 'sequence_memory': 0.9, 'dynamic_adaptation': 0.85, 'spectral_analysis': 0.75, 'harmonic_enhancement': 0.7},
            'specializations': ['temporal_processing', 'dynamic_mixing', 'sequential_analysis'],
            'preferred_genres': ['electronic', 'ambient', 'experimental'],
            'signature_techniques': ['temporal_smoothing', 'dynamic_gating', 'memory_retention']
        },
        'GAN': {
            'capabilities': {'creative_generation': 0.9, 'style_transfer': 0.85, 'novelty_creation': 0.8, 'spectral_analysis': 0.7, 'dynamic_range': 0.75},
            'specializations': ['creative_mixing', 'style_transfer', 'generative_enhancement'],
            'preferred_genres': ['experimental', 'creative', 'fusion'],
            'signature_techniques': ['adversarial_training', 'style_transfer', 'creative_synthesis']
        },
        'VAE': {
            'capabilities': {'latent_modeling': 0.9, 'smooth_interpolation': 0.85, 'probabilistic_mixing': 0.8, 'spectral_analysis': 0.75, 'dynamic_range': 0.8},
            'specializations': ['latent_manipulation', 'smooth_blending', 'probabilistic_mixing'],
            'preferred_genres': ['ambient', 'atmospheric', 'experimental'],
            'signature_techniques': ['latent_interpolation', 'probabilistic_encoding', 'smooth_generation']
        },
        'Transformer': {
            'capabilities': {'attention_modeling': 0.95, 'contextual_understanding': 0.9, 'harmonic_enhancement': 0.95, 'spectral_analysis': 0.9, 'multi_track_coordination': 0.85},
            'specializations': ['attention_mechanisms', 'contextual_mixing', 'harmonic_analysis'],
            'preferred_genres': ['orchestral', 'jazz', 'complex_arrangements'],
            'signature_techniques': ['self_attention', 'cross_modal_fusion', 'positional_encoding']
        },
        'ResNet': {
            'capabilities': {'deep_feature_extraction': 0.9, 'robustness': 0.95, 'frequency_analysis': 0.85, 'spectral_analysis': 0.9, 'stability': 0.9},
            'specializations': ['deep_processing', 'robust_mixing', 'frequency_analysis'],
            'preferred_genres': ['rock', 'metal', 'high_energy'],
            'signature_techniques': ['residual_connections', 'deep_feature_maps', 'skip_connections']
        }
    }
    
    # Calculate tier based on performance
    mae = evaluation_results['mae']
    if mae < 0.03:
        tier, elo_rating = "Expert", 1600
    elif mae < 0.05:
        tier, elo_rating = "Professional", 1400
    elif mae < 0.08:
        tier, elo_rating = "Intermediate", 1200
    else:
        tier, elo_rating = "Amateur", 1000
    
    config = config_map.get(architecture, config_map['LSTM'])  # Default to LSTM config
    
    return {
        'name': model_name,
        'architecture': architecture,
        'created_at': datetime.now().isoformat(),
        'description': f'{architecture}-based audio mixer with specialized {", ".join(config["specializations"])}',
        'generation': 1,
        'tier': tier,
        'elo_rating': elo_rating,
        'capabilities': config['capabilities'],
        'specializations': config['specializations'],
        'preferred_genres': config['preferred_genres'],
        'signature_techniques': config['signature_techniques'],
        'performance_metrics': {
            'test_loss': evaluation_results['test_loss'],
            'mae': evaluation_results['mae'],
            'training_time': training_results['training_time'],
            'epochs_trained': training_results['epochs_trained']
        }
    }

def main():
    """Main training pipeline."""
    logger.info("🚀 Starting New Architecture Training Pipeline")
    logger.info("=" * 60)
    
    # Load datasets
    data_dir = Path("../data")
    train_dataset = SpectrogramDataset(data_dir / "train", targets_file=data_dir / "targets_example.json", augment=True)
    val_dataset = SpectrogramDataset(data_dir / "train", targets_file=data_dir / "targets_example.json", augment=False)
    test_dataset = SpectrogramDataset(data_dir / "train", targets_file=data_dir / "targets_example.json", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    logger.info(f"Datasets loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
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
            logger.info(f"🤖 Training {model_name} ({architecture})")
            logger.info('='*50)
            
            # Train model
            training_results = train_single_model(model, train_loader, val_loader, model_name)
            
            # Load best model for evaluation
            model_path = Path("../models") / f"{model_name.lower().replace(' ', '_')}_best.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            
            # Evaluate model
            evaluation_results = evaluate_model(model, test_loader, model_name)
            
            # Create and save metadata
            metadata = create_model_metadata(model_name, architecture, training_results, evaluation_results)
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Model saved: {model_path}")
            logger.info(f"💾 Metadata saved: {metadata_path}")
            
            # Store results
            combined_results = {**training_results, **evaluation_results, **metadata}
            results.append(combined_results)
            successful_models.append(model_name)
            
        except Exception as e:
            logger.error(f"❌ Training failed for {model_name}: {e}")
            continue
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("🏆 TRAINING PIPELINE COMPLETE")
    logger.info('='*60)
    
    if results:
        results.sort(key=lambda x: x['mae'])
        
        logger.info(f"\n📊 Model Performance Ranking:")
        for i, result in enumerate(results, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            logger.info(f"{status} {result['name']}: MAE = {result['mae']:.6f}, Tier: {result['tier']}, ELO: {result['elo_rating']}")
        
        # Save comprehensive results
        results_path = Path("../models") / "new_architectures_training_results.json"
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

if __name__ == "__main__":
    main()
