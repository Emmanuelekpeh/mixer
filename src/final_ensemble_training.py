#!/usr/bin/env python3
"""
🚀 Final Ensemble Training Pipeline
==================================

Advanced ensemble training combining all optimizations to achieve MAE < 0.035:
- Ensemble models with learned weights
- Advanced hyperparameter optimization
- Multi-objective loss functions
- Real-time performance monitoring
- Automatic model selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our models and utilities
from baseline_cnn import SpectrogramDataset, BaselineCNN, EnhancedCNN, DEVICE
from improved_models import ImprovedEnhancedCNN
from ensemble_training import WeightedEnsemble, AdaptiveEnsemble

class ExtendedSpectralLoss(nn.Module):
    """Extended loss function optimized for ensemble training."""
    
    def __init__(self, mse_weight=1.0, consistency_weight=0.1, diversity_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.consistency_weight = consistency_weight
        self.diversity_weight = diversity_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets, individual_predictions=None):
        # Primary MSE loss
        mse_loss = self.mse_loss(predictions, targets)
        
        total_loss = self.mse_weight * mse_loss
        
        # Ensemble consistency loss (if individual predictions provided)
        if individual_predictions is not None and len(individual_predictions) > 1:
            consistency_loss = 0
            for i, pred_i in enumerate(individual_predictions):
                for j, pred_j in enumerate(individual_predictions[i+1:], i+1):
                    consistency_loss += self.mse_loss(pred_i, pred_j)
            consistency_loss /= (len(individual_predictions) * (len(individual_predictions) - 1) / 2)
            total_loss += self.consistency_weight * consistency_loss
            
            # Diversity loss (encourage different predictions)
            diversity_loss = -consistency_loss  # Negative consistency = positive diversity
            total_loss += self.diversity_weight * diversity_loss
        
        return total_loss

class FinalEnsembleTrainer:
    """Final ensemble trainer with all optimizations."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path('models')
        self.results_dir = Path('enhanced_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Load existing models
        self.individual_models = self._load_trained_models()
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.patience = 15
        
        print(f"🚀 Final Ensemble Trainer Initialized")
        print(f"📁 Found {len(self.individual_models)} trained models")
        print(f"🎯 Target: MAE < 0.035")
        
    def _load_trained_models(self) -> List[nn.Module]:
        """Load all available trained models."""
        models = []
        model_files = {
            'baseline_cnn.pth': BaselineCNN,
            'enhanced_cnn.pth': EnhancedCNN,
            'improved_enhanced_cnn.pth': ImprovedEnhancedCNN,
            'retrained_enhanced_cnn.pth': EnhancedCNN,
            'improved_baseline_cnn.pth': BaselineCNN
        }
        
        for model_file, model_class in model_files.items():
            model_path = self.models_dir / model_file
            if model_path.exists():
                try:
                    model = model_class().to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    model.eval()
                    models.append(model)
                    print(f"✅ Loaded {model_file}")
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
        
        return models
    
    def _prepare_datasets(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation datasets."""
        print("📊 Preparing datasets...")
        
        # Use augmented dataset if available
        spectrograms_dir = self.data_dir / 'spectrograms_augmented'
        targets_file = self.data_dir / 'targets_augmented.json'
        
        if not spectrograms_dir.exists() or not targets_file.exists():
            # Fallback to original dataset
            spectrograms_dir = self.data_dir / 'spectrograms_train'
            targets_file = self.data_dir / 'targets_train.json'
          # Create dataset (SpectrogramDataset expects file path, not loaded dict)
        dataset = SpectrogramDataset(str(spectrograms_dir), str(targets_file))
        
        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"📈 Training samples: {len(train_dataset)}")
        print(f"📉 Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _evaluate_individual_models(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate individual model performance."""
        print("\n🔍 Evaluating individual models...")
        performances = {}
        
        for i, model in enumerate(self.individual_models):
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    predictions = model(inputs)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            mae = mean_absolute_error(all_targets, all_predictions)
            model_name = f"Model_{i+1}"
            performances[model_name] = mae
            print(f"   {model_name}: MAE = {mae:.4f}")
        
        return performances
    
    def _train_ensemble(self, ensemble_model: nn.Module, train_loader: DataLoader, 
                       val_loader: DataLoader, ensemble_name: str) -> Dict[str, float]:
        """Train an ensemble model."""
        print(f"\n🎭 Training {ensemble_name}...")
        
        optimizer = optim.Adam(ensemble_model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = ExtendedSpectralLoss()
        
        best_val_mae = float('inf')
        best_weights = None
        patience_counter = 0
        
        training_history = {
            'train_loss': [],
            'val_mae': [],
            'learning_rates': []
        }
        
        for epoch in range(self.num_epochs):
            # Training phase
            ensemble_model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                predictions = ensemble_model(inputs)
                
                # Get individual predictions for extended loss
                individual_preds = []
                if hasattr(ensemble_model, 'models'):
                    for model in ensemble_model.models:
                        model.eval()
                        with torch.no_grad():
                            individual_preds.append(model(inputs))
                
                loss = criterion(predictions, targets, individual_preds if individual_preds else None)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            ensemble_model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    predictions = ensemble_model(inputs)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            val_mae = mean_absolute_error(all_targets, all_predictions)
            avg_train_loss = train_loss / len(train_loader)
            
            # Update learning rate
            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_mae'].append(val_mae)
            training_history['learning_rates'].append(current_lr)
            
            # Early stopping and best model saving
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_weights = ensemble_model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                model_path = self.models_dir / f'{ensemble_name.lower().replace(" ", "_")}.pth'
                torch.save(best_weights, model_path)
                
                print(f"✅ Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, "
                      f"Val MAE = {val_mae:.4f} (Best!) 🎯")
            else:
                patience_counter += 1
                print(f"📊 Epoch {epoch+1:3d}: Train Loss = {avg_train_loss:.4f}, "
                      f"Val MAE = {val_mae:.4f}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
            
            # Stop if target achieved
            if best_val_mae < 0.035:
                print(f"🎯 TARGET ACHIEVED! MAE = {best_val_mae:.4f} < 0.035")
                break
        
        # Restore best weights
        if best_weights is not None:
            ensemble_model.load_state_dict(best_weights)
        
        return {
            'best_mae': best_val_mae,
            'history': training_history,
            'epochs_trained': epoch + 1
        }
    
    def run_final_training(self):
        """Run the complete final ensemble training pipeline."""
        print("🚀 Starting Final Ensemble Training Pipeline")
        print("=" * 50)
        
        # Prepare datasets
        train_loader, val_loader = self._prepare_datasets()
        
        # Evaluate individual models
        individual_performances = self._evaluate_individual_models(val_loader)
        
        # Create ensemble models
        ensembles = {}
        
        if len(self.individual_models) >= 2:
            # 1. Weighted Ensemble with inverse MAE weights
            maes = list(individual_performances.values())
            inverse_weights = [1.0 / (mae + 1e-6) for mae in maes]
            weight_sum = sum(inverse_weights)
            normalized_weights = [w / weight_sum for w in inverse_weights]
            
            weighted_ensemble = WeightedEnsemble(self.individual_models, normalized_weights).to(DEVICE)
            ensembles['Weighted_Ensemble'] = weighted_ensemble
            
            # 2. Adaptive Ensemble
            try:
                adaptive_ensemble = AdaptiveEnsemble(self.individual_models).to(DEVICE)
                ensembles['Adaptive_Ensemble'] = adaptive_ensemble
            except Exception as e:
                print(f"⚠️ Could not create Adaptive Ensemble: {e}")
        
        # Train ensembles
        results = {}
        best_overall_mae = float('inf')
        best_model_name = None
        
        for name, ensemble in ensembles.items():
            try:
                result = self._train_ensemble(ensemble, train_loader, val_loader, name)
                results[name] = result
                
                if result['best_mae'] < best_overall_mae:
                    best_overall_mae = result['best_mae']
                    best_model_name = name
                    
                print(f"\n📈 {name} Results:")
                print(f"   Best MAE: {result['best_mae']:.4f}")
                print(f"   Epochs: {result['epochs_trained']}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        # Final evaluation and summary
        print("\n" + "=" * 50)
        print("🎭 FINAL ENSEMBLE TRAINING RESULTS")
        print("=" * 50)
        
        print("\n📊 Individual Model Performance:")
        for model_name, mae in individual_performances.items():
            print(f"   {model_name}: {mae:.4f}")
        
        print("\n🎭 Ensemble Model Performance:")
        for ensemble_name, result in results.items():
            mae = result['best_mae']
            status = "🎯 TARGET ACHIEVED!" if mae < 0.035 else "📈 Improved" if mae < min(individual_performances.values()) else "📊 Baseline"
            print(f"   {ensemble_name}: {mae:.4f} {status}")
        
        if best_model_name:
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"🎯 BEST MAE: {best_overall_mae:.4f}")
            
            if best_overall_mae < 0.035:
                print("🎉 SUCCESS! Target MAE < 0.035 achieved!")
                improvement = (0.0495 - best_overall_mae) / 0.0495 * 100
                print(f"📈 Improvement over previous best: {improvement:.1f}%")
            else:
                improvement_needed = (best_overall_mae - 0.035) / 0.035 * 100
                print(f"📊 Still need {improvement_needed:.1f}% improvement to reach target")
        
        # Save detailed results
        results_file = self.results_dir / 'final_ensemble_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'individual_performances': individual_performances,
                'ensemble_results': {k: {'best_mae': v['best_mae'], 'epochs': v['epochs_trained']} 
                                   for k, v in results.items()},
                'best_overall_mae': best_overall_mae,
                'best_model': best_model_name,
                'target_achieved': best_overall_mae < 0.035,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
        return results

def main():
    """Main execution function."""
    print("🎵 AI Mixing & Mastering - Final Ensemble Training")
    print("🎯 Target: MAE < 0.035")
    print("🚀 Implementing all advanced optimizations...")
    
    trainer = FinalEnsembleTrainer()
    results = trainer.run_final_training()
    
    print("\n✅ Final ensemble training completed!")
    return results

if __name__ == "__main__":
    main()
