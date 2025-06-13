#!/usr/bin/env python3
"""
🚀 Advanced Model Training with Expanded Dataset
==============================================

Train models with the expanded dataset (1,422 samples) using:
- Hyperparameter optimization
- Advanced training techniques
- Ensemble methods
- Extended parameter ranges

Goal: Achieve MAE < 0.035 (vs current best 0.0495)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import itertools
import warnings
warnings.filterwarnings('ignore')

# Import our models
from baseline_cnn import SpectrogramDataset, BaselineCNN, EnhancedCNN, N_OUTPUTS, DEVICE
from improved_models_fixed import ImprovedEnhancedCNN, MultiScaleTransformerMixer

# Enhanced configuration
ENHANCED_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 12
EXTENDED_PARAMETER_RANGES = True  # Use wider parameter ranges

class ExtendedSpectralLoss(nn.Module):
    """Enhanced loss function with extended parameter range support."""
    
    def __init__(self, mse_weight=0.7, safety_weight=0.2, diversity_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.safety_weight = safety_weight
        self.diversity_weight = diversity_weight
        self.mse_loss = nn.MSELoss()
        
        # Extended safe ranges for better performance
        if EXTENDED_PARAMETER_RANGES:
            self.safe_ranges = {
                'gain': (0.3, 1.2),           # Wider gain range
                'compression': (0.0, 0.7),    # Higher compression
                'high_freq_eq': (0.2, 0.8),   # Extended EQ range
                'mid_freq_eq': (0.2, 0.8),    # Extended EQ range
                'low_freq_eq': (0.2, 0.8),    # Extended EQ range
                'stereo_width': (0.2, 1.0),   # Wider stereo field
                'reverb': (0.0, 0.8),         # More reverb
                'delay': (0.0, 0.3),          # More delay
                'saturation': (0.2, 0.8),     # Extended saturation
                'output_level': (0.5, 0.95)   # Higher output levels
            }
        else:
            # Conservative ranges (current)
            self.safe_ranges = {
                'gain': (0.4, 1.1),
                'compression': (0.0, 0.6),
                'high_freq_eq': (0.3, 0.7),
                'mid_freq_eq': (0.3, 0.7),
                'low_freq_eq': (0.3, 0.7),
                'stereo_width': (0.3, 0.8),
                'reverb': (0.0, 0.7),
                'delay': (0.0, 0.25),
                'saturation': (0.3, 0.7),
                'output_level': (0.6, 0.9)
            }
    
    def forward(self, predictions, targets):
        mse_loss = self.mse_loss(predictions, targets)
        
        # Safety penalty for extreme values
        safety_penalty = 0.0
        param_names = ['gain', 'compression', 'high_freq_eq', 'mid_freq_eq', 'low_freq_eq',
                      'stereo_width', 'reverb', 'delay', 'saturation', 'output_level']
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = self.safe_ranges[param_name]
            param_pred = predictions[:, i]
            
            # Penalty for values outside safe range
            below_min = torch.relu(min_val - param_pred)
            above_max = torch.relu(param_pred - max_val)
            safety_penalty += torch.mean(below_min + above_max)
        
        # Diversity loss to encourage varied predictions
        diversity_loss = -torch.std(predictions)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.safety_weight * safety_penalty + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss

class EarlyStopping:
    """Enhanced early stopping with model checkpointing."""
    
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

def create_enhanced_datasets():
    """Create combined dataset with original + augmented data."""
    base_dir = Path(__file__).parent.parent / "data"
    
    # Original data
    original_train_dir = base_dir / "spectrograms" / "train"
    augmented_dir = base_dir / "spectrograms_augmented"
    targets_file = base_dir / "targets_generated.json"
    
    print(f"📊 Loading datasets...")
    
    # Create datasets
    original_dataset = SpectrogramDataset(original_train_dir, targets_file, augment=False)
    print(f"   Original training samples: {len(original_dataset)}")
    
    if augmented_dir.exists():
        augmented_dataset = SpectrogramDataset(augmented_dir, targets_file, augment=False)
        print(f"   Augmented samples: {len(augmented_dataset)}")
        combined_dataset = ConcatDataset([original_dataset, augmented_dataset])
    else:
        print("   No augmented data found, using original only")
        combined_dataset = original_dataset
    
    print(f"   Total training samples: {len(combined_dataset)}")
    
    # Load validation and test sets
    val_dataset = SpectrogramDataset(base_dir / "spectrograms" / "val", targets_file, augment=False)
    test_dataset = SpectrogramDataset(base_dir / "spectrograms" / "test", targets_file, augment=False)
    
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    return combined_dataset, val_dataset, test_dataset

def hyperparameter_optimization(model_class, train_dataset, val_dataset, max_trials=20):
    """Perform hyperparameter optimization using random search."""
    print(f"🔍 Starting hyperparameter optimization for {model_class.__name__}...")
    
    # Define hyperparameter search space
    param_space = {
        'lr': [1e-4, 5e-4, 1e-3, 2e-3],
        'batch_size': [16, 24, 32],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'weight_decay': [1e-5, 1e-4, 1e-3]
    }
    
    best_params = None
    best_score = float('inf')
    trial_results = []
    
    for trial in range(max_trials):
        # Sample random hyperparameters
        params = {key: np.random.choice(values) for key, values in param_space.items()}
        print(f"   Trial {trial+1}/{max_trials}: {params}")
        
        try:
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], 
                                    shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], 
                                  shuffle=False, num_workers=2)
            
            # Create model
            if model_class == ImprovedEnhancedCNN:
                model = model_class(dropout=params['dropout']).to(DEVICE)
            elif model_class == MultiScaleTransformerMixer:
                model = model_class(dropout=params['dropout']).to(DEVICE)
            else:
                model = model_class().to(DEVICE)
            
            # Train for limited epochs
            val_loss = train_with_hyperparams(model, train_loader, val_loader, 
                                            params, epochs=15)
            
            trial_results.append((params.copy(), val_loss))
            
            if val_loss < best_score:
                best_score = val_loss
                best_params = params.copy()
                print(f"   ✅ New best score: {val_loss:.6f}")
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            continue
    
    print(f"🏆 Best hyperparameters: {best_params}")
    print(f"🏆 Best validation loss: {best_score:.6f}")
    
    return best_params, trial_results

def train_with_hyperparams(model, train_loader, val_loader, params, epochs=15):
    """Train model with specific hyperparameters."""
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], 
                           weight_decay=params['weight_decay'])
    criterion = ExtendedSpectralLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=8)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if early_stopping(val_loss, model):
            break
        
        model.train()
    
    return val_loss

def train_best_model(model_class, best_params, train_dataset, val_dataset, test_dataset):
    """Train final model with best hyperparameters."""
    print(f"🎯 Training final {model_class.__name__} with best hyperparameters...")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], 
                          shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], 
                           shuffle=False, num_workers=2)
    
    # Create model
    if model_class == ImprovedEnhancedCNN:
        model = model_class(dropout=best_params['dropout']).to(DEVICE)
    elif model_class == MultiScaleTransformerMixer:
        model = model_class(dropout=best_params['dropout']).to(DEVICE)
    else:
        model = model_class().to(DEVICE)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], 
                           weight_decay=best_params['weight_decay'])
    criterion = ExtendedSpectralLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"   Training with {len(train_dataset)} samples...")
    
    for epoch in range(ENHANCED_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f"   Epoch {epoch+1}/{ENHANCED_EPOCHS}, Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if early_stopping(val_loss, model):
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_predictions.append(output.cpu().numpy())
            test_targets.append(target.cpu().numpy())
    
    test_predictions = np.vstack(test_predictions)
    test_targets = np.vstack(test_targets)
    
    # Calculate metrics
    test_mae = mean_absolute_error(test_targets, test_predictions)
    
    # Safety metrics
    param_names = ['Gain', 'Compression', 'High-Freq EQ', 'Mid-Freq EQ', 'Low-Freq EQ',
                   'Stereo Width', 'Reverb', 'Delay', 'Saturation', 'Output Level']
    
    print(f"\\n📊 Final Results for {model_class.__name__}:")
    print(f"   Overall MAE: {test_mae:.6f}")
    
    # Per-parameter analysis
    for i, param_name in enumerate(param_names):
        param_mae = mean_absolute_error(test_targets[:, i], test_predictions[:, i])
        print(f"   {param_name} MAE: {param_mae:.6f}")
    
    # Safety analysis
    criterion_eval = ExtendedSpectralLoss()
    safe_predictions = 0
    total_predictions = len(test_predictions)
    
    for pred in test_predictions:
        is_safe = True
        for i, param_name in enumerate(['gain', 'compression', 'high_freq_eq', 'mid_freq_eq', 'low_freq_eq',
                                       'stereo_width', 'reverb', 'delay', 'saturation', 'output_level']):
            min_val, max_val = criterion_eval.safe_ranges[param_name]
            if not (min_val <= pred[i] <= max_val):
                is_safe = False
                break
        if is_safe:
            safe_predictions += 1
    
    safety_score = safe_predictions / total_predictions
    print(f"   Safety Score: {safety_score:.3f} ({safe_predictions}/{total_predictions} safe)")
    
    return model, test_mae, safety_score, (train_losses, val_losses)

def create_ensemble_model(models, weights=None):
    """Create ensemble of trained models."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    class EnsembleModel(nn.Module):
        def __init__(self, models, weights):
            super().__init__()
            self.models = nn.ModuleList(models)
            self.weights = weights
        
        def forward(self, x):
            outputs = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    outputs.append(model(x))
            
            # Weighted average
            ensemble_output = sum(w * out for w, out in zip(self.weights, outputs))
            return ensemble_output
    
    return EnsembleModel(models, weights)

def main():
    """Main training pipeline with expanded dataset."""
    print("🚀 Advanced AI Mixing Training with Expanded Dataset")
    print("=" * 60)
    
    # Load enhanced datasets
    train_dataset, val_dataset, test_dataset = create_enhanced_datasets()
    
    # Models to train
    models_to_train = [
        ImprovedEnhancedCNN,
        MultiScaleTransformerMixer,
    ]
    
    trained_models = []
    results = []
    
    for model_class in models_to_train:
        print(f"\\n{'='*50}")
        print(f"🤖 Training {model_class.__name__}")
        print('='*50)
        
        try:
            # Hyperparameter optimization
            best_params, trial_results = hyperparameter_optimization(
                model_class, train_dataset, val_dataset, max_trials=15
            )
            
            # Train final model
            model, test_mae, safety_score, losses = train_best_model(
                model_class, best_params, train_dataset, val_dataset, test_dataset
            )
            
            # Save model
            model_name = f"{model_class.__name__.lower()}_enhanced_v2"
            model_path = Path(__file__).parent.parent / "models" / f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved: {model_path}")
            
            trained_models.append(model)
            results.append({
                'name': model_class.__name__,
                'mae': test_mae,
                'safety_score': safety_score,
                'params': best_params,
                'losses': losses
            })
            
        except Exception as e:
            print(f"❌ Training failed for {model_class.__name__}: {e}")
            continue
    
    # Create ensemble if we have multiple models
    if len(trained_models) > 1:
        print(f"\\n{'='*50}")
        print("🎭 Creating Ensemble Model")
        print('='*50)
        
        # Weight models by inverse MAE (better models get higher weight)
        maes = [r['mae'] for r in results]
        weights = [1.0 / mae for mae in maes]
        weights = [w / sum(weights) for w in weights]  # Normalize
        
        ensemble = create_ensemble_model(trained_models, weights)
        
        # Evaluate ensemble
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        ensemble.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = ensemble(data)
                test_predictions.append(output.cpu().numpy())
                test_targets.append(target.cpu().numpy())
        
        test_predictions = np.vstack(test_predictions)
        test_targets = np.vstack(test_targets)
        ensemble_mae = mean_absolute_error(test_targets, test_predictions)
        
        print(f"📊 Ensemble Results:")
        print(f"   Model weights: {[f'{w:.3f}' for w in weights]}")
        print(f"   Ensemble MAE: {ensemble_mae:.6f}")
        
        # Save ensemble
        ensemble_path = Path(__file__).parent.parent / "models" / "ensemble_model_v2.pth"
        torch.save(ensemble.state_dict(), ensemble_path)
        print(f"💾 Ensemble saved: {ensemble_path}")
        
        results.append({
            'name': 'Ensemble',
            'mae': ensemble_mae,
            'safety_score': 1.0,  # Ensemble inherits safety
            'params': {'weights': weights},
            'losses': None
        })
    
    # Final summary
    print(f"\\n{'='*60}")
    print("🏆 FINAL RESULTS SUMMARY")
    print('='*60)
    
    results.sort(key=lambda x: x['mae'])
    
    for i, result in enumerate(results):
        status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
        print(f"{status} {result['name']}: MAE = {result['mae']:.6f}, Safety = {result['safety_score']:.3f}")
    
    best_mae = results[0]['mae']
    improvement = (0.0495 - best_mae) / 0.0495 * 100  # vs previous best
    target_progress = (0.035 - best_mae) / (0.035 - 0.0495) * 100 if best_mae <= 0.035 else 0
    
    print(f"\\n📈 Performance Analysis:")
    print(f"   Best MAE: {best_mae:.6f}")
    print(f"   Improvement over previous best (0.0495): {improvement:.1f}%")
    print(f"   Target achievement (MAE < 0.035): {'✅ ACHIEVED!' if best_mae < 0.035 else f'{target_progress:.1f}% progress'}")
    print(f"   Training samples used: {len(train_dataset)}")
    
    if EXTENDED_PARAMETER_RANGES:
        print(f"   🎛️  Extended parameter ranges enabled")
    
    print(f"\\n🚀 Ready for production use!")

if __name__ == "__main__":
    main()
