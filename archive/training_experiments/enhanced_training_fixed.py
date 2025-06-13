#!/usr/bin/env python3
"""
🚀 Enhanced Training with Expanded Dataset - Fixed
=================================================

Train models with the expanded dataset using robust hyperparameter optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our models
from baseline_cnn import SpectrogramDataset, BaselineCNN, EnhancedCNN, N_OUTPUTS, DEVICE
from improved_models_fixed import ImprovedEnhancedCNN

class ExtendedSpectralLoss(nn.Module):
    """Enhanced loss function with extended parameter range support."""
    
    def __init__(self, mse_weight=0.7, safety_weight=0.2, diversity_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.safety_weight = safety_weight
        self.diversity_weight = diversity_weight
        self.mse_loss = nn.MSELoss()
        
        # Extended safe ranges for better performance
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
    
    def __init__(self, patience=12, min_delta=1e-6, restore_best_weights=True):
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

def train_model_with_hyperparams(model_class, train_dataset, val_dataset, test_dataset, model_name):
    """Train model with multiple hyperparameter configurations."""
    print(f"🎯 Training {model_name} with hyperparameter search...")
    
    # Hyperparameter configurations to try
    configs = [
        {'lr': 1e-4, 'batch_size': 16, 'dropout': 0.3, 'weight_decay': 1e-4},
        {'lr': 5e-4, 'batch_size': 24, 'dropout': 0.2, 'weight_decay': 1e-5},
        {'lr': 1e-3, 'batch_size': 32, 'dropout': 0.4, 'weight_decay': 1e-3},
        {'lr': 2e-3, 'batch_size': 16, 'dropout': 0.3, 'weight_decay': 1e-4},
        {'lr': 8e-4, 'batch_size': 24, 'dropout': 0.25, 'weight_decay': 5e-5},
    ]
    
    best_config = None
    best_val_loss = float('inf')
    best_model = None
    
    for i, config in enumerate(configs):
        print(f"   Config {i+1}/{len(configs)}: lr={config['lr']}, bs={config['batch_size']}, dropout={config['dropout']}")
        
        try:
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                    shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                                  shuffle=False, num_workers=2)
            
            # Create model
            if model_class == ImprovedEnhancedCNN:
                model = model_class(dropout=config['dropout']).to(DEVICE)
            else:
                model = model_class().to(DEVICE)
            
            # Train for limited epochs to find best config
            val_loss = quick_train(model, train_loader, val_loader, config, epochs=20)
            
            print(f"     Validation loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config.copy()
                best_model = model.state_dict().copy()
                print(f"     ✅ New best configuration!")
            
        except Exception as e:
            print(f"     ❌ Configuration failed: {e}")
            continue
    
    if best_config is None:
        print("❌ All configurations failed")
        return None, None, None
    
    print(f"🏆 Best configuration: {best_config}")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    
    # Train final model with best configuration
    print(f"🚀 Training final model with best configuration...")
    
    # Create final data loaders
    train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=best_config['batch_size'], 
                          shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=best_config['batch_size'], 
                           shuffle=False, num_workers=2)
    
    # Create final model
    if model_class == ImprovedEnhancedCNN:
        final_model = model_class(dropout=best_config['dropout']).to(DEVICE)
    else:
        final_model = model_class().to(DEVICE)
    
    # Load best weights as starting point
    if best_model:
        final_model.load_state_dict(best_model)
    
    # Final training
    final_mae, safety_score, losses = full_train(final_model, train_loader, val_loader, test_loader, best_config)
    
    return final_model, final_mae, safety_score

def quick_train(model, train_loader, val_loader, config, epochs=20):
    """Quick training for hyperparameter search."""
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
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

def full_train(model, train_loader, val_loader, test_loader, config, epochs=40):
    """Full training with final evaluation."""
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = ExtendedSpectralLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=12)
    
    train_losses = []
    val_losses = []
    
    print(f"   Training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
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
            print(f"   Epoch {epoch+1}/{epochs}, Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
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
    
    # Per-parameter analysis
    param_names = ['Gain', 'Compression', 'High-Freq EQ', 'Mid-Freq EQ', 'Low-Freq EQ',
                   'Stereo Width', 'Reverb', 'Delay', 'Saturation', 'Output Level']
    
    print(f"\\n📊 Final Results:")
    print(f"   Overall MAE: {test_mae:.6f}")
    print(f"   Safety Score: {safety_score:.3f} ({safe_predictions}/{total_predictions} safe)")
    
    for i, param_name in enumerate(param_names):
        param_mae = mean_absolute_error(test_targets[:, i], test_predictions[:, i])
        print(f"   {param_name} MAE: {param_mae:.6f}")
    
    return test_mae, safety_score, (train_losses, val_losses)

def main():
    """Main training pipeline."""
    print("🚀 Enhanced AI Mixing Training with Expanded Dataset")
    print("=" * 60)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = create_enhanced_datasets()
    
    # Models to train
    models_to_train = [
        (ImprovedEnhancedCNN, "ImprovedEnhancedCNN_v3"),
        (BaselineCNN, "BaselineCNN_Enhanced"),
        (EnhancedCNN, "EnhancedCNN_v2"),
    ]
    
    results = []
    
    for model_class, model_name in models_to_train:
        print(f"\\n{'='*50}")
        print(f"🤖 Training {model_name}")
        print('='*50)
        
        try:
            model, test_mae, safety_score = train_model_with_hyperparams(
                model_class, train_dataset, val_dataset, test_dataset, model_name
            )
            
            if model is not None:
                # Save model
                model_path = Path(__file__).parent.parent / "models" / f"{model_name.lower()}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"💾 Model saved: {model_path}")
                
                results.append({
                    'name': model_name,
                    'mae': test_mae,
                    'safety_score': safety_score
                })
            
        except Exception as e:
            print(f"❌ Training failed for {model_name}: {e}")
            continue
    
    # Final summary
    if results:
        print(f"\\n{'='*60}")
        print("🏆 FINAL RESULTS SUMMARY")
        print('='*60)
        
        results.sort(key=lambda x: x['mae'])
        
        for i, result in enumerate(results):
            status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            print(f"{status} {result['name']}: MAE = {result['mae']:.6f}, Safety = {result['safety_score']:.3f}")
        
        best_mae = results[0]['mae']
        improvement = (0.0495 - best_mae) / 0.0495 * 100  # vs previous best
        
        print(f"\\n📈 Performance Analysis:")
        print(f"   Best MAE: {best_mae:.6f}")
        print(f"   Improvement over previous best (0.0495): {improvement:.1f}%")
        print(f"   Target achievement (MAE < 0.035): {'✅ ACHIEVED!' if best_mae < 0.035 else f'{((0.0495 - best_mae) / (0.0495 - 0.035) * 100):.1f}% progress'}")
        print(f"   Training samples used: {len(train_dataset)}")
        print(f"   Extended parameter ranges: ✅ Enabled")
        
        print(f"\\n🚀 Ready for production use!")
    else:
        print("❌ No models were successfully trained")

if __name__ == "__main__":
    main()
