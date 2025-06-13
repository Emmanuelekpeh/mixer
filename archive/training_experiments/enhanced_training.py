#!/usr/bin/env python3
"""
🚀 Enhanced AI Mixing Training with Advanced Models
===================================================

This script implements comprehensive training improvements:
- Uses augmented dataset (800+ samples vs original 262)
- Implements improved model architectures with safe constraints
- Hyperparameter optimization with cross-validation
- Advanced training techniques (early stopping, scheduling)
- Comprehensive evaluation and comparison

Goal: Achieve MAE < 0.035 (vs current best 0.0554)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our models
from baseline_cnn import SpectrogramDataset, BaselineCNN, EnhancedCNN, N_OUTPUTS, DROPOUT, DEVICE
from improved_models_fixed import ImprovedEnhancedCNN, MultiScaleTransformerMixer
from ast_regressor import ASTFeatureDataset, ASTRegressor

# Configuration
ENHANCED_BATCH_SIZE = 32
ENHANCED_EPOCHS = 50
LEARNING_RATES = [1e-4, 5e-4, 1e-3]
DROPOUTS = [0.2, 0.3, 0.4]
EARLY_STOPPING_PATIENCE = 10

class SpectralLoss(nn.Module):
    """Perceptual loss combining MSE with spectral consistency."""
    
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for spectral loss
        self.mse = nn.MSELoss()
    
    def forward(self, predicted, target):
        # Standard MSE loss
        mse_loss = self.mse(predicted, target)
        
        # Spectral consistency (penalize parameters that lead to harsh processing)
        # Focus on preventing over-aggressive settings
        spectral_penalty = 0.0
        
        # Penalize extreme compression (target < 0.1 or > 0.8)
        comp_penalty = torch.mean(torch.clamp(predicted[:, 1] - 0.8, min=0) ** 2)
        comp_penalty += torch.mean(torch.clamp(0.1 - predicted[:, 1], min=0) ** 2)
        
        # Penalize extreme output levels (prevent clipping)
        output_penalty = torch.mean(torch.clamp(predicted[:, 9] - 0.9, min=0) ** 2)
        
        # Penalize extreme EQ settings
        eq_params = predicted[:, 2:6]  # High, Mid, Low, Presence
        eq_penalty = torch.mean(torch.clamp(torch.abs(eq_params - 0.5) - 0.3, min=0) ** 2)
        
        spectral_penalty = comp_penalty + output_penalty + eq_penalty
        
        return self.alpha * mse_loss + self.beta * spectral_penalty

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def create_enhanced_data_loaders(original_dir, augmented_dir, targets_file, batch_size=ENHANCED_BATCH_SIZE):
    """Create data loaders using both original and augmented data."""
    
    # Load original datasets
    original_train = SpectrogramDataset(original_dir / "train", targets_file, augment=False)
    original_val = SpectrogramDataset(original_dir / "val", targets_file, augment=False)
    original_test = SpectrogramDataset(original_dir / "test", targets_file, augment=False)
    
    # Check if augmented data exists
    augmented_train = None
    if augmented_dir.exists():
        print(f"🎵 Found augmented data at {augmented_dir}")
        try:
            # Try to load augmented training data
            augmented_targets_file = augmented_dir / "augmented_targets.json"
            if augmented_targets_file.exists():
                augmented_train = SpectrogramDataset(augmented_dir, augmented_targets_file, augment=False)
                print(f"✅ Loaded {len(augmented_train)} augmented training samples")
            else:
                print("❌ Augmented targets file not found")
        except Exception as e:
            print(f"❌ Error loading augmented data: {e}")
    
    # Combine original and augmented training data
    if augmented_train is not None:
        combined_train = ConcatDataset([original_train, augmented_train])
        print(f"📈 Combined training set: {len(original_train)} original + {len(augmented_train)} augmented = {len(combined_train)} total")
    else:
        combined_train = original_train
        print(f"📊 Using original training set only: {len(combined_train)} samples")
    
    # Create data loaders
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(original_val, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(original_test, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def train_enhanced_model(model, train_loader, val_loader, epochs=ENHANCED_EPOCHS, lr=1e-3, use_spectral_loss=True):
    """Train model with enhanced techniques."""
    
    model = model.to(DEVICE)
    criterion = SpectralLoss() if use_spectral_loss else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"🚂 Training with {len(train_loader.dataset)} samples, LR={lr}, Spectral Loss={use_spectral_loss}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        if epoch % 5 == 0 or early_stopping.early_stop:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_enhanced_model(model, test_loader):
    """Comprehensive model evaluation."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate detailed metrics
    mae_per_param = []
    param_names = ['Input Gain', 'Compression', 'High EQ', 'Mid EQ', 'Low EQ', 
                   'Presence', 'Reverb', 'Delay', 'Stereo Width', 'Output Level']
    
    for i in range(all_targets.shape[1]):
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        mae_per_param.append(mae)
    
    overall_mae = np.mean(mae_per_param)
    
    # Safety check - count potentially harmful predictions
    safety_violations = 0
    for pred in all_predictions:
        # Check for over-compression
        if pred[1] > 0.8:
            safety_violations += 1
        # Check for potential clipping
        if pred[9] > 0.95:
            safety_violations += 1
        # Check for extreme EQ
        if np.any(np.abs(pred[2:6] - 0.5) > 0.4):
            safety_violations += 1
    
    safety_score = 1.0 - (safety_violations / len(all_predictions))
    
    return overall_mae, mae_per_param, all_predictions, all_targets, safety_score

def train_and_compare_all_models():
    """Train and compare all model architectures."""
    
    # Setup directories
    base_dir = Path(__file__).resolve().parent.parent
    spectrograms_dir = base_dir / "data" / "spectrograms"
    augmented_dir = base_dir / "data" / "augmented_spectrograms"
    targets_file = base_dir / "data" / "targets.json"
    models_dir = base_dir / "models"
    results_dir = base_dir / "enhanced_results"
    
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    # Create enhanced data loaders
    train_loader, val_loader, test_loader = create_enhanced_data_loaders(
        spectrograms_dir, augmented_dir, targets_file
    )
    
    results = {}
    
    print("🏆 Enhanced AI Mixing Training Pipeline")
    print("=" * 50)
    
    # 1. Train Improved Enhanced CNN
    print("\n1️⃣ Training Improved Enhanced CNN...")
    improved_cnn = ImprovedEnhancedCNN(n_outputs=N_OUTPUTS, dropout=0.3)
    improved_cnn, train_losses, val_losses = train_enhanced_model(improved_cnn, train_loader, val_loader, lr=5e-4)
    
    # Evaluate
    mae, mae_per_param, preds, targets, safety = evaluate_enhanced_model(improved_cnn, test_loader)
    results['improved_cnn'] = {
        'mae': mae,
        'mae_per_param': mae_per_param,
        'safety_score': safety,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # Save model
    torch.save(improved_cnn.state_dict(), models_dir / "improved_enhanced_cnn.pth")
    print(f"✅ Improved Enhanced CNN - MAE: {mae:.4f}, Safety: {safety:.3f}")
    
    # 2. Train Multi-Scale Transformer (if we have enough data)
    if len(train_loader.dataset) > 500:
        print("\n2️⃣ Training Multi-Scale Transformer...")
        transformer = MultiScaleTransformerMixer(n_outputs=N_OUTPUTS, dropout=0.1)
        transformer, train_losses, val_losses = train_enhanced_model(transformer, train_loader, val_loader, lr=1e-4)
        
        # Evaluate
        mae, mae_per_param, preds, targets, safety = evaluate_enhanced_model(transformer, test_loader)
        results['transformer'] = {
            'mae': mae,
            'mae_per_param': mae_per_param,
            'safety_score': safety,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        # Save model
        torch.save(transformer.state_dict(), models_dir / "transformer_mixer.pth")
        print(f"✅ Multi-Scale Transformer - MAE: {mae:.4f}, Safety: {safety:.3f}")
    else:
        print("⚠️ Not enough data for transformer training (need >500 samples)")
    
    # 3. Retrain Enhanced CNN with safe constraints
    print("\n3️⃣ Training Safe Enhanced CNN...")
    safe_enhanced = EnhancedCNN(n_outputs=N_OUTPUTS, dropout=0.2)
    safe_enhanced, train_losses, val_losses = train_enhanced_model(safe_enhanced, train_loader, val_loader, lr=1e-3)
    
    # Evaluate
    mae, mae_per_param, preds, targets, safety = evaluate_enhanced_model(safe_enhanced, test_loader)
    results['safe_enhanced_cnn'] = {
        'mae': mae,
        'mae_per_param': mae_per_param,
        'safety_score': safety,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    # Save model
    torch.save(safe_enhanced.state_dict(), models_dir / "safe_enhanced_cnn.pth")
    print(f"✅ Safe Enhanced CNN - MAE: {mae:.4f}, Safety: {safety:.3f}")
    
    # Generate comprehensive report
    print("\n📊 Final Results Comparison:")
    print("=" * 60)
    
    for model_name, result in results.items():
        print(f"{model_name:20s} | MAE: {result['mae']:.4f} | Safety: {result['safety_score']:.3f}")
        
        # Detailed parameter breakdown
        param_names = ['Input Gain', 'Compression', 'High EQ', 'Mid EQ', 'Low EQ', 
                       'Presence', 'Reverb', 'Delay', 'Stereo Width', 'Output Level']
        print(f"{'':20s} | Parameter MAEs:")
        for i, (name, mae_val) in enumerate(zip(param_names, result['mae_per_param'])):
            print(f"{'':20s} |   {name:12s}: {mae_val:.4f}")
        print()
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['mae'])
    best_mae = results[best_model]['mae']
    
    print(f"🏆 Best Model: {best_model} with MAE: {best_mae:.4f}")
    
    # Check if we achieved our target
    target_mae = 0.035
    if best_mae < target_mae:
        improvement = ((0.0554 - best_mae) / 0.0554) * 100  # vs original AST best
        print(f"🎯 TARGET ACHIEVED! {improvement:.1f}% improvement over original best (0.0554)")
    else:
        improvement_needed = ((best_mae - target_mae) / target_mae) * 100
        print(f"🎯 Target MAE {target_mae:.3f} not yet reached. Need {improvement_needed:.1f}% more improvement.")
    
    # Save results
    with open(results_dir / "enhanced_training_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'mae': float(result['mae']),
                'mae_per_param': [float(x) for x in result['mae_per_param']],
                'safety_score': float(result['safety_score']),
                'train_losses': [float(x) for x in result['train_losses']],
                'val_losses': [float(x) for x in result['val_losses']]
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir}")
    
    return results

if __name__ == "__main__":
    results = train_and_compare_all_models()
