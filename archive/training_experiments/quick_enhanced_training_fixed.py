#!/usr/bin/env python3
"""
🚀 Quick Enhanced Training with Improved Models
===============================================

Simplified version that works with existing data structure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our models
from baseline_cnn import SpectrogramDataset, BaselineCNN, EnhancedCNN, N_OUTPUTS, DROPOUT, DEVICE, SPECTROGRAMS_FOLDER, TARGETS_FILE
from improved_models_fixed import ImprovedEnhancedCNN

class SpectralLoss(nn.Module):
    """Enhanced loss function for better mixing parameter prediction."""
    
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, predicted, target):
        mse_loss = self.mse(predicted, target)
        
        # Penalty for extreme values that could cause audio artifacts
        extreme_penalty = 0.0
        
        # Compression penalty (avoid over-compression)
        comp_penalty = torch.mean(torch.clamp(predicted[:, 1] - 0.7, min=0) ** 2)
        
        # Output level penalty (prevent clipping)
        output_penalty = torch.mean(torch.clamp(predicted[:, 9] - 0.85, min=0) ** 2)
        
        extreme_penalty = comp_penalty + output_penalty
        
        return self.alpha * mse_loss + self.beta * extreme_penalty

def train_improved_model(model, train_loader, val_loader, epochs=30, lr=1e-3):
    """Train model with improved techniques."""
    
    model = model.to(DEVICE)
    criterion = SpectralLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"🚂 Training with {len(train_loader.dataset)} samples")
    
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
            
            # Gradient clipping
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
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if patience_counter >= 8:  # Early stopping
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_improved_model(model, test_loader):
    """Evaluate model with safety metrics."""
    
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
    
    # Calculate MAE per parameter
    param_names = ['Input Gain', 'Compression', 'High EQ', 'Mid EQ', 'Low EQ', 
                   'Presence', 'Reverb', 'Delay', 'Stereo Width', 'Output Level']
    
    mae_per_param = []
    for i in range(all_targets.shape[1]):
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        mae_per_param.append(mae)
    
    overall_mae = np.mean(mae_per_param)
    
    # Safety analysis
    safety_violations = 0
    for pred in all_predictions:
        if pred[1] > 0.7:  # Over-compression
            safety_violations += 1
        if pred[9] > 0.9:  # Potential clipping
            safety_violations += 1
    
    safety_score = 1.0 - (safety_violations / len(all_predictions))
    
    return overall_mae, mae_per_param, safety_score, all_predictions, all_targets

def main():
    """Main training and comparison function."""
    
    print("🚀 Enhanced AI Mixing Training")
    print("=" * 40)
    
    # Create data loaders
    from baseline_cnn import create_data_loaders
    train_loader, val_loader, test_loader = create_data_loaders(SPECTROGRAMS_FOLDER, TARGETS_FILE, batch_size=32)
    
    print(f"📊 Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    results = {}
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # 1. Train Improved Enhanced CNN
    print("\n1️⃣ Training Improved Enhanced CNN...")
    improved_model = ImprovedEnhancedCNN(n_outputs=N_OUTPUTS, dropout=0.3)
    improved_model = train_improved_model(improved_model, train_loader, val_loader, lr=5e-4)
    
    # Evaluate
    mae, mae_per_param, safety, preds, targets = evaluate_improved_model(improved_model, test_loader)
    results['improved_enhanced_cnn'] = {
        'mae': mae,
        'mae_per_param': mae_per_param,
        'safety_score': safety
    }
    
    # Save model
    torch.save(improved_model.state_dict(), models_dir / "improved_enhanced_cnn.pth")
    print(f"✅ Improved Enhanced CNN - MAE: {mae:.4f}, Safety: {safety:.3f}")
    
    # 2. Retrain original Enhanced CNN with better techniques
    print("\n2️⃣ Retraining Enhanced CNN with improved techniques...")
    enhanced_model = EnhancedCNN(n_outputs=N_OUTPUTS, dropout=0.25)
    enhanced_model = train_improved_model(enhanced_model, train_loader, val_loader, lr=1e-3)
    
    # Evaluate
    mae, mae_per_param, safety, preds, targets = evaluate_improved_model(enhanced_model, test_loader)
    results['retrained_enhanced_cnn'] = {
        'mae': mae,
        'mae_per_param': mae_per_param,
        'safety_score': safety
    }
    
    # Save model
    torch.save(enhanced_model.state_dict(), models_dir / "retrained_enhanced_cnn.pth")
    print(f"✅ Retrained Enhanced CNN - MAE: {mae:.4f}, Safety: {safety:.3f}")
    
    # 3. Compare with baseline for reference
    print("\n3️⃣ Training Baseline CNN for comparison...")
    baseline_model = BaselineCNN(n_outputs=N_OUTPUTS, dropout=DROPOUT, n_conv_layers=3)
    baseline_model = train_improved_model(baseline_model, train_loader, val_loader, lr=1e-3)
    
    # Evaluate
    mae, mae_per_param, safety, preds, targets = evaluate_improved_model(baseline_model, test_loader)
    results['improved_baseline_cnn'] = {
        'mae': mae,
        'mae_per_param': mae_per_param,
        'safety_score': safety
    }
    
    # Save model
    torch.save(baseline_model.state_dict(), models_dir / "improved_baseline_cnn.pth")
    print(f"✅ Improved Baseline CNN - MAE: {mae:.4f}, Safety: {safety:.3f}")
    
    # Final comparison
    print("\n📊 ENHANCED TRAINING RESULTS")
    print("=" * 50)
    
    param_names = ['Input Gain', 'Compression', 'High EQ', 'Mid EQ', 'Low EQ', 
                   'Presence', 'Reverb', 'Delay', 'Stereo Width', 'Output Level']
    
    for model_name, result in results.items():
        print(f"\n🎯 {model_name.upper().replace('_', ' ')}:")
        print(f"   Overall MAE: {result['mae']:.4f}")
        print(f"   Safety Score: {result['safety_score']:.3f}")
        print("   Parameter MAEs:")
        for i, (param, mae_val) in enumerate(zip(param_names, result['mae_per_param'])):
            print(f"     {param:12s}: {mae_val:.4f}")
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['mae'])
    best_mae = results[best_model]['mae']
    
    print(f"\n🏆 BEST MODEL: {best_model.upper().replace('_', ' ')}")
    print(f"🎯 Best MAE: {best_mae:.4f}")
    
    # Compare with original performance
    original_best = 0.0554  # AST Regressor from conversation summary
    if best_mae < original_best:
        improvement = ((original_best - best_mae) / original_best) * 100
        print(f"🚀 IMPROVEMENT: {improvement:.1f}% better than original best!")
    else:
        print(f"📈 Current best still needs improvement vs original AST ({original_best:.4f})")
    
    # Check if target achieved
    target_mae = 0.035
    if best_mae < target_mae:
        print(f"🎯 TARGET ACHIEVED! MAE < {target_mae:.3f}")
    else:
        remaining = ((best_mae - target_mae) / target_mae) * 100
        print(f"🎯 Target progress: {remaining:.1f}% improvement still needed")
    
    return results

if __name__ == "__main__":
    results = main()
