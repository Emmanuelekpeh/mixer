#!/usr/bin/env python3
"""
🚀 Efficient Enhanced Training Pipeline
======================================

Streamlined training with clear progress tracking and improved models.
Goal: Achieve MAE < 0.035 with the expanded dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import json
import time
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import our models
from baseline_cnn import SpectrogramDataset, BaselineCNN, EnhancedCNN, N_OUTPUTS, DEVICE, SPECTROGRAMS_FOLDER, TARGETS_FILE
from improved_models_fixed import ImprovedEnhancedCNN, MultiScaleTransformerMixer

class AdvancedLoss(nn.Module):
    """Enhanced loss function with safety constraints."""
    
    def __init__(self, mse_weight=0.7, safety_weight=0.2, smoothness_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.safety_weight = safety_weight
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        # Base MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Safety penalty for extreme values
        extreme_penalty = torch.mean(torch.relu(predictions - 1.0) + torch.relu(-predictions))
        
        # Smoothness penalty (L1 regularization)
        smoothness_penalty = torch.mean(torch.abs(predictions))
        
        # Combined loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.safety_weight * extreme_penalty + 
                     self.smoothness_weight * smoothness_penalty)
        
        return total_loss

def create_enhanced_data_loaders(batch_size=32):
    """Create data loaders with augmented data."""
    print("📊 Loading datasets...")
    
    # Original data
    original_train = SpectrogramDataset(SPECTROGRAMS_FOLDER, TARGETS_FILE, augment=False)
    
    # Check for augmented data
    augmented_dir = Path(SPECTROGRAMS_FOLDER).parent / "spectrograms_augmented" 
    augmented_targets_file = Path(TARGETS_FILE).parent / "targets_augmented.json"
    
    train_datasets = [original_train]
    
    if augmented_dir.exists() and augmented_targets_file.exists():
        print(f"✅ Found augmented data: {augmented_dir}")
        augmented_train = SpectrogramDataset(augmented_dir, augmented_targets_file, augment=False)
        train_datasets.append(augmented_train)
        print(f"📈 Combined training: {len(original_train)} + {len(augmented_train)} = {len(original_train) + len(augmented_train)} samples")
    else:
        print(f"⚠️ No augmented data found, using original only: {len(original_train)} samples")
    
    # Combine datasets
    combined_train = ConcatDataset(train_datasets)
    
    # Create validation and test sets from original data
    total_original = len(original_train)
    val_size = int(0.15 * total_original)
    test_size = int(0.15 * total_original)
    train_size = total_original - val_size - test_size
    
    # Split original data for val/test
    indices = list(range(total_original))
    np.random.shuffle(indices)
    
    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size + test_size]
    
    # Create validation and test datasets
    val_dataset = SpectrogramDataset(SPECTROGRAMS_FOLDER, TARGETS_FILE, augment=False)
    val_dataset.samples = [original_train.samples[i] for i in val_indices]
    
    test_dataset = SpectrogramDataset(SPECTROGRAMS_FOLDER, TARGETS_FILE, augment=False)
    test_dataset.samples = [original_train.samples[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset splits: Train={len(combined_train)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def train_model_efficiently(model, train_loader, val_loader, epochs=25, lr=1e-3, model_name="model"):
    """Train model with clear progress tracking."""
    print(f"\n🚀 Training {model_name}...")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"🎯 Target: MAE < 0.035")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    criterion = AdvancedLoss()
    
    best_val_mae = float('inf')
    patience_counter = 0
    patience = 8
    
    train_losses = []
    val_maes = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            
            # Progress update every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}", end='\r')
        
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                outputs = model(data)
                
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_maes.append(val_mae)
        scheduler.step(val_mae)
        
        # Progress report
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} | Time: {elapsed:.1f}s")
        
        # Early stopping check
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            # Save best model
            model_path = Path("../models") / f"{model_name}_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"    ✅ New best MAE: {val_mae:.4f} - Model saved!")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"    ⏹️ Early stopping after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete! Best Val MAE: {best_val_mae:.4f} in {total_time:.1f}s")
    
    return best_val_mae, train_losses, val_maes

def evaluate_model_thoroughly(model, test_loader, model_name="model"):
    """Comprehensive model evaluation."""
    print(f"\n🔍 Evaluating {model_name}...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Overall metrics
    mae = mean_absolute_error(targets, predictions)
    mse = np.mean((predictions - targets) ** 2)
    
    # Parameter-specific MAE
    param_names = ["Gain", "Compression", "High_EQ", "Mid_EQ", "Low_EQ", "Reverb", 
                   "Delay", "Stereo_Width", "Output_Level", "Harmonic_Exciter"]
    
    param_maes = [mean_absolute_error(targets[:, i], predictions[:, i]) for i in range(N_OUTPUTS)]
    
    # Safety analysis
    clipped_predictions = np.clip(predictions, 0, 1)
    safety_violations = np.sum(predictions != clipped_predictions)
    safety_score = 1.0 - (safety_violations / predictions.size)
    
    print(f"📊 Overall Results:")
    print(f"   MAE: {mae:.4f} {'🎯' if mae < 0.035 else '⚠️' if mae < 0.05 else '❌'}")
    print(f"   MSE: {mse:.4f}")
    print(f"   Safety Score: {safety_score:.3f}")
    
    print(f"\n📈 Parameter-specific MAE:")
    for i, (name, param_mae) in enumerate(zip(param_names, param_maes)):
        status = "🎯" if param_mae < 0.03 else "✅" if param_mae < 0.05 else "⚠️"
        print(f"   {name:15}: {param_mae:.4f} {status}")
    
    return mae, param_maes, safety_score

def main():
    """Run enhanced training pipeline."""
    print("🚀 Efficient Enhanced Training Pipeline")
    print("=" * 50)
    
    # Load data
    train_loader, val_loader, test_loader = create_enhanced_data_loaders(batch_size=32)
    
    # Models to train
    models_config = [
        {
            "name": "improved_enhanced_cnn_v2",
            "model": ImprovedEnhancedCNN(N_OUTPUTS),
            "lr": 1e-3,
            "epochs": 30
        },
        {
            "name": "improved_baseline_cnn_v2", 
            "model": BaselineCNN(N_OUTPUTS),
            "lr": 5e-4,
            "epochs": 25
        }
    ]
    
    # Add transformer if we have enough data
    if len(train_loader.dataset) > 500:
        models_config.append({
            "name": "transformer_mixer",
            "model": MultiScaleTransformerMixer(input_dim=128*251, output_dim=N_OUTPUTS),
            "lr": 1e-4,
            "epochs": 20
        })
        print("🤖 Added Transformer model (sufficient training data)")
    else:
        print("⚠️ Skipping Transformer (need >500 samples, have {})".format(len(train_loader.dataset)))
    
    results = {}
    
    # Train each model
    for config in models_config:
        print(f"\n" + "="*60)
        model = config["model"].to(DEVICE)
        
        try:
            best_mae, train_losses, val_maes = train_model_efficiently(
                model, train_loader, val_loader, 
                epochs=config["epochs"], 
                lr=config["lr"],
                model_name=config["name"]
            )
            
            # Load best model and evaluate
            model_path = Path("../models") / f"{config['name']}_best.pth"
            if model_path.exists():
                model.load_state_dict(torch.load(model_path))
            
            test_mae, param_maes, safety_score = evaluate_model_thoroughly(
                model, test_loader, config["name"]
            )
            
            results[config["name"]] = {
                "val_mae": best_mae,
                "test_mae": test_mae,
                "param_maes": param_maes,
                "safety_score": safety_score
            }
            
        except Exception as e:
            print(f"❌ Error training {config['name']}: {e}")
            continue
    
    # Results summary
    print(f"\n" + "="*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*60)
    
    best_model = None
    best_mae = float('inf')
    
    for name, result in results.items():
        mae = result["test_mae"]
        safety = result["safety_score"]
        status = "🎯 TARGET ACHIEVED!" if mae < 0.035 else "✅ Good" if mae < 0.05 else "⚠️ Needs work"
        
        print(f"{name:25}: MAE={mae:.4f} Safety={safety:.3f} {status}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = name
    
    print(f"\n🏆 Best Model: {best_model} (MAE: {best_mae:.4f})")
    
    if best_mae < 0.035:
        print("🎯 🎉 TARGET ACHIEVED! MAE < 0.035! 🎉")
    else:
        improvement_needed = ((best_mae - 0.035) / 0.035) * 100
        print(f"📈 Need {improvement_needed:.1f}% more improvement to reach target")
    
    return results

if __name__ == "__main__":
    main()
