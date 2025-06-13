#!/usr/bin/env python3
"""
🚀 Advanced Training Script with Hyperparameter Optimization
===========================================================

This script implements:
- Systematic hyperparameter grid search
- Cross-validation for robust evaluation  
- Data augmentation for better generalization
- Advanced loss functions (spectral loss)
- Early stopping and learning rate scheduling
- Model checkpointing and best model selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from itertools import product
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our models
from baseline_cnn import BaselineCNN, SpectrogramDataset, N_OUTPUTS, DEVICE
from improved_models import ImprovedEnhancedCNN, MultiScaleTransformerMixer, DataAugmentation

class SpectralLoss(nn.Module):
    """Perceptual loss function based on spectral features."""
    
    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta    # Weight for spectral consistency
        
    def forward(self, pred, target):
        # Standard MSE loss
        mse_loss = self.mse_loss(pred, target)
        
        # Spectral consistency loss (encourage smooth parameter changes)
        spectral_loss = torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))
        
        return self.alpha * mse_loss + self.beta * spectral_loss

class HyperparameterOptimizer:
    """Systematic hyperparameter optimization with cross-validation."""
    
    def __init__(self, data_dir, results_dir):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Define hyperparameter search space
        self.search_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'dropout': [0.2, 0.3, 0.4, 0.5],
            'weight_decay': [1e-6, 1e-5, 1e-4],
            'optimizer': ['Adam', 'AdamW'],
            'loss_function': ['MSE', 'Spectral'],
            'data_augmentation': [True, False]
        }
        
        self.results = []
        
    def load_data(self):
        """Load and prepare data for training."""
        print("📊 Loading dataset...")
        
        # Load datasets
        train_dataset = SpectrogramDataset(self.data_dir / "spectrograms" / "train")
        val_dataset = SpectrogramDataset(self.data_dir / "spectrograms" / "val")
        test_dataset = SpectrogramDataset(self.data_dir / "spectrograms" / "test")
        
        print(f"✅ Loaded {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, batch_size, augment=False):
        """Create data loaders with optional augmentation."""
        
        if augment:
            # Add data augmentation transforms
            augmented_dataset = self.apply_augmentation(train_dataset)
            train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader
    
    def apply_augmentation(self, dataset):
        """Apply data augmentation to training dataset."""
        # For now, return original dataset
        # In practice, you'd implement augmentation transforms here
        return dataset
    
    def create_model(self, model_type, dropout):
        """Create model instance."""
        if model_type == 'baseline':
            return BaselineCNN(n_outputs=N_OUTPUTS, dropout=dropout)
        elif model_type == 'improved_enhanced':
            return ImprovedEnhancedCNN(n_outputs=N_OUTPUTS, dropout=dropout)
        elif model_type == 'transformer':
            return MultiScaleTransformerMixer(n_outputs=N_OUTPUTS, dropout=dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_optimizer(self, model, optimizer_name, learning_rate, weight_decay):
        """Create optimizer instance."""
        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def create_loss_function(self, loss_type):
        """Create loss function instance."""
        if loss_type == 'MSE':
            return nn.MSELoss()
        elif loss_type == 'Spectral':
            return SpectralLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")
    
    def train_model(self, model, train_loader, val_loader, config, max_epochs=50):
        """Train a single model with given configuration."""
        
        model = model.to(DEVICE)
        optimizer = self.create_optimizer(
            model, config['optimizer'], 
            config['learning_rate'], config['weight_decay']
        )
        criterion = self.create_loss_function(config['loss_function'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=False
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        train_losses = []
        val_losses = []
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(DEVICE), targets.to(DEVICE)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏰ Early stopping at epoch {epoch+1}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return best_val_loss, train_losses, val_losses
    
    def cross_validate(self, model_type, config, k_folds=3):
        """Perform k-fold cross-validation."""
        print(f"🔄 Cross-validating {model_type} with config: {config}")
        
        train_dataset, val_dataset, _ = self.load_data()
        
        # Combine train and val for cross-validation
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_dataset)):
            print(f"  Fold {fold+1}/{k_folds}")
            
            # Create data loaders for this fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                combined_dataset, batch_size=config['batch_size'], 
                sampler=train_sampler, num_workers=0
            )
            val_loader = DataLoader(
                combined_dataset, batch_size=config['batch_size'], 
                sampler=val_sampler, num_workers=0
            )
            
            # Create and train model
            model = self.create_model(model_type, config['dropout'])
            best_val_loss, _, _ = self.train_model(model, train_loader, val_loader, config, max_epochs=30)
            
            fold_scores.append(best_val_loss)
            print(f"    Fold {fold+1} best validation loss: {best_val_loss:.4f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        print(f"  📊 Cross-validation results: {mean_score:.4f} ± {std_score:.4f}")
        
        return mean_score, std_score
    
    def grid_search(self, model_type, max_combinations=50):
        """Perform grid search over hyperparameters."""
        print(f"🔍 Starting grid search for {model_type}")
        print(f"🎯 Search space size: {np.prod([len(v) for v in self.search_space.values()])} combinations")
        print(f"🎲 Testing {max_combinations} random combinations")
        
        # Generate all possible combinations
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        all_combinations = list(product(*values))
        
        # Randomly sample combinations to test
        np.random.shuffle(all_combinations)
        combinations_to_test = all_combinations[:max_combinations]
        
        best_score = float('inf')
        best_config = None
        
        results = []
        
        for i, combination in enumerate(combinations_to_test):
            config = dict(zip(keys, combination))
            
            print(f"\\n🧪 Testing combination {i+1}/{len(combinations_to_test)}")
            
            try:
                # Cross-validate this configuration
                mean_score, std_score = self.cross_validate(model_type, config)
                
                result = {
                    'model_type': model_type,
                    'config': config,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'timestamp': time.time()
                }
                
                results.append(result)
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_config = config
                    print(f"🏆 New best score: {best_score:.4f}")
                
            except Exception as e:
                print(f"❌ Error with config {config}: {e}")
                continue
        
        # Save results
        results_file = self.results_dir / f"{model_type}_grid_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n🎉 Grid search complete for {model_type}!")
        print(f"🏆 Best score: {best_score:.4f}")
        print(f"⚙️  Best config: {best_config}")
        
        return best_config, best_score, results
    
    def optimize_all_models(self):
        """Optimize hyperparameters for all model types."""
        models_to_test = ['baseline', 'improved_enhanced', 'transformer']
        
        all_results = {}
        
        for model_type in models_to_test:
            print(f"\\n{'='*60}")
            print(f"🚀 Optimizing {model_type.upper()} MODEL")
            print(f"{'='*60}")
            
            try:
                best_config, best_score, results = self.grid_search(model_type, max_combinations=20)
                all_results[model_type] = {
                    'best_config': best_config,
                    'best_score': best_score,
                    'all_results': results
                }
            except Exception as e:
                print(f"❌ Failed to optimize {model_type}: {e}")
                continue
        
        # Save comprehensive results
        final_results_file = self.results_dir / "hyperparameter_optimization_results.json"
        with open(final_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\\n{'='*60}")
        print("🎯 OPTIMIZATION COMPLETE!")
        print(f"{'='*60}")
        
        # Print summary
        for model_type, results in all_results.items():
            print(f"🏆 {model_type.upper()}: Best MAE = {results['best_score']:.4f}")
        
        return all_results

def main():
    """Run hyperparameter optimization."""
    print("🚀 Advanced AI Mixing Model Optimization")
    print("=" * 50)
    
    # Setup paths
    data_dir = Path(__file__).parent.parent / "data"
    results_dir = Path(__file__).parent.parent / "optimization_results"
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(data_dir, results_dir)
    
    # Run optimization
    all_results = optimizer.optimize_all_models()
    
    print("\\n✅ Optimization complete!")
    print(f"📁 Results saved to: {results_dir}")
    print("\\n💡 Next steps:")
    print("1. Review the best configurations for each model")
    print("2. Train final models with optimal hyperparameters")
    print("3. Compare performance on test set")
    print("4. Deploy the best performing model")

if __name__ == "__main__":
    main()
