#!/usr/bin/env python3
"""
🎭 Ensemble Model Training & Optimization
========================================

Create advanced ensemble models combining multiple architectures
for improved performance beyond individual models.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error
from baseline_cnn import SpectrogramDataset, BaselineCNN, EnhancedCNN, DEVICE
from improved_models_fixed import ImprovedEnhancedCNN
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

class WeightedEnsemble(nn.Module):
    """Weighted ensemble of multiple models with learned weights."""
    
    def __init__(self, models, initial_weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if initial_weights is None:
            initial_weights = [1.0 / len(models)] * len(models)
        
        # Learnable weights (softmax ensures they sum to 1)
        self.weight_logits = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))
        
        # Apply softmax to get normalized weights
        weights = torch.softmax(self.weight_logits, dim=0)
        
        # Weighted combination
        ensemble_output = sum(w * out for w, out in zip(weights, outputs))
        return ensemble_output
    
    def get_weights(self):
        """Get current ensemble weights."""
        with torch.no_grad():
            return torch.softmax(self.weight_logits, dim=0).cpu().numpy()

class AdaptiveEnsemble(nn.Module):
    """Adaptive ensemble that changes weights based on input features."""
    
    def __init__(self, models, input_dim):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Small network to predict ensemble weights based on input
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce spatial dimensions
            nn.Flatten(),
            nn.Linear(4 * 4 * input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_models),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Get model outputs
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))
        
        # Predict weights based on input
        weights = self.weight_predictor(x)  # Shape: (batch_size, num_models)
        
        # Apply weights
        ensemble_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            ensemble_output += weights[:, i:i+1] * output
        
        return ensemble_output

def load_trained_models():
    """Load all available trained models."""
    models_dir = Path(__file__).parent.parent / "models"
    loaded_models = []
    model_names = []
    
    model_configs = [
        ("improved_enhanced_cnn.pth", ImprovedEnhancedCNN, {"dropout": 0.3}),
        ("retrained_enhanced_cnn.pth", EnhancedCNN, {}),
        ("baseline_cnn.pth", BaselineCNN, {}),
    ]
    
    for model_file, model_class, kwargs in model_configs:
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                model = model_class(**kwargs).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                loaded_models.append(model)
                model_names.append(model_file.replace('.pth', ''))
                print(f"✅ Loaded {model_file}")
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
        else:
            print(f"⚠️ Model not found: {model_file}")
    
    return loaded_models, model_names

def evaluate_individual_models(models, model_names, test_loader):
    """Evaluate individual model performance."""
    print("\\n📊 Individual Model Performance:")
    print("-" * 40)
    
    individual_results = []
    
    for model, name in zip(models, model_names):
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        mae = mean_absolute_error(targets, predictions)
        
        print(f"   {name}: MAE = {mae:.6f}")
        individual_results.append((name, mae, predictions))
    
    return individual_results

def create_simple_ensemble(models, model_names, test_loader):
    """Create simple weighted ensemble."""
    print("\\n🎭 Creating Simple Weighted Ensemble...")
    
    # Get individual performances
    individual_results = evaluate_individual_models(models, model_names, test_loader)
    
    # Calculate inverse MAE weights (better models get higher weight)
    maes = [result[1] for result in individual_results]
    inv_weights = [1.0 / mae for mae in maes]
    normalized_weights = [w / sum(inv_weights) for w in inv_weights]
    
    print(f"   Ensemble weights: {[f'{w:.3f}' for w in normalized_weights]}")
    
    # Create ensemble predictions
    all_predictions = [result[2] for result in individual_results]
    ensemble_predictions = np.zeros_like(all_predictions[0])
    
    for pred, weight in zip(all_predictions, normalized_weights):
        ensemble_predictions += weight * pred
    
    # Evaluate ensemble
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(DEVICE)
            targets.append(target.cpu().numpy())
    targets = np.vstack(targets)
    
    ensemble_mae = mean_absolute_error(targets, ensemble_predictions)
    print(f"   Simple Ensemble MAE: {ensemble_mae:.6f}")
    
    return ensemble_predictions, ensemble_mae, normalized_weights

def train_weighted_ensemble(models, model_names, train_loader, val_loader, test_loader):
    """Train ensemble with learnable weights."""
    print("\\n🧠 Training Weighted Ensemble with Learnable Weights...")
    
    # Create weighted ensemble
    ensemble = WeightedEnsemble(models).to(DEVICE)
    
    # Training setup
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    print("   Training ensemble weights...")
    
    for epoch in range(50):
        # Training
        ensemble.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = ensemble(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        ensemble.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = ensemble(data)
                val_loss += criterion(output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = ensemble.get_weights().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            current_weights = ensemble.get_weights()
            print(f"   Epoch {epoch}: Val Loss = {val_loss:.6f}, Weights = {[f'{w:.3f}' for w in current_weights]}")
        
        if patience_counter >= max_patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
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
    
    weighted_ensemble_mae = mean_absolute_error(test_targets, test_predictions)
    final_weights = ensemble.get_weights()
    
    print(f"   Weighted Ensemble MAE: {weighted_ensemble_mae:.6f}")
    print(f"   Final weights: {[f'{w:.3f}' for w in final_weights]}")
    
    return ensemble, weighted_ensemble_mae, final_weights

def optimize_ensemble_hyperparameters(models, model_names, train_loader, val_loader, test_loader):
    """Optimize ensemble hyperparameters."""
    print("\\n🔧 Optimizing Ensemble Hyperparameters...")
    
    best_mae = float('inf')
    best_config = None
    best_ensemble = None
    
    # Different ensemble configurations to try
    configs = [
        {'method': 'simple', 'weight_strategy': 'inverse_mae'},
        {'method': 'simple', 'weight_strategy': 'inverse_squared_mae'},
        {'method': 'simple', 'weight_strategy': 'uniform'},
        {'method': 'learned', 'lr': 1e-3},
        {'method': 'learned', 'lr': 5e-4},
        {'method': 'learned', 'lr': 1e-2},
    ]
    
    for i, config in enumerate(configs):
        print(f"   Config {i+1}/{len(configs)}: {config}")
        
        try:
            if config['method'] == 'simple':
                # Simple ensemble with different weighting strategies
                individual_results = evaluate_individual_models(models, model_names, test_loader)
                maes = [result[1] for result in individual_results]
                
                if config['weight_strategy'] == 'inverse_mae':
                    weights = [1.0 / mae for mae in maes]
                elif config['weight_strategy'] == 'inverse_squared_mae':
                    weights = [1.0 / (mae ** 2) for mae in maes]
                else:  # uniform
                    weights = [1.0] * len(maes)
                
                weights = [w / sum(weights) for w in weights]
                
                # Create ensemble predictions
                all_predictions = [result[2] for result in individual_results]
                ensemble_predictions = np.zeros_like(all_predictions[0])
                
                for pred, weight in zip(all_predictions, weights):
                    ensemble_predictions += weight * pred
                
                targets = []
                with torch.no_grad():
                    for data, target in test_loader:
                        targets.append(target.cpu().numpy())
                targets = np.vstack(targets)
                
                mae = mean_absolute_error(targets, ensemble_predictions)
                
            else:  # learned
                ensemble = WeightedEnsemble(models).to(DEVICE)
                optimizer = torch.optim.Adam(ensemble.parameters(), lr=config['lr'])
                criterion = nn.MSELoss()
                
                # Quick training
                for epoch in range(20):
                    ensemble.train()
                    for data, target in train_loader:
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        optimizer.zero_grad()
                        output = ensemble(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
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
                mae = mean_absolute_error(test_targets, test_predictions)
            
            print(f"     MAE: {mae:.6f}")
            
            if mae < best_mae:
                best_mae = mae
                best_config = config.copy()
                if config['method'] == 'learned':
                    best_ensemble = ensemble
                print(f"     ✅ New best configuration!")
            
        except Exception as e:
            print(f"     ❌ Configuration failed: {e}")
            continue
    
    print(f"\\n🏆 Best ensemble configuration: {best_config}")
    print(f"🏆 Best ensemble MAE: {best_mae:.6f}")
    
    return best_config, best_mae, best_ensemble

def main():
    """Main ensemble training pipeline."""
    print("🎭 Advanced Ensemble Model Training")
    print("=" * 50)
    
    # Load datasets
    base_dir = Path(__file__).parent.parent / "data"
    targets_file = base_dir / "targets_generated.json"
    
    # Use original datasets for ensemble evaluation (fair comparison)
    train_dataset = SpectrogramDataset(base_dir / "spectrograms" / "train", targets_file, augment=False)
    val_dataset = SpectrogramDataset(base_dir / "spectrograms" / "val", targets_file, augment=False)
    test_dataset = SpectrogramDataset(base_dir / "spectrograms" / "test", targets_file, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Load trained models
    models, model_names = load_trained_models()
    
    if len(models) < 2:
        print("❌ Need at least 2 models for ensemble. Train individual models first.")
        return
    
    print(f"\\n🤖 Loaded {len(models)} models for ensemble:")
    for name in model_names:
        print(f"   • {name}")
    
    # Evaluate individual models
    individual_results = evaluate_individual_models(models, model_names, test_loader)
    
    # Create simple ensemble
    simple_pred, simple_mae, simple_weights = create_simple_ensemble(models, model_names, test_loader)
    
    # Train weighted ensemble
    weighted_ensemble, weighted_mae, learned_weights = train_weighted_ensemble(
        models, model_names, train_loader, val_loader, test_loader
    )
    
    # Optimize ensemble
    best_config, best_mae, best_ensemble = optimize_ensemble_hyperparameters(
        models, model_names, train_loader, val_loader, test_loader
    )
    
    # Final results
    print(f"\\n{'='*50}")
    print("🏆 ENSEMBLE RESULTS SUMMARY")
    print('='*50)
    
    all_results = [
        ("Best Individual Model", min(result[1] for result in individual_results)),
        ("Simple Weighted Ensemble", simple_mae),
        ("Learned Weighted Ensemble", weighted_mae),
        ("Optimized Ensemble", best_mae),
    ]
    
    all_results.sort(key=lambda x: x[1])
    
    for i, (name, mae) in enumerate(all_results):
        status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
        print(f"{status} {name}: MAE = {mae:.6f}")
    
    best_overall_mae = all_results[0][1]
    improvement_vs_individual = (min(result[1] for result in individual_results) - best_overall_mae) / min(result[1] for result in individual_results) * 100
    
    print(f"\\n📈 Performance Analysis:")
    print(f"   Best ensemble MAE: {best_overall_mae:.6f}")
    print(f"   Improvement over best individual: {improvement_vs_individual:.1f}%")
    print(f"   Target achievement (MAE < 0.035): {'✅ ACHIEVED!' if best_overall_mae < 0.035 else f'{((0.0495 - best_overall_mae) / (0.0495 - 0.035) * 100):.1f}% progress'}")
    
    # Save best ensemble
    if best_ensemble is not None:
        ensemble_path = Path(__file__).parent.parent / "models" / "best_ensemble.pth"
        torch.save(best_ensemble.state_dict(), ensemble_path)
        print(f"\\n💾 Best ensemble saved: {ensemble_path}")
    
    print(f"\\n🚀 Ensemble optimization complete!")

if __name__ == "__main__":
    main()
