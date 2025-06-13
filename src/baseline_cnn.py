import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error
import random

# Configuration
SPECTROGRAMS_FOLDER = Path(__file__).resolve().parent.parent / "data" / "spectrograms"
TARGETS_FILE = Path(__file__).resolve().parent.parent / "data" / "targets_generated.json"
BATCH_SIZE = 16
N_MELS = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model/augmentation options
N_OUTPUTS = 10
DROPOUT = 0.3
AUGMENT = True
AUG_TIME_MASK = 0.1  # Fraction of time steps to mask
AUG_FREQ_MASK = 0.1  # Fraction of freq bins to mask
AUG_NOISE_STD = 0.01 # Std of Gaussian noise
N_CONV_LAYERS = 3    # Number of conv layers (2 or 3)

class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, targets_file, n_outputs=N_OUTPUTS, augment=False):
        self.samples = []
        self.targets = json.load(open(targets_file))
        for track_dir in Path(spectrogram_dir).rglob("*.npy"):
            self.samples.append(track_dir)
        self.n_outputs = n_outputs
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def time_mask(self, spec, mask_frac):
        t = spec.shape[1]
        mask_len = int(t * mask_frac)
        if mask_len > 0:
            start = random.randint(0, t - mask_len)
            spec[:, start:start+mask_len] = 0
        return spec

    def freq_mask(self, spec, mask_frac):
        f = spec.shape[0]
        mask_len = int(f * mask_frac)
        if mask_len > 0:
            start = random.randint(0, f - mask_len)
            spec[start:start+mask_len, :] = 0
        return spec

    def add_noise(self, spec, std):
        return spec + np.random.normal(0, std, spec.shape)

    def __getitem__(self, idx):
        spec_path = self.samples[idx]
        spec = np.load(spec_path)
        # Normalize
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        
        # Fixed time dimension (crop or pad to consistent length)
        target_time_steps = 1000  # Fixed length
        if spec.shape[1] > target_time_steps:
            # Crop from center
            start = (spec.shape[1] - target_time_steps) // 2
            spec = spec[:, start:start + target_time_steps]
        elif spec.shape[1] < target_time_steps:
            # Pad with zeros
            pad_width = target_time_steps - spec.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        # Data augmentation
        if self.augment:
            if random.random() < 0.5:
                spec = self.time_mask(spec, AUG_TIME_MASK)
            if random.random() < 0.5:
                spec = self.freq_mask(spec, AUG_FREQ_MASK)
            if random.random() < 0.5:
                spec = self.add_noise(spec, AUG_NOISE_STD)
        
        # Add channel dimension
        spec = np.expand_dims(spec, axis=0)
        
        # Extract track name for target lookup
        track_name = spec_path.parent.name
        
        # Use real targets if available, else zeros
        target = self.targets.get(track_name, [0.0]*self.n_outputs)
        target = np.array(target, dtype=np.float32)
        
        return torch.tensor(spec, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=1)  # Always stride=1 for second conv
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class BaselineCNN(nn.Module):
    def __init__(self, n_outputs=N_OUTPUTS, dropout=DROPOUT, n_conv_layers=N_CONV_LAYERS):
        super(BaselineCNN, self).__init__()
          # Define the sequential CNN layers
        layers = []
        in_channels = 1
        out_channels = 32
        
        for i in range(n_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout2d(dropout))
            
            in_channels = out_channels
            out_channels = min(out_channels * 2, 128)  # Cap at 128 channels to match actual output
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calculate the size after convolutions
        # Input: (1, 128, 1000) -> after 3 conv+pool: (128, 16, 125) -> AdaptivePool: (128, 4, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers - fix the input size calculation
        # After 3 layers: 32->64->128 channels, so final is 128 * 4 * 4 = 2048
        fc_input_size = 128 * 4 * 4 if n_conv_layers >= 3 else (32 * (2**(n_conv_layers-1))) * 4 * 4
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class EnhancedCNN(nn.Module):
    def __init__(self, n_outputs=N_OUTPUTS, dropout=DROPOUT):
        super(EnhancedCNN, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
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
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
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
    
    # Calculate MAE for each output parameter
    mae_per_param = []
    for i in range(all_targets.shape[1]):
        mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
        mae_per_param.append(mae)
    
    overall_mae = np.mean(mae_per_param)
    
    return overall_mae, mae_per_param, all_predictions, all_targets

def create_data_loaders(spectrograms_folder, targets_file, batch_size=BATCH_SIZE, train_split=0.7, val_split=0.15):
    """Create train/val/test data loaders"""
    
    # Create full dataset
    full_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=False)
    
    # Split indices
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets with augmentation for training
    train_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=AUGMENT)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    
    val_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
    
    test_dataset = SpectrogramDataset(spectrograms_folder, targets_file, augment=False)
    test_dataset.samples = [full_dataset.samples[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def main():
    print(f"Using device: {DEVICE}")
    print(f"Spectrograms folder: {SPECTROGRAMS_FOLDER}")
    print(f"Targets file: {TARGETS_FILE}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(SPECTROGRAMS_FOLDER, TARGETS_FILE)
    
    print(f"Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Train baseline CNN
    print("\n=== Training Baseline CNN ===")
    baseline_model = BaselineCNN(n_outputs=N_OUTPUTS, dropout=DROPOUT, n_conv_layers=N_CONV_LAYERS)
    baseline_train_losses, baseline_val_losses = train_model(baseline_model, train_loader, val_loader)
    
    # Evaluate baseline CNN
    baseline_mae, baseline_mae_per_param, _, _ = evaluate_model(baseline_model, test_loader)
    print(f"Baseline CNN - Overall MAE: {baseline_mae:.4f}")
    print(f"Baseline CNN - MAE per parameter: {baseline_mae_per_param}")
    
    # Train enhanced CNN
    print("\n=== Training Enhanced CNN ===")
    enhanced_model = EnhancedCNN(n_outputs=N_OUTPUTS, dropout=DROPOUT)
    enhanced_train_losses, enhanced_val_losses = train_model(enhanced_model, train_loader, val_loader)
    
    # Evaluate enhanced CNN
    enhanced_mae, enhanced_mae_per_param, _, _ = evaluate_model(enhanced_model, test_loader)
    print(f"Enhanced CNN - Overall MAE: {enhanced_mae:.4f}")
    print(f"Enhanced CNN - MAE per parameter: {enhanced_mae_per_param}")
    
    # Save models
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    torch.save(baseline_model.state_dict(), models_dir / "baseline_cnn.pth")
    torch.save(enhanced_model.state_dict(), models_dir / "enhanced_cnn.pth")
    
    print(f"\nModels saved to {models_dir}")
    
    # Compare results
    print("\n=== Model Comparison ===")
    print(f"Baseline CNN MAE: {baseline_mae:.4f}")
    print(f"Enhanced CNN MAE: {enhanced_mae:.4f}")
    
    if enhanced_mae < baseline_mae:
        improvement = ((baseline_mae - enhanced_mae) / baseline_mae) * 100
        print(f"Enhanced CNN performs {improvement:.2f}% better than Baseline CNN")
    else:
        degradation = ((enhanced_mae - baseline_mae) / baseline_mae) * 100
        print(f"Enhanced CNN performs {degradation:.2f}% worse than Baseline CNN")

if __name__ == "__main__":
    main()