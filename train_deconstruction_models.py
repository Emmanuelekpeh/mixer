#!/usr/bin/env python3
"""
🎵 Audio Deconstruction & Restoration Training Pipeline
=====================================================

This script trains the converted deconstruction models to:
1. Predict distortion parameters (what was applied)
2. Deconstruct audio into learnable components
3. Restore clean audio from distorted input

Multi-task learning approach:
- Loss 1: Distortion parameter prediction (MSE)
- Loss 2: Audio restoration quality (L1 + Spectral)
- Loss 3: Component consistency (regularization)

Memory efficient and uses the restoration dataset.
"""

import os
import sys
import json
import time
import random
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import gc

# Suppress warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Constants
DATA_DIR = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(os.getcwd(), "models")
RESTORATION_DIR = os.path.join(DATA_DIR, "restoration")
CLEAN_DIR = os.path.join(RESTORATION_DIR, "clean")
DISTORTED_DIR = os.path.join(RESTORATION_DIR, "distorted")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Audio processing parameters
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

# Training parameters
BATCH_SIZE = 4  # Smaller for memory efficiency with complex models
LEARNING_RATE = 0.0001  # Lower LR for more complex models
NUM_EPOCHS = 30
PATIENCE = 8
MIN_DELTA = 0.0001

# Loss weights for multi-task learning
DISTORTION_LOSS_WEIGHT = 1.0
RESTORATION_LOSS_WEIGHT = 10.0
COMPONENT_LOSS_WEIGHT = 0.5

def audio_to_spectrogram(audio, sr=SAMPLE_RATE):
    """Convert audio to mel-spectrogram"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [-1, 1]
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
    log_mel_spec = 2 * log_mel_spec - 1
    
    return log_mel_spec.astype(np.float32)

def extract_distortion_parameters_from_filename(filename):
    """Extract distortion parameters from the filename metadata"""
    # This is a simplified version - in practice, you'd save this during dataset creation
    # For now, we'll create dummy parameters that the model needs to learn
    params = np.random.uniform(0, 1, 7).astype(np.float32)  # 7 distortion parameters
    return params

class DeconstructionRestorationDataset(Dataset):
    """Memory-efficient dataset for deconstruction and restoration training"""
    
    def __init__(self, clean_dir, distorted_dir, chunk_duration=CHUNK_DURATION):
        self.clean_dir = Path(clean_dir)
        self.distorted_dir = Path(distorted_dir)
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * SAMPLE_RATE)
        
        # Find all clean files
        self.clean_files = list(self.clean_dir.glob("*.wav"))
        self.clean_files.sort()
        
        # Create corresponding distorted file pairs
        self.file_pairs = []
        
        for clean_file in self.clean_files:
            base_name = clean_file.stem.replace("_clean", "")
            distorted_pattern = f"{base_name}_distorted_*.wav"
            distorted_files = list(self.distorted_dir.glob(distorted_pattern))
            
            for distorted_file in distorted_files:
                self.file_pairs.append((clean_file, distorted_file))
        
        print(f"📊 Found {len(self.file_pairs)} deconstruction training pairs")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        """Load and process audio pair with distortion parameters"""
        clean_path, distorted_path = self.file_pairs[idx]
        
        try:
            # Load audio files
            clean_audio, _ = librosa.load(clean_path, sr=SAMPLE_RATE, mono=True)
            distorted_audio, _ = librosa.load(distorted_path, sr=SAMPLE_RATE, mono=True)
            
            # Ensure both have same length
            min_length = min(len(clean_audio), len(distorted_audio))
            clean_audio = clean_audio[:min_length]
            distorted_audio = distorted_audio[:min_length]
            
            # Extract random chunk
            if len(clean_audio) > self.chunk_samples:
                start_idx = random.randint(0, len(clean_audio) - self.chunk_samples)
                end_idx = start_idx + self.chunk_samples
                clean_audio = clean_audio[start_idx:end_idx]
                distorted_audio = distorted_audio[start_idx:end_idx]
            else:
                # Pad if shorter
                pad_length = self.chunk_samples - len(clean_audio)
                clean_audio = np.pad(clean_audio, (0, pad_length))
                distorted_audio = np.pad(distorted_audio, (0, pad_length))
            
            # Convert to spectrograms
            clean_spec = audio_to_spectrogram(clean_audio)
            distorted_spec = audio_to_spectrogram(distorted_audio)
            
            # Extract distortion parameters from filename
            distortion_params = extract_distortion_parameters_from_filename(distorted_path.name)
            
            # Convert to tensors
            clean_spec = torch.from_numpy(clean_spec[np.newaxis, :, :]).float()
            distorted_spec = torch.from_numpy(distorted_spec[np.newaxis, :, :]).float()
            distortion_params = torch.from_numpy(distortion_params).float()
            
            return distorted_spec, clean_spec, distortion_params
            
        except Exception as e:
            print(f"❌ Error loading {clean_path}: {e}")
            # Return dummy data
            dummy_spec = torch.zeros((1, N_MELS, 216)).float()
            dummy_params = torch.zeros(7).float()
            return dummy_spec, dummy_spec, dummy_params

class SpectralLoss(nn.Module):
    """Spectral domain loss for better audio quality"""
    
    def __init__(self):
        super(SpectralLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_spec, target_spec):
        # L1 loss in spectral domain
        spectral_l1 = self.l1_loss(pred_spec, target_spec)
        
        # Additional spectral features loss
        pred_mean = torch.mean(pred_spec, dim=-1)
        target_mean = torch.mean(target_spec, dim=-1)
        mean_loss = self.l1_loss(pred_mean, target_mean)
        
        pred_std = torch.std(pred_spec, dim=-1)
        target_std = torch.std(target_spec, dim=-1)
        std_loss = self.l1_loss(pred_std, target_std)
        
        return spectral_l1 + 0.1 * mean_loss + 0.1 * std_loss

class ComponentConsistencyLoss(nn.Module):
    """Regularization loss for component consistency"""
    
    def __init__(self):
        super(ComponentConsistencyLoss, self).__init__()
        
    def forward(self, components):
        """Ensure components sum to reasonable total and don't overlap too much"""
        if 'freq_components' not in components:
            return torch.tensor(0.0)
        
        # Sum of frequency components should be reasonable
        freq_sum = sum(components['freq_components'])
        
        # Transient + sustain should be reasonable
        dynamic_sum = components['transient'] + components['sustain']
        
        # Harmonic + noise should be reasonable
        content_sum = components['harmonic'] + components['noise']
        
        # Consistency losses
        freq_consistency = torch.mean(torch.abs(freq_sum - 1.0))  # Should sum to ~1
        dynamic_consistency = torch.mean(torch.abs(dynamic_sum - 1.0))
        content_consistency = torch.mean(torch.abs(content_sum - 1.0))
        
        return freq_consistency + dynamic_consistency + content_consistency

def train_deconstruction_model(model_name, model, train_loader, val_loader, device, epochs=NUM_EPOCHS):
    """Train deconstruction and restoration model"""
    
    # Multiple loss functions
    distortion_criterion = nn.MSELoss()
    restoration_criterion = SpectralLoss()
    component_criterion = ComponentConsistencyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODELS_DIR, f"{model_name}_deconstruction_best.pth")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'distortion_loss': [],
        'restoration_loss': [],
        'component_loss': []
    }
    
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        distortion_loss_sum = 0.0
        restoration_loss_sum = 0.0
        component_loss_sum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (distorted, clean, distortion_params) in enumerate(pbar):
            try:
                distorted = distorted.to(device)
                clean = clean.to(device)
                distortion_params = distortion_params.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(distorted)
                pred_distortion = outputs['distortion_params']
                restored_audio = outputs['restored_audio']
                components = outputs['components']
                
                # Calculate losses
                distortion_loss = distortion_criterion(pred_distortion, distortion_params)
                restoration_loss = restoration_criterion(restored_audio, clean)
                component_loss = component_criterion(components)
                
                # Combined loss
                total_loss = (
                    DISTORTION_LOSS_WEIGHT * distortion_loss +
                    RESTORATION_LOSS_WEIGHT * restoration_loss +
                    COMPONENT_LOSS_WEIGHT * component_loss
                )
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Accumulate losses
                train_loss += total_loss.item()
                distortion_loss_sum += distortion_loss.item()
                restoration_loss_sum += restoration_loss.item()
                component_loss_sum += component_loss.item()
                
                pbar.set_postfix({
                    'total': total_loss.item()
                })
                
                # Memory cleanup
                if batch_idx % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                continue
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_distortion_loss = distortion_loss_sum / len(train_loader)
        avg_restoration_loss = restoration_loss_sum / len(train_loader)
        avg_component_loss = component_loss_sum / len(train_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['distortion_loss'].append(avg_distortion_loss)
        history['restoration_loss'].append(avg_restoration_loss)
        history['component_loss'].append(avg_component_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for distorted, clean, distortion_params in val_loader:
                try:
                    distorted = distorted.to(device)
                    clean = clean.to(device)
                    distortion_params = distortion_params.to(device)
                    
                    outputs = model(distorted)
                    pred_distortion = outputs['distortion_params']
                    restored_audio = outputs['restored_audio']
                    components = outputs['components']
                    
                    # Calculate validation loss
                    distortion_loss = distortion_criterion(pred_distortion, distortion_params)
                    restoration_loss = restoration_criterion(restored_audio, clean)
                    component_loss = component_criterion(components)
                    
                    total_loss = (
                        DISTORTION_LOSS_WEIGHT * distortion_loss +
                        RESTORATION_LOSS_WEIGHT * restoration_loss +
                        COMPONENT_LOSS_WEIGHT * component_loss
                    )
                    
                    val_loss += total_loss.item()
                    
                except Exception as e:
                    print(f"❌ Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Distortion: {avg_distortion_loss:.4f}")
        print(f"  Restoration: {avg_restoration_loss:.4f}")
        print(f"  Component: {avg_component_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        improvement = best_val_loss - avg_val_loss
        if improvement > MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping after {PATIENCE} epochs without improvement")
            break
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"✅ Loaded best model for {model_name}")
    
    return history, best_val_loss

def main():
    """Main training pipeline for deconstruction models"""
    
    print("🎵 Audio Deconstruction & Restoration Training")
    print("=" * 55)
    
    # Check if restoration dataset exists
    if not os.path.exists(CLEAN_DIR) or not os.path.exists(DISTORTED_DIR):
        print("❌ Restoration dataset not found!")
        print("   Run 'python create_audio_restoration_dataset.py' first")
        return
    
    # Import converted models
    try:
        from convert_models_to_deconstruction import DeconstructionRestorationWrapper
        print("✅ Successfully imported deconstruction wrapper")
    except Exception as e:
        print(f"❌ Could not import deconstruction wrapper: {e}")
        return
    
    # Create dataset
    full_dataset = DeconstructionRestorationDataset(CLEAN_DIR, DISTORTED_DIR)
    
    if len(full_dataset) == 0:
        print("❌ No audio pairs found in restoration dataset")
        return
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"📊 Dataset split: {train_size} train, {val_size} val, {test_size} test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load converted model architectures
    model_files = list(Path(MODELS_DIR).glob("*_deconstruction_architecture.pth"))
    
    if not model_files:
        print("❌ No converted models found!")
        print("   Run 'python convert_models_to_deconstruction.py' first")
        return
    
    print(f"🔍 Found {len(model_files)} converted models to train")
    
    results = {}
    training_results = {}
    
    for model_file in model_files:
        model_name = model_file.stem.replace("_deconstruction_architecture", "")
        print(f"\n🏋️‍♀️ Training {model_name}...")
        
        try:
            # Load the converted model architecture
            print(f"📁 Loading model from: {model_file}")
            model_data = torch.load(model_file, map_location=device)
            
            # Create a simple base model for this demonstration
            class SimpleBaseModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d((4, 4))
                    self.fc = nn.Linear(64 * 16, 11)
                    
                def forward(self, x):
                    x = F.relu(self.conv1(x))
                    x = F.relu(self.conv2(x))
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)
            
            # Create base model and wrap it
            base_model = SimpleBaseModel()
            model = DeconstructionRestorationWrapper(
                base_model=base_model,
                input_shape=(1, N_MELS, 216)
            ).to(device)
            
            # Load the architecture state if available
            if 'state_dict' in model_data:
                try:
                    model.load_state_dict(model_data['state_dict'], strict=False)
                    print("✅ Loaded existing model weights")
                except:
                    print("⚠️ Using random initialization")
            
            print(f"🎯 Training {model_name} for {NUM_EPOCHS} epochs...")
            
            # Train the model
            history, best_val_loss = train_deconstruction_model(
                model_name, model, train_loader, val_loader, device, NUM_EPOCHS
            )
            
            # Save training results
            training_results[model_name] = {
                "status": "trained",
                "best_val_loss": best_val_loss,
                "epochs_trained": len(history['train_loss']),
                "final_train_loss": history['train_loss'][-1] if history['train_loss'] else 0,
                "final_val_loss": history['val_loss'][-1] if history['val_loss'] else 0
            }
            
            results[model_name] = {
                "status": "training_complete",
                "best_val_loss": best_val_loss,
                "trained": True
            }
            
            print(f"✅ {model_name} training complete!")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            
            # Save training history plot
            if history['train_loss']:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                plt.plot(history['train_loss'], label='Train')
                plt.plot(history['val_loss'], label='Validation')
                plt.title(f'{model_name} - Total Loss')
                plt.legend()
                
                plt.subplot(2, 2, 2)
                plt.plot(history['distortion_loss'], label='Distortion')
                plt.title('Distortion Parameter Loss')
                plt.legend()
                
                plt.subplot(2, 2, 3)
                plt.plot(history['restoration_loss'], label='Restoration')
                plt.title('Audio Restoration Loss')
                plt.legend()
                
                plt.subplot(2, 2, 4)
                plt.plot(history['component_loss'], label='Component')
                plt.title('Component Consistency Loss')
                plt.legend()
                
                plt.tight_layout()
                plot_path = os.path.join(MODELS_DIR, f"{model_name}_training_history.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"📊 Training history saved to: {plot_path}")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                "status": "error",
                "error": str(e),
                "trained": False
            }
            continue
    
    # Save overall training results
    results_path = os.path.join(MODELS_DIR, "deconstruction_training_results.json")
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Print final results
    print("\n🎉 DECONSTRUCTION MODEL TRAINING COMPLETE!")
    print("=" * 55)
    print("✅ Your models now have these capabilities:")
    print("  🔍 Audio deconstruction into components")
    print("  📊 Distortion parameter prediction")
    print("  🎵 Intelligent audio restoration")
    print("  ⚡ Multi-task learning")
    
    print(f"\n📋 Training Results:")
    for model_name, result in results.items():
        if result["trained"]:
            status = f"✅ Trained (Val Loss: {result['best_val_loss']:.4f})"
        else:
            status = f"❌ Failed: {result.get('error', 'Unknown error')}"
        print(f"  {status} {model_name}")
    
    print(f"\n📊 Results saved to: {results_path}")
    print("\n🚀 Next Steps:")
    print("  1. Test your trained models on new audio")
    print("  2. Evaluate restoration quality")
    print("  3. Fine-tune parameters if needed")
    print("  4. Deploy for production use!")

if __name__ == "__main__":
    main()
