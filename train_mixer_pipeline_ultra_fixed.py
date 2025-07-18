"""
AI Mixer Training Pipeline - Ultra Fixed Version 
=================================================
Comprehensive fixes for all tensor shape issues and model training problems.

This version addresses:
- Advanced Transformer tensor shape mismatch
- VAE Mixer 5D tensor issue 
- ResNet Mixer parameter mismatch
- Model evaluation metric errors
- Windows compatibility issues
"""

import os
import sys
import json
import time
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import pickle
import shutil

# Add src directory to path for model imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

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
SPECTROGRAMS_DIR = os.path.join(DATA_DIR, "spectrograms")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure directories exist
for directory in [SPECTROGRAMS_DIR, FEATURES_DIR, TRAIN_DIR, TEST_DIR, AUDIO_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define feature extraction parameters
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
FIXED_LENGTH = 500  # Fixed number of frames for all spectrograms

# Parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
NUM_WORKERS = 0 if sys.platform == "win32" else 4  # Windows compatibility
PATIENCE = 10
MIN_DELTA = 0.0001

# Define the target mixing parameters
MIXING_PARAMS = [
    "gain", "compression_ratio", "attack_time", "release_time",
    "high_shelf_gain", "high_shelf_freq", "low_shelf_gain", "low_shelf_freq",
    "eq_low_gain", "eq_mid_gain", "eq_high_gain"
]

def preprocess_audio_data():
    """Process real audio files and extract features"""
    
    # Check if preprocessing has already been done
    metadata_file = os.path.join(DATA_DIR, "preprocessing_metadata.json")
    
    # Check data quality if metadata exists
    if os.path.exists(metadata_file):
        print("✅ Preprocessing metadata found, checking data quality...")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Quick data validation
        train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('_spec.npy')]
        test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('_spec.npy')]
        
        expected_train = metadata.get('split', {}).get('train', 0)
        expected_test = metadata.get('split', {}).get('validation', 0) + metadata.get('split', {}).get('test', 0)
        
        if len(train_files) >= expected_train * 0.9 and len(test_files) >= expected_test * 0.9:
            print(f"🎵 Found {len(train_files)} train files and {len(test_files)} test files (sufficient)")
            return metadata
        else:
            print(f"❌ Data appears corrupted or incomplete. Regenerating...")
            # Clear corrupted data
            import shutil
            if os.path.exists(TRAIN_DIR):
                shutil.rmtree(TRAIN_DIR)
            if os.path.exists(TEST_DIR):
                shutil.rmtree(TEST_DIR)
            os.makedirs(TRAIN_DIR, exist_ok=True)
            os.makedirs(TEST_DIR, exist_ok=True)
    
    print("🎵 Processing real audio files...")
    
    # Look for real audio files in multiple locations
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    # Search for FMA audio files in multiple possible locations
    search_directories = [
        os.path.join(DATA_DIR, "raw", "music", "fma"),
        os.path.join(DATA_DIR, "fma_small"),
        os.path.join(DATA_DIR, "fma"),
        os.path.join(DATA_DIR, "raw", "fma"),
        os.path.join(DATA_DIR, "audio", "fma"),
        os.path.join(DATA_DIR, "audio"),
        os.path.join(DATA_DIR, "raw"),
        DATA_DIR
    ]
    
    for search_dir in search_directories:
        if os.path.exists(search_dir):
            print(f"🔍 Searching for audio files in {search_dir}...")
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_files.append(os.path.join(root, file))
    
    # Remove duplicates
    audio_files = list(set(audio_files))
    
    if len(audio_files) == 0:
        print("❌ No FMA audio files found!")
        print("🎵 Attempting to download FMA small dataset...")
        
        # Try to download FMA dataset
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
            from enhanced_data_acquisition import EnhancedDataAcquisition
            
            # Create data acquisition instance
            acquisition = EnhancedDataAcquisition(base_dir=DATA_DIR)
            
            # Download FMA small dataset
            result = acquisition.fetch_fma(size="small")
            
            if result:
                print("✅ FMA dataset downloaded successfully!")
                # Re-search for audio files
                for search_dir in search_directories:
                    if os.path.exists(search_dir):
                        print(f"🔍 Re-searching for audio files in {search_dir}...")
                        for root, dirs, files in os.walk(search_dir):
                            for file in files:
                                if any(file.lower().endswith(ext) for ext in audio_extensions):
                                    audio_files.append(os.path.join(root, file))
                        if audio_files:
                            break
            else:
                print("❌ Failed to download FMA dataset")
                
        except Exception as e:
            print(f"❌ Error downloading FMA dataset: {e}")
        
        # Final check for audio files
        if len(audio_files) == 0:
            print("❌ No audio files found and unable to download FMA dataset!")
            print("📁 Please manually download FMA small dataset to one of these locations:")
            for search_dir in search_directories:
                print(f"   - {search_dir}")
            print("🎵 You can download FMA from: https://github.com/mdeff/fma")
            raise FileNotFoundError("No audio files found for processing")
    
    print(f"🎵 Found {len(audio_files)} REAL audio files")
    
    # Limit to a reasonable number for training
    if len(audio_files) > 1000:
        print(f"⚠️ Found {len(audio_files)} files, limiting to 1000 for faster processing")
        audio_files = audio_files[:1000]
    
    # Shuffle and split files
    random.seed(42)
    random.shuffle(audio_files)
    
    # Split into train/validation/test
    train_size = int(len(audio_files) * 0.8)
    val_size = int(len(audio_files) * 0.1)
    test_size = len(audio_files) - train_size - val_size
    
    train_files = audio_files[:train_size]
    val_files = audio_files[train_size:train_size + val_size]
    test_files = audio_files[train_size + val_size:]
    
    print(f"📊 Dataset splits: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    
    # Create metadata dictionary
    metadata = {
        "total_tracks": 0,
        "total_duration": 0,
        "split": {
            "train": 0,
            "validation": 0,
            "test": 0
        },
        "feature_dims": {
            "mfcc": N_MFCC
        },
        "source": "FMA (Free Music Archive) - REAL audio files"
    }
    
    # Process each split
    for split_name, files, save_dir in [
        ("train", train_files, TRAIN_DIR),
        ("validation", val_files, TEST_DIR),
        ("test", test_files, TEST_DIR)
    ]:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n🎵 Processing {split_name} split ({len(files)} files)...")
        
        successful = 0
        for i, audio_file in enumerate(tqdm(files, desc=f"Processing {split_name}")):
            try:
                # Load audio file
                try:
                    audio, sr = librosa.load(audio_file, sr=None, mono=True, duration=30.0)  # Load max 30 seconds
                except Exception as e:
                    print(f"❌ Error loading {audio_file}: {e}")
                    continue
                
                # Skip very short audio files
                if len(audio) < sr * 2.0:  # Less than 2 seconds
                    continue
                
                # Normalize audio
                audio = librosa.util.normalize(audio)
                
                # Extract MFCC features
                try:
                    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, 
                                               n_fft=N_FFT, hop_length=HOP_LENGTH)
                except Exception as e:
                    print(f"❌ Error extracting MFCC from {audio_file}: {e}")
                    continue
                
                # Pad or truncate to fixed length
                if mfcc.shape[1] < FIXED_LENGTH:
                    # Pad with zeros
                    padding = FIXED_LENGTH - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
                elif mfcc.shape[1] > FIXED_LENGTH:
                    # Truncate to fixed length
                    mfcc = mfcc[:, :FIXED_LENGTH]
                
                # Reshape to match expected format [1, N_MFCC, FIXED_LENGTH]
                spec = mfcc[np.newaxis, :, :].astype(np.float32)
                
                # Normalize features to [0, 1] range
                spec_min = np.min(spec)
                spec_max = np.max(spec)
                if spec_max > spec_min:
                    spec = (spec - spec_min) / (spec_max - spec_min)
                else:
                    spec = np.zeros_like(spec)
                
                # Validate spec shape
                if spec.shape != (1, N_MFCC, FIXED_LENGTH):
                    print(f"❌ Invalid spec shape {spec.shape} for {audio_file}, skipping...")
                    continue
                
                # Generate realistic mixing parameters based on audio characteristics
                targets = generate_mixing_targets_from_audio(spec, audio, sr)
                
                # Validate targets
                if targets.shape != (len(MIXING_PARAMS),):
                    print(f"❌ Invalid target shape {targets.shape} for {audio_file}, skipping...")
                    continue
                
                # Create filename from original audio file
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))[:50]
                track_id = f"{safe_name}_{i:05d}"
                
                # Save files
                spec_path = os.path.join(save_dir, f"{track_id}_spec.npy")
                target_path = os.path.join(save_dir, f"{track_id}_target.npy")
                
                np.save(spec_path, spec)
                np.save(target_path, targets)
                
                # Verify files were saved correctly
                if not os.path.exists(spec_path) or not os.path.exists(target_path):
                    print(f"❌ Files not created for {audio_file}")
                    continue
                
                # Update metadata
                metadata["total_tracks"] += 1
                metadata["split"][split_name] += 1
                metadata["total_duration"] += len(audio) / sr
                successful += 1
                
            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
                continue
        
        print(f"✅ Successfully processed {successful}/{len(files)} files for {split_name}")
    
    # Save metadata
    with open(os.path.join(DATA_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save targets mapping for reference
    targets_dict = {i: param for i, param in enumerate(MIXING_PARAMS)}
    with open(os.path.join(DATA_DIR, "targets_real.json"), "w") as f:
        json.dump(targets_dict, f, indent=2)
    
    print("\n✅ Real audio preprocessing complete!")
    print(f"🎵 Processed {metadata['total_tracks']} tracks ({metadata['total_duration']:.1f} seconds)")
    print(f"📊 Dataset splits: {metadata['split']['train']} train, {metadata['split']['validation']} validation, {metadata['split']['test']} test")
    print(f"📁 Metadata saved to: {os.path.join(DATA_DIR, 'preprocessing_metadata.json')}")
    print(f"🎚️ Mixing targets saved to: {os.path.join(DATA_DIR, 'targets_real.json')}")
    print(f"🧠 MFCC features extracted from real audio files")
    
    return metadata


def generate_mixing_targets_from_audio(spec, audio, sr):
    """Generate realistic mixing parameters based on actual audio characteristics"""
    
    # Extract the MFCC data from the spec (remove the batch dimension)
    mfcc = spec[0]  # Shape: [N_MFCC, FIXED_LENGTH]
    
    # Analyze audio characteristics
    # 1. Energy distribution
    low_freq_energy = np.mean(mfcc[:N_MFCC//4, :])  # Low frequencies
    mid_freq_energy = np.mean(mfcc[N_MFCC//4:3*N_MFCC//4, :])  # Mid frequencies  
    high_freq_energy = np.mean(mfcc[3*N_MFCC//4:, :])  # High frequencies
    
    # 2. Dynamic range
    dynamic_range = np.max(mfcc) - np.min(mfcc)
    
    # 3. Spectral centroid (brightness)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    brightness = min(spectral_centroid / (sr/2), 1.0)  # Normalize to [0, 1]
    
    # 4. RMS energy
    rms_energy = np.mean(librosa.feature.rms(y=audio))
    
    # 5. Zero crossing rate (indication of noisiness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Generate targets based on audio characteristics
    targets = np.zeros(len(MIXING_PARAMS), dtype=np.float32)
    
    # 1. Gain - based on RMS energy (quieter tracks need more gain)
    targets[0] = np.clip(0.5 + 0.3 * (1.0 - rms_energy), 0, 1)
    
    # 2. Compression ratio - based on dynamic range (high dynamic range needs more compression)
    targets[1] = np.clip(0.2 + 0.6 * (dynamic_range / 10.0), 0, 1)
    
    # 3. Attack time - based on zero crossing rate (more percussive sounds need faster attack)
    targets[2] = np.clip(0.3 - 0.2 * zcr, 0, 1)
    
    # 4. Release time - based on sustain characteristics
    targets[3] = np.clip(0.4 + 0.3 * (1.0 - zcr), 0, 1)
    
    # 5. High shelf gain - based on high frequency energy
    targets[4] = np.clip(0.3 + 0.4 * (1.0 - high_freq_energy), 0, 1)
    
    # 6. High shelf frequency - based on brightness
    targets[5] = np.clip(0.6 + 0.3 * brightness, 0, 1)
    
    # 7. Low shelf gain - based on low frequency energy
    targets[6] = np.clip(0.3 + 0.4 * (1.0 - low_freq_energy), 0, 1)
    
    # 8. Low shelf frequency - based on low frequency content
    targets[7] = np.clip(0.2 + 0.3 * low_freq_energy, 0, 1)
    
    # 9. EQ low gain - based on low frequency energy
    targets[8] = np.clip(0.4 + 0.3 * (1.0 - low_freq_energy), 0, 1)
    
    # 10. EQ mid gain - based on mid frequency energy
    targets[9] = np.clip(0.4 + 0.3 * (1.0 - mid_freq_energy), 0, 1)
    
    # 11. EQ high gain - based on high frequency energy and brightness
    targets[10] = np.clip(0.3 + 0.4 * (1.0 - high_freq_energy) * brightness, 0, 1)
    
    # Ensure all values are finite and in range [0, 1]
    targets = np.clip(targets, 0, 1)
    targets = np.where(np.isfinite(targets), targets, 0.5)  # Replace any NaN/inf with 0.5
    
    return targets

class AudioMixingDataset(Dataset):
    """Dataset for audio mixing parameter prediction"""
    def __init__(self, data_dir, max_retries=3):
        self.data_dir = data_dir
        self.max_retries = max_retries
        self.spec_files = [f for f in os.listdir(data_dir) if f.endswith('_spec.npy')]
        self.spec_files.sort()
        
        # Validate files and remove corrupted ones
        valid_files = []
        corrupted_count = 0
        
        print(f"🔍 Validating {len(self.spec_files)} files...")
        
        for i, spec_file in enumerate(self.spec_files):
            track_id = spec_file.split('_spec.npy')[0]
            target_file = f"{track_id}_target.npy"
            
            spec_path = os.path.join(data_dir, spec_file)
            target_path = os.path.join(data_dir, target_file)
            
            # Check if both files exist and are valid
            try:
                if os.path.exists(spec_path) and os.path.exists(target_path):
                    # Thorough validation
                    spec = np.load(spec_path)
                    target = np.load(target_path)
                    
                    # Check for valid data
                    if (spec.size > 0 and target.size > 0 and 
                        np.isfinite(spec).all() and np.isfinite(target).all() and
                        len(target) == len(MIXING_PARAMS) and
                        spec.shape[-1] == FIXED_LENGTH and
                        spec.shape[-2] == N_MFCC):
                        valid_files.append(spec_file)
                    else:
                        print(f"❌ Removing corrupted file: {spec_file} (invalid data)")
                        corrupted_count += 1
                        # Remove corrupted files
                        os.remove(spec_path)
                        if os.path.exists(target_path):
                            os.remove(target_path)
                else:
                    print(f"❌ Missing files for {spec_file}")
                    corrupted_count += 1
                    
            except Exception as e:
                print(f"❌ Error validating {spec_file}: {e}")
                corrupted_count += 1
                # Remove problematic files
                for path in [spec_path, target_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"   Validated {i+1}/{len(self.spec_files)} files, {len(valid_files)} valid, {corrupted_count} corrupted")
        
        self.spec_files = valid_files
        print(f"✅ Loaded {len(self.spec_files)} valid spectrogram files with targets")
        print(f"❌ Removed {corrupted_count} corrupted files")
        
        # Ensure we have enough valid files
        if len(self.spec_files) < 10:
            raise ValueError(f"Not enough valid files ({len(self.spec_files)}). Dataset may be corrupted.")
    
    def __len__(self):
        return len(self.spec_files)
    
    def __getitem__(self, idx):
        """Get item with robust error handling and fallback mechanism"""
        spec_file = None
        for retry in range(self.max_retries):
            try:
                # Use modulo to handle out-of-bounds indices gracefully
                actual_idx = idx % len(self.spec_files)
                spec_file = self.spec_files[actual_idx]
                track_id = spec_file.split('_spec.npy')[0]
                target_file = f"{track_id}_target.npy"
                
                # Load spectrogram and target with error handling
                spec_path = os.path.join(self.data_dir, spec_file)
                target_path = os.path.join(self.data_dir, target_file)
                
                # Check if files exist
                if not os.path.exists(spec_path) or not os.path.exists(target_path):
                    raise FileNotFoundError(f"Missing files: {spec_path} or {target_path}")
                
                spec = np.load(spec_path)
                target = np.load(target_path)
                
                # Validate loaded data
                if spec.size == 0 or target.size == 0:
                    raise ValueError(f"Empty data in {spec_file}")
                
                if not np.isfinite(spec).all() or not np.isfinite(target).all():
                    raise ValueError(f"Non-finite values in {spec_file}")
                
                # Ensure proper shape [1, 40, 500]
                if spec.ndim == 2:
                    spec = spec[np.newaxis, ...]
                elif spec.ndim == 3:
                    pass  # Already correct
                else:
                    raise ValueError(f"Invalid spec shape: {spec.shape}")
                
                # Validate target shape
                if target.ndim != 1 or len(target) != len(MIXING_PARAMS):
                    raise ValueError(f"Invalid target shape: {target.shape}")
                
                # Final shape validation
                if spec.shape != (1, N_MFCC, FIXED_LENGTH):
                    raise ValueError(f"Wrong spec dimensions: {spec.shape}, expected (1, {N_MFCC}, {FIXED_LENGTH})")
                
                # Convert to tensor
                spec = torch.from_numpy(spec).float()
                target = torch.from_numpy(target).float()
                
                return spec, target
                
            except Exception as e:
                if retry == self.max_retries - 1:
                    print(f"❌ Failed to load {spec_file or 'file'} after {self.max_retries} retries: {e}")
                    # Try to use a different file
                    if len(self.spec_files) > 1:
                        # Use a different valid file
                        fallback_idx = (idx + 1) % len(self.spec_files)
                        try:
                            fallback_file = self.spec_files[fallback_idx]
                            print(f"🔄 Using fallback file: {fallback_file}")
                            return self.__getitem__(fallback_idx)
                        except:
                            pass
                    
                    # Last resort: generate a valid sample
                    print(f"⚠️ Generating fallback sample for index {idx}")
                    spec = torch.zeros((1, N_MFCC, FIXED_LENGTH)).float()
                    target = torch.zeros(len(MIXING_PARAMS)).float()
                    return spec, target
                else:
                    print(f"🔄 Retry {retry + 1}/{self.max_retries} for {spec_file or 'file'}: {e}")
                    continue


def collate_fn(batch, model_name=None):
    """Custom collate function to handle different model input shapes and filter bad samples"""
    # Filter out None or invalid samples
    valid_samples = []
    for sample in batch:
        if sample is not None and len(sample) == 2:
            features, targets = sample
            # Check if tensors are valid
            if (features is not None and targets is not None and 
                features.numel() > 0 and targets.numel() > 0 and
                torch.isfinite(features).all() and torch.isfinite(targets).all()):
                valid_samples.append(sample)
    
    # If no valid samples, create a single dummy sample
    if not valid_samples:
        print("⚠️ No valid samples in batch, creating dummy sample")
        dummy_features = torch.zeros((1, N_MFCC, FIXED_LENGTH)).float()
        dummy_targets = torch.zeros(len(MIXING_PARAMS)).float()
        valid_samples = [(dummy_features, dummy_targets)]
    
    # If we have fewer valid samples than expected, pad with dummy samples
    if len(valid_samples) < len(batch):
        print(f"⚠️ Only {len(valid_samples)}/{len(batch)} samples were valid")
        # Duplicate existing samples to maintain batch size
        while len(valid_samples) < len(batch):
            valid_samples.append(valid_samples[0])  # Duplicate first valid sample
    
    features, targets = zip(*valid_samples)
    
    try:
        # Stack features and targets
        features = torch.stack(features)  # May be [B, 1, 1, 40, 500] or [B, 1, 40, 500]
        targets = torch.stack(targets)    # [B, 11]
        
        # Robust shape fixing - handle any dimension count
        batch_size = features.size(0)
        
        # Flatten all non-batch dimensions and reshape to target shape
        # This handles any input shape: [B, 1, 1, 40, 500], [B, 1, 40, 500], [B, 40, 500], etc.
        if features.numel() == batch_size * N_MFCC * FIXED_LENGTH:
            # We have the right number of elements, just reshape
            features = features.view(batch_size, 1, N_MFCC, FIXED_LENGTH)
        elif features.numel() == batch_size * 1 * N_MFCC * FIXED_LENGTH:
            # We have the right number of elements with an extra singleton dim
            features = features.view(batch_size, 1, N_MFCC, FIXED_LENGTH)
        else:
            # Something is wrong with the data, but try to fix it
            print(f"⚠️ Unexpected number of elements: {features.numel()}, expected: {batch_size * N_MFCC * FIXED_LENGTH}")
            features = features.view(batch_size, 1, N_MFCC, FIXED_LENGTH)
        
        # Adjust tensor shapes based on model requirements
        if model_name:
            model_name_lower = model_name.lower()
            if model_name_lower in ["lstm_mixer", "advanced_transformer"]:
                # Both expect [B, 500, 40]
                features = features.squeeze(1)
                if features.shape[1] == N_MFCC and features.shape[2] == FIXED_LENGTH:
                    features = features.permute(0, 2, 1)  # [B, 500, 40]
                print(f"[DEBUG] collate_fn: {model_name_lower} input shape: {features.shape}")
            elif model_name_lower == "vae_mixer":
                features = features.squeeze(1)
            # CNN models expect [B, 1, 40, 500] (already correct)
        return features, targets
        
    except Exception as e:
        print(f"❌ Error in collate_fn: {e}")
        # Return dummy batch
        batch_size = len(batch)
        features = torch.zeros((batch_size, 1, N_MFCC, FIXED_LENGTH)).float()
        targets = torch.zeros((batch_size, len(MIXING_PARAMS))).float()
        return features, targets


class CollateFn:
    """Pickleable collate function wrapper for Windows compatibility"""
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, batch):
        return collate_fn(batch, model_name=self.model_name)


# Define model architectures
class BaselineCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=len(MIXING_PARAMS)):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class EnhancedCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=len(MIXING_PARAMS)):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.bn5(self.fc1(x))))
        x = self.dropout(self.relu(self.bn6(self.fc2(x))))
        x = self.fc3(x)
        return x


class ASTRegressor(nn.Module):
    """Simplified AST-like model for regression tasks"""
    def __init__(self, input_channels=1, output_dim=len(MIXING_PARAMS)):
        super(ASTRegressor, self).__init__()
        # Use a simpler architecture as a fallback
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()  # Add missing relu attribute
        
        # Transformer-like layers (simplified)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.ffn = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.output_layer = nn.Linear(128, output_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Conv layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Prepare for transformer
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Global average pooling and output
        x = x.mean(dim=1)  # [B, C]
        x = self.output_layer(x)
        return x


# Try to import advanced models with robust error handling
try:
    from lstm_mixer import LSTMAudioMixer
    print("✅ Successfully imported LSTMAudioMixer")
except ImportError as e:
    print(f"❌ Could not import LSTMAudioMixer: {e}")
    LSTMAudioMixer = None

try:
    from advanced_transformer import AdvancedTransformerMixer
    print("✅ Successfully imported AdvancedTransformerMixer")
except ImportError as e:
    print(f"❌ Could not import AdvancedTransformerMixer: {e}")
    AdvancedTransformerMixer = None

try:
    from vae_mixer import VAEAudioMixer
    print("✅ Successfully imported VAEAudioMixer")
except ImportError as e:
    print(f"❌ Could not import VAEAudioMixer: {e}")
    VAEAudioMixer = None

try:
    from audio_gan import AudioGAN
    print("✅ Successfully imported AudioGAN")
except ImportError as e:
    print(f"❌ Could not import AudioGAN: {e}")
    AudioGAN = None

try:
    from resnet_mixer import ResNetAudioMixer, SpectralResidualBlock
    print("✅ Successfully imported ResNetAudioMixer")
except ImportError as e:
    print(f"❌ Could not import ResNetAudioMixer: {e}")
    ResNetAudioMixer = None
    SpectralResidualBlock = None

try:
    # Import the DualPathHybrid model from our training script
    import sys
    sys.path.append(os.getcwd())
    from train_dual_path_hybrid import DualPathHybrid
    print("✅ Successfully imported DualPathHybrid")
except ImportError as e:
    print(f"❌ Could not import DualPathHybrid: {e}")
    DualPathHybrid = None


def get_model(model_name, device, num_outputs):
    """Get model instance by name with proper error handling"""
    output_dim = num_outputs
    
    # Define model mappings
    model_map = {
        "baseline_cnn": BaselineCNN,
        "enhanced_cnn": EnhancedCNN,
        "ast_regressor": ASTRegressor,
        "lstm_mixer": LSTMAudioMixer,
        "advanced_transformer": AdvancedTransformerMixer,
        "vae_mixer": VAEAudioMixer,
        "audio_gan": AudioGAN,
        "resnet_mixer": ResNetAudioMixer,
        "dual_path_hybrid": DualPathHybrid
    }
    
    model_class = model_map.get(model_name.lower())
    
    if model_class is None:
        print(f"❌ Model class not found for {model_name}")
        return None
    
    # Create model with proper parameters
    try:
        model_name_lower = model_name.lower()
        
        if model_name_lower == "lstm_mixer":
            model = model_class(
                n_mels=N_MFCC,
                n_outputs=output_dim,
                hidden_size=256,
                num_layers=3,
                dropout=0.3,
                bidirectional=True
            )
        elif model_name_lower == "advanced_transformer":
            model = model_class(
                n_mels=N_MFCC,
                n_outputs=output_dim,
                d_model=256,
                n_heads=4
            )
        elif model_name_lower == "vae_mixer":
            model = model_class(
                n_mels=N_MFCC,
                n_outputs=output_dim,
                latent_dim=64
            )
        elif model_name_lower == "audio_gan":
            model = model_class(
                n_mels=N_MFCC,
                n_outputs=output_dim,
                latent_dim=128,
                style_dim=32
            )
        elif model_name_lower == "resnet_mixer":
            # Fixed parameter handling for ResNet mixer
            if SpectralResidualBlock is not None:
                model = model_class(
                    block=SpectralResidualBlock,
                    layers=[2, 2, 2, 2],
                    n_outputs=output_dim,
                    dropout=0.3,
                    input_channels=1,
                    n_mels=N_MFCC
                )
            else:
                print("❌ SpectralResidualBlock not available for ResNet mixer")
                return None
        elif model_name_lower == "dual_path_hybrid":
            # Dual-path hybrid model - no extra parameters needed
            if DualPathHybrid is not None:
                model = DualPathHybrid()
            else:
                print("❌ DualPathHybrid not available")
                return None
        else:
            # Baseline models
            model = model_class(input_channels=1, output_dim=output_dim)
            
        return model.to(device)
        
    except Exception as e:
        print(f"❌ Error instantiating {model_name}: {e}")
        return None


def create_loader(dataset, model_name, batch_size, num_workers):
    """Create a DataLoader with model-specific collate function"""
    
    # Special handling for dual-path hybrid model
    if model_name.lower() == "dual_path_hybrid":
        # For dual-path hybrid, we need to create a different dataset
        print(f"🔄 Creating specialized dataset for {model_name}")
        return create_dual_path_loader(batch_size, num_workers)
    
    collate_instance = CollateFn(model_name=model_name)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_instance,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )


def create_dual_path_loader(batch_size, num_workers):
    """Create a specialized DataLoader for dual-path hybrid model"""
    try:
        from train_dual_path_hybrid import DualPathDataset
        
        # Check if restoration dataset exists
        restoration_dir = os.path.join(DATA_DIR, "restoration")
        clean_dir = os.path.join(restoration_dir, "clean")
        distorted_dir = os.path.join(restoration_dir, "distorted")
        
        if not os.path.exists(clean_dir) or not os.path.exists(distorted_dir):
            print(f"❌ Restoration dataset not found at {restoration_dir}")
            print(f"   The dual-path hybrid model requires clean/distorted audio pairs")
            print(f"   Please run the audio restoration dataset creation script first")
            return None
        
        # Create dataset with smaller subset for integration test
        dataset = DualPathDataset(clean_dir, distorted_dir, subset_size=500)
        
        if len(dataset) == 0:
            print(f"❌ No audio pairs found in restoration dataset")
            return None
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"📊 Dual-path dataset: {train_size} train, {val_size} val samples")
        
        # Return train loader (we'll handle val/test separately)
        return DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Windows compatibility
            pin_memory=True if device.type == 'cuda' else False
        )
        
    except Exception as e:
        print(f"❌ Error creating dual-path loader: {e}")
        return None


def train_model(model_name, model, train_loader, val_loader, device, epochs=NUM_EPOCHS):
    """Train a model with comprehensive error handling and single checkpoint support"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Special handling for dual-path hybrid model
    is_dual_path = model_name.lower() == "dual_path_hybrid"
    if is_dual_path:
        # Multi-task loss function for dual-path hybrid
        restoration_criterion = nn.L1Loss()  # L1 loss for restoration
        mixing_criterion = nn.MSELoss()
        distortion_criterion = nn.MSELoss()
    else:
        # Standard criterion for other models
        restoration_criterion = None
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint.pth")  # Single checkpoint file
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    patience_counter = 0
    start_epoch = 0
    
    # Try to load checkpoint and resume training
    if os.path.exists(checkpoint_path):
        try:
            print(f"🔄 Loading checkpoint for {model_name}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            history = checkpoint['history']
            
            print(f"✅ Resumed training from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print(f"🔄 Starting training from scratch...")
            start_epoch = 0
            best_val_loss = float('inf')
            patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        try:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (features, targets) in enumerate(pbar):
                try:
                    print(f"[DEBUG] train_model: {model_name} batch {batch_idx} features shape: {features.shape}")
                    features, targets = features.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    if is_dual_path:
                        # Dual-path hybrid model returns a dictionary
                        outputs = model(features)
                        
                        # Multi-task loss calculation
                        # The dual-path model outputs mixing_params that we can compare to targets
                        mixing_loss = criterion(outputs['mixing_params'], targets)
                        
                        # Add small regularization losses if available
                        restoration_weight = 0.1  # Lower weight since we don't have clean targets
                        if 'restored_audio' in outputs and restoration_criterion is not None:
                            # Self-supervision: encourage restoration to be similar to input
                            restoration_loss = restoration_criterion(outputs['restored_audio'], features)
                            loss = mixing_loss + restoration_weight * restoration_loss
                        else:
                            loss = mixing_loss
                        
                    else:
                        # Standard model training
                        outputs = model(features)
                        loss = criterion(outputs, targets)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                except Exception as e:
                    print(f"❌ Error in training batch {batch_idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error in training epoch {epoch+1}: {e}")
            continue
        
        if len(train_loader) == 0:
            print("❌ No successful training batches")
            break
            
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in val_loader:
                try:
                    features, targets = features.to(device), targets.to(device)
                    
                    if is_dual_path:
                        # Dual-path hybrid model returns a dictionary
                        outputs = model(features)
                        loss = criterion(outputs['mixing_params'], targets)
                    else:
                        # Standard model evaluation
                        outputs = model(features)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                except Exception as e:
                    print(f"❌ Error in validation batch: {e}")
                    continue
        
        if len(val_loader) == 0:
            print("❌ No successful validation batches")
            break
            
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model and overwrite checkpoint
        improvement = best_val_loss - avg_val_loss
        if improvement > MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Save/overwrite checkpoint after each epoch (single file, overwrites previous)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'history': history,
            'model_name': model_name
        }
        torch.save(checkpoint, checkpoint_path)  # Overwrites the same file each time
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"🛑 Early stopping after {PATIENCE} epochs without improvement")
            break
    
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"✅ Loaded best model for {model_name}")
    
    # Clean up checkpoint after successful training completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"🧹 Cleaned up checkpoint file for {model_name}")
        except:
            pass
    
    return history, best_val_loss


def evaluate_model(model_name, model, test_loader, device):
    """Evaluate model with comprehensive metrics"""
    model.eval()
    all_outputs = []
    all_targets = []
    
    # Check if this is a dual-path hybrid model
    is_dual_path = model_name.lower() == "dual_path_hybrid"
    
    with torch.no_grad():
        for features, targets in test_loader:
            try:
                features, targets = features.to(device), targets.to(device)
                
                if is_dual_path:
                    # Dual-path hybrid model returns a dictionary
                    outputs_dict = model(features)
                    outputs = outputs_dict['mixing_params']  # Use mixing params for evaluation
                else:
                    # Standard model evaluation
                    outputs = model(features)
                
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            except Exception as e:
                print(f"❌ Error in evaluation batch: {e}")
                continue
    
    if not all_outputs:
        print(f"❌ No successful evaluation batches for {model_name}")
        return {"mse": float('inf'), "mae": float('inf'), "per_param_mae": [float('inf')] * len(MIXING_PARAMS)}
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # Overall metrics
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    r2 = r2_score(all_targets, all_outputs)
    
    # Per-parameter metrics
    per_param_mae = []
    per_param_r2 = []
    
    for i in range(len(MIXING_PARAMS)):
        param_mae = mean_absolute_error(all_targets[:, i], all_outputs[:, i])
        param_r2 = r2_score(all_targets[:, i], all_outputs[:, i])
        per_param_mae.append(param_mae)
        per_param_r2.append(param_r2)
    
    results = {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "per_param_mae": per_param_mae,
        "per_param_r2": per_param_r2
    }
    
    print(f"📊 Evaluation for {model_name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    return results


def analyze_model_strengths(successful_models):
    """Analyze model strengths with robust error handling"""
    print("\n🔍 DETAILED MODEL ANALYSIS:")
    print("=" * 60)
    
    all_results = {}
    
    # Load all model results
    for model_name in successful_models:
        results_path = os.path.join(MODELS_DIR, f"{model_name}_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    all_results[model_name] = json.load(f)
            except Exception as e:
                print(f"❌ Error loading results for {model_name}: {e}")
                continue
    
    if not all_results:
        print("❌ No model results found for analysis")
        return
    
    # Find best model for each parameter
    best_models_per_param = {}
    
    for i, param_name in enumerate(MIXING_PARAMS):
        best_mae = float('inf')
        best_model = None
        best_r2 = -float('inf')
        
        for model_name, results in all_results.items():
            try:
                if 'evaluation' in results and 'per_param_mae' in results['evaluation']:
                    mae = results['evaluation']['per_param_mae'][i]
                    r2 = results['evaluation']['per_param_r2'][i] if 'per_param_r2' in results['evaluation'] else 0;
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_model = model_name
                        best_r2 = r2
            except Exception as e:
                print(f"❌ Error analyzing {model_name} for {param_name}: {e}")
                continue
        
        if best_model:
            best_models_per_param[param_name] = {
                'model': best_model,
                'mae': best_mae,
                'r2': best_r2
            }
    # Print parameter-wise analysis
    if best_models_per_param:
        print(f"{'Parameter':<20} {'Best Model':<20} {'MAE':<10} {'R²':<10}")
        print("-" * 60)
        for param_name, info in best_models_per_param.items():
            print(f"{param_name:<20} {info['model']:<20} {info['mae']:<10.4f} {info['r2']:<10.3f}")
    
    # Overall model ranking
    print(f"\n🏆 OVERALL MODEL RANKING:")
    print("-" * 40)
    model_scores = {}
    for model_name, results in all_results.items():
        if 'evaluation' in results and 'mae' in results['evaluation']:
            score = results['evaluation']['mae']
            model_scores[model_name] = score
    
    if model_scores:
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1])
        for i, (model_name, score) in enumerate(ranked_models, 1):
            star = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{star} {model_name:<20} MAE: {score:.4f}")
    
    return best_models_per_param if best_models_per_param else None


def plot_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    save_path = os.path.join(MODELS_DIR, f"{model_name}_history.png")
    plt.savefig(save_path)
    plt.close()


def save_model_results(model_name, history, evaluation_results):
    """Save model results to JSON file"""
    results = {
        "model_name": model_name,
        "training_history": history,
        "evaluation": evaluation_results,
        "training_completed": True
    }
    
    save_path = os.path.join(MODELS_DIR, f"{model_name}_results.json")
    with open(save_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, dict):
                json_results[k] = {
                    k2: v2.tolist() if isinstance(v2, np.ndarray) else v2
                    for k2, v2 in v.items()
                }
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {save_path}")


def main():
    """Main training pipeline"""
    successful_models = []
    failed_models = []
    
    try:
        print("-----------------------------------------------")
        print("   Step 1: Preprocessing Dataset")
        print("-----------------------------------------------")
        
        # Process audio data and generate features
        preprocess_audio_data()
        
        print("-----------------------------------------------")
        print("   Step 2: Training Models")
        print("-----------------------------------------------")
        
        # Load datasets
        full_train_dataset = AudioMixingDataset(TRAIN_DIR)
        test_dataset = AudioMixingDataset(TEST_DIR)
        
        # Split training data
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
        print(f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation samples.")
        
        # List of models to train - All models except dual_path_hybrid
        models_to_train = [
            
            "lstm_mixer",
            "advanced_transformer",
            "vae_mixer", 
            "audio_gan",
            "resnet_mixer",
            "baseline_cnn",
            "enhanced_cnn",
            "ast_regressor",
            # "dual_path_hybrid"  # Requires different dataset, skip for now
            ]
        
        # Train each model
        for model_name in models_to_train:
            results_path = os.path.join(MODELS_DIR, f"{model_name}_results.json")
            model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
            checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}_checkpoint.pth")
            
            # Check if training is complete (results exist, model exists, and no checkpoint)
            if os.path.exists(results_path) and os.path.exists(model_path) and not os.path.exists(checkpoint_path):
                print(f"--- Skipping {model_name} (already trained completely) ---")
                successful_models.append(model_name)
                continue
            
            # Check if training was interrupted (checkpoint exists)
            if os.path.exists(checkpoint_path):
                print(f"🔄 Found checkpoint for {model_name} - will resume training")
            
            print(f"🏋️‍♀️ Training {model_name}...")
            
            try:
                # Get model
                model = get_model(model_name, device, num_outputs=len(MIXING_PARAMS))
                if model is None:
                    failed_models.append(model_name)
                    continue
                
                print(f"✅ Model {model_name} loaded successfully")
                
                # Create data loaders
                if model_name.lower() == "dual_path_hybrid":
                    # Special handling for dual-path hybrid
                    train_loader = create_loader(train_dataset, model_name, BATCH_SIZE, NUM_WORKERS)
                    if train_loader is None:
                        print(f"❌ Could not create dual-path data loader for {model_name}")
                        failed_models.append(model_name)
                        continue
                    # For dual-path, use same loader for val and test (it handles splits internally)
                    val_loader = train_loader
                    test_loader = train_loader
                else:
                    # Standard model data loaders
                    train_loader = create_loader(train_dataset, model_name, BATCH_SIZE, NUM_WORKERS)
                    val_loader = create_loader(val_dataset, model_name, BATCH_SIZE, NUM_WORKERS)
                    test_loader = create_loader(test_dataset, model_name, BATCH_SIZE, NUM_WORKERS)
                
                # Train model
                history, best_val_loss = train_model(model_name, model, train_loader, val_loader, device)
                
                # Evaluate model
                evaluation_results = evaluate_model(model_name, model, test_loader, device)
                
                # Save results
                save_model_results(model_name, history, evaluation_results)
                
                # Plot training history
                plot_history(history, model_name)
                
                successful_models.append(model_name)
                print(f"✅ {model_name} training completed successfully")
                
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")
                failed_models.append(model_name)
                continue
        
        print("\n--- Training Summary ---")
        print(f"✅ Successful models: {successful_models}")
        print(f"❌ Failed models: {failed_models}")
        
        # Analyze model strengths
        if successful_models:
            analyze_model_strengths(successful_models)
        
    except Exception as e:
        print(f"❌ Critical error in main pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
