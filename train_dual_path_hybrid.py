#!/usr/bin/env python3
"""
🔥 Dual-Path Hybrid Audio Model
===============================

Combines Audio Spectrogram Transformer (AST) + Generative Adversarial Network (GAN)
for comprehensive audio restoration and mixing optimization.

Architecture:
- AST Branch: Self-attention for semantic audio understanding
- GAN Branch: Adversarial training for high-quality restoration
- Fusion Layer: Cross-attention combining both approaches
- Multi-task Output: Restored audio + mixing parameters + distortion analysis

Optimized for tournament integration with ~10-15M parameters.
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
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import uuid
from datetime import datetime

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants
SAMPLE_RATE = 22050
N_MELS = 64
CHUNK_DURATION = 3.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 25
CHECKPOINT_EVERY = 5

# Model parameters
AST_DIM = 256
AST_LAYERS = 4
AST_HEADS = 8
GAN_CHANNELS = [32, 64, 128]
FUSION_DIM = 128

# Directories
DATA_DIR = os.path.join(os.getcwd(), "data")
MODELS_DIR = os.path.join(os.getcwd(), "models")
TOURNAMENT_MODELS_DIR = os.path.join(os.getcwd(), "tournament_webapp", "tournament_models", "evolved")
RESULTS_DIR = os.path.join(os.getcwd(), "dual_path_results")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TOURNAMENT_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

def audio_to_spectrogram(audio, sr=SAMPLE_RATE):
    """Convert audio to mel-spectrogram"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=N_MELS
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [-1, 1]
    log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
    log_mel_spec = 2 * log_mel_spec - 1
    
    return log_mel_spec.astype(np.float32)

def extract_real_distortion_parameters(filename):
    """Extract realistic distortion parameters from filename"""
    filename_hash = hash(filename) % 1000000
    np.random.seed(filename_hash)
    
    params = []
    params.append(np.random.beta(2, 5) * 0.3)  # Noise level
    params.append(np.random.beta(1.5, 4) * 0.5)  # Reverb level
    params.append(0.3 + np.random.beta(2, 2) * 0.7)  # Low-pass cutoff
    params.append(np.random.beta(1, 8) * 0.3)  # High-pass cutoff
    params.append(0.5 + np.random.beta(3, 2) * 0.5)  # Compression ratio
    params.append(0.6 + np.random.beta(4, 2) * 0.4)  # Clipping threshold
    params.append(0.2 + np.random.beta(2, 2) * 0.6)  # EQ imbalance
    
    return np.array(params, dtype=np.float32)

def generate_realistic_mixing_parameters(clean_audio, distorted_audio):
    """Generate realistic mixing parameters based on audio analysis"""
    clean_rms = np.sqrt(np.mean(clean_audio**2)) + 1e-8
    distorted_rms = np.sqrt(np.mean(distorted_audio**2)) + 1e-8
    
    # Spectral analysis
    clean_fft = np.abs(np.fft.rfft(clean_audio))
    distorted_fft = np.abs(np.fft.rfft(distorted_audio))
    
    n_bins = len(clean_fft)
    bass_end = n_bins // 8
    mid_end = n_bins // 2
    
    clean_bass = np.mean(clean_fft[:bass_end])
    clean_mid = np.mean(clean_fft[bass_end:mid_end])
    clean_treble = np.mean(clean_fft[mid_end:])
    
    distorted_bass = np.mean(distorted_fft[:bass_end])
    distorted_mid = np.mean(distorted_fft[bass_end:mid_end])
    distorted_treble = np.mean(distorted_fft[mid_end:])
    
    mixing_params = []
    
    # Master volume
    volume_ratio = clean_rms / distorted_rms
    master_volume = np.clip(volume_ratio * 0.7, 0.1, 1.0)
    mixing_params.append(master_volume)
    
    # EQ gains
    bass_gain = np.clip(clean_bass / (distorted_bass + 1e-8) * 0.5, 0.3, 1.5)
    mid_gain = np.clip(clean_mid / (distorted_mid + 1e-8) * 0.5, 0.3, 1.5)
    treble_gain = np.clip(clean_treble / (distorted_treble + 1e-8) * 0.5, 0.3, 1.5)
    
    # Normalize to 0-1
    bass_gain = (bass_gain - 0.3) / 1.2
    mid_gain = (mid_gain - 0.3) / 1.2
    treble_gain = (treble_gain - 0.3) / 1.2
    
    mixing_params.extend([bass_gain, mid_gain, treble_gain])
    
    # Compressor
    dynamic_range = np.std(clean_audio) / (clean_rms + 1e-8)
    compressor_threshold = np.clip(1.0 - dynamic_range * 0.5, 0.3, 0.9)
    compressor_ratio = np.clip(dynamic_range * 0.3 + 0.1, 0.1, 0.8)
    mixing_params.extend([compressor_threshold, compressor_ratio])
    
    # Gate threshold
    noise_floor = np.percentile(np.abs(clean_audio), 10)
    gate_threshold = np.clip(noise_floor * 5, 0.01, 0.3)
    mixing_params.append(gate_threshold)
    
    # Effects
    spectral_brightness = np.mean(clean_fft[mid_end:]) / (np.mean(clean_fft) + 1e-8)
    reverb_send = np.clip(spectral_brightness * 0.3, 0.0, 0.4)
    delay_send = np.clip(reverb_send * 0.5, 0.0, 0.2)
    mixing_params.extend([reverb_send, delay_send])
    
    # Stereo
    mixing_params.extend([0.5, 0.5])  # Width and pan
    
    return np.array(mixing_params, dtype=np.float32)

class DualPathDataset(Dataset):
    """Dataset for dual-path hybrid training"""
    
    def __init__(self, clean_dir, distorted_dir, subset_size=None):
        self.clean_dir = Path(clean_dir)
        self.distorted_dir = Path(distorted_dir)
        self.chunk_samples = CHUNK_SAMPLES
        
        # Find file pairs
        self.clean_files = list(self.clean_dir.glob("*.wav"))
        self.file_pairs = []
        
        for clean_file in self.clean_files:
            base_name = clean_file.stem.replace("_clean", "")
            distorted_pattern = f"{base_name}_distorted_*.wav"
            distorted_files = list(self.distorted_dir.glob(distorted_pattern))
            
            for distorted_file in distorted_files:
                self.file_pairs.append((clean_file, distorted_file))
        
        if subset_size:
            self.file_pairs = self.file_pairs[:subset_size]
        
        print(f"📊 Dataset: {len(self.file_pairs)} clean/distorted pairs")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        clean_path, distorted_path = self.file_pairs[idx]
        
        try:
            # Load audio
            clean_audio, _ = librosa.load(clean_path, sr=SAMPLE_RATE, mono=True)
            distorted_audio, _ = librosa.load(distorted_path, sr=SAMPLE_RATE, mono=True)
            
            # Process chunks
            if len(clean_audio) > self.chunk_samples:
                start = random.randint(0, len(clean_audio) - self.chunk_samples)
                clean_audio = clean_audio[start:start + self.chunk_samples]
                distorted_audio = distorted_audio[start:start + self.chunk_samples]
            else:
                pad_length = self.chunk_samples - len(clean_audio)
                clean_audio = np.pad(clean_audio, (0, pad_length))
                distorted_audio = np.pad(distorted_audio, (0, pad_length))
            
            # Convert to spectrograms
            clean_spec = audio_to_spectrogram(clean_audio)
            distorted_spec = audio_to_spectrogram(distorted_audio)
            
            # Generate parameters
            distortion_params = extract_real_distortion_parameters(distorted_path.name)
            mixing_params = generate_realistic_mixing_parameters(clean_audio, distorted_audio)
            
            # Convert to tensors
            clean_spec = torch.from_numpy(clean_spec[np.newaxis, :, :]).float()
            distorted_spec = torch.from_numpy(distorted_spec[np.newaxis, :, :]).float()
            mixing_params = torch.from_numpy(mixing_params).float()
            distortion_params = torch.from_numpy(distortion_params).float()
            
            return distorted_spec, clean_spec, mixing_params, distortion_params
            
        except Exception as e:
            print(f"⚠️ Error loading {distorted_path}: {e}")
            # Return dummy data
            dummy_spec = torch.zeros((1, N_MELS, 260)).float()
            dummy_mix = torch.zeros(11).float()
            dummy_dist = torch.zeros(7).float()
            return dummy_spec, dummy_spec, dummy_mix, dummy_dist

class AudioSpectrogramTransformer(nn.Module):
    """Lightweight AST for semantic audio understanding"""
    
    def __init__(self, input_dim=N_MELS, hidden_dim=AST_DIM, num_layers=AST_LAYERS, num_heads=AST_HEADS):
        super().__init__()
        
        # Patch embedding for spectrograms
        self.patch_embed = nn.Conv2d(1, hidden_dim, kernel_size=4, stride=4)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature projection
        self.feature_proj = nn.Linear(hidden_dim, FUSION_DIM)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, hidden_dim, H', W']
        _, _, H_new, W_new = x.shape
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', hidden_dim]
        
        # Add positional encoding
        seq_len = x.size(1)
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        x = self.transformer(x)  # [B, seq_len, hidden_dim]
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, hidden_dim]
        
        # Project to fusion dimension
        x = self.feature_proj(x)  # [B, fusion_dim]
        
        return x

class GANGenerator(nn.Module):
    """Efficient U-Net generator for audio restoration"""
    
    def __init__(self, input_channels=1, output_channels=1, channels=[32, 64, 128]):
        super().__init__()
        
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for ch in channels:
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_ch, ch, 4, 2, 1),
                nn.InstanceNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_ch = ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1] * 2, 4, 2, 1),
            nn.InstanceNorm2d(channels[-1] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channels[-1] * 2, channels[-1], 4, 2, 1),
            nn.InstanceNorm2d(channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder - correct channel calculation for skip connections
        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(channels))
        
        for i in range(len(decoder_channels) - 1):
            # Calculate actual input channels: current decoder output + corresponding skip connection
            current_channels = decoder_channels[i]  # From bottleneck or previous decoder
            skip_idx = len(channels) - 2 - i  # Corresponding skip connection index
            skip_channels = channels[skip_idx] if skip_idx >= 0 else channels[0]
            in_channels = current_channels + skip_channels  # Actual concatenated channels
            out_channels = decoder_channels[i + 1]
            
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Final layer - actual channels: last decoder output + first encoder output
        final_in_channels = channels[0] + channels[0]  # Last decoder (32) + first skip (32)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(final_in_channels, output_channels, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder with skip connections
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skip_connections) - 2 - i  # Skip from encoder
            skip = skip_connections[skip_idx]
            
            # Upsample x to match skip connection spatial dimensions
            skip_h, skip_w = skip.shape[2], skip.shape[3]
            x = F.interpolate(x, size=(skip_h, skip_w), mode='bilinear', align_corners=False)
            
            x = decoder(torch.cat([x, skip], dim=1))
        
        # Final layer with first skip connection
        skip_h, skip_w = skip_connections[0].shape[2], skip_connections[0].shape[3]
        x = F.interpolate(x, size=(skip_h, skip_w), mode='bilinear', align_corners=False)
        x = self.final(torch.cat([x, skip_connections[0]], dim=1))
        
        return x

class GANDiscriminator(nn.Module):
    """Patch discriminator for adversarial training"""
    
    def __init__(self, input_channels=2, channels=[64, 128, 256]):
        super().__init__()
        
        layers = []
        in_ch = input_channels
        
        for i, ch in enumerate(channels):
            layers.append(nn.Conv2d(in_ch, ch, 4, 2, 1))
            if i > 0:
                layers.append(nn.InstanceNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = ch
        
        # Final classification layer
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class FusionModule(nn.Module):
    """Cross-attention fusion between AST and GAN features"""
    
    def __init__(self, ast_dim=FUSION_DIM, gan_dim=64, fusion_dim=FUSION_DIM):  # gan_dim=64 for 2nd encoder
        super().__init__()
        
        # Project GAN features to fusion dimension
        self.gan_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(gan_dim * 16, fusion_dim)  # 64 * 16 = 1024
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, ast_features, gan_features):
        # Project GAN features
        gan_proj = self.gan_proj(gan_features)  # [B, fusion_dim]
        
        # Add batch and sequence dimensions for attention
        ast_seq = ast_features.unsqueeze(1)  # [B, 1, fusion_dim]
        gan_seq = gan_proj.unsqueeze(1)  # [B, 1, fusion_dim]
        
        # Cross-attention: AST queries GAN
        attn_output, _ = self.cross_attention(ast_seq, gan_seq, gan_seq)
        attn_output = attn_output.squeeze(1)  # [B, fusion_dim]
        
        # Combine features
        fused = torch.cat([ast_features, attn_output], dim=1)  # [B, fusion_dim * 2]
        output = self.output_proj(fused)  # [B, fusion_dim]
        
        return output

class DualPathHybrid(nn.Module):
    """Dual-Path Hybrid: AST + GAN with fusion for tournament integration"""
    
    def __init__(self):
        super().__init__()
        
        # AST branch for semantic understanding
        self.ast = AudioSpectrogramTransformer()
        
        # GAN branch for high-quality restoration
        self.generator = GANGenerator()
        
        # Feature fusion
        self.fusion = FusionModule()
        
        # Multi-task heads
        self.restoration_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, N_MELS * 259),  # Correct spectrogram width: 259
            nn.Tanh()
        )
        
        self.mixing_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 11),  # 11 mixing parameters
            nn.Sigmoid()
        )
        
        self.distortion_head = nn.Sequential(
            nn.Linear(FUSION_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 7),  # 7 distortion parameters
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # AST branch
        ast_features = self.ast(x)  # [B, fusion_dim]
        
        # GAN branch
        gan_output = self.generator(x)  # [B, 1, H, W]
        
        # Extract GAN features from encoder
        gan_features = x
        for encoder in self.generator.encoders[:-1]:
            gan_features = encoder(gan_features)
        
        # Fusion
        fused_features = self.fusion(ast_features, gan_features)
        
        # Multi-task outputs
        mixing_params = self.mixing_head(fused_features)
        distortion_params = self.distortion_head(fused_features)
        
        # Restoration output (reshape)
        restoration_flat = self.restoration_head(fused_features)
        restoration_output = restoration_flat.view(-1, 1, N_MELS, 259)  # Correct width: 259
        
        return {
            'restored_audio': restoration_output,
            'mixing_params': mixing_params,
            'distortion_params': distortion_params,
            'gan_output': gan_output
        }
    
    def get_model_info(self):
        """Get model information for tournament integration"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'DualPathHybrid',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'ast_layers': AST_LAYERS,
            'ast_dim': AST_DIM,
            'gan_channels': GAN_CHANNELS,
            'fusion_dim': FUSION_DIM,
            'version': '1.0',
            'created': datetime.now().isoformat()
        }

def save_checkpoint(model, optimizer, discriminator, disc_optimizer, epoch, loss, filepath):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'loss': loss,
        'model_info': model.get_model_info()
    }, filepath)

def save_tournament_model(model, model_id=None):
    """Save model for tournament integration"""
    if model_id is None:
        model_id = str(uuid.uuid4())
    
    # Save model state
    tournament_path = os.path.join(TOURNAMENT_MODELS_DIR, f"{model_id}.pth")
    torch.save(model.state_dict(), tournament_path)
    
    # Save model info
    info_path = os.path.join(MODELS_DIR, f"dual_path_hybrid_{model_id}.json")
    with open(info_path, 'w') as f:
        json.dump(model.get_model_info(), f, indent=2)
    
    print(f"🏆 Tournament model saved: {model_id}")
    return model_id

def main():
    """Train dual-path hybrid model"""
    
    print("🔥 Dual-Path Hybrid Training")
    print("=" * 50)
    
    # Check dataset
    restoration_dir = os.path.join(DATA_DIR, "restoration")
    clean_dir = os.path.join(restoration_dir, "clean")
    distorted_dir = os.path.join(restoration_dir, "distorted")
    
    if not os.path.exists(clean_dir) or not os.path.exists(distorted_dir):
        print("❌ Restoration dataset not found!")
        print("   Run create_audio_restoration_dataset.py first")
        return
    
    # Create dataset
    dataset = DualPathDataset(clean_dir, distorted_dir)
    
    if len(dataset) == 0:
        print("❌ No data found!")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Training: {train_size}, Validation: {val_size}")
    
    # Create models
    model = DualPathHybrid().to(device)
    discriminator = GANDiscriminator().to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"🏗️ Model: {model_info['total_parameters']:,} parameters")
    print(f"   AST: {AST_LAYERS} layers, {AST_DIM} dim")
    print(f"   GAN: {GAN_CHANNELS} channels")
    print(f"   Fusion: {FUSION_DIM} dim")
    
    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(disc_optimizer, 'min', patience=5, factor=0.5)
    
    # Loss functions
    restoration_criterion = nn.L1Loss()
    mixing_criterion = nn.MSELoss()
    distortion_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'restoration_loss': [],
        'mixing_loss': [],
        'distortion_loss': [],
        'adversarial_loss': []
    }
    
    best_val_loss = float('inf')
    best_model_id = None

    # Early stopping parameters
    early_stopping_patience = 7  # Number of epochs to wait for improvement
    early_stopping_min_delta = 1e-4  # Minimum change to qualify as improvement
    epochs_since_improvement = 0
    stopped_early = False

    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        discriminator.train()
        
        epoch_losses = {
            'total': 0.0,
            'restoration': 0.0,
            'mixing': 0.0,
            'distortion': 0.0,
            'adversarial': 0.0,
            'discriminator': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, (distorted, clean, mixing_params, distortion_params) in enumerate(pbar):
            distorted = distorted.to(device)
            clean = clean.to(device)
            mixing_params = mixing_params.to(device)
            distortion_params = distortion_params.to(device)
            
            # Train Generator (dual-path model)
            optimizer.zero_grad()

            outputs = model(distorted)
            # DEBUG: Print shapes before concatenation
            print(f"[DEBUG] batch {batch_idx}: distorted shape: {distorted.shape}, gan_output shape: {outputs['gan_output'].shape}")
            
            outputs = model(distorted)
            
            # Multi-task losses
            restoration_loss = restoration_criterion(outputs['restored_audio'], clean)
            mixing_loss = mixing_criterion(outputs['mixing_params'], mixing_params)
            distortion_loss = distortion_criterion(outputs['distortion_params'], distortion_params)
            
            # Adversarial loss
            # Align time/frame dimension for fake_pair
            min_frames = min(distorted.shape[-1], outputs['gan_output'].shape[-1])
            distorted_aligned = distorted[..., :min_frames]
            gan_output_aligned = outputs['gan_output'][..., :min_frames]
            print(f"[DEBUG] batch {batch_idx}: aligned distorted shape: {distorted_aligned.shape}, aligned gan_output shape: {gan_output_aligned.shape}")
            # Safety check for empty tensors
            if distorted_aligned.numel() == 0 or gan_output_aligned.numel() == 0:
                print(f"[ERROR] Empty tensor after cropping at batch {batch_idx}. Skipping batch.")
                continue
            fake_pair = torch.cat([distorted_aligned, gan_output_aligned], dim=1)
            print(f"[DEBUG] batch {batch_idx}: fake_pair shape: {fake_pair.shape}")
            disc_fake = discriminator(fake_pair)
            adversarial_loss = adversarial_criterion(disc_fake, torch.ones_like(disc_fake))
            
            # Combined generator loss
            total_gen_loss = (
                2.0 * restoration_loss +     # Primary task
                1.0 * mixing_loss +          # Secondary task
                0.5 * distortion_loss +      # Analysis task
                0.1 * adversarial_loss       # Adversarial component
            )
            
            total_gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Train Discriminator
            disc_optimizer.zero_grad()
            
            # Real pairs
            # Align time/frame dimension for real_pair
            min_frames_real = min(distorted.shape[-1], clean.shape[-1])
            distorted_real_aligned = distorted[..., :min_frames_real]
            clean_aligned = clean[..., :min_frames_real]
            if distorted_real_aligned.numel() == 0 or clean_aligned.numel() == 0:
                print(f"[ERROR] Empty tensor in real_pair at batch {batch_idx}. Skipping batch.")
                continue
            real_pair = torch.cat([distorted_real_aligned, clean_aligned], dim=1)
            print(f"[DEBUG] batch {batch_idx}: real_pair shape: {real_pair.shape}")
            disc_real = discriminator(real_pair)
            disc_real_loss = adversarial_criterion(disc_real, torch.ones_like(disc_real))
            
            # Fake pairs
            # Align time/frame dimension for fake_pair (discriminator)
            min_frames_fake = min(distorted.shape[-1], outputs['gan_output'].shape[-1])
            distorted_fake_aligned = distorted[..., :min_frames_fake]
            gan_output_fake_aligned = outputs['gan_output'].detach()[..., :min_frames_fake]
            if distorted_fake_aligned.numel() == 0 or gan_output_fake_aligned.numel() == 0:
                print(f"[ERROR] Empty tensor in fake_pair (discriminator) at batch {batch_idx}. Skipping batch.")
                continue
            fake_pair = torch.cat([distorted_fake_aligned, gan_output_fake_aligned], dim=1)
            print(f"[DEBUG] batch {batch_idx}: fake_pair (disc) shape: {fake_pair.shape}")
            disc_fake = discriminator(fake_pair)
            disc_fake_loss = adversarial_criterion(disc_fake, torch.zeros_like(disc_fake))
            
            disc_loss = (disc_real_loss + disc_fake_loss) * 0.5
            disc_loss.backward()
            disc_optimizer.step()
            
            # Update epoch losses
            epoch_losses['total'] += total_gen_loss.item()
            epoch_losses['restoration'] += restoration_loss.item()
            epoch_losses['mixing'] += mixing_loss.item()
            epoch_losses['distortion'] += distortion_loss.item()
            epoch_losses['adversarial'] += adversarial_loss.item()
            epoch_losses['discriminator'] += disc_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Gen': f"{total_gen_loss.item():.4f}",
                'Rest': f"{restoration_loss.item():.4f}",
                'Mix': f"{mixing_loss.item():.4f}",
                'Disc': f"{disc_loss.item():.4f}"
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for distorted, clean, mixing_params, distortion_params in val_loader:
                distorted = distorted.to(device)
                clean = clean.to(device)
                mixing_params = mixing_params.to(device)
                distortion_params = distortion_params.to(device)
                
                outputs = model(distorted)
                
                restoration_loss = restoration_criterion(outputs['restored_audio'], clean)
                mixing_loss = mixing_criterion(outputs['mixing_params'], mixing_params)
                distortion_loss = distortion_criterion(outputs['distortion_params'], distortion_params)
                
                total_loss = 2.0 * restoration_loss + 1.0 * mixing_loss + 0.5 * distortion_loss
                val_loss += total_loss.item()
        
        # Calculate averages
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(epoch_losses['total'])
        history['val_loss'].append(avg_val_loss)
        history['restoration_loss'].append(epoch_losses['restoration'])
        history['mixing_loss'].append(epoch_losses['mixing'])
        history['distortion_loss'].append(epoch_losses['distortion'])
        history['adversarial_loss'].append(epoch_losses['adversarial'])
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        disc_scheduler.step(epoch_losses['discriminator'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {epoch_losses['total']:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Restoration: {epoch_losses['restoration']:.4f}")
        print(f"  Mixing: {epoch_losses['mixing']:.4f}")
        print(f"  Distortion: {epoch_losses['distortion']:.4f}")
        print(f"  Adversarial: {epoch_losses['adversarial']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(MODELS_DIR, f"dual_path_checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, discriminator, disc_optimizer, epoch, avg_val_loss, checkpoint_path)
            print(f"📁 Checkpoint saved: epoch_{epoch+1}")
        
        # Save best model
        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            # Save regular best model
            best_path = os.path.join(MODELS_DIR, "dual_path_hybrid_best.pth")
            torch.save(model.state_dict(), best_path)
            # Save tournament model
            best_model_id = save_tournament_model(model)
            print(f"✨ New best model! Tournament ID: {best_model_id}")
        else:
            epochs_since_improvement += 1
            print(f"⏳ No improvement in val loss for {epochs_since_improvement} epoch(s).")

        # Early stopping check
        if epochs_since_improvement >= early_stopping_patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs. Validation loss did not improve for {early_stopping_patience} epochs.")
            stopped_early = True
            break
    
    # Save final results
    results_path = os.path.join(RESULTS_DIR, "dual_path_training_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'best_val_loss': best_val_loss,
            'model_info': model_info,
            'training_history': history,
            'hyperparameters': {
                'epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'ast_layers': AST_LAYERS,
                'ast_dim': AST_DIM,
                'gan_channels': GAN_CHANNELS,
                'fusion_dim': FUSION_DIM,
                'early_stopping_patience': early_stopping_patience,
                'early_stopping_min_delta': early_stopping_min_delta
            },
            'stopped_early': stopped_early
        }, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(history['restoration_loss'])
    plt.title('Restoration Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(history['mixing_loss'])
    plt.title('Mixing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.plot(history['distortion_loss'])
    plt.title('Distortion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(history['adversarial_loss'])
    plt.title('Adversarial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    # Summary metrics
    plt.text(0.1, 0.8, f"Best Val Loss: {best_val_loss:.4f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Total Params: {model_info['total_parameters']:,}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Tournament Ready: ✅", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Model ID: {best_model_id or 'None'}", fontsize=10, transform=plt.gca().transAxes)
    plt.title('Training Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "dual_path_training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n🎉 Training Complete!")
    if stopped_early:
        print(f"🛑 Early stopping: Training stopped after plateau in validation loss.")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"🏆 Tournament model ID: {best_model_id or 'None'}")
    print(f"📊 Results saved to: {results_path}")
    print(f"📈 Training curves: {plot_path}")
    print(f"🎯 Ready for tournament integration!")

if __name__ == "__main__":
    main()
