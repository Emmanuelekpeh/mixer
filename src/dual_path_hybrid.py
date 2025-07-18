#!/usr/bin/env python3
"""
🔄 Dual-Path Hybrid Model
========================

Sophisticated AST + GAN hybrid for audio restoration and mixing parameter prediction.
Combines Audio Spectrogram Transformer with adversarial training for high-quality results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer layers"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class PatchEmbedding(nn.Module):
    """Convert spectrogram patches to embeddings like in AST"""
    
    def __init__(self, img_size=(64, 500), patch_size=(4, 4), in_channels=1, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure input size matches expected
        if H != self.img_size[0] or W != self.img_size[1]:
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        return x

class AudioSpectrogramTransformer(nn.Module):
    """Audio Spectrogram Transformer branch"""
    
    def __init__(self, img_size=(64, 500), patch_size=(4, 4), embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=0.1) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Feature extraction heads
        self.feature_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4)
        )
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Use class token for global representation
        cls_features = x[:, 0]  # (B, embed_dim)
        global_features = self.feature_head(cls_features)  # (B, embed_dim//4)
        
        # Also get patch features for spatial information
        patch_features = x[:, 1:]  # (B, num_patches, embed_dim)
        
        return global_features, patch_features

class ResidualBlock(nn.Module):
    """Residual block for GAN generator"""
    
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class GANGenerator(nn.Module):
    """GAN-based generator for audio restoration"""
    
    def __init__(self, input_channels=1, output_channels=1, base_channels=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleList([
            # Down 1: 64x500 -> 32x250
            nn.Sequential(
                nn.Conv2d(input_channels, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2)
            ),
            # Down 2: 32x250 -> 16x125
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2)
            ),
            # Down 3: 16x125 -> 8x62 (handle odd dimension)
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2)
            ),
            # Down 4: 8x62 -> 4x31
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
        )
        
        # Decoder
        self.decoder = nn.ModuleList([
            # Up 1: 4x31 -> 8x62
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 4),
                nn.ReLU()
            ),
            # Up 2: 8x62 -> 16x125 (handle odd dimension)
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU()
            ),
            # Up 3: 16x125 -> 32x250
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU()
            ),
            # Up 4: 32x250 -> 64x500
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 2, output_channels, 4, 2, 1),
                nn.Tanh()
            )
        ])
        
    def forward(self, x):
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i < len(self.decoder) - 1:  # Skip the final output layer
                x = layer(x)
                # Add skip connection from corresponding encoder layer
                skip_idx = len(encoder_features) - 1 - i
                if i < len(encoder_features):
                    skip_features = encoder_features[skip_idx]
                    # Handle size mismatches
                    if x.shape[-2:] != skip_features.shape[-2:]:
                        skip_features = F.interpolate(skip_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    x = torch.cat([x, skip_features], dim=1)
            else:
                # Final layer
                x = layer(x)
        
        return x

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between AST and GAN features"""
    
    def __init__(self, ast_dim, gan_dim, fusion_dim):
        super().__init__()
        self.ast_proj = nn.Linear(ast_dim, fusion_dim)
        self.gan_proj = nn.Linear(gan_dim, fusion_dim)
        
        self.cross_attn = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
    def forward(self, ast_features, gan_features):
        # Project to common dimension
        ast_proj = self.ast_proj(ast_features)  # (B, fusion_dim)
        gan_proj = self.gan_proj(gan_features)   # (B, H*W, fusion_dim)
        
        # Add sequence dimension to AST features
        ast_proj = ast_proj.unsqueeze(1)  # (B, 1, fusion_dim)
        
        # Cross-attention: AST queries GAN features
        fused, _ = self.cross_attn(ast_proj, gan_proj, gan_proj)
        fused = self.norm1(fused + ast_proj)
        
        # Feed-forward
        fused = fused + self.ffn(self.norm2(fused))
        
        return fused.squeeze(1)  # (B, fusion_dim)

class DualPathHybrid(nn.Module):
    """
    Dual-Path Hybrid Model: AST + GAN with sophisticated fusion
    
    Architecture:
    1. AST Branch: Semantic understanding via transformer attention
    2. GAN Branch: High-quality restoration via adversarial training
    3. Cross-Attention Fusion: Intelligent feature combination
    4. Multi-task Output: Restored audio + mixing parameters + distortion analysis
    """
    
    def __init__(self, n_mels=64, sequence_length=500, n_mixing_params=11, n_distortion_params=7):
        super().__init__()
        
        self.n_mels = n_mels
        self.sequence_length = sequence_length
        
        # AST Branch
        self.ast = AudioSpectrogramTransformer(
            img_size=(n_mels, sequence_length),
            patch_size=(4, 4),
            embed_dim=384,
            depth=6,
            num_heads=6
        )
        
        # GAN Branch
        self.generator = GANGenerator(
            input_channels=1,
            output_channels=1,
            base_channels=64
        )
        
        # Feature extraction from GAN features
        self.gan_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Cross-attention fusion
        ast_feature_dim = 384 // 4  # From AST feature head
        gan_feature_dim = 256       # From GAN feature extractor
        fusion_dim = 256
        
        self.fusion = CrossAttentionFusion(ast_feature_dim, gan_feature_dim, fusion_dim)
        
        # Task-specific heads
        self.restoration_refiner = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_mels * sequence_length),
            nn.Tanh()
        )
        
        self.mixing_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_mixing_params),
            nn.Sigmoid()
        )
        
        self.distortion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_distortion_params),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_intermediate=False):
        """
        Args:
            x: Input spectrogram (B, 1, n_mels, sequence_length)
            return_intermediate: Whether to return intermediate features
        
        Returns:
            dict with keys: 'restored_audio', 'mixing_params', 'distortion_params'
        """
        # Ensure input is 4D [batch, channels, height, width]
        if x.dim() == 5:
            # Remove extra dimensions
            while x.dim() > 4 and x.shape[1] == 1:
                x = x.squeeze(1)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Final check to ensure we have exactly 4 dimensions
        while x.dim() > 4:
            x = x.squeeze(1)
        
        batch_size = x.shape[0]
        
        # AST Branch: Semantic understanding
        ast_global_features, ast_patch_features = self.ast(x)  # Global: (B, 96), Patches: (B, num_patches, 384)
        
        # GAN Branch: High-quality restoration
        gan_restored = self.generator(x)  # (B, 1, n_mels, sequence_length)
        
        # Extract GAN features for fusion
        gan_features = self.gan_feature_extractor(gan_restored)  # (B, 256)
        
        # Prepare GAN features for cross-attention (add spatial dimension)
        gan_features_spatial = gan_features.unsqueeze(1)  # (B, 1, 256)
        
        # Cross-attention fusion
        fused_features = self.fusion(ast_global_features, gan_features_spatial)  # (B, 256)
        
        # Task-specific outputs
        # 1. Refined restoration (combining GAN output with fused intelligence)
        restoration_refinement = self.restoration_refiner(fused_features)  # (B, n_mels * sequence_length)
        restoration_refinement = restoration_refinement.view(batch_size, 1, self.n_mels, self.sequence_length)
        
        # Combine GAN restoration with intelligent refinement
        restored_audio = gan_restored + 0.1 * restoration_refinement  # Subtle refinement
        restored_audio = torch.tanh(restored_audio)  # Ensure valid range
        
        # 2. Mixing parameter prediction
        mixing_params = self.mixing_head(fused_features)  # (B, n_mixing_params)
        
        # 3. Distortion analysis
        distortion_params = self.distortion_head(fused_features)  # (B, n_distortion_params)
        
        outputs = {
            'restored_audio': restored_audio,
            'mixing_params': mixing_params,
            'distortion_params': distortion_params
        }
        
        if return_intermediate:
            outputs.update({
                'ast_features': ast_global_features,
                'gan_features': gan_features,
                'fused_features': fused_features,
                'gan_restored': gan_restored,
                'restoration_refinement': restoration_refinement
            })
        
        return outputs

class DualPathDiscriminator(nn.Module):
    """Discriminator for adversarial training of the dual-path model"""
    
    def __init__(self, input_channels=1, base_channels=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # Layer 1: 64x500 -> 32x250
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 32x250 -> 16x125
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 16x125 -> 8x62
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 8x62 -> 4x31
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x).view(x.size(0), -1).mean(dim=1)

def test_dual_path_hybrid():
    """Test the dual-path hybrid model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = DualPathHybrid(n_mels=64, sequence_length=500).to(device)
    discriminator = DualPathDiscriminator().to(device)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 1, 64, 500).to(device)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, return_intermediate=True)
    
    print(f"Restored audio shape: {outputs['restored_audio'].shape}")
    print(f"Mixing params shape: {outputs['mixing_params'].shape}")
    print(f"Distortion params shape: {outputs['distortion_params'].shape}")
    
    # Test discriminator
    with torch.no_grad():
        disc_out = discriminator(outputs['restored_audio'])
    print(f"Discriminator output shape: {disc_out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model, discriminator

if __name__ == "__main__":
    test_dual_path_hybrid()
