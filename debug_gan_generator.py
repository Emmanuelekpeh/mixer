#!/usr/bin/env python3
"""
Debug GAN Generator dimensions
"""

import torch
import torch.nn as nn

def debug_gan_generator():
    """Debug the GAN generator step by step"""
    
    print("🔍 Debugging GAN Generator Dimensions")
    print("=" * 50)
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 64, 260)
    print(f"Input shape: {input_tensor.shape}")
    
    # Channel configuration
    channels = [32, 64, 128]
    
    # Test encoder manually
    print("\n📊 Encoder outputs:")
    x = input_tensor
    skip_connections = []
    
    in_ch = 1
    for i, ch in enumerate(channels):
        encoder = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        x = encoder(x)
        skip_connections.append(x)
        print(f"  Encoder {i+1}: {x.shape} (channels: {in_ch} -> {ch})")
        in_ch = ch
    
    # Test bottleneck
    print(f"\n🔄 Bottleneck:")
    print(f"  Before bottleneck: {x.shape}")
    
    bottleneck = nn.Sequential(
        nn.Conv2d(channels[-1], channels[-1] * 2, 4, 2, 1),
        nn.InstanceNorm2d(channels[-1] * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ConvTranspose2d(channels[-1] * 2, channels[-1], 4, 2, 1),
        nn.InstanceNorm2d(channels[-1]),
        nn.ReLU(inplace=True)
    )
    x = bottleneck(x)
    print(f"  After bottleneck: {x.shape}")
    
    # Test decoder manually
    print(f"\n📈 Decoder outputs:")
    decoder_channels = list(reversed(channels))
    print(f"  Decoder channels: {decoder_channels}")
    print(f"  Skip connection shapes: {[skip.shape for skip in skip_connections]}")
    
    for i in range(len(decoder_channels) - 1):
        in_channels = decoder_channels[i] * 2  # Current + skip
        out_channels = decoder_channels[i + 1]
        skip_idx = len(skip_connections) - 2 - i  # Skip from encoder
        skip = skip_connections[skip_idx]
        
        print(f"\n  Decoder {i+1}:")
        print(f"    Current x: {x.shape}")
        print(f"    Skip {skip_idx}: {skip.shape}")
        print(f"    Trying to concat: {x.shape} + {skip.shape}")
        
        # Check if concatenation will work
        if x.shape[2:] != skip.shape[2:]:
            print(f"    ❌ Spatial dimension mismatch!")
            print(f"       x spatial: {x.shape[2:]}, skip spatial: {skip.shape[2:]}")
            return False
        
        try:
            concat = torch.cat([x, skip], dim=1)
            print(f"    Concatenated: {concat.shape}")
            print(f"    Expected input channels: {in_channels}, actual: {concat.shape[1]}")
            
            if concat.shape[1] != in_channels:
                print(f"    ❌ Channel mismatch!")
                return False
            
            decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            x = decoder(concat)
            print(f"    Output: {x.shape}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    # Test final layer
    print(f"\n🎯 Final layer:")
    skip = skip_connections[0]
    print(f"  Current x: {x.shape}")
    print(f"  Skip 0: {skip.shape}")
    
    try:
        concat = torch.cat([x, skip], dim=1)
        print(f"  Final concatenated: {concat.shape}")
        
        final = nn.Sequential(
            nn.ConvTranspose2d(concat.shape[1], 1, 4, 2, 1),
            nn.Tanh()
        )
        output = final(concat)
        print(f"  Final output: {output.shape}")
        
        print("\n✅ GAN Generator architecture is valid!")
        return True
        
    except Exception as e:
        print(f"  ❌ Final layer error: {e}")
        return False

if __name__ == "__main__":
    debug_gan_generator()
