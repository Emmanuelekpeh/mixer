#!/usr/bin/env python3
"""
🏆 Test Tournament System with New Models
========================================

Test the tournament system with our 5 new model architectures.
"""

import sys
from pathlib import Path
import json
import torch

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "tournament_webapp" / "backend"))
sys.path.append(str(project_root / "src"))

# Import models
from lstm_mixer import LSTMAudioMixer
from audio_gan import AudioGANMixer
from vae_mixer import VAEAudioMixer
from advanced_transformer import AdvancedTransformerMixer
from resnet_mixer import ResNetAudioMixer

def test_model_loading():
    """Test loading models for tournament."""
    
    print("Testing Tournament Model Loading")
    print("=" * 50)
    
    models_dir = project_root / "models"
    
    # Model configurations
    model_configs = [
        ('lstm_audio_mixer', 'LSTM Audio Mixer', LSTMAudioMixer, (2, 128, 250)),
        ('audio_gan_mixer', 'Audio GAN Mixer', AudioGANMixer, (2, 128, 250)),
        ('vae_audio_mixer', 'VAE Audio Mixer', VAEAudioMixer, (2, 128, 250)),
        ('advanced_transformer_mixer', 'Advanced Transformer Mixer', AdvancedTransformerMixer, (2, 128, 1000)),
        ('resnet_audio_mixer', 'ResNet Audio Mixer', ResNetAudioMixer, (2, 128, 250))
    ]
    
    loaded_models = []
    
    for model_id, model_name, model_class, input_shape in model_configs:
        try:
            print(f"\nTesting {model_name}...")
            
            # Load metadata
            metadata_path = models_dir / f"{model_id}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"   Metadata loaded: ELO {metadata['elo_rating']}, Tier {metadata['tier']}")
            
            # Load model
            model_path = models_dir / f"{model_id}.pth"
            if model_path.exists():
                model = model_class()
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                print(f"   Model loaded successfully")
                
                # Test inference
                test_input = torch.randn(*input_shape)
                with torch.no_grad():
                    output = model(test_input)
                    # Handle tuple outputs (like VAE)
                    if isinstance(output, tuple):
                        output = output[0]
                
                print(f"   Inference test: Input {test_input.shape} -> Output {output.shape}")
                
                loaded_models.append({
                    'id': model_id,
                    'name': model_name,
                    'model': model,
                    'metadata': metadata,
                    'input_shape': input_shape
                })
                print(f"   ✅ {model_name} ready for tournament")
            else:
                print(f"   ❌ Model file not found: {model_path}")
                
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
    
    return loaded_models

def simulate_tournament_battle(models):
    """Simulate a tournament battle between models."""
    
    if len(models) < 2:
        print("Need at least 2 models for a battle")
        return
    
    print(f"\n🥊 Simulating Tournament Battle")
    print("=" * 50)
    
    # Select two models for battle
    model_a = models[0]
    model_b = models[1]
    
    print(f"⚔️ Battle: {model_a['name']} vs {model_b['name']}")
    print(f"   {model_a['name']}: ELO {model_a['metadata']['elo_rating']} ({model_a['metadata']['tier']})")
    print(f"   {model_b['name']}: ELO {model_b['metadata']['elo_rating']} ({model_b['metadata']['tier']})")
    
    # Simulate audio processing
    try:
        # Use the input shape for model A (they should be compatible)
        test_audio_input = torch.randn(*model_a['input_shape'])
        
        print(f"\n🎵 Processing audio...")
        
        # Model A processing
        with torch.no_grad():
            output_a = model_a['model'](test_audio_input)
            if isinstance(output_a, tuple):
                output_a = output_a[0]
        
        # Model B processing (adjust input shape if needed)
        if model_b['input_shape'] != model_a['input_shape']:
            test_audio_input_b = torch.randn(*model_b['input_shape'])
        else:
            test_audio_input_b = test_audio_input
            
        with torch.no_grad():
            output_b = model_b['model'](test_audio_input_b)
            if isinstance(output_b, tuple):
                output_b = output_b[0]
        
        print(f"   {model_a['name']} output: {output_a.shape}")
        print(f"   {model_b['name']} output: {output_b.shape}")
        
        # Compare outputs (simple difference as battle result)
        diff = torch.mean(torch.abs(output_a - output_b[:output_a.size(0)])).item()
        
        print(f"\n🏆 Battle Result:")
        print(f"   Parameter difference: {diff:.6f}")
        print(f"   Both models processed successfully!")
        print(f"   Ready for user voting in tournament webapp")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Battle simulation failed: {e}")
        return False

def main():
    """Main test pipeline."""
    print("🏆 Tournament System Test with New Models")
    print("=" * 60)
    
    # Test model loading
    models = test_model_loading()
    
    if not models:
        print("\n❌ No models loaded successfully!")
        return
    
    print(f"\n📊 Summary:")
    print(f"   Successfully loaded: {len(models)}/5 models")
    
    # Test battle simulation
    if len(models) >= 2:
        battle_success = simulate_tournament_battle(models)
        
        if battle_success:
            print(f"\n🎯 Tournament System Status:")
            print(f"   ✅ Model loading: Working")
            print(f"   ✅ Battle simulation: Working") 
            print(f"   ✅ New architectures: Integrated")
            print(f"   🚀 Ready for tournament webapp testing!")
        else:
            print(f"\n⚠️ Battle simulation had issues")
    else:
        print(f"\n⚠️ Need more models for battle testing")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Start tournament webapp")
    print(f"   2. Upload test audio")
    print(f"   3. Begin battles between new architectures")
    print(f"   4. Watch models evolve through competition!")

if __name__ == "__main__":
    main()
