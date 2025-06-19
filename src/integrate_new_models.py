#!/usr/bin/env python3
"""
🏆 Tournament Database Integration for New Models
===============================================

Add the 5 new AI model architectures to the tournament database:
- LSTM Audio Mixer
- Audio GAN Mixer
- VAE Audio Mixer
- Advanced Transformer Mixer
- ResNet Audio Mixer

This script integrates with the tournament system database.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add backend to path for database access
backend_path = Path(__file__).parent.parent / "tournament_webapp" / "backend"
sys.path.append(str(backend_path))

try:
    from database_service import DatabaseService
    from database import AIModel
    print("✅ Database modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import database modules: {e}")
    sys.exit(1)

def load_model_metadata(models_dir: Path) -> dict:
    """Load metadata for trained models."""
    metadata = {}
    
    # Expected model files
    model_files = [
        "lstm_audio_mixer_best.json",
        "audio_gan_mixer_best.json", 
        "vae_audio_mixer_best.json",
        "advanced_transformer_mixer_best.json",
        "resnet_audio_mixer_best.json"
    ]
    
    for model_file in model_files:
        metadata_path = models_dir / model_file
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    model_id = model_file.replace('_best.json', '')
                    metadata[model_id] = data
                    print(f"✅ Loaded metadata for {data['name']}")
            except Exception as e:
                print(f"⚠️ Failed to load {model_file}: {e}")
        else:
            print(f"⚠️ Metadata file not found: {metadata_path}")
    
    return metadata

def add_models_to_database(metadata: dict):
    """Add models to the tournament database."""
    
    print(f"\n🏗️ Adding {len(metadata)} models to tournament database...")
    
    try:
        with DatabaseService() as db:
            models_added = 0
            
            for model_id, data in metadata.items():
                try:
                    # Check if model already exists
                    existing_model = db.session.query(AIModel).filter_by(id=model_id).first()
                    
                    if existing_model:
                        print(f"⚠️ Model {model_id} already exists, updating...")
                        # Update existing model
                        existing_model.name = data['name']
                        existing_model.architecture = data['architecture']
                        existing_model.description = data['description']
                        existing_model.generation = data['generation']
                        existing_model.tier = data['tier']
                        existing_model.elo_rating = data['elo_rating']
                        existing_model.specializations = data['specializations']
                        existing_model.preferred_genres = data['preferred_genres']
                        existing_model.signature_techniques = data['signature_techniques']
                        existing_model.capabilities = data['capabilities']
                        existing_model.updated_at = datetime.now()
                        
                        # Update model file path
                        model_file_path = f"models/{model_id}_best.pth"
                        existing_model.model_file_path = model_file_path
                        
                    else:
                        # Create new model
                        new_model = AIModel(
                            id=model_id,
                            name=data['name'],
                            nickname=data['name'].split()[0],  # Use first word as nickname
                            architecture=data['architecture'],
                            description=data['description'],
                            generation=data['generation'],
                            tier=data['tier'],
                            elo_rating=data['elo_rating'],
                            specializations=data['specializations'],
                            preferred_genres=data['preferred_genres'],
                            signature_techniques=data['signature_techniques'],
                            capabilities=data['capabilities'],
                            model_file_path=f"models/{model_id}_best.pth",
                            is_active=True,
                            created_at=datetime.now()
                        )
                        
                        db.session.add(new_model)
                        print(f"✅ Added new model: {data['name']}")
                    
                    models_added += 1
                    
                except Exception as e:
                    print(f"❌ Failed to add model {model_id}: {e}")
                    continue
            
            # Commit all changes
            db.session.commit()
            print(f"\n🎉 Successfully processed {models_added} models!")
            
            # Verify the models were added
            print(f"\n📊 Current models in database:")
            all_models = db.session.query(AIModel).all()
            for model in all_models:
                print(f"   • {model.name} ({model.architecture}) - ELO: {model.elo_rating} - Tier: {model.tier}")
                
    except Exception as e:
        print(f"❌ Database operation failed: {e}")

def create_tournament_model_configs():
    """Create configuration files for tournament integration."""
    
    config_dir = Path(__file__).parent.parent / "tournament_webapp" / "tournament_models"
    config_dir.mkdir(exist_ok=True)
    
    # Update model genealogy
    genealogy_file = config_dir / "model_genealogy.json"
    
    genealogy_data = {
        "generations": {
            "1": {
                "base_models": [
                    "lstm_audio_mixer",
                    "audio_gan_mixer", 
                    "vae_audio_mixer",
                    "advanced_transformer_mixer",
                    "resnet_audio_mixer"
                ],
                "created_at": datetime.now().isoformat(),
                "description": "First generation of new architecture models"
            }
        },
        "architecture_families": {
            "LSTM": ["lstm_audio_mixer"],
            "GAN": ["audio_gan_mixer"],
            "VAE": ["vae_audio_mixer"], 
            "Transformer": ["advanced_transformer_mixer"],
            "ResNet": ["resnet_audio_mixer"]
        },
        "evolution_history": {},
        "last_updated": datetime.now().isoformat()
    }
    
    with open(genealogy_file, 'w') as f:
        json.dump(genealogy_data, f, indent=2)
    
    print(f"✅ Updated model genealogy: {genealogy_file}")
    
    # Create model compatibility matrix
    compatibility_file = config_dir / "architecture_compatibility.json"
    
    compatibility_data = {
        "cross_architecture_evolution": {
            "LSTM": ["GAN", "VAE", "Transformer"],
            "GAN": ["LSTM", "VAE", "ResNet"],
            "VAE": ["LSTM", "GAN", "Transformer"],
            "Transformer": ["LSTM", "VAE", "ResNet"],
            "ResNet": ["GAN", "Transformer", "LSTM"]
        },
        "evolution_strategies": {
            "LSTM_GAN": "temporal_creative_fusion",
            "VAE_Transformer": "latent_attention_blend",
            "ResNet_GAN": "robust_creative_enhancement",
            "Transformer_LSTM": "attention_temporal_hybrid"
        },
        "last_updated": datetime.now().isoformat()
    }
    
    with open(compatibility_file, 'w') as f:
        json.dump(compatibility_data, f, indent=2)
    
    print(f"✅ Created compatibility matrix: {compatibility_file}")

def main():
    """Main integration pipeline."""
    print("🏆 Tournament Database Integration for New Models")
    print("=" * 60)
    
    # Load model metadata
    models_dir = Path(__file__).parent.parent / "models"
    metadata = load_model_metadata(models_dir)
    
    if not metadata:
        print("❌ No model metadata found! Please train models first.")
        print("   Run: python src/train_new_architectures_fixed.py")
        return
    
    print(f"\n📊 Found metadata for {len(metadata)} models:")
    for model_id, data in metadata.items():
        print(f"   • {data['name']} ({data['architecture']}) - MAE: {data['performance_metrics']['mae']:.6f}")
    
    # Add models to database
    add_models_to_database(metadata)
    
    # Create tournament configuration files
    create_tournament_model_configs()
    
    print(f"\n🎯 Integration Complete!")
    print(f"   ✅ Models added to tournament database")
    print(f"   ✅ Configuration files created")
    print(f"   ✅ Ready for tournament battles!")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Start tournament webapp: cd tournament_webapp && python backend/main.py")
    print(f"   2. Upload audio and begin battles")
    print(f"   3. Watch models evolve through competition!")

if __name__ == "__main__":
    main()
