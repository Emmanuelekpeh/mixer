#!/usr/bin/env python3
"""
Add new architecture models to tournament database
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from .database import get_db, AIModel

def add_models_to_tournament():
    """Add new models to tournament database."""
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    # Model files to add
    model_files = [
        "lstm_audio_mixer.json",
        "audio_gan_mixer.json", 
        "vae_audio_mixer.json",
        "advanced_transformer_mixer.json",
        "resnet_audio_mixer.json"
    ]
    
    print("Adding models to tournament database...")
    
    models_added = []
    with get_db() as db:
        for model_file in model_files:
            metadata_path = models_dir / model_file
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                new_model = AIModel(
                    id=data['id'],
                    name=data['name'],
                    architecture=data['architecture'],
                    description=data.get('description', ''),
                    elo_rating=data.get('elo_rating', 1200.0),
                    tier=data.get('tier', 'Amateur'),
                    generation=data.get('generation', 1),
                    parent_ids=data.get('parent_ids', []),
                    specializations=data.get('specializations', []),
                    capabilities=data.get('capabilities', {}),
                    created_at=datetime.now()
                )
                db.add(new_model)
                db.commit()
                models_added.append(data)
                print(f"   Added: {data['name']} - {data['architecture']} - ELO: {data['elo_rating']}")
            else:
                print(f"   Missing: {model_file}")
    
    print(f"\nTournament Integration Complete!")
    print(f"   {len(models_added)} models added to database!")
    return models_added

if __name__ == "__main__":
    add_models_to_tournament()
