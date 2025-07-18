#!/usr/bin/env python3
"""
🏆 Tournament Integration for Dual-Path Hybrid
==============================================

Integrates the trained dual-path hybrid model into the existing tournament system.
Creates fresh genealogy branch and enables iterative learning feedback.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from pathlib import Path
import torch

# Tournament integration functions
def generate_fresh_model_id():
    """Generate a new UUID for fresh genealogy branch"""
    return str(uuid.uuid4())

def update_model_genealogy(model_id, model_info):
    """Add new model to tournament genealogy with fresh branch"""
    
    genealogy_path = os.path.join(
        os.getcwd(), 
        "tournament_webapp", 
        "tournament_models", 
        "model_genealogy.json"
    )
    
    # Load existing genealogy
    if os.path.exists(genealogy_path):
        with open(genealogy_path, 'r') as f:
            genealogy = json.load(f)
    else:
        genealogy = {"evolution_tree": {}, "architecture_stats": {}}
    
    # Add new model as root of fresh branch
    genealogy["evolution_tree"][model_id] = {
        "parents": [],  # Fresh start - no parents
        "evolution_method": "initial_training",
        "generation": 0,  # Generation 0 for fresh start
        "context": {
            "architecture": "DualPathHybrid",
            "training_type": "restoration_dataset",
            "real_learning": True,  # Mark as real learning vs simulated
            "total_parameters": model_info.get('total_parameters', 0),
            "ast_layers": model_info.get('ast_layers', 4),
            "gan_channels": model_info.get('gan_channels', [64, 128, 256])
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Update architecture stats
    if "DualPathHybrid" not in genealogy["architecture_stats"]:
        genealogy["architecture_stats"]["DualPathHybrid"] = {
            "count": 0,
            "first_introduced": datetime.now().isoformat(),
            "latest_version": "1.0"
        }
    
    genealogy["architecture_stats"]["DualPathHybrid"]["count"] += 1
    
    # Save updated genealogy
    with open(genealogy_path, 'w') as f:
        json.dump(genealogy, f, indent=2)
    
    print(f"🧬 Added {model_id} to tournament genealogy")
    print(f"   Generation: 0 (fresh branch)")
    print(f"   Architecture: DualPathHybrid")
    print(f"   Parameters: {model_info.get('total_parameters', 0):,}")

def create_iterative_training_config(model_id):
    """Create configuration for iterative training and tournament feedback"""
    
    config = {
        "model_id": model_id,
        "training_mode": "iterative",
        "feedback_integration": {
            "enabled": True,
            "tournament_metrics": [
                "vote_confidence",
                "win_rate", 
                "performance_score"
            ],
            "learning_signals": [
                "success_patterns",
                "failure_modes",
                "user_preferences"
            ]
        },
        "off_premise_training": {
            "enabled": True,
            "export_format": "pytorch_state_dict",
            "privacy_mode": "differential_privacy",
            "reintegration_protocol": "weight_averaging"
        },
        "continuous_improvement": {
            "retrain_threshold": 0.1,  # Retrain if performance drops by 10%
            "update_frequency": "weekly",
            "backup_generations": 3
        },
        "created": datetime.now().isoformat()
    }
    
    config_path = os.path.join(
        os.getcwd(),
        "dual_path_results",
        f"iterative_config_{model_id}.json"
    )
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️ Iterative training config saved: {config_path}")
    return config

def validate_tournament_compatibility(model_path):
    """Validate that the model is compatible with tournament system"""
    
    try:
        # Load model
        model_state = torch.load(model_path, map_location='cpu')
        
        # Check if it's a state dict or full checkpoint
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            state_dict = model_state['model_state_dict']
            model_info = model_state.get('model_info', {})
        else:
            state_dict = model_state
            model_info = {}
        
        # Basic validation checks
        checks = {
            "state_dict_valid": isinstance(state_dict, dict) and len(state_dict) > 0,
            "reasonable_size": len(str(state_dict)) < 500_000_000,  # ~500MB max
            "has_parameters": any('weight' in key for key in state_dict.keys()),
            "architecture_info": bool(model_info)
        }
        
        all_passed = all(checks.values())
        
        print(f"🔍 Tournament Compatibility Check:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
        
        if all_passed:
            print(f"🏆 Model is tournament-ready!")
        else:
            print(f"⚠️ Model needs fixes before tournament integration")
        
        return all_passed, checks, model_info
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False, {}, {}

def integrate_dual_path_model(model_path=None, model_id=None):
    """Main integration function"""
    
    print("🏆 Dual-Path Hybrid Tournament Integration")
    print("=" * 50)
    
    # Find the best model if not specified
    if model_path is None:
        models_dir = os.path.join(os.getcwd(), "models")
        best_model_path = os.path.join(models_dir, "dual_path_hybrid_best.pth")
        
        if os.path.exists(best_model_path):
            model_path = best_model_path
            print(f"📁 Using best model: {best_model_path}")
        else:
            print("❌ No trained dual-path model found!")
            print("   Run train_dual_path_hybrid.py first")
            return None
    
    # Validate compatibility
    is_compatible, checks, model_info = validate_tournament_compatibility(model_path)
    
    if not is_compatible:
        print("❌ Model not compatible with tournament system")
        return None
    
    # Generate model ID if not provided
    if model_id is None:
        model_id = generate_fresh_model_id()
    
    # Copy model to tournament directory
    tournament_models_dir = os.path.join(
        os.getcwd(), 
        "tournament_webapp", 
        "tournament_models", 
        "evolved"
    )
    os.makedirs(tournament_models_dir, exist_ok=True)
    
    tournament_model_path = os.path.join(tournament_models_dir, f"{model_id}.pth")
    
    # Copy just the state dict to tournament location
    original_state = torch.load(model_path, map_location='cpu')
    if isinstance(original_state, dict) and 'model_state_dict' in original_state:
        state_dict = original_state['model_state_dict']
    else:
        state_dict = original_state
    
    torch.save(state_dict, tournament_model_path)
    print(f"💾 Model copied to tournament: {tournament_model_path}")
    
    # Update genealogy
    update_model_genealogy(model_id, model_info)
    
    # Create iterative training config
    config = create_iterative_training_config(model_id)
    
    # Create integration summary
    integration_summary = {
        "model_id": model_id,
        "original_path": model_path,
        "tournament_path": tournament_model_path,
        "integration_timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "compatibility_checks": checks,
        "iterative_config": config,
        "status": "integrated",
        "next_steps": [
            "Run tournament to test model performance",
            "Monitor vote confidence and win rate", 
            "Collect performance data for iterative training",
            "Enable off-premise training pipeline"
        ]
    }
    
    summary_path = os.path.join(
        os.getcwd(),
        "dual_path_results", 
        f"integration_summary_{model_id}.json"
    )
    
    with open(summary_path, 'w') as f:
        json.dump(integration_summary, f, indent=2)
    
    print(f"\n🎉 Integration Complete!")
    print(f"🆔 Model ID: {model_id}")
    print(f"📊 Parameters: {model_info.get('total_parameters', 0):,}")
    print(f"🧬 Genealogy: Fresh branch (Generation 0)")
    print(f"⚙️ Iterative learning: Enabled")
    print(f"📄 Summary: {summary_path}")
    
    return {
        "model_id": model_id,
        "tournament_path": tournament_model_path,
        "summary": integration_summary
    }

def check_tournament_readiness():
    """Check if tournament system can accept new models"""
    
    tournament_dir = os.path.join(
        os.getcwd(),
        "tournament_webapp",
        "tournament_models",
        "evolved"
    )
    
    if not os.path.exists(tournament_dir):
        print("❌ Tournament models directory not found")
        return False
    
    # Count existing models
    existing_models = [f for f in os.listdir(tournament_dir) if f.endswith('.pth')]
    
    print(f"🏟️ Tournament Status:")
    print(f"   Models directory: {tournament_dir}")
    print(f"   Existing models: {len(existing_models)}")
    
    if len(existing_models) > 0:
        print(f"   Latest models: {existing_models[-3:]}")
    
    return True

def main():
    """Main integration function"""
    
    # Check tournament readiness
    if not check_tournament_readiness():
        return
    
    # Integrate the dual-path model
    result = integrate_dual_path_model()
    
    if result:
        print(f"\n🚀 Ready for tournament competition!")
        print(f"   Start tournament webapp to test the new model")
        print(f"   Model ID: {result['model_id']}")
    else:
        print(f"\n❌ Integration failed")

if __name__ == "__main__":
    main()
