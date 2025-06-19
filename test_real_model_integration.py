#!/usr/bin/env python3
"""
Test Real Model Loading for Tournament System
"""

import sys
from pathlib import Path

# Add the backend path
backend_path = Path(__file__).parent / "tournament_webapp" / "backend"
sys.path.insert(0, str(backend_path))

def test_real_model_loading():
    """Test if real models are being loaded correctly"""
    print("🧪 Testing Real Model Loading...")
    print("=" * 50)
    
    try:
        from simplified_tournament_engine import AVAILABLE_MODELS, load_real_models
        
        print(f"📊 Found {len(AVAILABLE_MODELS)} models:")
        
        for i, model in enumerate(AVAILABLE_MODELS, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   ID: {model['id']}")
            print(f"   Architecture: {model['architecture']}")
            print(f"   ELO Rating: {model['elo_rating']}")
            print(f"   Tier: {model['tier']}")
            print(f"   Specializations: {', '.join(model.get('specializations', []))}")
            
            # Check if model files exist
            if 'model_path' in model:
                model_file = Path(model['model_path'])
                exists = "✅ EXISTS" if model_file.exists() else "❌ MISSING"
                print(f"   Model File: {exists} ({model_file})")
            
            if 'config_path' in model:
                config_file = Path(model['config_path'])
                exists = "✅ EXISTS" if config_file.exists() else "❌ MISSING"
                print(f"   Config File: {exists} ({config_file})")
        
        print("\n" + "=" * 50)
        if len(AVAILABLE_MODELS) > 5:
            print("🎉 SUCCESS: Real trained models are being loaded!")
            print(f"   Found {len(AVAILABLE_MODELS)} real models vs ~5 expected mock models")
            return True
        else:
            print("⚠️ WARNING: Still using mock models or limited model loading")
            print(f"   Only found {len(AVAILABLE_MODELS)} models")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Failed to load models: {e}")
        return False

def test_tournament_engine_creation():
    """Test creating tournament engine with real models"""
    print("\n🏗️ Testing Tournament Engine Creation...")
    print("=" * 50)
    
    try:
        from simplified_tournament_engine import EnhancedTournamentEngine
        
        engine = EnhancedTournamentEngine()
        
        print(f"✅ Tournament engine created successfully")
        print(f"📊 Engine has {len(engine.models)} models available")
        print(f"🔗 AI Mixer loaded: {hasattr(engine, 'ai_mixer') and engine.ai_mixer is not None}")
        print(f"🔗 Database available: {hasattr(engine, 'db_service') and engine.db_service is not None}")
        
        # Test getting model list
        model_list = engine.get_model_list()
        print(f"📋 get_model_list() returns {len(model_list)} models")
        
        if model_list:
            print("\nFirst model details:")
            first_model = model_list[0]
            for key, value in first_model.items():
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to create tournament engine: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Real Model Integration Test")
    print("Testing if tournament system is using real trained models...")
    
    success1 = test_real_model_loading()
    success2 = test_tournament_engine_creation()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 SUCCESS: Tournament system is properly integrated with real models!")
    else:
        print("❌ ISSUES FOUND: Tournament system needs fixes for real model integration")
