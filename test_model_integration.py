#!/usr/bin/env python3
"""
Test script to verify model integration between database and tournament system
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "tournament_webapp" / "backend"
sys.path.insert(0, str(backend_dir))

def test_database_models():
    """Test database model loading"""
    print("🗄️  Testing database models...")
    
    try:
        from database_service import DatabaseService
        from database import init_database, get_database_stats
        
        # Initialize database
        init_database()
        stats = get_database_stats()
        print(f"📊 Database stats: {stats}")
        
        # Test model retrieval
        db_service = DatabaseService()
        models = db_service.get_all_models()
        
        print(f"✅ Found {len(models)} models in database:")
        for model in models:
            print(f"  - {model.name} ({model.architecture}) ELO: {model.elo_rating}")
        
        db_service.close()
        return len(models)
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return 0

def test_tournament_engine():
    """Test tournament engine model integration"""
    print("\n🎯 Testing tournament engine...")
    
    try:
        from simplified_tournament_engine import EnhancedTournamentEngine
        
        engine = EnhancedTournamentEngine()
        models = engine.get_model_list()
        
        print(f"✅ Tournament engine has {len(models)} models:")
        for model in models[:5]:  # Show first 5
            print(f"  - {model['name']} ({model['architecture']}) ELO: {model['elo_rating']}")
        
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more models")
        
        return len(models)
        
    except Exception as e:
        print(f"❌ Tournament engine test failed: {e}")
        return 0

def test_api_models():
    """Test API model endpoint"""
    print("\n🌐 Testing API models endpoint...")
    
    try:
        from database_service import DatabaseService
        
        db_service = DatabaseService()
        models = db_service.get_all_models_cached()
        
        # Simulate API response format
        models_list = []
        for model in models:
            model_dict = {
                "id": model.id,
                "name": model.name,
                "nickname": model.nickname or model.name,
                "architecture": model.architecture,
                "generation": model.generation,
                "tier": model.tier,
                "elo_rating": round(model.elo_rating, 1),
                "total_battles": model.total_battles,
                "wins": model.wins,
                "losses": model.losses,
                "win_rate": round(model.win_rate * 100, 1),
                "specializations": model.specializations or [],
                "capabilities": model.capabilities or {},
                "is_active": model.is_active
            }
            models_list.append(model_dict)
        
        print(f"✅ API would return {len(models_list)} models:")
        for model in models_list[:3]:  # Show first 3
            print(f"  - {model['name']} (ELO: {model['elo_rating']}, Win Rate: {model['win_rate']}%)")
        
        db_service.close()
        return len(models_list)
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return 0

def test_file_system_models():
    """Test file system model discovery"""
    print("\n📁 Testing file system models...")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠️  Models directory not found")
        return 0
    
    pth_files = list(models_dir.glob("**/*.pth"))
    json_files = list(models_dir.glob("**/*.json"))
    
    print(f"📄 Found {len(pth_files)} .pth files:")
    for pth_file in pth_files[:5]:
        print(f"  - {pth_file.name}")
    
    print(f"📄 Found {len(json_files)} .json files:")
    for json_file in json_files[:5]:
        print(f"  - {json_file.name}")
    
    return len(pth_files)

def main():
    """Run all tests"""
    print("🚀 Model Integration Test Suite")
    print("=" * 50)
    
    # Test database models
    db_count = test_database_models()
    
    # Test tournament engine
    engine_count = test_tournament_engine()
    
    # Test API endpoint
    api_count = test_api_models()
    
    # Test file system
    fs_count = test_file_system_models()
    
    print("\n📊 Summary:")
    print(f"  Database models: {db_count}")
    print(f"  Tournament engine models: {engine_count}")
    print(f"  API models: {api_count}")
    print(f"  File system models: {fs_count}")
    
    if db_count == engine_count == api_count and db_count > 0:
        print("✅ All systems are using the same models - INTEGRATION SUCCESS!")
        return True
    else:
        print("⚠️  Model counts don't match - there may be integration issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)