#!/usr/bin/env python3
"""
Backend Integration Test
Verifies that all our Phase 1 fixes are working correctly
"""

import sys
import os
from pathlib import Path

# Add the project directory to the path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir / "tournament_webapp" / "backend"))

def test_backend_imports():
    """Test that all backend modules can be imported"""
    print("🧪 Testing Backend Module Imports...")
    
    try:
        print("  📦 Importing tournament_api...")
        import tournament_webapp.backend.tournament_api as api
        print("  ✅ tournament_api imported successfully")
        
        print("  📦 Importing tournament_model_manager...")
        from tournament_webapp.backend.tournament_model_manager import TournamentModelManager
        print("  ✅ TournamentModelManager imported successfully")
        
        print("  📦 Importing simplified_tournament_engine...")
        from tournament_webapp.backend.simplified_tournament_engine import EnhancedTournamentEngine
        print("  ✅ EnhancedTournamentEngine imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_model_manager():
    """Test the TournamentModelManager functionality"""
    print("\n🧪 Testing TournamentModelManager...")
    
    try:
        from tournament_webapp.backend.tournament_model_manager import TournamentModelManager
        from pathlib import Path
        
        # Test initialization
        manager = TournamentModelManager(models_dir=Path("../models"))
        print("  ✅ TournamentModelManager initialized")
        
        # Test model listing - use correct method name
        models = manager.available_models  # This is a property
        print(f"  ✅ Found {len(models)} available models")
        
        # Test the new update_model_metrics method
        if len(models) >= 2:
            manager.update_model_metrics(models[0]["id"], models[1]["id"], 0.8)
            print("  ✅ update_model_metrics method works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ TournamentModelManager test failed: {e}")
        return False

def test_tournament_engine():
    """Test the EnhancedTournamentEngine functionality"""
    print("\n🧪 Testing EnhancedTournamentEngine...")
    
    try:
        from tournament_webapp.backend.simplified_tournament_engine import EnhancedTournamentEngine
        
        # Test initialization
        engine = EnhancedTournamentEngine()
        print("  ✅ EnhancedTournamentEngine initialized")
        
        # Test tournament creation
        tournament = engine.create_tournament(
            tournament_id="test_tournament",
            user_id="test_user",
            max_rounds=3
        )
        print("  ✅ Tournament creation works")
        print(f"  📊 Tournament has {len(tournament['pairs'])} pairs")
        
        return True
        
    except Exception as e:
        print(f"  ❌ EnhancedTournamentEngine test failed: {e}")
        return False

def main():
    """Run all backend tests"""
    print("🚀 Backend Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_backend_imports,
        test_model_manager,
        test_tournament_engine,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Backend is ready for Phase 2!")
        print("✅ Phase 1 completion confirmed")
    else:
        print("❌ Some tests failed. Backend needs additional fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
