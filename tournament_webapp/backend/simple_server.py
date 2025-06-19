#!/usr/bin/env python3
"""
Direct server startup - no reload, just run
"""

import sys
import os
from pathlib import Path

# Fix path issues
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent))

if __name__ == "__main__":
    print("🚀 Starting Tournament Webapp Server (Simple Mode)...")
    
    try:
        # Import uvicorn
        import uvicorn
        print("✅ Uvicorn imported")
        
        # Import the app
        from tournament_api import app
        print("✅ Tournament API imported")
        
        print("")
        print("🌐 Starting server at http://localhost:10000")
        print("📚 API docs at http://localhost:10000/docs")
        print("🛑 Press Ctrl+C to stop")
        print("")
        
        # Start server with minimal config
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=10000,
            reload=False,
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Installing uvicorn...")
        os.system("pip install uvicorn")
        
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        import traceback
        traceback.print_exc()
