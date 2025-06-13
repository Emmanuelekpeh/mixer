#!/usr/bin/env python3
"""
🚀 Render Deployment Helper
=========================

Script to help prepare and deploy the AI Mixer Tournament app to Render.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def prepare_render_deployment():
    """Prepare the application for Render deployment"""
    print("🚀 Preparing for Render Deployment")
    print("=" * 50)
    
    # Check if the required files exist
    render_dir = Path(__file__).resolve().parent / "render"
    render_yaml = render_dir / "render.yaml"
    
    if not render_yaml.exists():
        print("❌ render.yaml not found in render directory!")
        return
    
    # Copy Render files to top level for deployment
    print("📂 Copying Render configuration files...")
    shutil.copy(render_yaml, Path(__file__).resolve().parent / "render.yaml")
    
    # Create an optimized frontend build
    print("🔨 Building frontend...")
    frontend_dir = Path(__file__).resolve().parent / "tournament_webapp" / "frontend"
    
    try:
        subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            check=True
        )
        print("✅ Frontend built successfully")
    except subprocess.CalledProcessError:
        print("❌ Frontend build failed")
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js and npm")
    
    # Check requirements.txt
    requirements = Path(__file__).resolve().parent / "requirements.txt"
    if requirements.exists():
        print("✅ requirements.txt found")
    else:
        print("❌ requirements.txt not found!")
    
    print("\n✨ Deployment preparation complete!")
    print("""
Next steps:
1. Create a new Web Service on Render.com
2. Connect to your GitHub repository
3. Use these settings:
   - Build Command: pip install -r requirements.txt
   - Start Command: cd tournament_webapp && uvicorn backend.tournament_api:app --host=0.0.0.0 --port=$PORT
4. Set up environment variables as specified in RENDER_DEPLOYMENT.md
5. Click "Create Web Service"
    """)

if __name__ == "__main__":
    prepare_render_deployment()
