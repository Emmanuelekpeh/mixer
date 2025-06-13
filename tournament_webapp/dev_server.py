#!/usr/bin/env python3
"""
🚀 AI Tournament Development Server
==================================

Development server for the AI Mixer Tournament system.
Starts both backend API and frontend React app in development mode.

Usage:
    python dev_server.py [--backend-only] [--frontend-only] [--production]
"""

import subprocess
import sys
import time
import signal
import threading
from pathlib import Path
import argparse
import os

class TournamentDevServer:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print("\n🛑 Shutting down development server...")
        self.stop_servers()
        sys.exit(0)
    
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🔧 Starting backend API server...")
        
        backend_dir = Path(__file__).parent / "tournament_webapp" / "backend"
        
        try:
            # Change to backend directory
            os.chdir(backend_dir)
            
            # Start FastAPI with uvicorn
            self.backend_process = subprocess.Popen([
                sys.executable, "tournament_api.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Monitor backend output
            def monitor_backend():
                for line in iter(self.backend_process.stdout.readline, ''):
                    if self.running:
                        print(f"[BACKEND] {line.strip()}")
                    else:
                        break
            
            backend_thread = threading.Thread(target=monitor_backend, daemon=True)
            backend_thread.start()
            
            print("✅ Backend server starting on http://localhost:8000")
            print("📚 API docs available at http://localhost:8000/docs")
            
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
        
        return True
    
    def start_frontend(self):
        """Start the React frontend development server"""
        print("⚛️  Starting React frontend...")
        
        frontend_dir = Path(__file__).parent / "tournament_webapp" / "frontend"
        
        try:
            # Install dependencies if node_modules doesn't exist
            if not (frontend_dir / "node_modules").exists():
                print("📦 Installing frontend dependencies...")
                npm_install = subprocess.run(
                    ["npm", "install"], 
                    cwd=frontend_dir, 
                    capture_output=True, 
                    text=True
                )
                
                if npm_install.returncode != 0:
                    print(f"❌ npm install failed: {npm_install.stderr}")
                    return False
                
                print("✅ Frontend dependencies installed")
            
            # Start React development server
            self.frontend_process = subprocess.Popen([
                "npm", "start"
            ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Monitor frontend output
            def monitor_frontend():
                for line in iter(self.frontend_process.stdout.readline, ''):
                    if self.running:
                        # Filter out some noisy React dev server output
                        if "webpack compiled" not in line.lower() and "hot update" not in line.lower():
                            print(f"[FRONTEND] {line.strip()}")
                    else:
                        break
            
            frontend_thread = threading.Thread(target=monitor_frontend, daemon=True)
            frontend_thread.start()
            
            print("✅ Frontend server starting on http://localhost:3000")
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
        
        return True
    
    def wait_for_backend(self, timeout=30):
        """Wait for backend to be ready"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8000/api/health", timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)
        
        return False
    
    def stop_servers(self):
        """Stop both servers gracefully"""
        self.running = False
        
        if self.backend_process:
            print("🔧 Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            print("⚛️  Stopping frontend server...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
        
        print("✅ Servers stopped")
    
    def run_development_server(self, backend_only=False, frontend_only=False):
        """Run the development server"""
        print("🏆 AI Mixer Tournament - Development Server")
        print("=" * 50)
        
        # Check dependencies
        self.check_dependencies()
        
        success = True
        
        if not frontend_only:
            success &= self.start_backend()
            
            if success:
                print("⏳ Waiting for backend to be ready...")
                if self.wait_for_backend():
                    print("✅ Backend is ready!")
                else:
                    print("⚠️  Backend may not be fully ready yet")
        
        if not backend_only and success:
            time.sleep(2)  # Give backend a moment to start
            success &= self.start_frontend()
        
        if success:
            print("\n🎉 Development server started successfully!")
            print("-" * 50)
            
            if not frontend_only:
                print("🔧 Backend API: http://localhost:8000")
                print("📚 API Docs: http://localhost:8000/docs")
            
            if not backend_only:
                print("⚛️  Frontend: http://localhost:3000")
            
            print("-" * 50)
            print("Press Ctrl+C to stop the server")
            
            # Keep the main thread alive
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print("❌ Failed to start development server")
            self.stop_servers()
            sys.exit(1)
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        # Check Python dependencies
        required_packages = [
            "fastapi", "uvicorn", "torch", "numpy", "pathlib"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing Python packages: {', '.join(missing_packages)}")
            print("Install with: pip install fastapi uvicorn torch numpy")
            sys.exit(1)
        
        # Check Node.js
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Node.js {result.stdout.strip()}")
            else:
                print("❌ Node.js not found")
                sys.exit(1)
        except FileNotFoundError:
            print("❌ Node.js not found. Please install Node.js to run the frontend.")
            sys.exit(1)
        
        # Check npm
        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ npm {result.stdout.strip()}")
            else:
                print("❌ npm not found")
                sys.exit(1)
        except FileNotFoundError:
            print("❌ npm not found")
            sys.exit(1)
        
        print("✅ All dependencies available")
      def run_production_server(self):
        """Run production server"""
        print("🚀 Starting production server...")
        
        # Build frontend
        frontend_dir = Path(__file__).parent / "frontend"
        print("📦 Building frontend for production...")
        
        build_result = subprocess.run(
            ["npm", "run", "build"], 
            cwd=frontend_dir,
            capture_output=True,
            text=True
        )
        
        if build_result.returncode != 0:
            print(f"❌ Frontend build failed: {build_result.stderr}")
            sys.exit(1)
        
        print("✅ Frontend built successfully")
        
        # Start production backend
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        # Start with gunicorn for production
        try:
            subprocess.run([
                "uvicorn", "tournament_api:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--workers", "4",
                "--access-log"
            ])
        except KeyboardInterrupt:
            print("\n🛑 Production server stopped")


def main():
    parser = argparse.ArgumentParser(description="AI Tournament Development Server")
    parser.add_argument("--backend-only", action="store_true", help="Start only the backend server")
    parser.add_argument("--frontend-only", action="store_true", help="Start only the frontend server")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    
    args = parser.parse_args()
    
    server = TournamentDevServer()
    
    if args.production:
        server.run_production_server()
    else:
        server.run_development_server(
            backend_only=args.backend_only,
            frontend_only=args.frontend_only
        )


if __name__ == "__main__":
    main()
