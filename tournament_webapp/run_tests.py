#!/usr/bin/env python3
"""
🎭 Playwright Test Runner for Tournament Webapp
===============================================

Automatically starts servers and runs Playwright tests
"""

import subprocess
import time
import sys
import os
import signal
import threading
import requests
from pathlib import Path

class TestRunner:
    def __init__(self):
        self.frontend_process = None
        self.backend_process = None
        self.base_dir = Path(__file__).parent
        
    def check_port(self, port, host='localhost'):
        """Check if a port is available"""
        try:
            response = requests.get(f'http://{host}:{port}', timeout=5)
            return True
        except:
            return False
    
    def start_frontend(self):
        """Start the React frontend server"""
        print("🌐 Starting frontend server...")
        frontend_dir = self.base_dir / 'frontend'
        
        if not frontend_dir.exists():
            print("❌ Frontend directory not found!")
            return False
        
        # Change to frontend directory and start
        self.frontend_process = subprocess.Popen(
            ['npm', 'start'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        # Wait for frontend to start
        print("⏳ Waiting for frontend to start on port 3000...")
        for i in range(30):  # Wait up to 30 seconds
            if self.check_port(3000):
                print("✅ Frontend started successfully!")
                return True
            time.sleep(1)
        
        print("❌ Frontend failed to start within 30 seconds")
        return False
    
    def start_backend(self):
        """Start the Python backend server"""
        print("🔧 Starting backend server...")
        backend_dir = self.base_dir / 'backend'
        
        if not backend_dir.exists():
            print("❌ Backend directory not found!")
            return False
        
        # Start backend
        self.backend_process = subprocess.Popen(
            ['python', 'simple_server.py'],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start on port 10000...")
        for i in range(20):  # Wait up to 20 seconds
            if self.check_port(10000):
                print("✅ Backend started successfully!")
                return True
            time.sleep(1)
        
        print("❌ Backend failed to start within 20 seconds")
        return False
    
    def run_tests(self, test_args=None):
        """Run Playwright tests"""
        print("🎭 Running Playwright tests...")
        
        cmd = ['npx', 'playwright', 'test']
        if test_args:
            cmd.extend(test_args)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=False,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def cleanup(self):
        """Stop all servers"""
        print("\n🧹 Cleaning up servers...")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except:
                self.frontend_process.kill()
            print("✅ Frontend server stopped")
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except:
                self.backend_process.kill()
            print("✅ Backend server stopped")
    
    def run_full_test_suite(self, test_args=None):
        """Run the complete test suite"""
        print("🚀 Starting Tournament Webapp Test Suite")
        print("=" * 50)
        
        try:
            # Check if servers are already running
            frontend_running = self.check_port(3000)
            backend_running = self.check_port(10000)
            
            if frontend_running:
                print("ℹ️ Frontend already running on port 3000")
            else:
                if not self.start_frontend():
                    return False
            
            if backend_running:
                print("ℹ️ Backend already running on port 10000")
            else:
                if not self.start_backend():
                    return False
            
            # Give servers a moment to fully initialize
            print("⏳ Allowing servers to fully initialize...")
            time.sleep(3)
            
            # Run tests
            success = self.run_tests(test_args)
            
            if success:
                print("\n🎉 All tests passed!")
            else:
                print("\n❌ Some tests failed. Check the output above.")
            
            return success
            
        except KeyboardInterrupt:
            print("\n⚠️ Test run interrupted by user")
            return False
        finally:
            # Only cleanup if we started the servers
            if not frontend_running or not backend_running:
                self.cleanup()

def main():
    """Main entry point"""
    runner = TestRunner()
    
    # Parse command line arguments
    test_args = sys.argv[1:] if len(sys.argv) > 1 else None
    
    # Handle special commands
    if test_args and test_args[0] in ['--help', '-h']:
        print("""
🎭 Tournament Webapp Test Runner

Usage:
  python run_tests.py [playwright_args...]

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --headed           # Run with visible browser
  python run_tests.py --ui               # Run with Playwright UI
  python run_tests.py smoke.spec.js      # Run specific test file
  python run_tests.py --debug            # Debug mode
  
Special commands:
  python run_tests.py --setup            # Setup Playwright only
  python run_tests.py --check-servers    # Check if servers are running
        """)
        return
    
    if test_args and test_args[0] == '--setup':
        print("📦 Setting up Playwright...")
        subprocess.run(['npm', 'install'], cwd=runner.base_dir)
        subprocess.run(['npx', 'playwright', 'install'], cwd=runner.base_dir)
        print("✅ Playwright setup complete!")
        return
    
    if test_args and test_args[0] == '--check-servers':
        frontend = "✅ Running" if runner.check_port(3000) else "❌ Not running"
        backend = "✅ Running" if runner.check_port(10000) else "❌ Not running"
        print(f"Frontend (3000): {frontend}")
        print(f"Backend (10000): {backend}")
        return
    
    # Run the test suite
    success = runner.run_full_test_suite(test_args)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
