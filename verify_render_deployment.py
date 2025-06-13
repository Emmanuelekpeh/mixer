#!/usr/bin/env python3
"""
🔍 Render Deployment Verification Script
======================================

This script verifies that your deployment on Render.com is working correctly.
It checks the API endpoints and model loading functionality.
"""

import requests
import sys
import json
import time
from pathlib import Path

def verify_render_deployment():
    """Verify that the deployment on Render is working correctly"""
    print("🔍 Verifying Render Deployment")
    print("=" * 50)
    
    # Get deployment URL from user
    print("Enter your Render deployment URL:")
    deployment_url = input("> ").strip()
    
    if not deployment_url:
        print("❌ No URL provided. Exiting.")
        return
    
    # Normalize URL
    if deployment_url.endswith("/"):
        deployment_url = deployment_url[:-1]
    
    if not (deployment_url.startswith("http://") or deployment_url.startswith("https://")):
        deployment_url = "https://" + deployment_url
    
    print(f"\n🌐 Testing connection to {deployment_url}")
    
    # Check basic connectivity
    try:
        response = requests.get(f"{deployment_url}/api/health", timeout=10)
        if response.status_code == 200:
            print("✅ API health endpoint responding")
            health_data = response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Environment: {health_data.get('environment', 'unknown')}")
        else:
            print(f"❌ API health endpoint returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {str(e)}")
        print("\nPossible issues:")
        print("1. The deployment might still be initializing")
        print("2. The URL might be incorrect")
        print("3. There might be a network issue")
        print("\nTry again in a few minutes or check the deployment logs on Render.")
        return
    
    # Check available models
    print("\n🤖 Checking available models")
    try:
        response = requests.get(f"{deployment_url}/api/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            print(f"✅ Found {len(models)} models:")
            for model in models:
                print(f"   - {model.get('name')} ({model.get('type')})")
        else:
            print(f"❌ Models endpoint returned status code: {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Failed to retrieve models information")
    
    # Check tournament functionality
    print("\n🏆 Checking tournament functionality")
    try:
        response = requests.get(f"{deployment_url}/api/tournaments/status", timeout=10)
        if response.status_code == 200:
            tournament_data = response.json()
            print("✅ Tournament system is operational")
            active_tournaments = tournament_data.get("active_tournaments", 0)
            print(f"   Active tournaments: {active_tournaments}")
        else:
            print(f"❌ Tournament status endpoint returned status code: {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Failed to retrieve tournament information")
    
    print("\n📊 Deployment Verification Summary")
    print("=" * 50)
    print("To complete verification:")
    print("1. Check Render logs for any error messages")
    print("2. Try creating a test tournament through the web interface")
    print("3. Verify model loading and inference through the API")
    print("\nIf issues persist, consider:")
    print("- Checking environment variables in Render dashboard")
    print("- Verifying model paths are correct")
    print("- Reviewing application logs for specific errors")

if __name__ == "__main__":
    verify_render_deployment()
