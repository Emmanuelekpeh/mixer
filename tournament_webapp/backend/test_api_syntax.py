#!/usr/bin/env python3
"""
Quick syntax test for the API
"""
import sys
from pathlib import Path

# Add path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

try:
    import tournament_api
    print("✅ API imports successfully!")
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    print(f"   File: {e.filename}")
    print(f"   Line: {e.lineno}")
    print(f"   Text: {e.text}")
except Exception as e:
    print(f"❌ Import Error: {e}")
