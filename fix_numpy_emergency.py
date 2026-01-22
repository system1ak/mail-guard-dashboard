#!/usr/bin/env python3
"""
Emergency NumPy Compatibility Fixer for Mail Guard
This script MUST be run in your local machine to fix the models
"""

import os
import sys
import pickle
from pathlib import Path

print("=" * 60)
print("Mail Guard Emergency NumPy Fixer")
print("=" * 60)

# Check if models directory exists
models_dir = Path("models")
if not models_dir.exists():
    print("❌ ERROR: models/ directory not found!")
    print("Run this from the project root directory")
    sys.exit(1)

print("\n📋 Checking environment...")
try:
    import numpy
    print(f"✓ NumPy version: {numpy.__version__}")
    
    # Check if numpy._core exists
    try:
        from numpy import _core
        print("✓ numpy._core module found")
    except ImportError:
        print("⚠️  numpy._core NOT found - you may have wrong NumPy version")
        print("   Expected: numpy >= 1.26.4")
        print(f"   Got: {numpy.__version__}")
        
except ImportError:
    print("❌ NumPy not installed!")
    sys.exit(1)

print("\n" + "=" * 60)
print("Attempting to load and re-pickle models...")
print("=" * 60)

model_files = [
    "stacking_model.pkl",
    "feature_extractor.pkl", 
    "best_threshold.pkl",
    "scaler.pkl"
]

fixed_count = 0
failed_files = []

for model_file in model_files:
    model_path = models_dir / model_file
    
    if not model_path.exists():
        print(f"\n⚠️  SKIP: {model_file} (not found)")
        continue
    
    print(f"\n🔄 Processing: {model_file}")
    print(f"   File size: {model_path.stat().st_size / 1024:.1f} KB")
    
    try:
        # Load the model
        print(f"   → Loading...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"   ✓ Loaded: {type(model).__name__}")
        
        # Re-save with protocol 4
        print(f"   → Re-pickling with protocol 4...")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        print(f"   ✓ FIXED: {model_file}")
        fixed_count += 1
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        failed_files.append((model_file, str(e)))

print("\n" + "=" * 60)
print(f"Results: {fixed_count}/{len(model_files)} models fixed")
print("=" * 60)

if failed_files:
    print("\n❌ Failed files:")
    for fname, error in failed_files:
        print(f"  - {fname}: {error}")
    sys.exit(1)
else:
    print("\n✅ All models fixed successfully!")
    print("\nNext steps:")
    print("1. Commit changes:")
    print("   git add models/")
    print("   git commit -m 'Fix NumPy compatibility in models'")
    print("\n2. Push to GitHub:")
    print("   git push")
    print("\n3. Deploy to Cloud Run:")
    print("   gcloud run deploy mail-guard-dashboard --source .")
    print("\nYour app should be live in 5-10 minutes!")
    sys.exit(0)
