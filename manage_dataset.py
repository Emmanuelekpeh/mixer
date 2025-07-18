#!/usr/bin/env python3
"""
🎵 Enhanced Dataset Manager
=========================

This script provides a command-line interface for managing the enhanced
dataset pipeline. It allows you to:

1. Set up the environment
2. Download datasets
3. Process raw audio
4. Create augmented versions
5. Generate complete dataset statistics

Usage:
    python manage_dataset.py setup
    python manage_dataset.py download --datasets fma damp rir
    python manage_dataset.py process --datasets fma --subset-size 100
    python manage_dataset.py augment --categories music --subset-size 50
    python manage_dataset.py stats
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import time

def run_script(script_name, args=None):
    """Run a Python script with arguments."""
    script_path = Path(__file__).resolve().parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    
    if args:
        cmd.extend(args)
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        process = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Completed in {elapsed_time:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running script: {e}")
        return False

def setup_environment():
    """Set up the environment for dataset processing."""
    print("🛠️ Setting up environment...")
    return run_script("src/setup_enhanced_dataset.py")

def download_datasets(args):
    """Download datasets."""
    print("📥 Downloading datasets...")
    
    # Prepare arguments
    script_args = []
    
    if args.datasets:
        script_args.extend(["--datasets"] + args.datasets)
    
    if args.fma_size:
        script_args.extend(["--fma-size", args.fma_size])
    
    if args.damp_size:
        script_args.extend(["--damp-size", str(args.damp_size)])
    
    return run_script("src/enhanced_data_acquisition.py", script_args)

def process_datasets(args):
    """Process raw datasets."""
    print("🔄 Processing datasets...")
    
    # Prepare arguments
    script_args = []
    
    if args.datasets:
        script_args.extend(["--datasets"] + args.datasets)
    
    if args.subset_size:
        script_args.extend(["--subset-size", str(args.subset_size)])
    
    if args.skip_features:
        script_args.append("--skip-features")
    
    return run_script("src/enhanced_audio_processor.py", script_args)

def augment_datasets(args):
    """Create augmented versions of processed datasets."""
    print("🔊 Augmenting datasets...")
    
    # Prepare arguments
    script_args = []
    
    if args.categories:
        script_args.extend(["--categories"] + args.categories)
    
    if args.subset_size:
        script_args.extend(["--subset-size", str(args.subset_size)])
    
    return run_script("src/enhanced_audio_augmentor.py", script_args)

def generate_statistics():
    """Generate comprehensive dataset statistics."""
    print("📊 Generating dataset statistics...")
    
    base_dir = Path(__file__).resolve().parent / "data"
    metadata_dir = base_dir / "metadata"
    stats_path = metadata_dir / "dataset_statistics.json"
    
    # Collect metadata from various sources
    metadata_files = {
        "dataset": metadata_dir / "dataset_metadata.json",
        "processing": metadata_dir / "processing_metadata.json",
        "augmentation": metadata_dir / "augmentation_metadata.json"
    }
    
    # Check that metadata files exist
    missing_files = [f for f, p in metadata_files.items() if not p.exists()]
    if missing_files:
        print(f"❌ Missing metadata files: {', '.join(missing_files)}")
        print("   Please run the corresponding pipeline steps first.")
        return False
    
    # Load metadata
    metadata = {}
    for name, path in metadata_files.items():
        try:
            with open(path, 'r') as f:
                metadata[name] = json.load(f)
        except Exception as e:
            print(f"❌ Error loading {name} metadata: {e}")
            return False
    
    # Count files in directories
    dir_counts = {}
    
    for subdir in ["raw", "processed", "features"]:
        dir_path = base_dir / subdir
        if dir_path.exists():
            file_count = sum(1 for _ in dir_path.glob("**/*") if _.is_file())
            dir_counts[subdir] = file_count
    
    # Generate comprehensive statistics
    statistics = {
        "dataset_summary": {
            "original_datasets": list(metadata.get("dataset", {}).get("datasets", {}).keys()),
            "total_original_tracks": metadata.get("processing", {}).get("processing_stats", {}).get("total_tracks", 0),
            "successful_processed": metadata.get("processing", {}).get("processing_stats", {}).get("successful", 0),
            "total_augmented_tracks": metadata.get("augmentation", {}).get("augmentation_stats", {}).get("total_augmented_tracks", 0),
            "augmentation_factor": metadata.get("augmentation", {}).get("augmentation_stats", {}).get("augmentation_factor", 0),
            "total_audio_hours": metadata.get("processing", {}).get("processing_stats", {}).get("duration_total", 0) / 3600
        },
        "file_counts": dir_counts,
        "feature_extraction": metadata.get("processing", {}).get("feature_extraction", {}),
        "augmentation_methods": metadata.get("augmentation", {}).get("augmentation_stats", {}).get("methods_used", {})
    }
    
    # Save statistics
    try:
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"✅ Statistics saved to {stats_path}")
        
        # Print summary
        print("\n📋 Dataset Summary:")
        print("=" * 50)
        print(f"Original datasets: {', '.join(statistics['dataset_summary']['original_datasets'])}")
        print(f"Original tracks: {statistics['dataset_summary']['total_original_tracks']}")
        print(f"Processed tracks: {statistics['dataset_summary']['successful_processed']}")
        print(f"Augmented tracks: {statistics['dataset_summary']['total_augmented_tracks']}")
        print(f"Augmentation factor: {statistics['dataset_summary']['augmentation_factor']:.1f}x")
        print(f"Total audio hours: {statistics['dataset_summary']['total_audio_hours']:.1f}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving statistics: {e}")
        return False

def main():
    """Main function to parse arguments and run commands."""
    parser = argparse.ArgumentParser(description="Enhanced Dataset Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up the environment")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("--datasets", nargs="+", choices=["fma", "damp", "rir", "all"],
                               default=["all"], help="Datasets to download")
    download_parser.add_argument("--fma-size", choices=["small", "medium", "large"],
                               default="small", help="Size of FMA dataset")
    download_parser.add_argument("--damp-size", type=int, default=100,
                               help="Number of DAMP tracks to download")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process raw datasets")
    process_parser.add_argument("--datasets", nargs="+", choices=["fma", "damp", "rir", "all"],
                              default=["all"], help="Datasets to process")
    process_parser.add_argument("--subset-size", type=int, default=None,
                              help="Max number of tracks to process per dataset")
    process_parser.add_argument("--skip-features", action="store_true",
                              help="Skip feature extraction")
    
    # Augment command
    augment_parser = subparsers.add_parser("augment", help="Create augmented versions")
    augment_parser.add_argument("--categories", nargs="+", choices=["music", "vocals", "all"],
                              default=["all"], help="Categories to augment")
    augment_parser.add_argument("--subset-size", type=int, default=None,
                              help="Max number of tracks to augment per category")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Generate dataset statistics")
    
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "setup":
        setup_environment()
    elif args.command == "download":
        download_datasets(args)
    elif args.command == "process":
        process_datasets(args)
    elif args.command == "augment":
        augment_datasets(args)
    elif args.command == "stats":
        generate_statistics()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
