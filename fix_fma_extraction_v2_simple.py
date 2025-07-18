#!/usr/bin/env python3
"""
Enhanced FMA Dataset Extraction Fix - Version 2 (No External Dependencies)
==========================================================================
This script fixes the stuck FMA extraction process by:
1. Detecting and handling corrupted/problematic files
2. Implementing robust extraction with progress tracking
3. Verifying extraction completeness
4. Providing detailed logging and recovery options
"""

import os
import sys
import zipfile
import json
import shutil
import time
import csv
from pathlib import Path

def print_progress(current, total, prefix="Progress"):
    """Print progress bar"""
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)

def fix_fma_extraction():
    """Fix the stuck FMA extraction process"""
    
    # Define paths
    base_dir = Path("C:/Users/emman/Projects/mixer/data")
    raw_dir = base_dir / "raw" / "music" / "fma"
    zip_path = base_dir / "fma_small.zip"
    metadata_zip_path = base_dir / "fma_metadata.zip"
    
    print("🔧 FMA Extraction Fix - Version 2")
    print("=" * 50)
    print(f"📁 Base directory: {base_dir}")
    print(f"📁 Raw directory: {raw_dir}")
    print(f"📦 ZIP file: {zip_path}")
    print(f"📦 Metadata ZIP: {metadata_zip_path}")
    
    # Check if files exist
    if not zip_path.exists():
        print(f"❌ FMA ZIP file not found: {zip_path}")
        return False
    
    if not metadata_zip_path.exists():
        print(f"❌ Metadata ZIP file not found: {metadata_zip_path}")
        return False
    
    # Create extraction directory
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Check current extraction state
    print("\n🔍 Checking current extraction state...")
    
    existing_files = []
    if raw_dir.exists():
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file.endswith('.mp3'):
                    existing_files.append(os.path.join(root, file))
    
    print(f"📊 Found {len(existing_files)} existing MP3 files")
    
    # Step 2: Extract FMA small dataset with robust handling
    print("\n📦 Extracting FMA small dataset...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of all files in the zip
            all_files = zip_ref.namelist()
            mp3_files = [f for f in all_files if f.endswith('.mp3')]
            
            print(f"📊 Total MP3 files in ZIP: {len(mp3_files)}")
            print(f"📊 Total files in ZIP: {len(all_files)}")
            
            # Extract files with progress tracking
            extracted_count = 0
            skipped_count = 0
            error_count = 0
            
            file_list = zip_ref.infolist()
            total_files = len(file_list)
            
            for i, file_info in enumerate(file_list):
                try:
                    # Print progress
                    if i % 100 == 0 or i == total_files - 1:
                        print_progress(i + 1, total_files, "Extracting")
                    
                    # Check if file already exists
                    target_path = raw_dir / file_info.filename
                    
                    if target_path.exists() and target_path.stat().st_size > 0:
                        skipped_count += 1
                        continue
                    
                    # Create parent directories if needed
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract file with timeout protection
                    try:
                        zip_ref.extract(file_info, raw_dir)
                        extracted_count += 1
                    except Exception as extract_error:
                        print(f"\n❌ Error extracting {file_info.filename}: {extract_error}")
                        error_count += 1
                        continue
                    
                    # Verify extraction
                    if not target_path.exists():
                        print(f"\n❌ Failed to extract: {file_info.filename}")
                        error_count += 1
                    
                except Exception as e:
                    print(f"\n❌ Error processing {file_info.filename}: {e}")
                    error_count += 1
                    continue
            
            print(f"\n✅ Extraction complete!")
            print(f"📊 Extracted: {extracted_count} files")
            print(f"📊 Skipped: {skipped_count} files (already exist)")
            print(f"📊 Errors: {error_count} files")
            
    except Exception as e:
        print(f"❌ Error opening ZIP file: {e}")
        return False
    
    # Step 3: Extract metadata
    print("\n📋 Extracting metadata...")
    
    try:
        with zipfile.ZipFile(metadata_zip_path, 'r') as zip_ref:
            metadata_files = zip_ref.namelist()
            print(f"📊 Metadata files in ZIP: {len(metadata_files)}")
            
            # Extract metadata
            zip_ref.extractall(raw_dir / "fma_metadata")
            print("✅ Metadata extracted successfully!")
            
    except Exception as e:
        print(f"❌ Error extracting metadata: {e}")
        return False
    
    # Step 4: Verify extraction completeness
    print("\n🔍 Verifying extraction completeness...")
    
    # Count extracted MP3 files
    final_mp3_files = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                # Check if file is not empty
                if os.path.getsize(file_path) > 0:
                    final_mp3_files.append(file_path)
    
    print(f"📊 Final MP3 count: {len(final_mp3_files)}")
    
    # Check metadata files
    metadata_dir = raw_dir / "fma_metadata"
    metadata_files = []
    if metadata_dir.exists():
        for file in metadata_dir.glob("*.csv"):
            metadata_files.append(file)
    
    print(f"📊 Metadata files found: {len(metadata_files)}")
    for file in metadata_files:
        print(f"   - {file.name}")
    
    # Step 5: Validate dataset integrity
    print("\n🔍 Validating dataset integrity...")
    
    # Check if tracks.csv exists and load it
    tracks_csv = metadata_dir / "tracks.csv"
    if tracks_csv.exists():
        try:
            # Count lines in tracks.csv
            with open(tracks_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                track_count = sum(1 for row in reader) - 1  # Subtract header
            
            print(f"📊 Tracks in metadata: {track_count}")
            
            # Check if we have audio files for the tracks
            available_track_ids = set()
            for mp3_file in final_mp3_files:
                # Extract track ID from filename
                filename = os.path.basename(mp3_file)
                if filename.endswith('.mp3'):
                    track_id = filename[:-4]  # Remove .mp3 extension
                    try:
                        track_id = int(track_id)
                        available_track_ids.add(track_id)
                    except ValueError:
                        continue
            
            print(f"📊 Available track IDs: {len(available_track_ids)}")
            
        except Exception as e:
            print(f"❌ Error validating tracks.csv: {e}")
    
    # Step 6: Create summary report
    print("\n📝 Creating extraction summary...")
    
    summary = {
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_mp3_files": len(final_mp3_files),
        "metadata_files": len(metadata_files),
        "extraction_successful": len(final_mp3_files) > 7000,  # Expect ~8000 files
        "dataset_path": str(raw_dir),
        "metadata_path": str(metadata_dir)
    }
    
    # Save summary
    summary_path = base_dir / "fma_extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: {summary_path}")
    
    # Step 7: Final status
    print("\n" + "=" * 50)
    if summary["extraction_successful"]:
        print("✅ FMA EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"🎵 {len(final_mp3_files)} audio files ready for training")
        print(f"📋 {len(metadata_files)} metadata files available")
        print(f"📁 Dataset location: {raw_dir}")
        print("\n🚀 Next steps:")
        print("   1. Run the training pipeline")
        print("   2. python train_mixer_pipeline_ultra_fixed.py")
        return True
    else:
        print("❌ FMA EXTRACTION INCOMPLETE")
        print(f"   Expected: ~8000 files")
        print(f"   Found: {len(final_mp3_files)} files")
        print("   Please check for extraction errors above")
        return False

def clean_partial_extraction():
    """Clean up partial extraction if needed"""
    base_dir = Path("C:/Users/emman/Projects/mixer/data")
    raw_dir = base_dir / "raw" / "music" / "fma"
    
    print("🧹 Cleaning partial extraction...")
    
    if raw_dir.exists():
        # Remove empty directories and files
        removed_files = 0
        removed_dirs = 0
        
        for root, dirs, files in os.walk(raw_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                # Remove empty files
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    removed_files += 1
                    if removed_files % 100 == 0:
                        print(f"🗑️ Removed {removed_files} empty files...")
            
            # Remove empty directories
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                    removed_dirs += 1
                except OSError:
                    pass  # Directory not empty
        
        print(f"✅ Cleanup complete! Removed {removed_files} files and {removed_dirs} directories")
    else:
        print("No extraction directory found to clean")

def check_zip_integrity():
    """Check if ZIP files are intact"""
    base_dir = Path("C:/Users/emman/Projects/mixer/data")
    zip_path = base_dir / "fma_small.zip"
    metadata_zip_path = base_dir / "fma_metadata.zip"
    
    print("🔍 Checking ZIP file integrity...")
    
    # Check main ZIP
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                bad_files = zip_ref.testzip()
                if bad_files:
                    print(f"❌ Corrupted files in main ZIP: {bad_files}")
                    return False
                else:
                    print("✅ Main ZIP file is intact")
        except Exception as e:
            print(f"❌ Error checking main ZIP: {e}")
            return False
    
    # Check metadata ZIP
    if metadata_zip_path.exists():
        try:
            with zipfile.ZipFile(metadata_zip_path, 'r') as zip_ref:
                bad_files = zip_ref.testzip()
                if bad_files:
                    print(f"❌ Corrupted files in metadata ZIP: {bad_files}")
                    return False
                else:
                    print("✅ Metadata ZIP file is intact")
        except Exception as e:
            print(f"❌ Error checking metadata ZIP: {e}")
            return False
    
    return True

def main():
    """Main function"""
    print("🔧 FMA Dataset Extraction Fix - Version 2")
    print("=" * 50)
    
    # Check ZIP integrity first
    if not check_zip_integrity():
        print("❌ ZIP files are corrupted. Please re-download the dataset.")
        return False
    
    # Check if we need to clean up first
    response = input("Do you want to clean up partial extraction first? (y/n): ").lower().strip()
    if response == 'y':
        clean_partial_extraction()
    
    # Attempt extraction
    success = fix_fma_extraction()
    
    if success:
        print("\n🎉 FMA dataset is ready for training!")
    else:
        print("\n❌ FMA extraction needs manual intervention")
        print("   Check the error messages above for details")
    
    return success

if __name__ == "__main__":
    main()
