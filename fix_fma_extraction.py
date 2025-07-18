#!/usr/bin/env python3
"""
🎵 Audio Restoration Training Dataset Generator
=============================================

This script creates training data for audio restoration/enhancement by:
1. Using clean audio as ground truth
2. Applying realistic distortions (reverb, noise, compression artifacts, etc.)
3. Training models to restore distorted audio back to original quality
"""

import os
import sys
import zipfile
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import urllib.request

def extract_fma_efficiently(base_dir):
    """Extract FMA dataset efficiently without getting stuck"""
    
    data_dir = Path(base_dir) / "data"
    fma_dir = data_dir / "raw" / "music" / "fma"
    
    # Check if zip files exist
    fma_zip = fma_dir / "fma_small.zip"
    metadata_zip = fma_dir / "fma_metadata.zip"
    
    if not fma_zip.exists():
        print(f"❌ FMA zip file not found at {fma_zip}")
        return False
    
    if not metadata_zip.exists():
        print(f"❌ FMA metadata zip file not found at {metadata_zip}")
        return False
    
    print(f"🎵 Found FMA dataset files")
    print(f"📦 Audio: {fma_zip} ({fma_zip.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"📊 Metadata: {metadata_zip} ({metadata_zip.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Check if already extracted
    audio_extracted = fma_dir / "fma_small"
    metadata_extracted = fma_dir / "fma_metadata"
    
    if audio_extracted.exists() and len(list(audio_extracted.rglob("*.mp3"))) > 7000:
        print("✅ FMA audio files already extracted")
        extract_audio = False
    else:
        print("🔄 Need to extract FMA audio files")
        extract_audio = True
        
    if metadata_extracted.exists() and (metadata_extracted / "tracks.csv").exists():
        print("✅ FMA metadata already extracted")
        extract_metadata = False
    else:
        print("🔄 Need to extract FMA metadata")
        extract_metadata = True
    
    # Extract audio files with progress bar
    if extract_audio:
        print("\n📦 Extracting FMA audio files...")
        try:
            with zipfile.ZipFile(fma_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                total_files = len(file_list)
                
                # Extract with progress bar
                with tqdm(total=total_files, desc="Extracting audio") as pbar:
                    for i, file in enumerate(file_list):
                        try:
                            zip_ref.extract(file, fma_dir)
                            pbar.update(1)
                            
                            # Update description every 100 files
                            if i % 100 == 0:
                                pbar.set_description(f"Extracting audio ({i}/{total_files})")
                                
                        except Exception as e:
                            print(f"⚠️ Error extracting {file}: {e}")
                            continue
                
                print("✅ FMA audio files extracted successfully")
                
        except Exception as e:
            print(f"❌ Error extracting FMA audio: {e}")
            return False
    
    # Extract metadata files
    if extract_metadata:
        print("\n📊 Extracting FMA metadata...")
        try:
            with zipfile.ZipFile(metadata_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                with tqdm(total=len(file_list), desc="Extracting metadata") as pbar:
                    for file in file_list:
                        try:
                            zip_ref.extract(file, fma_dir)
                            pbar.update(1)
                        except Exception as e:
                            print(f"⚠️ Error extracting {file}: {e}")
                            continue
                
                print("✅ FMA metadata extracted successfully")
                
        except Exception as e:
            print(f"❌ Error extracting FMA metadata: {e}")
            return False
    
    # Verify extraction
    print("\n🔍 Verifying extraction...")
    
    # Count audio files
    audio_files = list(fma_dir.rglob("*.mp3"))
    print(f"🎵 Found {len(audio_files)} audio files")
    
    # Check metadata
    tracks_csv = None
    for csv_file in fma_dir.rglob("tracks.csv"):
        tracks_csv = csv_file
        break
    
    if tracks_csv:
        print(f"📊 Found metadata file: {tracks_csv}")
        try:
            # Load tracks metadata
            tracks_df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
            print(f"📈 Metadata contains {len(tracks_df)} track entries")
            
            # Save metadata info
            metadata_info = {
                "dataset": "FMA small",
                "audio_files": len(audio_files),
                "metadata_entries": len(tracks_df),
                "extraction_complete": True,
                "base_path": str(fma_dir)
            }
            
            with open(data_dir / "fma_extraction_info.json", "w") as f:
                json.dump(metadata_info, f, indent=2)
            
            print("✅ FMA dataset extraction completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
            return False
    else:
        print("❌ No tracks.csv metadata file found")
        return False

def download_musdb_dataset(data_dir):
    """Download MUSDB18 dataset if not already present"""
    import urllib.request
    
    musdb_dir = data_dir / "raw" / "music" / "musdb18"
    musdb_dir.mkdir(parents=True, exist_ok=True)
    
    musdb_zip = musdb_dir / "musdb18.zip"
    
    # Check if already downloaded
    if musdb_zip.exists() and musdb_zip.stat().st_size > 1000000:  # > 1MB
        print(f"✅ MUSDB18 zip already exists: {musdb_zip}")
        return True
    
    print("🎵 Downloading MUSDB18 dataset...")
    print("ℹ️ This is a large dataset (~4GB), download may take a while...")
    print("📋 Dataset includes 150 full songs with separated stems (vocals, drums, bass, other)")
    
    # MUSDB18 download URLs (try multiple sources)
    download_urls = [
        "https://zenodo.org/record/1117372/files/musdb18.zip",  # Primary source
        "https://sigsep.github.io/datasets/musdb/musdb18.zip",  # Alternative
    ]
    
    for i, musdb_url in enumerate(download_urls):
        try:
            print(f"\n🔄 Attempting download from source {i+1}/{len(download_urls)}...")
            print(f"🌐 URL: {musdb_url}")
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded / total_size) * 100)
                    downloaded_mb = downloaded / 1024 / 1024
                    total_mb = total_size / 1024 / 1024
                    print(f"\r📦 Downloading: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
                else:
                    downloaded_mb = (block_num * block_size) / 1024 / 1024
                    print(f"\r📦 Downloading: {downloaded_mb:.1f} MB", end="", flush=True)
            
            urllib.request.urlretrieve(musdb_url, musdb_zip, reporthook=show_progress)
            print(f"\n✅ MUSDB18 downloaded successfully from source {i+1}!")
            
            # Verify download
            if musdb_zip.exists() and musdb_zip.stat().st_size > 100000000:  # > 100MB
                print(f"📊 Downloaded file size: {musdb_zip.stat().st_size / 1024 / 1024:.1f} MB")
                return True
            else:
                print(f"❌ Downloaded file seems too small, trying next source...")
                if musdb_zip.exists():
                    musdb_zip.unlink()  # Remove incomplete download
                continue
                
        except Exception as e:
            print(f"\n❌ Error downloading from source {i+1}: {e}")
            if musdb_zip.exists():
                musdb_zip.unlink()  # Remove incomplete download
            continue
    
    print(f"\n❌ Failed to download MUSDB18 from all sources")
    print("ℹ️ You can manually download MUSDB18 from:")
    print("   - https://sigsep.github.io/datasets/musdb.html")
    print("   - https://zenodo.org/record/1117372")
    print(f"   Place the downloaded musdb18.zip in: {musdb_zip}")
    return False

def extract_musdb_efficiently(base_dir):
    """Extract MUSDB18 dataset efficiently"""
    
    data_dir = Path(base_dir) / "data"
    musdb_dir = data_dir / "raw" / "music" / "musdb18"
    musdb_zip = musdb_dir / "musdb18.zip"
    
    # Check alternate path where files might have been extracted  
    alt_musdb_dir = Path(base_dir) / "raw" / "music" / "musdb18"
    
    # Check if zip exists
    if not musdb_zip.exists():
        print("🎵 MUSDB18 zip not found, attempting to download...")
        if not download_musdb_dataset(data_dir):
            return False
    
    print(f"🎵 Found MUSDB18 dataset file")
    print(f"📦 Audio: {musdb_zip} ({musdb_zip.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Check if already extracted (in either location)
    stem_files = []
    if alt_musdb_dir.exists():
        stem_files = list(alt_musdb_dir.rglob("*.stem.mp4"))
        print(f"✅ Found existing MUSDB18 files in: {alt_musdb_dir}")
    
    musdb_extracted = musdb_dir / "musdb18"
    if musdb_extracted.exists():
        stem_files.extend(list(musdb_extracted.rglob("*.stem.mp4")))
        
    if len(stem_files) > 100:
        print(f"✅ MUSDB18 files already extracted ({len(stem_files)} stem files)")
        return True
    
    print("🔄 Need to extract MUSDB18 files")
    
    # Extract MUSDB18 files
    print("\n📦 Extracting MUSDB18 dataset...")
    try:
        with zipfile.ZipFile(musdb_zip, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # Extract with progress bar
            with tqdm(total=total_files, desc="Extracting MUSDB18") as pbar:
                for i, file in enumerate(file_list):
                    try:
                        zip_ref.extract(file, musdb_dir)
                        pbar.update(1)
                        
                        # Update description every 10 files (MUSDB has fewer files)
                        if i % 10 == 0:
                            pbar.set_description(f"Extracting MUSDB18 ({i}/{total_files})")
                            
                    except Exception as e:
                        print(f"⚠️ Error extracting {file}: {e}")
                        continue
            
            print("✅ MUSDB18 files extracted successfully")
            
    except Exception as e:
        print(f"❌ Error extracting MUSDB18: {e}")
        return False
    
    # Verify extraction
    print("🔍 Verifying MUSDB18 extraction...")
    
    # Count stem files (MUSDB uses .stem.mp4 format, not .wav)
    stem_files = list(musdb_dir.rglob("*.stem.mp4"))
    
    # Also check alternate location if files exist there
    if alt_musdb_dir.exists():
        alt_stem_files = list(alt_musdb_dir.rglob("*.stem.mp4"))
        print(f"🎵 Found {len(alt_stem_files)} MUSDB18 stem files in alternate location")
        if len(alt_stem_files) > len(stem_files):
            stem_files = alt_stem_files
            
    print(f"🎵 Found {len(stem_files)} MUSDB18 stem files")
    
    if len(stem_files) > 100:  # MUSDB18 should have ~150 stem files (86 train + 50 test)
        print("✅ MUSDB18 dataset extraction completed successfully!")
        
        # Create extraction info
        extraction_info = {
            "dataset": "MUSDB18",
            "stem_files": len(stem_files),
            "extraction_complete": True,
            "base_path": str(musdb_dir),
            "format": "stem.mp4"
        }
        
        with open(data_dir / "musdb18_extraction_info.json", "w") as f:
            json.dump(extraction_info, f, indent=2)
            
        return True
    else:
        print("❌ MUSDB18 extraction appears incomplete")
        return False

def main():
    """Main function"""
    base_dir = Path(__file__).parent
    
    print("🎵 FMA and MUSDB Dataset Extraction")
    print("=" * 40)
    
    # Extract FMA dataset (required)
    print("\n📦 Extracting FMA dataset...")
    success_fma = extract_fma_efficiently(base_dir)
    
    if not success_fma:
        print("❌ FMA extraction failed - this is required for training")
        return
    
    # Extract MUSDB dataset (optional, adds variety)
    print("\n📦 Extracting MUSDB dataset...")
    print("ℹ️ MUSDB is optional but adds high-quality professional music data")
    print("🔄 Attempting to download and extract MUSDB18 automatically...")
    
    success_musdb = extract_musdb_efficiently(base_dir)
    
    if success_musdb:
        print("\n✅ Both FMA and MUSDB datasets are ready for training!")
        print("🎵 You now have maximum variety for AI mixing training")
    else:
        print("\n⚠️ MUSDB extraction failed, but FMA is ready")
        print("🎵 You can still train with FMA dataset alone")
    
    print("\n🚀 Next steps:")
    print("   1. Run the training pipeline:")
    print("   2. python train_mixer_pipeline_ultra_fixed.py")

if __name__ == "__main__":
    main()
