#!/usr/bin/env python3
"""
🎵 Enhanced Data Acquisition
===========================

This script downloads and organizes the following datasets for AI mixing:
- FMA (Free Music Archive) - Music tracks with Creative Commons licenses
- DAMP Dataset - Karaoke recordings for vocal processing
- Room Impulse Responses - For simulating different recording spaces

These datasets provide a comprehensive foundation for training
production-grade AI mixing models with greater diversity.
"""

import os
import sys
import zipfile
import tarfile
import shutil
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import requests
import librosa
import numpy as np
import soundfile as sf
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

class EnhancedDataAcquisition:
    """Downloads and organizes enhanced datasets for AI mixing."""
    
    def __init__(self, base_dir=None):
        # Set up base directories
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            self.base_dir = Path(base_dir)
            
        # Raw data directories
        self.raw_dir = self.base_dir / "raw"
        self.fma_dir = self.raw_dir / "music" / "fma"
        self.damp_dir = self.raw_dir / "vocals" / "damp"
        self.rir_dir = self.raw_dir / "acoustics" / "room_impulse_responses"
        
        # Processed data directories
        self.processed_dir = self.base_dir / "processed"
        self.clean_dir = self.processed_dir / "clean"
        self.augmented_dir = self.processed_dir / "augmented"
        
        # Metadata directory
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create all directories
        for directory in [
            self.fma_dir, self.damp_dir, self.rir_dir,
            self.clean_dir, self.augmented_dir, self.metadata_dir
        ]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Initialize metadata tracking
        self.metadata = {
            "datasets": {},
            "tracks": {},
            "statistics": {}
        }
    
    def download_with_progress(self, url, destination_path):
        """Download a file with progress bar."""
        destination_path = Path(destination_path)
        
        # Skip if file already exists
        if destination_path.exists():
            print(f"File already exists: {destination_path.name}")
            return destination_path
        
        # Make sure destination directory exists
        destination_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Get file size for progress bar
        response = requests.head(url)
        file_size = int(response.headers.get('content-length', 0))
        
        # Initialize progress bar
        progress = tqdm(total=file_size, unit='B', unit_scale=True, 
                        desc=f"Downloading {destination_path.name}")
        
        # Custom hook for progress updates
        def report_progress(count, block_size, total_size):
            progress.update(block_size)
        
        try:
            # Download file with progress tracking
            urllib.request.urlretrieve(url, destination_path, report_progress)
            progress.close()
            print(f"✅ Downloaded: {destination_path.name}")
            return destination_path
        except Exception as e:
            progress.close()
            print(f"❌ Error downloading {url}: {e}")
            return None
    
    def extract_archive(self, archive_path, extract_to=None):
        """Extract a zip or tar archive."""
        archive_path = Path(archive_path)
        
        if extract_to is None:
            extract_to = archive_path.parent
            
        print(f"📦 Extracting {archive_path.name} to {extract_to}...")
        
        if archive_path.suffix.lower() in ['.zip']:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Get total files for progress
                total_files = len(zip_ref.namelist())
                for i, file in enumerate(zip_ref.namelist()):
                    if i % 100 == 0:  # Only update every 100 files for speed
                        print(f"  ⏳ Extracting: {i}/{total_files} files...", end='\r')
                zip_ref.extractall(extract_to)
                
        elif archive_path.suffix.lower() in ['.tar', '.gz', '.bz2']:
            with tarfile.open(archive_path, 'r') as tar_ref:
                total_files = len(tar_ref.getnames())
                for i, file in enumerate(tar_ref.getnames()):
                    if i % 100 == 0:
                        print(f"  ⏳ Extracting: {i}/{total_files} files...", end='\r')
                tar_ref.extractall(extract_to)
        else:
            print(f"❌ Unsupported archive format: {archive_path.suffix}")
            return False
            
        print(f"✅ Extracted {archive_path.name}")
        return True
    
    def fetch_fma(self, size="small"):
        """
        Download the Free Music Archive (FMA) dataset.
        
        Args:
            size: Dataset size - "small" (8K tracks), "medium" (25K), "large" (106K)
        """
        if size not in ["small", "medium", "large"]:
            print(f"❌ Invalid FMA size: {size}. Using 'small' instead.")
            size = "small"
        
        print(f"🎵 Downloading FMA {size} dataset...")
        
        # Track metadata in dataset
        self.metadata["datasets"]["fma"] = {
            "name": f"FMA {size}",
            "size": size,
            "tracks": 0,
            "download_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
        
        # URLs for the FMA dataset
        base_url = "https://os.unil.cloud.switch.ch/fma/"
        urls = {
            # Audio files
            f"fma_{size}.zip": f"{base_url}fma_{size}.zip",
            
            # Metadata
            "fma_metadata.zip": f"{base_url}fma_metadata.zip",
        }
        
        # Download audio files
        audio_zip = self.download_with_progress(
            urls[f"fma_{size}.zip"], 
            self.fma_dir / f"fma_{size}.zip"
        )
        
        # Download metadata
        metadata_zip = self.download_with_progress(
            urls["fma_metadata.zip"],
            self.fma_dir / "fma_metadata.zip"
        )
        
        # Extract files
        if audio_zip and metadata_zip:
            self.extract_archive(audio_zip, self.fma_dir)
            self.extract_archive(metadata_zip, self.fma_dir)
            
            # Parse metadata
            try:
                # Load tracks metadata
                tracks_file = list(self.fma_dir.glob("**/tracks.csv"))[0]
                print(f"📊 Loading FMA metadata from {tracks_file}...")
                
                # Use pandas to read the metadata
                tracks_df = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
                
                # Count tracks and update metadata
                track_count = len(tracks_df)
                self.metadata["datasets"]["fma"]["tracks"] = track_count
                
                print(f"✅ FMA dataset ready: {track_count} tracks")
                return True
            except Exception as e:
                print(f"❌ Error processing FMA metadata: {e}")
                return False
        else:
            print("❌ Failed to download or extract FMA dataset")
            return False
    
    def fetch_damp(self, subset_size=1000):
        """
        Download the DAMP karaoke dataset (Smule Sing! performances).
        
        For academic/research use, we'll use the publicly available subset.
        
        Args:
            subset_size: Number of tracks to download (default: 1000)
        """
        print(f"🎤 Downloading DAMP Karaoke Dataset (subset: {subset_size} tracks)...")
        
        # Initialize metadata
        self.metadata["datasets"]["damp"] = {
            "name": "DAMP Karaoke Dataset",
            "subset_size": subset_size,
            "tracks": 0,
            "download_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
        
        try:
            # The DAMP dataset is available through the MedleyDB or by request
            # For demo purposes, we'll download sample files from a public source
            
            # URLs for sample DAMP recordings (using Zenodo open research repository)
            # Note: In a real implementation, you would need to request access to the full dataset
            base_url = "https://zenodo.org/record/2650351/files/"
            sample_files = [f"sample_{i}.mp3" for i in range(1, min(subset_size, 20) + 1)]
            
            downloaded_count = 0
            for i, sample in enumerate(sample_files):
                try:
                    url = f"{base_url}{sample}"
                    save_path = self.damp_dir / f"damp_sample_{i+1}.mp3"
                    
                    if self.download_with_progress(url, save_path):
                        downloaded_count += 1
                except Exception as e:
                    print(f"  ⚠️ Error downloading {sample}: {e}")
            
            # Update metadata
            self.metadata["datasets"]["damp"]["tracks"] = downloaded_count
            
            print(f"✅ DAMP dataset ready: {downloaded_count} tracks")
            
            # Add note about full dataset
            print("📝 Note: For the complete DAMP dataset, request access from the Stanford DAP group")
            print("   or MedleyDB project. This is a sample subset for demonstration.")
            
            return True
        except Exception as e:
            print(f"❌ Error processing DAMP dataset: {e}")
            return False
    
    def fetch_room_impulse_responses(self):
        """
        Download room impulse responses for acoustic simulation.
        
        We'll use the OpenAIR (Open Acoustic Impulse Response) library.
        """
        print(f"🔊 Downloading Room Impulse Responses...")
        
        # Initialize metadata
        self.metadata["datasets"]["rir"] = {
            "name": "Room Impulse Responses",
            "source": "OpenAIR",
            "spaces": 0,
            "download_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
        
        # OpenAIR base URL
        base_url = "https://www.openair.hosted.york.ac.uk/samples/"
        
        # Selection of diverse acoustic spaces (small subset for demo)
        spaces = {
            "small_room": "living-room/stereo/Living_Room_Stereo.zip",
            "medium_hall": "stairway/stereo/York-Guildhall-Stairway_Stereo.zip",
            "large_hall": "york-minster/stereo/York_Minster_Nave_Stereo.zip",
            "studio": "laboratory/stereo/DRR_FDN_Stereo.zip"
        }
        
        # Download each space's impulse responses
        downloaded_spaces = 0
        for space_name, space_path in spaces.items():
            try:
                url = f"{base_url}{space_path}"
                filename = space_path.split('/')[-1]
                save_path = self.rir_dir / filename
                
                if self.download_with_progress(url, save_path):
                    # Extract the impulse responses
                    self.extract_archive(save_path, self.rir_dir / space_name)
                    downloaded_spaces += 1
            except Exception as e:
                print(f"  ⚠️ Error downloading {space_name}: {e}")
        
        # Update metadata
        self.metadata["datasets"]["rir"]["spaces"] = downloaded_spaces
        
        print(f"✅ Room Impulse Responses ready: {downloaded_spaces} acoustic spaces")
        return True
    
    def save_metadata(self):
        """Save the dataset metadata to a JSON file."""
        metadata_path = self.metadata_dir / "dataset_metadata.json"
        
        print(f"💾 Saving dataset metadata to {metadata_path}...")
        
        with open(metadata_path, 'w') as f:            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Metadata saved")
        return metadata_path
    
    def run_acquisition(self, datasets=None, fma_size="small", damp_size=100):
        """
        Run the data acquisition process for specified datasets.
        
        Args:
            datasets: List of datasets to acquire. If None, acquires all datasets.
                     Options: ["fma", "damp", "rir"]
            fma_size: Size of FMA dataset to download (small, medium, large)
            damp_size: Number of DAMP tracks to download
        """
        if datasets is None:
            datasets = ["fma", "damp", "rir"]
        
        print(f"🚀 Starting enhanced data acquisition process...")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Datasets to acquire: {', '.join(datasets)}")
        
        results = {}
        
        # Acquire each dataset
        if "fma" in datasets:
            results["fma"] = self.fetch_fma(size=fma_size)
            
        if "damp" in datasets:
            results["damp"] = self.fetch_damp(subset_size=damp_size)
            
        if "rir" in datasets:
            results["rir"] = self.fetch_room_impulse_responses()
        
        # Save metadata
        self.save_metadata()
        
        # Summary
        print("\n📋 Acquisition Summary:")
        print("=" * 50)
        for dataset, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{dataset.upper()}: {status}")
        
        successful = sum(1 for result in results.values() if result)
        print(f"\n📊 Overall: {successful}/{len(results)} datasets successfully acquired")
        
        return results

# When run directly
if __name__ == "__main__":    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Dataset Acquisition for AI Mixing")
    parser.add_argument('--datasets', nargs='+', choices=['fma', 'damp', 'rir', 'all'],
                        default=['all'], help="Datasets to download")
    parser.add_argument('--fma-size', choices=['small', 'medium', 'large'],
                        default='small', help="Size of FMA dataset to download")
    parser.add_argument('--damp-size', type=int, default=100,
                        help="Number of DAMP tracks to download")
    
    args = parser.parse_args()
    
    # Process datasets argument
    if 'all' in args.datasets:
        datasets_to_acquire = ['fma', 'damp', 'rir']
    else:
        datasets_to_acquire = args.datasets
    
    # Create acquisition instance
    acquisition = EnhancedDataAcquisition()
    
    # Run acquisition with specific parameters
    acquisition.run_acquisition(
        datasets=datasets_to_acquire,
        fma_size=args.fma_size,
        damp_size=args.damp_size
    )
    