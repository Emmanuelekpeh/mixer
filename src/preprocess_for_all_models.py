#!/usr/bin/env python3
"""
🎵 Complete Audio Preprocessing Pipeline
=======================================

This script handles the entire preprocessing workflow for all models:
1. Process FMA dataset audio files (already downloaded)
2. Create spectrograms with optimal parameters for all model types
3. Generate dataset splits (train/validation/test)
4. Create additional model-specific features (MFCC, etc.)
5. Set up metadata for all models
6. Prepare directory structure for training

Usage:
    python preprocess_for_all_models.py

Options:
    --fma-size=small     Size of FMA dataset (small/medium/large)
    --sr=22050           Sample rate for processing
    --n-mels=128         Number of mel bands for spectrograms
    --split-ratio=0.8    Train/test split ratio
    --n-jobs=4           Number of parallel jobs
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from pathlib import Path
from tqdm import tqdm
import json
import random
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import warnings
warnings.filterwarnings('ignore')

class CompletePreprocessor:
    """Complete preprocessing pipeline for all model architectures."""
    
    def __init__(self, 
                 base_dir=None, 
                 sr=22050, 
                 n_fft=2048, 
                 hop_length=512, 
                 n_mels=128,
                 split_ratio=0.8,
                 fma_size="small",
                 n_jobs=None):
        """Initialize the preprocessor with parameters."""        # Set up base directories
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            self.base_dir = Path(base_dir)
        
        # Audio processing parameters
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.split_ratio = split_ratio
        self.fma_size = fma_size
        self.generate_ast = generate_ast and AST_AVAILABLE
        self.ast_extractor = None
        
        # Number of parallel jobs
        if n_jobs is None:
            self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_jobs = n_jobs
        
        # Set up directory structure
        self.setup_directories()
        
        # Initialize metadata
        self.metadata = {
            "preprocessing_info": {
                "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "sample_rate": self.sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "split_ratio": self.split_ratio,
                "fma_size": self.fma_size
            },
            "dataset_stats": {
                "total_tracks": 0,
                "train_tracks": 0,
                "val_tracks": 0,
                "test_tracks": 0,
                "total_duration_hours": 0,
                "genres": {}
            },
            "track_info": {}
        }        # Initialize AST feature extractor if available
        if self.generate_ast:
            try:
                if 'ASTFeatureExtractor' in globals():
                    self.ast_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
                    print("✅ AST feature extractor loaded successfully")
                else:
                    print("⚠️ AST feature extractor not available")
                    self.generate_ast = False
            except Exception as e:
                print(f"⚠️ Failed to load AST feature extractor: {e}")
                print("⚠️ Falling back to MFCC features")
                self.generate_ast = False
        
    def setup_directories(self):
        """Set up all required directories for the preprocessing pipeline."""
        # FMA dataset location
        self.fma_dir = self.base_dir / "raw" / "music" / "fma"
        self.fma_audio_dir = self.fma_dir / "fma_small"  # Adjust based on fma_size
        
        # Output directories for all model types
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val" 
        self.test_dir = self.base_dir / "test"
        
        # Spectrogram directories
        self.spec_dir = self.base_dir / "spectrograms"
        self.spec_train_dir = self.spec_dir / "train"
        self.spec_val_dir = self.spec_dir / "val"
        self.spec_test_dir = self.spec_dir / "test"
        
        # AST feature directories
        self.ast_dir = self.base_dir / "ast_features"
        self.ast_train_dir = self.ast_dir / "train"
        self.ast_val_dir = self.ast_dir / "val"
        self.ast_test_dir = self.ast_dir / "test"
        
        # Create all directories
        for directory in [
            self.train_dir, self.val_dir, self.test_dir,
            self.spec_train_dir, self.spec_val_dir, self.spec_test_dir,
            self.ast_train_dir, self.ast_val_dir, self.ast_test_dir
        ]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def find_fma_audio_files(self):
        """Find all MP3 files in the FMA dataset."""
        if not self.fma_audio_dir.exists():
            raise FileNotFoundError(f"FMA audio directory not found at {self.fma_audio_dir}")
        
        print(f"🔍 Finding audio files in {self.fma_audio_dir}...")
        
        # The FMA dataset organizes files in a nested structure
        audio_files = list(self.fma_audio_dir.glob("**/*.mp3"))
        
        if not audio_files:
            raise FileNotFoundError(f"No MP3 files found in {self.fma_audio_dir}")
        
        print(f"✅ Found {len(audio_files)} audio files")
        return audio_files
    
    def load_fma_metadata(self):
        """Load metadata from the FMA dataset."""
        print("📊 Loading FMA metadata...")
        
        # Find the metadata files
        metadata_dir = self.fma_dir
        tracks_file = list(metadata_dir.glob("**/tracks.csv"))
        
        if not tracks_file:
            print("⚠️ Tracks metadata file not found. Some features will be limited.")
            return {}
        
        # Load tracks metadata using pandas
        try:
            tracks_df = pd.read_csv(tracks_file[0], index_col=0, header=[0, 1])
            # Extract relevant metadata (genre, etc.)
            track_metadata = {}
            
            # Extract genre information
            if ('track', 'genre_top') in tracks_df.columns:
                genres = tracks_df['track', 'genre_top'].dropna()
                
                # Count genres for statistics
                genre_counts = genres.value_counts().to_dict()
                self.metadata["dataset_stats"]["genres"] = genre_counts
                
                # Store genre per track
                for track_id, genre in genres.items():
                    track_metadata[str(track_id)] = {"genre": genre}
            
            print(f"✅ Loaded metadata for {len(track_metadata)} tracks")
            return track_metadata
            
        except Exception as e:
            print(f"⚠️ Error loading metadata: {e}")
            return {}
    
    def create_train_test_split(self, audio_files, metadata):
        """Create train/validation/test splits."""
        print("🔀 Creating dataset splits...")
        
        # Shuffle files for random split
        random.shuffle(audio_files)
        
        # Split into train, validation, and test sets
        train_size = int(len(audio_files) * self.split_ratio)
        val_size = int((len(audio_files) - train_size) / 2)
        
        train_files = audio_files[:train_size]
        val_files = audio_files[train_size:train_size + val_size]
        test_files = audio_files[train_size + val_size:]
        
        # Update metadata
        self.metadata["dataset_stats"]["train_tracks"] = len(train_files)
        self.metadata["dataset_stats"]["val_tracks"] = len(val_files)
        self.metadata["dataset_stats"]["test_tracks"] = len(test_files)
        self.metadata["dataset_stats"]["total_tracks"] = len(audio_files)
        
        print(f"✅ Split dataset: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
        
        return train_files, val_files, test_files
    
    def process_audio_file(self, audio_file, output_dir, spectrogram_dir, ast_dir=None):
        """Process a single audio file: load, normalize, extract features, save."""
        try:
            # Extract track ID from FMA file structure (varies by FMA version)
            track_id = audio_file.stem
            
            # Prepare output paths
            output_path = output_dir / f"{track_id}.wav"
            spec_path = spectrogram_dir / f"{track_id}.npy"
            
            # Skip if already processed
            if output_path.exists() and spec_path.exists():
                return track_id, "already_processed"
            
            # Load audio (resampling to target sr)
            try:
                audio, _ = librosa.load(audio_file, sr=self.sr, mono=True)
            except Exception as e:
                return track_id, f"load_error: {e}"
            
            # Check if audio is too short
            if len(audio) < self.sr:  # Less than 1 second
                return track_id, "too_short"
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Save preprocessed audio
            sf.write(output_path, audio, self.sr)
            
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sr, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                n_mels=self.n_mels
            )
            
            # Convert to dB scale
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Save spectrogram
            np.save(spec_path, S_db)
              # Generate AST features if requested
            if ast_dir is not None and self.generate_ast:
                ast_path = ast_dir / f"{track_id}_ast_cls.npy"
                if not ast_path.exists():
                    try:
                        if AST_AVAILABLE:
                            # Use the AST feature extractor
                            inputs = self.ast_extractor(
                                audio, 
                                sampling_rate=self.sr, 
                                return_tensors="pt"
                            )
                            # Save the features
                            np.save(ast_path, inputs.input_values.numpy())
                        else:
                            # Fallback: use MFCC features instead
                            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=40)
                            # Add delta features for more context
                            mfcc_delta = librosa.feature.delta(mfcc)
                            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                            # Stack all features
                            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
                            # Reshape to match AST input shape expectation
                            mfcc_features = mfcc_features.reshape(1, -1)
                            # Save the features
                            np.save(ast_path, mfcc_features)
                    except Exception as e:
                        print(f"⚠️ Error generating features for {track_id}: {e}")
            
            # Calculate duration
            duration = len(audio) / self.sr
            
            # Track info
            track_info = {
                "duration": duration,
                "sample_rate": self.sr,
                "n_samples": len(audio)
            }
            
            return track_id, track_info
            
        except Exception as e:
            return audio_file.stem, f"error: {e}"
    
    def process_dataset_split(self, files, output_dir, spectrogram_dir, ast_dir=None):
        """Process all files in a dataset split using parallel execution."""
        # Create a partial function with fixed parameters
        process_fn = partial(
            self.process_audio_file, 
            output_dir=output_dir, 
            spectrogram_dir=spectrogram_dir,
            ast_dir=ast_dir
        )
        
        # Process files in parallel
        results = {}
        errors = []
        total_duration = 0
        
        print(f"⏳ Processing {len(files)} files...")
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            with tqdm(total=len(files)) as pbar:
                for file, result in zip(files, executor.map(process_fn, files)):
                    track_id, info = result
                    
                    if isinstance(info, dict):  # Successful processing
                        results[track_id] = info
                        total_duration += info["duration"]
                    else:  # Error
                        errors.append((track_id, info))
                    
                    pbar.update(1)
        
        # Print summary
        print(f"✅ Processed {len(results)} files successfully")
        if errors:
            print(f"⚠️ Encountered {len(errors)} errors")
            
        return results, total_duration, errors
    
    def generate_fake_mixing_targets(self, track_ids):
        """
        Generate fake mixing parameters for training.
        In a real scenario, these would be derived from professional mixes.
        """
        print("🎚️ Generating synthetic mixing targets...")
        
        # Initialize targets dictionary
        targets = {}
        
        # Parameter names for reference
        param_names = [
            "Input Gain", "Compression Ratio", "Compression Attack", "Compression Release",
            "Low Shelf (80Hz)", "Low Mid (200Hz)", "Mid (1kHz)", "High Mid (4kHz)", 
            "High Shelf (12kHz)", "Presence (8kHz)", "Reverb Send", "Reverb Type",
            "Delay Send", "Delay Time", "Stereo Width", "Bass Mono", "Output Level"
        ]
        
        # Create synthetic values for each track (17 parameters)
        for track_id in track_ids:
            # Generate random values between 0.0 and 1.0 with some constraints
            params = []
            
            # Input Gain (0.6-0.9)
            params.append(0.6 + 0.3 * random.random())
            
            # Compression Ratio (0.1-0.7)
            params.append(0.1 + 0.6 * random.random())
            
            # Compression Attack (0.2-0.8)
            params.append(0.2 + 0.6 * random.random())
            
            # Compression Release (0.3-0.7)
            params.append(0.3 + 0.4 * random.random())
            
            # EQ parameters (more variety)
            for _ in range(5):
                params.append(0.2 + 0.6 * random.random())
            
            # Presence (0.3-0.7)
            params.append(0.3 + 0.4 * random.random())
            
            # Reverb Send (0.1-0.5)
            params.append(0.1 + 0.4 * random.random())
            
            # Reverb Type (0.0-1.0)
            params.append(random.random())
            
            # Delay Send (0.05-0.3)
            params.append(0.05 + 0.25 * random.random())
            
            # Delay Time (0.0-1.0)
            params.append(random.random())
            
            # Stereo Width (0.4-0.8)
            params.append(0.4 + 0.4 * random.random())
            
            # Bass Mono (0.6-1.0)
            params.append(0.6 + 0.4 * random.random())
            
            # Output Level (0.7-0.95)
            params.append(0.7 + 0.25 * random.random())
            
            # Store in targets dictionary
            targets[track_id] = params
        
        # Save targets to file
        targets_file = self.base_dir / "targets_generated.json"
        with open(targets_file, 'w') as f:
            json.dump(targets, f, indent=2)
        
        print(f"✅ Generated mixing targets for {len(targets)} tracks")
        return targets
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        start_time = time.time()
        print("🚀 Starting complete preprocessing pipeline...")
        
        # 1. Find all audio files
        audio_files = self.find_fma_audio_files()
        
        # 2. Load metadata
        track_metadata = self.load_fma_metadata()
        
        # 3. Create dataset splits
        train_files, val_files, test_files = self.create_train_test_split(audio_files, track_metadata)
        
        # 4. Process each split
        print("\n📊 Processing training set...")
        train_results, train_duration, train_errors = self.process_dataset_split(
            train_files, self.train_dir, self.spec_train_dir, self.ast_train_dir
        )
        
        print("\n📊 Processing validation set...")
        val_results, val_duration, val_errors = self.process_dataset_split(
            val_files, self.val_dir, self.spec_val_dir, self.ast_val_dir
        )
        
        print("\n📊 Processing test set...")
        test_results, test_duration, test_errors = self.process_dataset_split(
            test_files, self.test_dir, self.spec_test_dir, self.ast_test_dir
        )
        
        # 5. Update metadata
        self.metadata["track_info"].update(train_results)
        self.metadata["track_info"].update(val_results)
        self.metadata["track_info"].update(test_results)
        
        total_duration_hours = (train_duration + val_duration + test_duration) / 3600
        self.metadata["dataset_stats"]["total_duration_hours"] = total_duration_hours
        
        # 6. Generate mixing targets
        all_track_ids = list(train_results.keys()) + list(val_results.keys()) + list(test_results.keys())
        targets = self.generate_fake_mixing_targets(all_track_ids)
        
        # 7. Save metadata
        metadata_file = self.base_dir / "preprocessing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # 8. Print summary
        elapsed_time = time.time() - start_time
        print("\n✅ Preprocessing complete!")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"🎵 Processed {len(all_track_ids)} tracks ({total_duration_hours:.1f} hours)")
        print(f"📊 Dataset splits: {len(train_results)} train, {len(val_results)} validation, {len(test_results)} test")
        print(f"📁 Metadata saved to: {metadata_file}")
        print(f"🎚️ Mixing targets saved to: {self.base_dir / 'targets_generated.json'}")
        
        if self.generate_ast:
            print(f"🧠 AST features generated for all tracks")
        
        return self.metadata

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Complete preprocessing pipeline for all models")
    parser.add_argument("--fma-size", choices=["small", "medium", "large"], default="small",
                        help="Size of FMA dataset (default: small)")
    parser.add_argument("--sr", type=int, default=22050,
                        help="Sample rate for processing (default: 22050)")
    parser.add_argument("--n-mels", type=int, default=128,
                        help="Number of mel bands for spectrograms (default: 128)")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Train/test split ratio (default: 0.8)")
    parser.add_argument("--no-ast", action="store_true",
                        help="Skip AST feature generation")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Number of parallel jobs (default: CPU count - 1)")
    
    args = parser.parse_args()
    
    # Create and run preprocessor
    preprocessor = CompletePreprocessor(
        sr=args.sr,
        n_mels=args.n_mels,
        split_ratio=args.split_ratio,
        fma_size=args.fma_size,
        generate_ast=not args.no_ast,
        n_jobs=args.n_jobs
    )
    
    preprocessor.run_preprocessing()
