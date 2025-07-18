#!/usr/bin/env python3
"""
🎵 Enhanced Audio Processing Pipeline
===================================

This script processes the downloaded datasets for AI mixing:
- Normalizes audio formats and sample rates
- Extracts spectrograms and features
- Applies initial preprocessing
- Creates standardized metadata

The goal is to prepare a production-grade dataset with high quality
and consistency for training advanced AI mixing models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings
warnings.filterwarnings('ignore')

class EnhancedAudioProcessor:
    """Process raw audio files into standardized formats for AI mixing."""
    
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
        self.clean_music_dir = self.clean_dir / "music"
        self.clean_vocals_dir = self.clean_dir / "vocals"
        self.clean_acoustics_dir = self.clean_dir / "acoustics"
        
        # Feature directories
        self.features_dir = self.base_dir / "features"
        self.spectrograms_dir = self.features_dir / "spectrograms"
        self.ast_features_dir = self.features_dir / "ast_features"
        
        # Metadata directory
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create all directories
        for directory in [
            self.clean_music_dir, self.clean_vocals_dir, self.clean_acoustics_dir,
            self.spectrograms_dir, self.ast_features_dir
        ]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Standard processing parameters
        self.target_sr = 44100  # Standard sample rate
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length
        self.n_mels = 128  # Number of mel bands
        
        # Initialize metadata
        self.metadata = {
            "processed_tracks": {
                "music": [],
                "vocals": [],
                "acoustics": []
            },
            "processing_stats": {
                "total_tracks": 0,
                "successful": 0,
                "failed": 0,
                "duration_total": 0,
                "duration_average": 0
            },
            "feature_extraction": {
                "spectrograms": 0,
                "ast_features": 0
            }
        }
        
        # Load existing metadata if available
        metadata_path = self.metadata_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.dataset_metadata = json.load(f)
        else:
            self.dataset_metadata = {"datasets": {}}
    
    def process_audio_file(self, file_path, output_subdir=None, target_duration=None):
        """
        Process a single audio file to standard format.
        
        Args:
            file_path: Path to the audio file
            output_subdir: Subdirectory for output (music, vocals, acoustics)
            target_duration: Optional duration to trim/pad to (in seconds)
            
        Returns:
            Dict with processing results
        """
        file_path = Path(file_path)
        
        if output_subdir is None:
            # Determine output subdir from file path
            if "fma" in str(file_path).lower():
                output_subdir = "music"
            elif "damp" in str(file_path).lower():
                output_subdir = "vocals"
            elif "impulse" in str(file_path).lower() or "rir" in str(file_path).lower():
                output_subdir = "acoustics"
            else:
                output_subdir = "other"
        
        # Set output directory
        if output_subdir == "music":
            output_dir = self.clean_music_dir
        elif output_subdir == "vocals":
            output_dir = self.clean_vocals_dir
        elif output_subdir == "acoustics":
            output_dir = self.clean_acoustics_dir
        else:
            output_dir = self.clean_dir / output_subdir
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create output filename
        output_filename = f"{file_path.stem}_standardized.wav"
        output_path = output_dir / output_filename
        
        # Skip if already processed
        if output_path.exists():
            return {
                "file": str(file_path),
                "output": str(output_path),
                "status": "skipped",
                "sr": self.target_sr,
                "duration": None
            }
        
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=False)
            
            # Convert to stereo if mono
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            # Get original duration
            duration = audio.shape[1] / self.target_sr
            
            # Trim or pad to target duration if specified
            if target_duration is not None:
                target_samples = int(target_duration * self.target_sr)
                if audio.shape[1] > target_samples:
                    # Trim to target duration
                    audio = audio[:, :target_samples]
                elif audio.shape[1] < target_samples:
                    # Pad with silence
                    padding = np.zeros((2, target_samples - audio.shape[1]))
                    audio = np.concatenate([audio, padding], axis=1)
            
            # Normalize amplitude
            max_amp = np.max(np.abs(audio))
            if max_amp > 0:
                audio = audio / max_amp * 0.9  # Normalize to 90% of max amplitude
            
            # Save processed audio
            sf.write(output_path, audio.T, self.target_sr)
            
            return {
                "file": str(file_path),
                "output": str(output_path),
                "status": "success",
                "sr": self.target_sr,
                "duration": duration,
                "channels": audio.shape[0],
                "samples": audio.shape[1],
                "category": output_subdir
            }
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
            return {
                "file": str(file_path),
                "status": "failed",
                "error": str(e)
            }
    
    def extract_spectrogram(self, file_path, output_dir=None):
        """
        Extract mel spectrogram from audio file.
        
        Args:
            file_path: Path to audio file
            output_dir: Directory to save the spectrogram
            
        Returns:
            Dict with extraction results
        """
        file_path = Path(file_path)
        
        if output_dir is None:
            output_dir = self.spectrograms_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = output_dir / f"{file_path.stem}_mel.npy"
        
        # Skip if already extracted
        if output_path.exists():
            return {
                "file": str(file_path),
                "spectrogram": str(output_path),
                "status": "skipped"
            }
        
        try:
            # Load audio file (convert to mono for spectrogram)
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_fft=self.n_fft, 
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Save spectrogram
            np.save(output_path, mel_spec_db)
            
            return {
                "file": str(file_path),
                "spectrogram": str(output_path),
                "status": "success",
                "shape": mel_spec_db.shape,
                "min": np.min(mel_spec_db),
                "max": np.max(mel_spec_db),
                "mean": np.mean(mel_spec_db)
            }
            
        except Exception as e:
            print(f"❌ Error extracting spectrogram from {file_path.name}: {e}")
            return {
                "file": str(file_path),
                "status": "failed",
                "error": str(e)
            }
    
    def extract_ast_features(self, file_path, output_dir=None):
        """
        Extract Audio Spectrogram Transformer features.
        
        Args:
            file_path: Path to audio file
            output_dir: Directory to save the features
            
        Returns:
            Dict with extraction results
        """
        file_path = Path(file_path)
        
        if output_dir is None:
            output_dir = self.ast_features_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = output_dir / f"{file_path.stem}_ast.npy"
        
        # Skip if already extracted
        if output_path.exists():
            return {
                "file": str(file_path),
                "ast_features": str(output_path),
                "status": "skipped"
            }
        
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            
            # Extract various audio features
            # Note: This is a simplified version of AST features
            features = {
                # Spectral features
                "mfcc": librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20),
                "spectral_centroid": librosa.feature.spectral_centroid(y=audio, sr=sr),
                "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=audio, sr=sr),
                "spectral_contrast": librosa.feature.spectral_contrast(y=audio, sr=sr),
                "spectral_rolloff": librosa.feature.spectral_rolloff(y=audio, sr=sr),
                
                # Rhythm features
                "tempogram": librosa.feature.tempogram(y=audio, sr=sr),
                
                # Amplitude features
                "rms": librosa.feature.rms(y=audio),
                
                # Onset features
                "onset_env": librosa.onset.onset_strength(y=audio, sr=sr),
                
                # Pitch features
                "pitch_tuning": librosa.piptrack(y=audio, sr=sr)[1],
            }
            
            # Combine features
            feature_list = []
            for name, feature in features.items():
                # Subsample or truncate to a standard size
                if feature.shape[1] > 1000:
                    indices = np.linspace(0, feature.shape[1]-1, 1000, dtype=int)
                    feature = feature[:, indices]
                
                feature_list.append(feature)
            
            # Save features
            np.save(output_path, feature_list)
            
            return {
                "file": str(file_path),
                "ast_features": str(output_path),
                "status": "success",
                "features": list(features.keys())
            }
            
        except Exception as e:
            print(f"❌ Error extracting AST features from {file_path.name}: {e}")
            return {
                "file": str(file_path),
                "status": "failed",
                "error": str(e)
            }
    
    def process_fma_dataset(self, subset_size=None):
        """
        Process the FMA dataset.
        
        Args:
            subset_size: Number of tracks to process (None = all)
            
        Returns:
            Dict with processing results
        """
        print(f"🎵 Processing FMA dataset...")
        
        # Look for FMA tracks
        fma_tracks = []
        for ext in ['.mp3', '.wav']:
            fma_tracks.extend(list(self.fma_dir.glob(f"**/*{ext}")))
        
        # Limit subset size if specified
        if subset_size is not None and subset_size < len(fma_tracks):
            print(f"⚠️ Limiting to {subset_size} tracks (total: {len(fma_tracks)})")
            fma_tracks = fma_tracks[:subset_size]
        
        # Process each track
        print(f"🔄 Processing {len(fma_tracks)} FMA tracks...")
        
        results = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            process_func = partial(self.process_audio_file, output_subdir="music", target_duration=30)
            for result in tqdm(executor.map(process_func, fma_tracks), total=len(fma_tracks)):
                results.append(result)
        
        # Count successes and failures
        success_count = sum(1 for r in results if r.get('status') == 'success')
        skip_count = sum(1 for r in results if r.get('status') == 'skipped')
        fail_count = sum(1 for r in results if r.get('status') == 'failed')
        
        print(f"✅ Processed FMA tracks: {success_count} successes, {skip_count} skipped, {fail_count} failures")
        
        # Add to metadata
        self.metadata["processed_tracks"]["music"].extend([r for r in results if r.get('status') in ('success', 'skipped')])
        self.metadata["processing_stats"]["total_tracks"] += len(results)
        self.metadata["processing_stats"]["successful"] += success_count
        self.metadata["processing_stats"]["failed"] += fail_count
        
        return results
    
    def process_damp_dataset(self, subset_size=None):
        """
        Process the DAMP karaoke dataset.
        
        Args:
            subset_size: Number of tracks to process (None = all)
            
        Returns:
            Dict with processing results
        """
        print(f"🎤 Processing DAMP dataset...")
        
        # Look for DAMP tracks
        damp_tracks = []
        for ext in ['.mp3', '.wav']:
            damp_tracks.extend(list(self.damp_dir.glob(f"**/*{ext}")))
        
        # Limit subset size if specified
        if subset_size is not None and subset_size < len(damp_tracks):
            print(f"⚠️ Limiting to {subset_size} tracks (total: {len(damp_tracks)})")
            damp_tracks = damp_tracks[:subset_size]
        
        # Process each track
        print(f"🔄 Processing {len(damp_tracks)} DAMP tracks...")
        
        results = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            process_func = partial(self.process_audio_file, output_subdir="vocals", target_duration=20)
            for result in tqdm(executor.map(process_func, damp_tracks), total=len(damp_tracks)):
                results.append(result)
        
        # Count successes and failures
        success_count = sum(1 for r in results if r.get('status') == 'success')
        skip_count = sum(1 for r in results if r.get('status') == 'skipped')
        fail_count = sum(1 for r in results if r.get('status') == 'failed')
        
        print(f"✅ Processed DAMP tracks: {success_count} successes, {skip_count} skipped, {fail_count} failures")
        
        # Add to metadata
        self.metadata["processed_tracks"]["vocals"].extend([r for r in results if r.get('status') in ('success', 'skipped')])
        self.metadata["processing_stats"]["total_tracks"] += len(results)
        self.metadata["processing_stats"]["successful"] += success_count
        self.metadata["processing_stats"]["failed"] += fail_count
        
        return results
    
    def process_room_impulse_responses(self):
        """
        Process room impulse response files.
        
        Returns:
            Dict with processing results
        """
        print(f"🔊 Processing room impulse responses...")
        
        # Look for impulse response files (typically WAV)
        ir_files = list(self.rir_dir.glob("**/*.wav"))
        
        # Process each impulse response
        print(f"🔄 Processing {len(ir_files)} impulse responses...")
        
        results = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            process_func = partial(self.process_audio_file, output_subdir="acoustics")
            for result in tqdm(executor.map(process_func, ir_files), total=len(ir_files)):
                results.append(result)
        
        # Count successes and failures
        success_count = sum(1 for r in results if r.get('status') == 'success')
        skip_count = sum(1 for r in results if r.get('status') == 'skipped')
        fail_count = sum(1 for r in results if r.get('status') == 'failed')
        
        print(f"✅ Processed impulse responses: {success_count} successes, {skip_count} skipped, {fail_count} failures")
        
        # Add to metadata
        self.metadata["processed_tracks"]["acoustics"].extend([r for r in results if r.get('status') in ('success', 'skipped')])
        self.metadata["processing_stats"]["total_tracks"] += len(results)
        self.metadata["processing_stats"]["successful"] += success_count
        self.metadata["processing_stats"]["failed"] += fail_count
        
        return results
    
    def extract_all_features(self, processed_files=None):
        """
        Extract features from all processed audio files.
        
        Args:
            processed_files: List of processed file paths (None = auto-detect)
            
        Returns:
            Dict with extraction results
        """
        print(f"📊 Extracting features from processed audio files...")
        
        # Collect all processed files if not specified
        if processed_files is None:
            processed_files = []
            for category in ["music", "vocals"]:  # Skip acoustics for now
                cat_dir = self.clean_dir / category
                processed_files.extend(list(cat_dir.glob("**/*.wav")))
        
        # Extract spectrograms
        print(f"🔄 Extracting spectrograms from {len(processed_files)} files...")
        
        spec_results = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for result in tqdm(executor.map(self.extract_spectrogram, processed_files), total=len(processed_files)):
                spec_results.append(result)
        
        spec_success = sum(1 for r in spec_results if r.get('status') == 'success')
        spec_skip = sum(1 for r in spec_results if r.get('status') == 'skipped')
        
        print(f"✅ Extracted spectrograms: {spec_success} successes, {spec_skip} skipped")
        
        # Extract AST features
        print(f"🔄 Extracting AST features from {len(processed_files)} files...")
        
        ast_results = []
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for result in tqdm(executor.map(self.extract_ast_features, processed_files), total=len(processed_files)):
                ast_results.append(result)
        
        ast_success = sum(1 for r in ast_results if r.get('status') == 'success')
        ast_skip = sum(1 for r in ast_results if r.get('status') == 'skipped')
        
        print(f"✅ Extracted AST features: {ast_success} successes, {ast_skip} skipped")
        
        # Update metadata
        self.metadata["feature_extraction"]["spectrograms"] = spec_success + spec_skip
        self.metadata["feature_extraction"]["ast_features"] = ast_success + ast_skip
        
        return {
            "spectrograms": spec_results,
            "ast_features": ast_results
        }
    
    def save_processing_metadata(self):
        """Save processing metadata to a JSON file."""
        metadata_path = self.metadata_dir / "processing_metadata.json"
        
        print(f"💾 Saving processing metadata to {metadata_path}...")
        
        # Calculate some statistics
        if self.metadata["processing_stats"]["successful"] > 0:
            # Calculate average duration
            all_durations = [
                track.get('duration', 0) 
                for category in self.metadata["processed_tracks"].values()
                for track in category 
                if track.get('status') == 'success' and track.get('duration') is not None
            ]
            
            total_duration = sum(all_durations)
            avg_duration = total_duration / len(all_durations) if all_durations else 0
            
            self.metadata["processing_stats"]["duration_total"] = total_duration
            self.metadata["processing_stats"]["duration_average"] = avg_duration
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✅ Processing metadata saved")
        return metadata_path
    
    def run_processing(self, datasets=None, subset_size=None, extract_features=True):
        """
        Run the entire processing pipeline.
        
        Args:
            datasets: List of datasets to process. If None, processes all datasets.
                     Options: ["fma", "damp", "rir"]
            subset_size: Max number of tracks to process per dataset (None = all)
            extract_features: Whether to extract features
            
        Returns:
            Dict with processing results
        """
        if datasets is None:
            datasets = ["fma", "damp", "rir"]
        
        print(f"🚀 Starting enhanced audio processing pipeline...")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Datasets to process: {', '.join(datasets)}")
        
        results = {}
        processed_files = []
        
        # Process each dataset
        if "fma" in datasets:
            fma_results = self.process_fma_dataset(subset_size=subset_size)
            results["fma"] = fma_results
            processed_files.extend([
                Path(r["output"]) for r in fma_results 
                if r.get('status') in ('success', 'skipped') and 'output' in r
            ])
            
        if "damp" in datasets:
            damp_results = self.process_damp_dataset(subset_size=subset_size)
            results["damp"] = damp_results
            processed_files.extend([
                Path(r["output"]) for r in damp_results 
                if r.get('status') in ('success', 'skipped') and 'output' in r
            ])
            
        if "rir" in datasets:
            rir_results = self.process_room_impulse_responses()
            results["rir"] = rir_results
            # Note: We typically don't extract spectrograms/AST from impulse responses
        
        # Extract features if requested
        if extract_features and processed_files:
            feature_results = self.extract_all_features(processed_files)
            results["features"] = feature_results
        
        # Save metadata
        self.save_processing_metadata()
        
        # Summary
        print("\n📋 Processing Summary:")
        print("=" * 50)
        total_processed = self.metadata["processing_stats"]["successful"]
        total_failed = self.metadata["processing_stats"]["failed"]
        
        print(f"Total files processed: {total_processed} successes, {total_failed} failures")
        print(f"Spectrograms extracted: {self.metadata['feature_extraction']['spectrograms']}")
        print(f"AST features extracted: {self.metadata['feature_extraction']['ast_features']}")
        
        if self.metadata["processing_stats"]["duration_total"] > 0:
            total_duration = self.metadata["processing_stats"]["duration_total"]
            hours = total_duration // 3600
            minutes = (total_duration % 3600) // 60
            seconds = total_duration % 60
            
            print(f"Total audio duration: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")
            print(f"Average track duration: {self.metadata['processing_stats']['duration_average']:.1f}s")
        
        return results

# When run directly
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Audio Processing for AI Mixing")
    parser.add_argument('--datasets', nargs='+', choices=['fma', 'damp', 'rir', 'all'],
                        default=['all'], help="Datasets to process")
    parser.add_argument('--subset-size', type=int, default=None,
                        help="Max number of tracks to process per dataset")
    parser.add_argument('--skip-features', action='store_true',
                        help="Skip feature extraction")
    
    args = parser.parse_args()
    
    # Process datasets argument
    if 'all' in args.datasets:
        datasets_to_process = ['fma', 'damp', 'rir']
    else:
        datasets_to_process = args.datasets
    
    # Create processor instance
    processor = EnhancedAudioProcessor()
    
    # Run processing
    processor.run_processing(
        datasets=datasets_to_process,
        subset_size=args.subset_size,
        extract_features=not args.skip_features
    )
