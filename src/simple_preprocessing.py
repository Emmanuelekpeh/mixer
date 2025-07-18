#!/usr/bin/env python3
"""
🎵 Simple Audio Preprocessing Pipeline
=======================================

This script handles the entire preprocessing workflow without AST dependencies:
1. Process FMA dataset audio files (already downloaded)
2. Create spectrograms with optimal parameters for all model types
3. Generate dataset splits (train/validation/test)
4. Create MFCC features (alternative to AST features)
5. Set up metadata for all models
6. Prepare directory structure for training

Usage:
    python simple_preprocessing.py
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from pathlib import Path
from tqdm import tqdm
import json
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import warnings
warnings.filterwarnings('ignore')

def simple_preprocessing():
    """Run a simple preprocessing pipeline without AST dependencies."""
    print("🚀 Starting simple preprocessing pipeline...")
    
    # Parameters
    base_dir = Path('data')
    sr = 22050
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    split_ratio = 0.8
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # Set up directories
    dirs = [
        base_dir / 'train',
        base_dir / 'val', 
        base_dir / 'test',
        base_dir / 'spectrograms' / 'train',
        base_dir / 'spectrograms' / 'val',
        base_dir / 'spectrograms' / 'test',
        base_dir / 'features' / 'train',
        base_dir / 'features' / 'val',
        base_dir / 'features' / 'test'
    ]
    
    for d in dirs:
        d.mkdir(exist_ok=True, parents=True)
    
    # Find FMA audio files
    fma_dir = base_dir / 'raw' / 'music' / 'fma' / 'fma_small'
    print(f'Finding audio files in {fma_dir}...')
    
    if not fma_dir.exists():
        print(f'FMA directory not found at {fma_dir}. Please download the FMA dataset first.')
        sys.exit(1)
    
    audio_files = list(fma_dir.glob('**/*.mp3'))
    print(f'Found {len(audio_files)} audio files')
    
    # Split into train/val/test
    random.seed(42)
    random.shuffle(audio_files)
    train_size = int(len(audio_files) * split_ratio)
    val_size = int((len(audio_files) - train_size) / 2)
    train_files = audio_files[:train_size]
    val_files = audio_files[train_size:train_size + val_size]
    test_files = audio_files[train_size + val_size:]
    print(f'Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test')
    
    # Process a single file
    def process_file(audio_file, output_dir, spec_dir, feature_dir):
        try:
            track_id = audio_file.stem
            output_path = output_dir / f'{track_id}.wav'
            spec_path = spec_dir / f'{track_id}.npy'
            mfcc_path = feature_dir / f'{track_id}_mfcc.npy'
            
            # Skip if already processed
            if output_path.exists() and spec_path.exists():
                return track_id, {'status': 'already_processed'}
            
            # Load and normalize audio
            audio, _ = librosa.load(audio_file, sr=sr, mono=True)
            if len(audio) < sr:  # Skip very short clips
                return track_id, {'status': 'too_short'}
            
            audio = librosa.util.normalize(audio)
            
            # Save preprocessed audio
            sf.write(output_path, audio, sr)
            
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            np.save(spec_path, S_db)
            
            # Generate MFCC features (good alternative to AST)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            np.save(mfcc_path, mfcc_features)
            
            return track_id, {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'n_samples': len(audio),
                'status': 'success'
            }
        except Exception as e:
            return audio_file.stem, {'status': f'error: {e}'}
    
    # Process files in parallel
    all_track_info = {}
    total_duration = 0
    
    for name, files, out_dir, spec_dir, feat_dir in [
        ('Training', train_files, base_dir / 'train', base_dir / 'spectrograms' / 'train', base_dir / 'features' / 'train'),
        ('Validation', val_files, base_dir / 'val', base_dir / 'spectrograms' / 'val', base_dir / 'features' / 'val'),
        ('Test', test_files, base_dir / 'test', base_dir / 'spectrograms' / 'test', base_dir / 'features' / 'test')
    ]:
        print(f'\nProcessing {name} set...')
        
        # Create partial function
        process_fn = partial(process_file, output_dir=out_dir, spec_dir=spec_dir, feature_dir=feat_dir)
        
        # Process files with progress bar
        results = {}
        errors = []
          with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(process_fn, f): f for f in files}
            
            for i, future in enumerate(tqdm(as_completed(futures), total=len(files), desc=f'Processing {name} files')):
                file = futures[future]
                try:
                    track_id, info = future.result()
                    if info.get('status') == 'success':
                        results[track_id] = info
                        total_duration += info['duration']
                    elif info.get('status').startswith('error'):
                        errors.append((track_id, info['status']))
                except Exception as e:
                    errors.append((file.stem, f"processing_error: {e}"))
        
        print(f'Processed {len(results)} files successfully')
        if errors:
            print(f'Encountered {len(errors)} errors')
        
        all_track_info.update(results)
    
    # Generate fake mixing parameters
    print('\nGenerating synthetic mixing targets...')
    targets = {}
    param_names = [
        'Input Gain', 'Compression Ratio', 'Compression Attack', 'Compression Release',
        'Low Shelf (80Hz)', 'Low Mid (200Hz)', 'Mid (1kHz)', 'High Mid (4kHz)', 
        'High Shelf (12kHz)', 'Presence (8kHz)', 'Reverb Send', 'Reverb Type',
        'Delay Send', 'Delay Time', 'Stereo Width', 'Bass Mono', 'Output Level'
    ]
    
    for track_id in all_track_info:
        # Generate 17 parameters with realistic constraints
        params = []
        params.append(0.6 + 0.3 * random.random())  # Input Gain
        params.append(0.1 + 0.6 * random.random())  # Compression Ratio
        params.append(0.2 + 0.6 * random.random())  # Compression Attack
        params.append(0.3 + 0.4 * random.random())  # Compression Release
        for _ in range(5):  # EQ parameters
            params.append(0.2 + 0.6 * random.random())
        params.append(0.3 + 0.4 * random.random())  # Presence
        params.append(0.1 + 0.4 * random.random())  # Reverb Send
        params.append(random.random())             # Reverb Type
        params.append(0.05 + 0.25 * random.random()) # Delay Send
        params.append(random.random())             # Delay Time
        params.append(0.4 + 0.4 * random.random())  # Stereo Width
        params.append(0.6 + 0.4 * random.random())  # Bass Mono
        params.append(0.7 + 0.25 * random.random()) # Output Level
        targets[track_id] = params
    
    # Save targets
    targets_file = base_dir / 'targets_generated.json'
    with open(targets_file, 'w') as f:
        json.dump(targets, f, indent=2)
    
    # Save metadata
    metadata = {
        'preprocessing_info': {
            'date': time.strftime('%Y-%m-%d'),
            'sample_rate': sr,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'split_ratio': split_ratio
        },
        'dataset_stats': {
            'total_tracks': len(all_track_info),
            'train_tracks': len(train_files),
            'val_tracks': len(val_files),
            'test_tracks': len(test_files),
            'total_duration_hours': total_duration / 3600
        },
        'track_info': all_track_info
    }
    
    metadata_file = base_dir / 'preprocessing_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print('\n✅ Preprocessing complete!')
    print(f'⏱️  Total time: {time.time() - time.time():.1f} seconds')
    print(f'🎵 Processed {len(all_track_info)} tracks ({total_duration / 3600:.1f} hours)')
    print(f'📊 Dataset splits: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test')
    print(f'📁 Metadata saved to: {metadata_file}')
    print(f'🎚️ Mixing targets saved to: {targets_file}')
    print(f'🧠 MFCC features generated for all tracks (alternative to AST)')

if __name__ == "__main__":
    start_time = time.time()
    simple_preprocessing()
    print(f"\nTotal preprocessing time: {time.time() - start_time:.1f} seconds")
