@echo off
echo ===============================================
echo    AI MIXER - Complete Training Pipeline
echo ===============================================

echo.
echo This script will:
echo 1. Preprocess the FMA dataset for all models
echo 2. Train all 8 models sequentially
echo 3. Generate evaluation metrics
echo.

set PYTHON_CMD=python

REM Check if Python is installed
where %PYTHON_CMD% >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Make sure Python is installed and in your PATH.
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
%PYTHON_CMD% -c "import torch, librosa, soundfile, tqdm, numpy, matplotlib" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Some required packages are missing. Installing dependencies...
    %PYTHON_CMD% -m pip install -r requirements.txt
)

echo.
echo -----------------------------------------------
echo    Step 1: Preprocessing FMA Dataset
echo -----------------------------------------------
echo.

echo Running basic preprocessing for all models...
echo.
echo This will:
echo - Create mel spectrograms for all models
echo - Generate MFCC features (instead of AST features that have compatibility issues)
echo - Create train/validation/test splits
echo - Generate synthetic mixing parameters
echo.

REM Using custom preprocessing command without AST dependency
%PYTHON_CMD% -c "import os, sys, numpy as np, pandas as pd, librosa, soundfile as sf, torch, json, random, time, warnings; from pathlib import Path; from tqdm import tqdm; from concurrent.futures import ProcessPoolExecutor; from functools import partial; import multiprocessing; warnings.filterwarnings('ignore'); print('Starting custom preprocessing...');

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
    print(f'\\nProcessing {name} set...')
    
    # Create partial function
    process_fn = partial(process_file, output_dir=out_dir, spec_dir=spec_dir, feature_dir=feat_dir)
    
    # Process files with progress bar
    results = {}
    errors = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(process_fn, f): f for f in files}
        
        for i, future in enumerate(tqdm(futures, desc=f'Processing {name} files')):
            track_id, info = future.result()
            if info.get('status') == 'success':
                results[track_id] = info
                total_duration += info['duration']
            elif info.get('status').startswith('error'):
                errors.append((track_id, info['status']))
    
    print(f'Processed {len(results)} files successfully')
    if errors:
        print(f'Encountered {len(errors)} errors')
    
    all_track_info.update(results)

# Generate fake mixing parameters
print('\\nGenerating synthetic mixing targets...')
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

print('\\n✅ Preprocessing complete!')
print(f'⏱️  Total time: {time.time() - time.time():.1f} seconds')
print(f'🎵 Processed {len(all_track_info)} tracks ({total_duration / 3600:.1f} hours)')
print(f'📊 Dataset splits: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test')
print(f'📁 Metadata saved to: {metadata_file}')
print(f'🎚️ Mixing targets saved to: {targets_file}')
print(f'🧠 MFCC features generated for all tracks (alternative to AST)')"

if %ERRORLEVEL% NEQ 0 (
    echo Error during preprocessing. Please check the logs above.
    exit /b 1
)

echo.
echo -----------------------------------------------
echo    Step 2: Training All Models
echo -----------------------------------------------
echo.

echo Training models (this may take a while)...
%PYTHON_CMD% src\train_all_models.py --models=all --epochs=10 --batch-size=16

if %ERRORLEVEL% NEQ 0 (
    echo Error during training. Please check the logs above.
    exit /b 1
)

echo.
echo ===============================================
echo    Training Complete!
echo ===============================================
echo.
echo All models have been trained and evaluated.
echo Results are available in the training_results directory.
echo.
echo To use the models for mixing:
echo   python demo_ai_mixer.py path\to\your\audio.wav
echo.

pause
