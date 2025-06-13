import json
import numpy as np
from pathlib import Path
import librosa
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from audio_processing import list_tracks, load_stems

# Configuration for target generation
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGETS_FILE = DATA_DIR / "targets_generated.json"

def extract_audio_features(audio, sr=44100):
    """Extract basic audio features that could correlate with mixing parameters."""
    # RMS energy (correlates with compression/gain)
    rms = librosa.feature.rms(y=audio)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # Spectral centroid (correlates with EQ brightness)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    centroid_mean = np.mean(spectral_centroid)
    
    # Zero crossing rate (correlates with presence of high frequencies)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_mean = np.mean(zcr)
    
    # Dynamic range
    dynamic_range = np.max(np.abs(audio)) - np.mean(np.abs(audio))
    
    # Spectral rolloff (frequency where 85% of energy is contained)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    rolloff_mean = np.mean(rolloff)
    
    return {
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'centroid_mean': centroid_mean,
        'zcr_mean': zcr_mean,
        'dynamic_range': dynamic_range,
        'rolloff_mean': rolloff_mean
    }

def audio_features_to_mixing_params(features):
    """Convert audio features to plausible mixing parameters."""
    # Normalize features to [0, 1] range using heuristic scaling
    rms_norm = np.clip(features['rms_mean'] * 10, 0, 1)  # Gain/compression
    centroid_norm = np.clip(features['centroid_mean'] / 8000, 0, 1)  # High-freq EQ
    zcr_norm = np.clip(features['zcr_mean'] * 50, 0, 1)  # Presence/clarity
    dynamic_norm = np.clip(features['dynamic_range'] * 5, 0, 1)  # Compression ratio
    rolloff_norm = np.clip(features['rolloff_mean'] / 10000, 0, 1)  # Overall brightness
    
    # Generate 10 mixing parameters based on features
    mixing_params = [
        rms_norm,  # Input gain
        1.0 - dynamic_norm,  # Compression ratio (inverse of dynamic range)
        centroid_norm,  # High-freq EQ boost
        0.5 + 0.5 * (zcr_norm - 0.5),  # Mid-freq EQ
        0.3 + 0.4 * rms_norm,  # Low-freq EQ (based on energy)
        rolloff_norm,  # Presence/air frequencies
        0.2 + 0.6 * dynamic_norm,  # Reverb send (more for dynamic sources)
        0.1 + 0.3 * (1 - rms_norm),  # Delay send (less for loud sources)
        0.5 + 0.3 * centroid_norm,  # Stereo width
        0.7 + 0.3 * rms_norm  # Output level
    ]
    
    return mixing_params

def generate_targets_for_split(split="train"):
    """Generate mixing targets for all tracks in a split."""
    targets = {}
    tracks = list_tracks(split)
    
    print(f"Generating targets for {len(tracks)} tracks in {split} split...")
    
    for track_dir in tracks:
        track_name = track_dir.name
        print(f"Processing {track_name}...")
        
        try:
            stems = load_stems(track_dir, sr=22050)  # Lower sample rate for faster processing
            
            # Process each stem and compute mixing parameters
            stem_params = []
            for stem_name, audio in stems.items():
                if len(audio) > 0:  # Skip empty audio
                    features = extract_audio_features(audio, sr=22050)
                    params = audio_features_to_mixing_params(features)
                    stem_params.append(params)
            
            if stem_params:
                # Average parameters across all stems for the track
                track_params = np.mean(stem_params, axis=0).tolist()
                # Ensure all values are in [0, 1] range
                track_params = [max(0, min(1, param)) for param in track_params]
                targets[track_name] = track_params
            else:
                # Fallback to random values if no valid stems
                targets[track_name] = np.random.uniform(0, 1, 10).tolist()
                
        except Exception as e:
            print(f"Error processing {track_name}: {e}")
            # Fallback to random values
            targets[track_name] = np.random.uniform(0, 1, 10).tolist()
    
    return targets

def main():
    """Generate targets for all splits and save to JSON."""
    all_targets = {}
    
    # Generate targets for all splits
    for split in ["train", "val", "test"]:
        split_targets = generate_targets_for_split(split)
        all_targets.update(split_targets)
    
    # Save to file
    with open(TARGETS_FILE, 'w') as f:
        json.dump(all_targets, f, indent=2)
    
    print(f"Generated {len(all_targets)} target entries and saved to {TARGETS_FILE}")
    
    # Update baseline_cnn.py and ast_regressor.py to use the new targets file
    update_targets_path()

def update_targets_path():
    """Update the TARGETS_FILE path in model files to use generated targets."""
    files_to_update = [
        "src/baseline_cnn.py",
        "src/ast_regressor.py", 
        "src/train_compare.py"
    ]
    
    for file_path in files_to_update:
        full_path = Path(__file__).resolve().parent.parent / file_path
        if full_path.exists():
            content = full_path.read_text()
            content = content.replace(
                'TARGETS_FILE = Path(__file__).resolve().parent.parent / "data" / "targets_example.json"',
                'TARGETS_FILE = Path(__file__).resolve().parent.parent / "data" / "targets_generated.json"'
            )
            full_path.write_text(content)
            print(f"Updated {file_path} to use generated targets")

if __name__ == "__main__":
    main()
