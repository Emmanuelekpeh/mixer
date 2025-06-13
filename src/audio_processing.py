import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

MUSDB18_FOLDER = Path(__file__).resolve().parent.parent / "data"
SPECTROGRAMS_FOLDER = Path(__file__).resolve().parent.parent / "data" / "spectrograms"


def list_tracks(split="train"):
    """List all tracks in the MUSDB18 split (train/test)."""
    split_dir = MUSDB18_FOLDER / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} not found.")
    return [d for d in split_dir.iterdir() if d.is_dir()]


def load_stems(track_dir, sr=44100):
    """Load all stems (wav files) from a track directory. Returns a dict of stem_name: audio_array."""
    stems = {}
    for file in track_dir.glob("*.wav"):
        y, _ = librosa.load(file, sr=sr, mono=True)
        stems[file.stem] = y
    return stems


def audio_to_melspectrogram(y, sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """Convert audio array to mel-spectrogram (dB)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def save_spectrogram(spectrogram, out_path):
    """Save spectrogram as a numpy file."""
    np.save(out_path, spectrogram)


def process_and_save_all_spectrograms(split="train", sr=44100, n_fft=2048, hop_length=512, n_mels=128):
    """Process all tracks in a split, convert stems to mel-spectrograms, and save them."""
    SPECTROGRAMS_FOLDER.mkdir(exist_ok=True)
    tracks = list_tracks(split)
    for track_dir in tracks:
        stems = load_stems(track_dir, sr=sr)
        for stem_name, y in stems.items():
            S_db = audio_to_melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            out_dir = SPECTROGRAMS_FOLDER / split / track_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem_name}.npy"
            save_spectrogram(S_db, out_path)
            print(f"Saved spectrogram: {out_path}")

if __name__ == "__main__":
    # Process both train and test splits
    process_and_save_all_spectrograms(split="train")
    process_and_save_all_spectrograms(split="test") 