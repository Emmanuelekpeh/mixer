import os
from pathlib import Path
import shutil

# Organize .wav files in data/train and data/test into per-track subfolders

def organize_audio_files(split="train"):
    data_dir = Path(__file__).resolve().parent.parent / "data" / split
    for wav_file in data_dir.glob("*.wav"):
        # Use the base name (before first '-') as the track name, or full stem if no dash
        base = wav_file.stem
        if ' - ' in base:
            track_name = base.split(' - ')[0].strip()
        else:
            track_name = base
        track_dir = data_dir / track_name
        track_dir.mkdir(exist_ok=True)
        dest = track_dir / wav_file.name
        shutil.move(str(wav_file), str(dest))
        print(f"Moved {wav_file} -> {dest}")

if __name__ == "__main__":
    organize_audio_files("train")
    organize_audio_files("test") 