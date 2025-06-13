import os
import shutil
import random
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "train"
VAL_DIR = Path(__file__).resolve().parent.parent / "data" / "val"
SPECTRO_TRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "spectrograms" / "train"
SPECTRO_VAL_DIR = Path(__file__).resolve().parent.parent / "data" / "spectrograms" / "val"

VAL_DIR.mkdir(exist_ok=True)
SPECTRO_VAL_DIR.mkdir(parents=True, exist_ok=True)

# List all track folders in train (ignore files)
track_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]

# Select 10% of tracks for validation
random.seed(42)
n_val = max(1, int(0.1 * len(track_dirs)))
val_tracks = random.sample(track_dirs, n_val)

print(f"Moving {n_val} tracks to validation set:")
for track_dir in val_tracks:
    print(f"- {track_dir.name}")
    # Move audio folder
    dest = VAL_DIR / track_dir.name
    shutil.move(str(track_dir), str(dest))
    # Move spectrogram folder if exists
    spec_src = SPECTRO_TRAIN_DIR / track_dir.name
    spec_dest = SPECTRO_VAL_DIR / track_dir.name
    if spec_src.exists():
        shutil.move(str(spec_src), str(spec_dest)) 