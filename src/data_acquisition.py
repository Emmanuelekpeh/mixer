import os
import requests
import zipfile
from pathlib import Path

MUSDB18_URL = "https://zenodo.org/record/1117372/files/musdb18.zip?download=1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MUSDB18_ZIP = DATA_DIR / "musdb18.zip"
MUSDB18_FOLDER = DATA_DIR / "musdb18"


def download_musdb18():
    """Download the MUSDB18 dataset zip file if not already present."""
    DATA_DIR.mkdir(exist_ok=True)
    if MUSDB18_ZIP.exists():
        print("MUSDB18 zip already exists.")
        return
    print(f"Downloading MUSDB18 from {MUSDB18_URL}...")
    response = requests.get(MUSDB18_URL, stream=True)
    response.raise_for_status()
    with open(MUSDB18_ZIP, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

def extract_musdb18():
    """Extract the MUSDB18 zip file if not already extracted."""
    if MUSDB18_FOLDER.exists():
        print("MUSDB18 already extracted.")
        return
    print(f"Extracting {MUSDB18_ZIP}...")
    with zipfile.ZipFile(MUSDB18_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

def verify_musdb18():
    """Verify that the MUSDB18 dataset is present and contains expected files."""
    if not MUSDB18_FOLDER.exists():
        print("MUSDB18 folder not found.")
        return False
    # Check for a few expected subfolders/files
    expected = ["train", "test"]
    for sub in expected:
        if not (MUSDB18_FOLDER / sub).exists():
            print(f"Missing {sub} folder in MUSDB18.")
            return False
    print("MUSDB18 dataset verified.")
    return True

if __name__ == "__main__":
    download_musdb18()
    extract_musdb18()
    verify_musdb18() 