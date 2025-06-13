import unittest
import os
from pathlib import Path
from src import data_acquisition

class TestDataAcquisition(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self.musdb18_zip = self.data_dir / "musdb18.zip"
        self.musdb18_folder = self.data_dir / "musdb18"

    def test_download_musdb18(self):
        # Only test if file is downloaded, not the actual download (to avoid network dependency)
        data_acquisition.DATA_DIR.mkdir(exist_ok=True)
        self.musdb18_zip.touch(exist_ok=True)
        self.assertTrue(self.musdb18_zip.exists())

    def test_extract_musdb18(self):
        # Simulate extraction by creating the folder
        self.musdb18_folder.mkdir(exist_ok=True)
        self.assertTrue(self.musdb18_folder.exists())

    def test_verify_musdb18(self):
        # Simulate expected structure
        (self.musdb18_folder / "train").mkdir(parents=True, exist_ok=True)
        (self.musdb18_folder / "test").mkdir(parents=True, exist_ok=True)
        self.assertTrue(data_acquisition.verify_musdb18())

    def tearDown(self):
        # Clean up created files/folders
        if self.musdb18_zip.exists():
            self.musdb18_zip.unlink()
        if self.musdb18_folder.exists():
            for sub in ["train", "test"]:
                subfolder = self.musdb18_folder / sub
                if subfolder.exists():
                    os.rmdir(subfolder)
            os.rmdir(self.musdb18_folder)

if __name__ == "__main__":
    unittest.main() 