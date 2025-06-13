import unittest
import numpy as np
from pathlib import Path
from src import audio_processing

class TestAudioProcessing(unittest.TestCase):
    def setUp(self):
        # Create a mock directory structure and a dummy wav file if needed
        self.test_dir = Path(__file__).resolve().parent.parent / "data" / "musdb18" / "train" / "dummy_track"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        # Generate a dummy audio file
        self.dummy_wav = self.test_dir / "vocals.wav"
        sr = 22050
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 220 * t)
        import soundfile as sf
        sf.write(self.dummy_wav, y, sr)

    def test_list_tracks(self):
        tracks = audio_processing.list_tracks(split="train")
        self.assertTrue(any("dummy_track" in str(t) for t in tracks))

    def test_load_stems(self):
        stems = audio_processing.load_stems(self.test_dir, sr=22050)
        self.assertIn("vocals", stems)
        self.assertIsInstance(stems["vocals"], np.ndarray)

    def test_audio_to_melspectrogram(self):
        stems = audio_processing.load_stems(self.test_dir, sr=22050)
        y = stems["vocals"]
        S_db = audio_processing.audio_to_melspectrogram(y, sr=22050)
        self.assertIsInstance(S_db, np.ndarray)
        self.assertTrue(S_db.shape[0] > 0 and S_db.shape[1] > 0)

    def tearDown(self):
        # Remove dummy files and directories
        if self.dummy_wav.exists():
            self.dummy_wav.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
        train_dir = self.test_dir.parent
        if train_dir.exists() and not any(train_dir.iterdir()):
            train_dir.rmdir()

if __name__ == "__main__":
    unittest.main() 