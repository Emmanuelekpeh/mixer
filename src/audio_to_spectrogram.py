#!/usr/bin/env python3
"""
🔊 Audio Spectrogram Utility - Storage Optimization
==================================================

This utility converts audio files to spectrograms for efficient storage and processing.
It reduces storage requirements and speeds up model inference by pre-processing audio.

Usage:
    python audio_to_spectrogram.py input_audio.wav
    python audio_to_spectrogram.py --batch /path/to/audio/directory

Features:
- Converts audio files to mel spectrograms
- Supports batch processing
- Configurable spectrogram parameters
- Optimized for AI model input
"""

import os
import sys
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default spectrogram parameters
DEFAULT_PARAMS = {
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 20,
    "fmax": 8000,
    "sr": 22050  # Target sample rate
}

class SpectrogramConverter:
    """Converts audio files to spectrograms for efficient storage"""
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize with optional custom parameters"""
        self.params = params or DEFAULT_PARAMS
        logger.info(f"Initialized spectrogram converter with {self.params['n_mels']} mel bands")
    
    def audio_to_spectrogram(self, audio_path: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Convert audio file to mel spectrogram and save as numpy file
        
        Args:
            audio_path: Path to input audio file
            output_dir: Optional directory to save output, defaults to same directory as input
            
        Returns:
            Tuple of (spectrogram_path, metadata_path)
        """
        try:
            # Load audio file
            logger.info(f"Loading audio: {audio_path}")
            y, sr = librosa.load(audio_path, sr=self.params["sr"])
            
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.params["n_fft"],
                hop_length=self.params["hop_length"],
                n_mels=self.params["n_mels"],
                fmin=self.params["fmin"],
                fmax=self.params["fmax"]
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Setup output paths
            audio_path = Path(audio_path)
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = audio_path.parent
            
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Create base filename
            base_name = audio_path.stem
            spec_path = output_path / f"{base_name}_mel_spec.npy"
            meta_path = output_path / f"{base_name}_metadata.json"
            
            # Save spectrogram as numpy file
            np.save(spec_path, mel_spec_db)
            
            # Create and save metadata
            metadata = {
                "original_file": str(audio_path),
                "duration": float(len(y) / sr),
                "sample_rate": sr,
                "n_fft": self.params["n_fft"],
                "hop_length": self.params["hop_length"],
                "n_mels": self.params["n_mels"],
                "shape": mel_spec_db.shape,
                "created_at": str(Path(spec_path).stat().st_mtime)
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create visualization for reference
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spec_db, 
                x_axis='time', 
                y_axis='mel', 
                sr=sr, 
                fmin=self.params["fmin"], 
                fmax=self.params["fmax"]
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel spectrogram - {base_name}')
            vis_path = output_path / f"{base_name}_spectrogram.png"
            plt.savefig(vis_path)
            plt.close()
            
            logger.info(f"Saved spectrogram to {spec_path}")
            logger.info(f"Saved metadata to {meta_path}")
            logger.info(f"Saved visualization to {vis_path}")
            
            # Print storage savings
            original_size = Path(audio_path).stat().st_size
            spec_size = Path(spec_path).stat().st_size
            meta_size = Path(meta_path).stat().st_size
            viz_size = Path(vis_path).stat().st_size
            total_size = spec_size + meta_size + viz_size
            savings = (1 - (total_size / original_size)) * 100
            
            logger.info(f"Storage savings: {savings:.2f}% (Original: {original_size/1024:.2f}KB, New: {total_size/1024:.2f}KB)")
            
            return str(spec_path), str(meta_path)
            
        except Exception as e:
            logger.error(f"Error converting {audio_path}: {str(e)}")
            raise
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None) -> List[Tuple[str, str]]:
        """Process all audio files in a directory"""
        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f"**/*{ext}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files in {input_dir}")
        
        results = []
        for audio_file in audio_files:
            try:
                spec_path, meta_path = self.audio_to_spectrogram(str(audio_file), output_dir)
                results.append((spec_path, meta_path))
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {str(e)}")
        
        return results
    
    @staticmethod
    def load_spectrogram(spec_path: str) -> np.ndarray:
        """Load a spectrogram from a numpy file"""
        return np.load(spec_path)
    
    @staticmethod
    def load_metadata(meta_path: str) -> Dict:
        """Load metadata from a JSON file"""
        with open(meta_path, 'r') as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Convert audio files to spectrograms for efficient storage")
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--batch", "-b", action="store_true", help="Process input as a directory of audio files")
    parser.add_argument("--n_fft", type=int, default=DEFAULT_PARAMS["n_fft"], help="FFT window size")
    parser.add_argument("--hop_length", type=int, default=DEFAULT_PARAMS["hop_length"], help="Hop length")
    parser.add_argument("--n_mels", type=int, default=DEFAULT_PARAMS["n_mels"], help="Number of mel bands")
    parser.add_argument("--fmin", type=int, default=DEFAULT_PARAMS["fmin"], help="Minimum frequency")
    parser.add_argument("--fmax", type=int, default=DEFAULT_PARAMS["fmax"], help="Maximum frequency")
    
    args = parser.parse_args()
    
    # Create custom parameters
    params = {
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "sr": DEFAULT_PARAMS["sr"]
    }
    
    converter = SpectrogramConverter(params)
    
    if args.batch or os.path.isdir(args.input):
        converter.process_directory(args.input, args.output)
    else:
        if not os.path.isfile(args.input):
            logger.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        converter.audio_to_spectrogram(args.input, args.output)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
