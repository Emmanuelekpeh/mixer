#!/usr/bin/env python3
"""
Dataset Acquisition and Processing Script
----------------------------------------

This script helps download and process audio datasets for the AI Mixer project.
It provides instructions for obtaining datasets and processes them into the 
required format for training.
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create the necessary directory structure if it doesn't exist"""
    data_dir = Path("data")
    dirs = [
        data_dir / "raw",
        data_dir / "train",
        data_dir / "test",
        data_dir / "spectrograms"
    ]
    
    for directory in dirs:
        directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created directory: {directory}")

def download_musdb18():
    """
    Provide instructions for downloading MUSDB18 dataset
    
    Note: This doesn't actually download the dataset due to licensing,
    but provides instructions on how to obtain it.
    """
    print("\n" + "="*80)
    print("MUSDB18 Dataset Download Instructions")
    print("="*80)
    print("""
MUSDB18 is a dataset of 150 professionally mixed songs with isolated stems.
Due to licensing restrictions, this script cannot automatically download the dataset.

To obtain the dataset:

1. Visit: https://sigsep.github.io/datasets/musdb.html
2. Follow the instructions to purchase or license the dataset
3. Once downloaded, extract the contents to the 'data/raw/musdb18' directory
4. Run this script again with the --process flag to process the data:
   python dataset_acquisition.py --process musdb18

The dataset contains:
- 150 songs with full mixes and isolated stems
- Stems include: vocals, bass, drums, and other
- High-quality audio suitable for training mixing models
""")
    return False

def download_medleydb():
    """
    Provide instructions for downloading MedleyDB dataset
    
    Note: This doesn't actually download the dataset due to licensing,
    but provides instructions on how to obtain it.
    """
    print("\n" + "="*80)
    print("MedleyDB Dataset Download Instructions")
    print("="*80)
    print("""
MedleyDB is a dataset of multi-track audio recordings with annotations.
Due to licensing restrictions, this script cannot automatically download the dataset.

To obtain the dataset:

1. Visit: https://medleydb.weebly.com/
2. Follow the instructions to access or license the dataset
3. Once downloaded, extract the contents to the 'data/raw/medleydb' directory
4. Run this script again with the --process flag to process the data:
   python dataset_acquisition.py --process medleydb

The dataset contains:
- Multi-track recordings with raw stems
- Annotations for instrument and recording information
- Various musical genres suitable for diverse training
""")
    return False

def process_dataset(dataset_name):
    """
    Process the specified dataset into the format needed for training
    
    Args:
        dataset_name: Name of the dataset to process ('musdb18' or 'medleydb')
    
    Returns:
        bool: Success status
    """
    raw_dir = Path("data/raw") / dataset_name
    
    if not raw_dir.exists():
        logger.error(f"Dataset directory not found: {raw_dir}")
        logger.error("Please download the dataset first and extract it to the correct location")
        return False
    
    logger.info(f"Processing dataset: {dataset_name}")
    
    # In a real implementation, this would:
    # 1. Extract stems from the dataset
    # 2. Convert audio to spectrograms
    # 3. Split into train/test sets
    # 4. Generate target parameters
    
    # Placeholder for actual implementation
    logger.info("Dataset processing would normally:")
    logger.info("1. Extract stems from the dataset")
    logger.info("2. Convert audio to spectrograms")
    logger.info("3. Split into train/test sets")
    logger.info("4. Generate target parameters")
    
    logger.info(f"\nTo actually process the {dataset_name} dataset:")
    if dataset_name == "musdb18":
        logger.info("1. Run the stem extraction script:")
        logger.info("   python src/extract_stems.py --input data/raw/musdb18 --output data/raw/stems")
    elif dataset_name == "medleydb":
        logger.info("1. Run the stem aggregation script:")
        logger.info("   python src/aggregate_stems.py --input data/raw/medleydb --output data/raw/stems")
    
    logger.info("2. Convert to spectrograms:")
    logger.info("   python src/audio_to_spectrogram.py --input data/raw/stems --output data/spectrograms")
    
    logger.info("3. Split and prepare data:")
    logger.info("   python src/prepare_dataset.py --input data/spectrograms --train data/train --test data/test --split 0.8")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Dataset acquisition and processing tool")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", choices=["musdb18", "medleydb"], 
                      help="Get instructions for downloading a specific dataset")
    group.add_argument("--process", choices=["musdb18", "medleydb"],
                      help="Process a downloaded dataset")
    group.add_argument("--setup", action="store_true",
                      help="Set up directory structure only")
    
    args = parser.parse_args()
    
    # Create directories
    setup_directories()
    
    if args.setup:
        logger.info("Directory setup complete.")
        return 0
    
    if args.download:
        if args.download == "musdb18":
            download_musdb18()
        elif args.download == "medleydb":
            download_medleydb()
    
    if args.process:
        success = process_dataset(args.process)
        if not success:
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
