# Data Pipeline Enhancement Guide

This document explains how to use the new enhanced data pipeline for creating a production-grade dataset for AI mixing model training.

## Overview

The enhanced data pipeline adds several new capabilities to the mixer project:

1. **Multi-source dataset acquisition**: Combines diverse audio sources including FMA, DAMP, and room impulse responses
2. **Standardized processing**: Normalizes all audio for consistent training inputs
3. **Advanced augmentation**: Creates realistic variations with noise, room effects, EQ, and more
4. **Targeted parameter adjustment**: Automatically adjusts mixing parameters based on augmentations

## Directory Structure

The enhanced pipeline uses the following directory structure:

```
data/
├── raw/                       # Original unprocessed files
│   ├── music/                 # Music multi-tracks
│   │   └── fma/               # Free Music Archive
│   ├── vocals/                # Vocal-specific datasets
│   │   └── damp/              # DAMP karaoke recordings
│   └── acoustics/             # Room impulse responses
│       └── room_impulse_responses/
├── processed/                 # Pre-processed audio files
│   ├── clean/                 # Normalized, aligned clean sources
│   └── augmented/             # Augmented versions with noise/effects
├── features/                  # Extracted features for model training
│   ├── spectrograms/
│   └── ast_features/
└── metadata/                  # Comprehensive metadata
```

## Setup Instructions

1. **Initialize the directory structure and install dependencies**:

```bash
python src/setup_enhanced_dataset.py
```

2. **Download the datasets**:

```bash
# Download all datasets
python src/enhanced_data_acquisition.py

# Download specific datasets
python src/enhanced_data_acquisition.py --datasets fma damp
```

3. **Process the raw data**:

```bash
# Process all datasets
python src/enhanced_audio_processor.py

# Process specific datasets with limited size
python src/enhanced_audio_processor.py --datasets fma --subset-size 100
```

4. **Create augmented versions**:

```bash
# Augment all processed files
python src/enhanced_audio_augmentor.py

# Augment specific categories
python src/enhanced_audio_augmentor.py --categories music --subset-size 50
```

## Dataset Descriptions

### FMA (Free Music Archive)

The [Free Music Archive](https://github.com/mdeff/fma) is a collection of Creative Commons licensed music with over 106,000 tracks. We use this for training on diverse musical material across genres.

Available sizes:
- Small: 8,000 tracks (30-second clips)
- Medium: 25,000 tracks
- Large: 106,000 tracks

### DAMP Karaoke Dataset

The [DAMP dataset](https://ccrma.stanford.edu/damp/) contains karaoke recordings from the Smule Sing! app, providing thousands of vocal performances in diverse acoustic environments. This is perfect for training models that can process singing in varied conditions.

### Room Impulse Responses

From the [OpenAIR library](https://www.openair.hosted.york.ac.uk/), these impulse responses capture the acoustic characteristics of different spaces, from small rooms to large halls. We use these for simulating diverse recording environments.

## Advanced Usage

### Custom Augmentation

You can modify the augmentation types and parameters in `enhanced_audio_augmentor.py` to create custom variations:

```python
# Available augmentation types
augmentation_types = [
    "noise",       # Environmental noise injection
    "room",        # Room acoustics simulation
    "compression", # Dynamic range compression
    "eq",          # Frequency response variations
    "pitch",       # Pitch shifting
    "time"         # Time stretching
]
```

### Mixing Parameter Adjustment

The augmentation process automatically adjusts mixing parameters based on applied modifications. This ensures that target parameters for training remain consistent with the audio characteristics.

## Limitations and Future Work

- **Large dataset handling**: For very large datasets, consider processing in batches
- **Memory usage**: Feature extraction can be memory-intensive; adjust batch sizes as needed
- **Future expansion**: Additional datasets like AudioSet could be integrated for even more variety

## Help and Troubleshooting

If you encounter issues:

1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Ensure you have sufficient disk space for dataset storage
3. For memory errors, try reducing batch sizes or using subset processing
