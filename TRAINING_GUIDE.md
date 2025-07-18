# 🎛️ AI Mixer Training Guide

This guide explains how to preprocess the FMA dataset and train all the models in the AI Mixer project.

## Prerequisites

- Python 3.8+ installed
- PyTorch installed
- Required packages (see requirements.txt)
- FMA dataset (Small) downloaded and extracted to `data/raw/music/fma/fma_small`

## Quick Start

For a completely automated process, simply run the training batch script:

```bash
train_models.bat   # Windows
```

This will:
1. Preprocess the FMA dataset for all models
2. Train all 8 model architectures
3. Create the weighted ensemble model
4. Generate evaluation metrics

## Manual Steps

If you prefer to run the process step by step:

### 1. Preprocess the FMA Dataset

```bash
python src/preprocess_for_all_models.py --fma-size=small
```

Options:
- `--fma-size`: Size of FMA dataset to use (small/medium/large)
- `--sr`: Sample rate for processing (default: 22050)
- `--n-mels`: Number of mel bands for spectrograms (default: 128)
- `--split-ratio`: Train/test split ratio (default: 0.8)
- `--no-ast`: Skip AST feature generation
- `--n-jobs`: Number of parallel jobs (default: CPU count - 1)

### 2. Train Individual Models

Train specific models:

```bash
# Train just the baseline CNN
python src/train_all_models.py --models=baseline

# Train multiple models
python src/train_all_models.py --models=baseline,enhanced,ast

# Train all models
python src/train_all_models.py --models=all
```

Training options:
- `--models`: Models to train (all/baseline/enhanced/ast/lstm/transformer/vae/gan/resnet/ensemble)
- `--epochs`: Number of epochs for training (default: 10)
- `--batch-size`: Batch size for training (default: 16)
- `--learning-rate`: Learning rate for training (default: 0.001)
- `--device`: Device to use (cuda/cpu/auto)
- `--no-checkpoints`: Disable saving checkpoints during training
- `--patience`: Early stopping patience (default: 5)
- `--no-augment`: Disable data augmentation during training

## Directory Structure After Training

```
mixer/
├── 📂 data/                   # Dataset and processed files
│   ├── train/                # Training audio files
│   ├── val/                  # Validation audio files 
│   ├── test/                 # Test audio files
│   ├── spectrograms/         # Mel spectrograms for all models
│   ├── ast_features/         # AST features for the AST Regressor
│   └── targets_generated.json # Ground truth mixing parameters
├── 📂 models/                 # Trained model weights
│   ├── baseline_cnn.pth      # Baseline CNN weights
│   ├── enhanced_cnn.pth      # Enhanced CNN weights
│   ├── ast_regressor.pth     # AST Regressor weights
│   ├── lstm_mixer.pth        # LSTM model weights
│   ├── transformer_mixer.pth # Transformer model weights
│   ├── vae_mixer.pth         # VAE model weights
│   ├── gan_mixer.pth         # GAN model weights
│   ├── resnet_mixer.pth      # ResNet model weights
│   └── weighted_ensemble.pth # Weighted ensemble model
└── 📂 training_results/       # Training history and metrics
    ├── training_history.json # Loss history for all models
    ├── model_evaluation_results.json # Evaluation metrics
    └── *_training.png        # Training loss plots for each model
```

## Testing Your Trained Models

After training, you can use the models to mix audio:

```bash
# Use the best individual model (AST Regressor)
python demo_ai_mixer.py path/to/your/song.wav

# Compare all models on the same song
python src/comprehensive_mixer.py path/to/your/song.wav
```

## Model Performance

After training, the expected performance ranking (from best to worst) is:

1. Weighted Ensemble (MAE: ~0.035)
2. AST Regressor (MAE: ~0.055)
3. Baseline CNN (MAE: ~0.069)
4. ResNet Mixer (MAE: ~0.070)
5. Transformer Mixer (MAE: ~0.071)
6. LSTM Mixer (MAE: ~0.072)
7. VAE Mixer (MAE: ~0.085)
8. GAN Mixer (MAE: ~0.089)
9. Enhanced CNN (MAE: ~0.137)

The actual results may vary depending on your training setup and dataset.
