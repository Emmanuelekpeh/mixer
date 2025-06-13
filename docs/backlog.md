# Backlog: AI Mixing & Mastering System

## Phase 1: Data & Preprocessing
- [ ] **Dataset Acquisition**: Download/curate a dataset of raw stems and reference mixes (e.g., MUSDB18, MedleyDB).
- [ ] **Audio Loading Script**: Write scripts to load multi-track audio files (stems) and reference mixes.
- [ ] **Spectrogram Conversion**: Convert audio to mel-spectrograms and normalize for model input.
- [ ] **Data Split**: Split data into training, validation, and test sets.
- [ ] **Automated Tests: Data Pipeline**: Write tests to ensure data loading and preprocessing are correct.

## Phase 2: Baseline Model
- [ ] **Baseline CNN Model**: Implement a simple CNN to predict EQ/compression settings from spectrograms.
- [ ] **Training Script**: Write code to train the baseline model on the dataset.
- [ ] **Evaluation Script**: Evaluate model performance using spectral loss and basic metrics.
- [ ] **Automated Tests: Model Output**: Test model predictions for shape, type, and basic sanity.

## Phase 3: Model Improvements
- [ ] **Temporal Modeling**: Add RNN/LSTM layers to capture time dependencies in the mix.
- [ ] **Attention Mechanisms**: Integrate attention layers to prioritize key elements (e.g., vocals).
- [ ] **Generative Models**: Experiment with GANs/VAEs for creative mix variations.
- [ ] **Transfer Learning**: Fine-tune pre-trained models (e.g., VGGish) for mixing tasks.
- [ ] **Reinforcement Learning**: Prototype RL for optimizing mix quality based on defined rewards.
- [ ] **Patch-level Pre-trained Models**: Research and integrate pre-trained patch-level models (e.g., Audio Spectrogram Transformer, ViT) for feature extraction on spectrograms.
- [ ] **Anomaly Detection for Mix Quality**: Experiment with using anomaly detection models to identify mix issues or guide effect application.

## Phase 4: Evaluation & Metrics
- [ ] **Perceptual Metrics**: Implement Fréchet Audio Distance (FAD) and other perceptual quality metrics.
- [ ] **Listening Tests**: Set up a process for manual or crowd-sourced listening tests (optional).

## Phase 5: Real-Time/Edge Optimization
- [ ] **Model Pruning/Quantization**: Optimize models for fast inference and low memory usage.
- [ ] **Edge Deployment**: Test model on edge devices or with real-time audio streams.

## General
- [ ] **Documentation**: Maintain clear documentation for setup, usage, and contribution.
- [ ] **Project Structure**: Organize codebase (src/, data/, tests/, etc.) for clarity and scalability.
- [ ] **Integrate Real Target Extraction**: Update dataset and pipeline to use real EQ/compression targets from mix settings or annotations (CSV/JSON mapping).
- [ ] **Validation and Test Splits**: Organize data and update training loop to support validation and test splits, including evaluation metrics (MSE, MAE).
- [ ] **Model Complexity & Regularization**: Experiment with CNN model depth, dropout, kernel sizes, and regularization techniques to improve generalization.
- [ ] **Data Augmentation**: Implement and test spectrogram augmentations (time/frequency masking, noise) to improve robustness.
- [ ] **Compare with Pre-trained Models**: Use AST/ViT as feature extractors and compare performance with baseline CNN. 