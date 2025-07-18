@echo off
echo ==============================================
echo     Training New Model Architectures
echo ==============================================
echo This will train the 5 additional models:
echo - LSTM Audio Mixer
echo - Audio GAN Mixer
echo - VAE Audio Mixer
echo - Advanced Transformer Mixer
echo - ResNet Audio Mixer
echo ==============================================
cd src
python train_new_architectures_fixed.py
cd ..
echo ==============================================
echo Training complete!
echo ==============================================
pause
