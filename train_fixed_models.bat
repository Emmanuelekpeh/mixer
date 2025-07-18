@echo off
echo ==============================================
echo     Training All Model Architectures
echo ==============================================
echo This will train the 5 new architectures:
echo - LSTM Audio Mixer
echo - Audio GAN Mixer
echo - VAE Audio Mixer
echo - Advanced Transformer Mixer
echo - ResNet Audio Mixer
echo ==============================================
python src/fixed_train_models.py
echo ==============================================
echo Training complete!
echo ==============================================
pause
