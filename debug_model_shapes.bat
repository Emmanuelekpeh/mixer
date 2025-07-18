@echo off
echo ==============================================
echo     Debugging Model Shape Issues
echo ==============================================
echo This will analyze shape issues with the 5 models:
echo - LSTM Audio Mixer
echo - Audio GAN Mixer
echo - VAE Audio Mixer
echo - Advanced Transformer Mixer
echo - ResNet Audio Mixer
echo ==============================================
python src/debug_model_shapes.py
echo ==============================================
echo Debug complete!
echo ==============================================
pause
