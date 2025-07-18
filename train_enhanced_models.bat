@echo off
echo ===============================================
echo    AI MIXER - Enhanced Training Pipeline
echo         Professional Model Development
echo ===============================================

echo.
echo 🚀 ENHANCED TRAINING FEATURES:
echo ✅ Resume from checkpoints (continues previous training)
echo ✅ 50 epochs per model (much more concrete training)
echo ✅ Early stopping (prevents overfitting)
echo ✅ Gradient clipping (training stability)
echo ✅ 16,000 varied synthetic samples (2x more data)
echo ✅ Realistic audio patterns (helps models specialize)
echo ✅ Detailed per-parameter analysis
echo ✅ Model strength profiling
echo.
echo ⏱️ Expected time: 2-4 hours (depending on hardware)
echo 💾 Can be interrupted and resumed safely
echo.

set PYTHON_CMD=python

REM Check if Python is installed
where %PYTHON_CMD% >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error: Python not found. Make sure Python is installed and in your PATH.
    exit /b 1
)

REM Check if we have existing checkpoints
if exist "models\*_checkpoint.pth" (
    echo 🔄 Found existing training checkpoints!
    echo    Training will resume from where it left off.
    echo.
)

echo 🔍 Checking required packages...
%PYTHON_CMD% -c "import torch, numpy, matplotlib, tqdm, librosa, sklearn" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 📦 Installing missing packages...
    %PYTHON_CMD% -m pip install torch numpy matplotlib tqdm scikit-learn librosa
)

echo.
echo 🏋️‍♀️ Starting enhanced training of all 8 models...
echo ================================================
echo.
echo Models to train:
echo 1. Baseline CNN         - Foundation model
echo 2. Enhanced CNN         - Advanced convolutions  
echo 3. AST Regressor        - Transformer-based
echo 4. LSTM Mixer          - Temporal sequences
echo 5. Advanced Transformer - Multi-head attention
echo 6. VAE Mixer           - Latent space mixing
echo 7. Audio GAN          - Generative mixing
echo 8. ResNet Mixer       - Deep residual networks
echo.

%PYTHON_CMD% train_mixer_pipeline_fixed.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠️ Some models may have failed, but check the output above.
    echo ✅ Successfully trained models are available in models/ directory.
    echo 🔄 You can re-run this script to resume failed models.
) else (
    echo.
    echo ===============================================
    echo     🎉 ENHANCED TRAINING COMPLETE! 🎉
    echo ===============================================
)

echo.
echo 📊 RESULTS AVAILABLE:
echo   models/*_best.pth         - Best trained models
echo   models/*_results.json     - Detailed performance metrics  
echo   models/*_history.png      - Training curve visualizations
echo.
echo 🎯 MODEL ANALYSIS:
echo   • Individual model strengths identified
echo   • Per-parameter specialization analysis
echo   • Overall performance ranking
echo.
echo 🎵 TO USE YOUR TRAINED MODELS:
echo   python demo_ai_mixer.py path\to\your\audio.wav
echo.
echo 🏆 TO START TOURNAMENT BATTLES:
echo   cd tournament_webapp
echo   python dev_server.py
echo.
echo 💡 TIP: Models now have distinct personalities and strengths!
echo     Each model specializes in different mixing parameters.
echo.

pause
