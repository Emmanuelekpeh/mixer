@echo off
echo ===============================================
echo    AI MIXER - Train All 8 Models Pipeline
echo ===============================================

echo.
echo This script will train all 8 AI mixing models:
echo 1. Baseline CNN
echo 2. Enhanced CNN
echo 3. AST Regressor (Transformer)
echo 4. LSTM Mixer
echo 5. Advanced Transformer
echo 6. VAE Mixer
echo 7. Audio GAN Mixer
echo 8. ResNet Mixer
echo.
echo Features:
echo - Uses synthetic data (no dataset download needed)
echo - Continues training even if some models fail
echo - Saves results and training history for each model
echo - Takes approximately 60-90 minutes to complete
echo.

set PYTHON_CMD=python

REM Check if Python is installed
where %PYTHON_CMD% >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Make sure Python is installed and in your PATH.
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
%PYTHON_CMD% -c "import torch, numpy, matplotlib, tqdm, librosa, sklearn" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Some required packages are missing. Installing dependencies...
    %PYTHON_CMD% -m pip install torch numpy matplotlib tqdm scikit-learn librosa
)

echo.
echo Starting training of all 8 models...
echo =====================================
%PYTHON_CMD% train_mixer_pipeline_fixed.py

if %ERRORLEVEL% NEQ 0 (
    echo Some models may have failed, but check the output above for details.
    echo Successfully trained models are available in the models/ directory.
) else (
    echo.
    echo ===============================================
    echo    All Models Training Complete!
    echo ===============================================
)

echo.
echo Results available in:
echo   models/                    - Trained model files (.pth)
echo   models/*_results.json      - Performance metrics
echo   models/*_history.png       - Training curves
echo.
echo To use the trained models:
echo   python demo_ai_mixer.py path\to\your\audio.wav
echo.
echo To start the tournament webapp:
echo   cd tournament_webapp
echo   python dev_server.py
echo.

pause
