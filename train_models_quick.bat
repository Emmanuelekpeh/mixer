@echo off
echo ===============================================
echo    AI MIXER - Fixed Training Pipeline
echo ===============================================

echo.
echo This script will:
echo 1. Generate synthetic data (bypassing torchaudio issues)
echo 2. Train models with fixed-length spectrograms
echo 3. Generate evaluation metrics
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
echo Running fixed training pipeline...
%PYTHON_CMD% train_mixer_pipeline_fixed.py

if %ERRORLEVEL% NEQ 0 (
    echo Error during training. Please check the logs above.
    exit /b 1
)

echo.
echo ===============================================
echo    Training Complete!
echo ===============================================
echo.
echo All models have been trained and evaluated.
echo Results are available in the training_results directory.
echo.
echo To use the models for mixing:
echo   python demo_ai_mixer.py path\to\your\audio.wav
echo.

pause
