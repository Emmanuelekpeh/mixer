@echo off
echo ===============================================
echo    AI MIXER - Complete Training Pipeline
echo ===============================================

echo.
echo This script will:
echo 1. Preprocess the FMA dataset for all models
echo 2. Train all 8 models sequentially
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
%PYTHON_CMD% -c "import torch, librosa, soundfile, tqdm, numpy, matplotlib" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Some required packages are missing. Installing dependencies...
    %PYTHON_CMD% -m pip install -r requirements.txt
)

echo.
echo -----------------------------------------------
echo    Step 1: Preprocessing FMA Dataset
echo -----------------------------------------------
echo.

echo Running simple preprocessing for all models...
echo.
echo This will:
echo - Create mel spectrograms for all models
echo - Generate MFCC features (instead of AST features that have compatibility issues)
echo - Create train/validation/test splits
echo - Generate synthetic mixing parameters
echo.

REM Run the simple preprocessing script
%PYTHON_CMD% src\simple_preprocessing.py

if %ERRORLEVEL% NEQ 0 (
    echo Error during preprocessing. Please check the logs above.
    exit /b 1
)

echo.
echo -----------------------------------------------
echo    Step 2: Training All Models
echo -----------------------------------------------
echo.

echo Training models (this may take a while)...
%PYTHON_CMD% src\train_all_models.py --models=all --epochs=10 --batch-size=16

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
