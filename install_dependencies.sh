#!/bin/bash
# Script to install all necessary dependencies

# Update pip
pip install --upgrade pip

# Install requirements from requirements.txt
pip install -r requirements.txt

# Check for specific packages that might cause issues
pip install librosa soundfile torchaudio

echo "All dependencies installed successfully."
