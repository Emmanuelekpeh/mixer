# PowerShell script to install all necessary dependencies

# Update pip
python -m pip install --upgrade pip

# Install requirements from requirements.txt
python -m pip install -r requirements.txt

# Check for specific packages that might cause issues
python -m pip install librosa soundfile torchaudio

Write-Host "All dependencies installed successfully." -ForegroundColor Green
