# 🎛️ Mixture - Quick Start Guide

## 🚀 **READY TO USE! Here's How:**

Your Mixture AI mixing system is **completely set up and working**. Here are the three ways to use it:

---

## 🎵 **1. QUICK DEMO (Recommended First Step)**

Use your best AI model (AST Regressor) on any audio file:

```bash
python demo_ai_mixer.py path/to/your/song.wav
```

**What it does:**
- Uses the champion AST Regressor model (MAE: 0.0554)
- Shows AI parameter predictions
- Generates professionally mixed audio
- Processing time: ~3-5 seconds

**Example output:**
```
🎛️ AST Regressor Predictions:
• Input Gain    : 0.954
• Compression   : 0.000
• High-Freq EQ  : 0.335
• Mid-Freq EQ   : 0.750
• Low-Freq EQ   : 0.681
• Presence/Air  : 0.554
• Reverb Send   : 0.800
• Delay Send    : 0.114
• Stereo Width  : 0.600
• Output Level  : 0.986

✅ Mixing Complete!
📁 Output File: mixed_outputs/song_ast_demo_mixed.wav
```

---

## 📊 **2. FULL MODEL COMPARISON**

Test ALL your AI models on the same song for A/B comparison:

```bash
python src/comprehensive_mixer.py
```

**What it does:**
- Uses ALL 6+ trained models
- Generates mixed versions from each model
- Creates comparison charts and analysis
- Saves detailed performance metrics

**Generated files:**
- `original.wav` - Source audio
- `baseline_cnn_mixed.wav` - Conservative mixing
- `enhanced_cnn_mixed.wav` - Aggressive processing
- `ast_regressor_mixed.wav` - Best balanced ⭐
- `weighted_ensemble_mixed.wav` - Ultimate quality ⭐
- `comparison.png` - Visual parameter chart
- `mixing_comparison.json` - Detailed metrics

---

## 🔧 **3. CUSTOM INTEGRATION**

Use the AI mixer in your own Python code:

```python
from src.ai_mixer import AudioMixer
import numpy as np

# Initialize the mixer
mixer = AudioMixer()

# Get AI predictions for any audio file
predictions = mixer.predict_mixing_parameters("song.wav")

# Use the best model (AST Regressor)
ast_params = predictions['AST Regressor']
print(f"AI suggests: {ast_params}")

# Apply the mixing to your audio
import librosa
audio, sr = librosa.load("song.wav", sr=22050, mono=False)
mixed_audio = mixer.apply_mixing_parameters(audio, sr, ast_params)

# Save the result
import soundfile as sf
sf.write("my_ai_mixed_song.wav", mixed_audio.T, sr)
```

---

## 🎧 **LISTEN TO YOUR RESULTS:**

After running any of the above, you'll find mixed audio files in:
- `mixed_outputs/` directory
- Compare original vs AI mixed versions
- **AST Regressor** and **Weighted Ensemble** typically sound best

---

## 🎯 **WHICH MODEL TO USE:**

| Model | When to Use | Characteristics |
|-------|-------------|-----------------|
| **AST Regressor** | 🥇 **Production/Default** | Balanced, professional, fast |
| **Weighted Ensemble** | 🏆 **Highest Quality** | Best performance, slower |
| **Baseline CNN** | 🛡️ **Conservative** | Safe, minimal processing |
| **Enhanced CNN** | 🧪 **Experimental** | Creative, aggressive changes |

---

## 📁 **KEY FILES REFERENCE:**

### 🎵 **Main Scripts:**
- `demo_ai_mixer.py` - Quick demo with best model
- `src/ai_mixer.py` - Core mixing engine
- `src/comprehensive_mixer.py` - Full model comparison

### 🤖 **Trained Models:**
- `models/baseline_cnn.pth` - Baseline CNN weights
- `models/enhanced_cnn.pth` - Enhanced CNN weights  
- `models/weighted_ensemble.pth` - Best ensemble model ⭐

### 📊 **Results & Analysis:**
- `mixed_outputs/` - Generated audio files
- `enhanced_results/` - Performance metrics
- `PROJECT_OVERVIEW.md` - Complete project summary

---

## 🚨 **REQUIREMENTS:**

Make sure you have:
```bash
pip install torch librosa soundfile scikit-learn numpy scipy matplotlib
```

---

## 🎉 **YOU'RE READY!**

Your Mixture system is **fully operational**. Start with the demo script and explore from there!

```bash
# Try it right now:
python demo_ai_mixer.py
```

**🎵 Welcome to Mixture - AI-powered music mixing! 🎛️**

## Updated Quick-Start (Async Pipeline 2025-07-19)

### Prerequisites
1. Docker & Docker Compose ≥ v2
2. Railway account (or local dev tools) / Redis & Postgres plugins if deploying

### Local Development
```bash
# Build & start API, Worker, Model-Manager, Redis
docker-compose up --build
# API       → http://localhost:10000
# Worker    → background logs only
# ModelMgr  → http://localhost:8090/health
```

Environment variables can be overridden in a `.env` file – see below:
```env
# .env
DATABASE_URL=postgresql://user:pass@dbhost:5432/mixer
REDIS_URL=redis://redis:6379/0
STORAGE_ROOT=/app/processed_audio
STORAGE_BACKEND=local   # switch to s3 later
```

### Deployment on Railway
1. Add **Redis** plugin – note the auto-generated `REDIS_URL` secret.
2. Add **Postgres** plugin – copy `DATABASE_URL` secret.
3. Main service (FastAPI):
   * Start command: `/app/entrypoint.sh`
   * Port: 8080 (exposed via 10000 in compose)
4. Worker service:
   * Start command: `/app/worker_entrypoint.sh`
5. Model-Manager service:
   * Start command: `/app/model_manager_entrypoint.sh`
   * Port 8090 (optional)
6. Add a **Volume** named `processed_audio` mounted at `/app/processed_audio` for persistent mixes.

### Cancelling a Mix Job
```bash
POST /api/mix-jobs/{job_id}/cancel
```
Response:
```json
{"cancelled": true, "status": "cancelled"}
```

### Health Endpoints
* API: `/api/health`, `/api/health/redis`
* Worker: logs only (checks Redis)
* Model-Manager: `/health`

### Running Tests
```bash
pip install -r requirements.txt
pytest -q
```
