# 🎛️ AI Mixing System - Quick Start Guide

## 🚀 **READY TO USE! Here's How:**

Your AI mixing system is **completely set up and working**. Here are the three ways to use it:

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

Your AI mixing system is **fully operational**. Start with the demo script and explore from there!

```bash
# Try it right now:
python demo_ai_mixer.py
```

**🎵 Welcome to AI-powered music mixing! 🎛️**
