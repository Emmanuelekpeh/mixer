# Audio Comparison Results

## File Naming Convention

For each sample, you'll find these files:

1. **01_original_distorted.wav** - The actual distorted input audio (ground truth)
2. **02_target_clean.wav** - The actual clean target audio (ground truth)  
3. **03_model_input_distorted.wav** - Distorted audio reconstructed from spectrogram
4. **04_model_target_clean.wav** - Clean audio reconstructed from spectrogram
5. **05_model_restored.wav** - Audio restored by the model (restoration/hybrid only)
5. **05_model_mixed_result.wav** - Audio with mixing parameters applied (mixing approach only)
6. **parameters.txt** - Predicted mixing and distortion parameters

## Approaches Compared

### MIXING
- **Goal**: Predict optimal mixing parameters from distorted audio
- **Output**: Mixing parameters + Applied result audio (05_model_mixed_result.wav)
- **Best for**: Understanding and hearing what mixing adjustments improve the audio

### RESTORATION  
- **Goal**: Restore clean audio from distorted input
- **Output**: Restored audio + distortion parameter analysis
- **Best for**: Audio enhancement and cleaning

### HYBRID
- **Goal**: Complete pipeline - analyze distortions -> restore audio -> optimize mixing
- **Output**: Restored audio + mixing parameters + distortion analysis
- **Best for**: End-to-end audio processing

## How to Listen

1. **Start with files 01 & 02** - Compare the actual ground truth (distorted vs clean)
2. **Listen to file 05** - This is what the model actually produced:
   - **restoration/hybrid**: 05_model_restored.wav (restored audio)
   - **mixing**: 05_model_mixed_result.wav (original + applied mixing parameters)
3. **Compare with file 02** - How close did the model get to the clean target?
4. **Check parameters.txt** - What did the model think was wrong and how to fix it?

## Quality Assessment

- **MIXING**: Low loss (0.0162) - Good at predicting mixing parameters
- **RESTORATION**: Medium loss (0.1336) - Decent audio restoration  
- **HYBRID**: Higher loss (0.2460) but handles ALL tasks - Best overall approach

The HYBRID model is recommended because it provides a complete audio processing pipeline.
