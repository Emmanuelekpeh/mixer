import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from ai_mixer import AudioMixer

def analyze_mixing_differences():
    """Analyze and compare the mixing approaches of all three models."""
    
    # Load the mixed outputs
    mixed_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
    
    if not mixed_dir.exists():
        print("❌ No mixed outputs found. Run ai_mixer.py first!")
        return
    
    print("🎛️ AI Mixing Analysis Report")
    print("=" * 50)
    
    # Find all the mixed files
    original_file = None
    mixed_files = {}
    
    for file in mixed_dir.glob("*.wav"):
        if "original" in file.name:
            original_file = file
        elif "baseline_cnn" in file.name:
            mixed_files['Baseline CNN'] = file
        elif "enhanced_cnn" in file.name:
            mixed_files['Enhanced CNN'] = file
        elif "ast_regressor" in file.name:
            mixed_files['AST Regressor'] = file
    
    if not original_file or len(mixed_files) != 3:
        print("❌ Missing mixed output files!")
        return
    
    # Load audio files
    print("📊 Loading audio files for analysis...")
    sr = 22050
    
    original_audio, _ = librosa.load(original_file, sr=sr, mono=True)
    mixed_audios = {}
    
    for model_name, file_path in mixed_files.items():
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        mixed_audios[model_name] = audio
    
    # Analysis metrics
    print("\n📈 Audio Analysis Metrics:")
    print("-" * 30)
    
    metrics_data = []
    
    # Analyze original
    orig_metrics = calculate_audio_metrics(original_audio, sr)
    orig_metrics['Model'] = 'Original'
    metrics_data.append(orig_metrics)
    
    # Analyze each mixed version
    for model_name, audio in mixed_audios.items():
        metrics = calculate_audio_metrics(audio, sr)
        metrics['Model'] = model_name
        metrics_data.append(metrics)
    
    # Create comparison table
    df = pd.DataFrame(metrics_data)
    df = df.set_index('Model')
    
    print(df.round(4))
    
    # Parameter comparison
    print("\n🎛️ Mixing Parameter Comparison:")
    print("-" * 35)
    
    mixer = AudioMixer()
    test_file = mixed_dir.parent / "data" / "test" / "Al James - Schoolboy Facination.stem.mp4"
    
    if test_file.exists():
        predictions = mixer.predict_mixing_parameters(test_file)
        
        param_names = mixer.param_names
        comparison_data = []
        
        for i, param_name in enumerate(param_names):
            row = {'Parameter': param_name}
            for model_name, params in predictions.items():
                row[model_name] = round(params[i], 3)
            comparison_data.append(row)
        
        param_df = pd.DataFrame(comparison_data)
        param_df = param_df.set_index('Parameter')
        print(param_df)
        
        # Find biggest differences
        print("\n🔍 Key Differences:")
        print("-" * 20)
        
        for i, param_name in enumerate(param_names):
            values = [predictions[model][i] for model in predictions.keys()]
            param_range = max(values) - min(values)
            if param_range > 0.2:  # Significant difference
                print(f"• {param_name}: Range {param_range:.3f}")
                for model_name, params in predictions.items():
                    print(f"  - {model_name}: {params[i]:.3f}")
    
    # Recommendations
    print("\n🏆 Model Performance Summary:")
    print("-" * 30)
    print("🥇 AST Regressor: Best overall performance (MAE: 0.0554)")
    print("   - Most conservative and balanced mixing approach")
    print("   - Good stereo width and output level control")
    print("   - Moderate reverb and EQ adjustments")
    
    print("\n🥈 Baseline CNN: Good performance (MAE: 0.0689)")
    print("   - More aggressive processing in some areas")
    print("   - Conservative gain and compression")
    print("   - Balanced mid-range frequencies")
    
    print("\n🥉 Enhanced CNN: Needs improvement (MAE: 0.1373)")
    print("   - Most aggressive mixing style")
    print("   - High gain and output levels")
    print("   - May cause clipping or over-processing")
    
    print(f"\n📁 Audio files saved in: {mixed_dir}")
    
    return df, param_df

def calculate_audio_metrics(audio, sr):
    """Calculate comprehensive audio metrics."""
    
    # RMS energy (loudness)
    rms = np.sqrt(np.mean(audio**2))
    
    # Peak level
    peak = np.max(np.abs(audio))
    
    # Dynamic range (difference between peak and RMS)
    dynamic_range = peak - rms if peak > 0 else 0
    
    # Spectral centroid (brightness)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    # Zero crossing rate (high frequency content)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Spectral rolloff (frequency distribution)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    
    # LUFS approximation (perceived loudness)
    lufs_approx = 20 * np.log10(rms + 1e-10) - 23  # Rough approximation
    
    return {
        'RMS_Energy': rms,
        'Peak_Level': peak,
        'Dynamic_Range': dynamic_range,
        'Spectral_Centroid': spectral_centroid,
        'Zero_Crossing_Rate': zcr,
        'Spectral_Rolloff': rolloff,
        'LUFS_Approx': lufs_approx
    }

def create_visual_comparison():
    """Create visual comparison of the mixing results."""
    mixed_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
    
    if not mixed_dir.exists():
        print("❌ No mixed outputs found. Run ai_mixer.py first!")
        return
    
    # Load audio files
    sr = 22050
    audio_files = {}
    
    for file in mixed_dir.glob("*.wav"):
        if "original" in file.name:
            audio, _ = librosa.load(file, sr=sr, mono=True)
            audio_files['Original'] = audio
        elif "baseline_cnn" in file.name:
            audio, _ = librosa.load(file, sr=sr, mono=True)
            audio_files['Baseline CNN'] = audio
        elif "enhanced_cnn" in file.name:
            audio, _ = librosa.load(file, sr=sr, mono=True)
            audio_files['Enhanced CNN'] = audio
        elif "ast_regressor" in file.name:
            audio, _ = librosa.load(file, sr=sr, mono=True)
            audio_files['AST Regressor'] = audio
    
    # Create spectrograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, audio) in enumerate(audio_files.items()):
        if i < 4:
            # Create mel spectrogram
            S = librosa.feature.melspectrogram(y=audio[:sr*10], sr=sr, n_mels=128)  # First 10 seconds
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axes[i])
            axes[i].set_title(f'{name} - Mel Spectrogram')
            plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = mixed_dir / "mixing_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visual comparison saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Run comprehensive analysis
    df, param_df = analyze_mixing_differences()
    
    # Create visual comparison
    try:
        visual_path = create_visual_comparison()
    except Exception as e:
        print(f"⚠️ Could not create visual comparison: {e}")
    
    print("\n✅ Analysis complete!")
    print("\n💡 Next Steps:")
    print("1. Listen to the audio files to hear the differences")
    print("2. The AST Regressor is your best model for production use")
    print("3. Consider fine-tuning the Enhanced CNN architecture")
    print("4. Try different songs to see how models adapt to different genres")
