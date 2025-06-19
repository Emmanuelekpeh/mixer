#!/usr/bin/env python3
"""
🎵 Comprehensive AI Mixer
========================

Advanced AI mixer that uses ALL available trained models for comparison:
- Baseline CNN
- Enhanced CNN  
- Improved Enhanced CNN
- Retrained Enhanced CNN
- Improved Baseline CNN
- Weighted Ensemble
- AST Regressor (feature-based)

This allows direct A/B testing of all model approaches.
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Import our models
from baseline_cnn import BaselineCNN, EnhancedCNN, SpectrogramDataset, N_OUTPUTS, N_CONV_LAYERS, DROPOUT, DEVICE
from improved_models import ImprovedEnhancedCNN
from ensemble_training import WeightedEnsemble
from ast_regressor import ASTRegressor, ASTFeatureDataset

class ComprehensiveAudioMixer:
    """Advanced AI mixer using all available trained models."""
    
    def __init__(self):
        self.sr = 22050  # Sample rate for processing
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.output_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all available models
        self.models = {}
        self.load_all_models()
        
        # Mixing parameter names for reference
        self.param_names = [
            "Input Gain", "Compression Ratio", "High-Freq EQ", "Mid-Freq EQ", 
            "Low-Freq EQ", "Presence/Air", "Reverb Send", "Delay Send", 
            "Stereo Width", "Output Level"
        ]
        
        print(f"🎵 Comprehensive AI Mixer Initialized")
        print(f"🤖 Loaded {len(self.models)} models for comparison")
    
    def load_all_models(self):
        """Load all available trained models."""
        print("🔄 Loading all available models...")
        
        # Model configurations
        model_configs = {
            'baseline_cnn.pth': {
                'class': BaselineCNN,
                'args': {'n_outputs': N_OUTPUTS, 'dropout': DROPOUT, 'n_conv_layers': N_CONV_LAYERS},
                'name': 'Baseline CNN'
            },
            'enhanced_cnn.pth': {
                'class': EnhancedCNN,
                'args': {'n_outputs': N_OUTPUTS, 'dropout': DROPOUT},
                'name': 'Enhanced CNN'
            },
            'improved_baseline_cnn.pth': {
                'class': BaselineCNN,
                'args': {'n_outputs': N_OUTPUTS, 'dropout': DROPOUT, 'n_conv_layers': N_CONV_LAYERS},
                'name': 'Improved Baseline CNN'
            },
            'improved_enhanced_cnn.pth': {
                'class': ImprovedEnhancedCNN,
                'args': {},
                'name': 'Improved Enhanced CNN'
            },
            'retrained_enhanced_cnn.pth': {
                'class': EnhancedCNN,
                'args': {'n_outputs': N_OUTPUTS, 'dropout': DROPOUT},
                'name': 'Retrained Enhanced CNN'
            },
            'weighted_ensemble.pth': {
                'class': None,  # Special handling for ensemble
                'args': {},
                'name': 'Weighted Ensemble'
            }
        }
        
        # Load individual models first
        individual_models = []
        
        for model_file, config in model_configs.items():
            model_path = self.models_dir / model_file
            if model_path.exists():
                try:
                    if config['class'] is None:
                        # Skip ensemble for now, load after individual models
                        continue
                        
                    model = config['class'](**config['args']).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    model.eval()
                    
                    self.models[config['name']] = model
                    individual_models.append(model)
                    print(f"✅ Loaded {config['name']}")
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_file}: {e}")
        
        # Load ensemble model if available and we have individual models
        if len(individual_models) >= 2:
            ensemble_path = self.models_dir / 'weighted_ensemble.pth'
            if ensemble_path.exists():
                try:
                    # Create ensemble with loaded models
                    weights = [1.0 / len(individual_models)] * len(individual_models)
                    ensemble = WeightedEnsemble(individual_models, weights).to(DEVICE)
                    ensemble.load_state_dict(torch.load(ensemble_path, map_location=DEVICE))
                    ensemble.eval()
                    
                    self.models['Weighted Ensemble'] = ensemble
                    print(f"✅ Loaded Weighted Ensemble")
                    
                except Exception as e:
                    print(f"❌ Failed to load Weighted Ensemble: {e}")
        
        # Add AST Regressor (feature-based prediction)
        self.models['AST Regressor'] = 'feature_based'
        print(f"✅ AST Regressor (feature-based)")
    
    def predict_mixing_parameters(self, audio_file_path):
        """Get mixing parameter predictions from all models."""
        audio_file_path = Path(audio_file_path)
        predictions = {}
        
        print(f"🎵 Analyzing: {audio_file_path.name}")
        
        # Prepare audio
        audio, sr = librosa.load(audio_file_path, sr=self.sr, mono=True)
        
        # Get predictions from all models
        for model_name, model in self.models.items():
            try:
                if model_name == 'AST Regressor':
                    # Feature-based prediction
                    pred = self._predict_from_ast_features(audio_file_path)
                else:
                    # Spectrogram-based prediction
                    pred = self._predict_from_spectrogram(audio, model)
                
                predictions[model_name] = pred
                print(f"✅ {model_name}: MAE prediction ready")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
        
        return predictions
    
    def _predict_from_spectrogram(self, audio, model):
        """Generate prediction using spectrogram-based model."""
        # Create mel spectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        spec = librosa.power_to_db(spec, ref=np.max)
        
        # Normalize
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        
        # Fixed time dimension (same as training)
        target_time_steps = 1000
        if spec.shape[1] > target_time_steps:
            start = (spec.shape[1] - target_time_steps) // 2
            spec = spec[:, start:start + target_time_steps]
        elif spec.shape[1] < target_time_steps:
            pad_width = target_time_steps - spec.shape[1]
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            spec = np.pad(spec, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        # Add batch and channel dimensions
        spec = np.expand_dims(spec, axis=0)  # Channel
        spec = np.expand_dims(spec, axis=0)  # Batch
        
        # Predict
        with torch.no_grad():
            spec_tensor = torch.tensor(spec, dtype=torch.float32).to(DEVICE)
            prediction = model(spec_tensor).cpu().numpy()[0]
        
        return prediction
    
    def _predict_from_ast_features(self, audio_file_path):
        """Generate prediction using AST features (simplified for demo)."""
        audio, sr = librosa.load(audio_file_path, sr=self.sr, mono=True)
        
        # Extract features similar to our target generation
        features = self._extract_mixing_features(audio, sr)
        
        # Convert to mixing parameters (similar to generate_targets.py logic)
        rms_norm = np.clip(features['rms_mean'] * 10, 0, 1)
        centroid_norm = np.clip(features['centroid_mean'] / 8000, 0, 1)
        zcr_norm = np.clip(features['zcr_mean'] * 50, 0, 1)
        dynamic_norm = np.clip(features['dynamic_range'] * 5, 0, 1)
        rolloff_norm = np.clip(features['rolloff_mean'] / 10000, 0, 1)
        
        mixing_params = [
            rms_norm,  # Input gain
            1.0 - dynamic_norm,  # Compression ratio
            centroid_norm,  # High-freq EQ boost
            0.5 + 0.5 * (zcr_norm - 0.5),  # Mid-freq EQ
            0.3 + 0.4 * rms_norm,  # Low-freq EQ
            rolloff_norm,  # Presence/air frequencies
            0.2 + 0.6 * dynamic_norm,  # Reverb send
            0.1 + 0.3 * (1 - rms_norm),  # Delay send
            0.5 + 0.3 * centroid_norm,  # Stereo width
            0.7 + 0.3 * rms_norm  # Output level
        ]
        
        return np.array(mixing_params)
    
    def _extract_mixing_features(self, audio, sr):
        """Extract features for mixing parameter prediction."""
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_mean = np.mean(spectral_centroid)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        
        # Dynamic range
        dynamic_range = np.max(np.abs(audio)) - np.mean(np.abs(audio))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        rolloff_mean = np.mean(rolloff)
        
        return {
            'rms_mean': rms_mean,
            'centroid_mean': centroid_mean,
            'zcr_mean': zcr_mean,
            'dynamic_range': dynamic_range,
            'rolloff_mean': rolloff_mean
        }
    
    def apply_mixing_parameters(self, audio, sr, mixing_params):
        """Apply mixing parameters to audio and return processed audio."""
        # Convert to stereo if mono
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        
        processed_audio = audio.copy()
        
        # 1. Input Gain
        input_gain = 0.1 + 1.5 * mixing_params[0]  # 0.1x to 1.6x gain
        processed_audio *= input_gain
        
        # 2. Compression Ratio
        compression_ratio = mixing_params[1]
        if compression_ratio > 0.1:
            processed_audio = self._apply_compression(processed_audio, compression_ratio)
        
        # 3-5. EQ (High, Mid, Low)
        high_freq_boost = mixing_params[2]
        mid_freq_boost = mixing_params[3] 
        low_freq_boost = mixing_params[4]
        processed_audio = self._apply_eq(processed_audio, sr, low_freq_boost, mid_freq_boost, high_freq_boost)
        
        # 6. Presence/Air frequencies (high shelf)
        presence = mixing_params[5]
        if presence > 0.1:
            processed_audio = self._apply_presence(processed_audio, sr, presence)
        
        # 7. Reverb Send
        reverb_send = mixing_params[6]
        if reverb_send > 0.1:
            processed_audio = self._apply_reverb(processed_audio, sr, reverb_send)
        
        # 8. Delay Send
        delay_send = mixing_params[7]
        if delay_send > 0.1:
            processed_audio = self._apply_delay(processed_audio, sr, delay_send)
        
        # 9. Stereo Width
        stereo_width = mixing_params[8]
        processed_audio = self._apply_stereo_width(processed_audio, stereo_width)
        
        # 10. Output Level
        output_level = 0.3 + 0.7 * mixing_params[9]  # 0.3x to 1.0x final level
        processed_audio *= output_level
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(processed_audio))
        if peak > 0.95:
            processed_audio = processed_audio * (0.95 / peak)
        
        return processed_audio
    
    # Audio processing methods (same as original ai_mixer.py)
    def _apply_compression(self, audio, ratio):
        """Simple soft compression."""
        threshold = 0.5
        for i in range(audio.shape[0]):
            channel = audio[i]
            above_threshold = np.abs(channel) > threshold
            compressed = np.sign(channel) * (threshold + (np.abs(channel) - threshold) * (1 - ratio))
            audio[i] = np.where(above_threshold, compressed, channel)
        return audio
      def _apply_eq(self, audio, sr, low_boost, mid_boost, high_boost):
        """Apply 3-band EQ using biquad filters."""
        # Low shelf (100 Hz)
        if abs(low_boost - 0.5) > 0.1:
            gain_db = (low_boost - 0.5) * 12  # ±6dB
            sos = signal.butter(2, 100, btype='lowpass', fs=sr, output='sos')
            for i in range(audio.shape[0]):
                if gain_db > 0:
                    filtered = signal.sosfilt(sos, audio[i])
                    audio[i] = audio[i] + gain_db/12 * filtered
        
        # Mid peak (1000 Hz)
        if abs(mid_boost - 0.5) > 0.1:            gain_db = (mid_boost - 0.5) * 12
            sos = signal.butter(2, [800, 1200], btype='bandpass', fs=sr, output='sos')
            for i in range(audio.shape[0]):
                filtered = signal.sosfilt(sos, audio[i])
                audio[i] = audio[i] + gain_db/12 * filtered
        
        # High shelf (8000 Hz)
        if abs(high_boost - 0.5) > 0.1:            gain_db = (high_boost - 0.5) * 12
            sos = signal.butter(2, 8000, btype='highpass', fs=sr, output='sos')
            for i in range(audio.shape[0]):
                if gain_db > 0:
                    filtered = signal.sosfilt(sos, audio[i])
                    audio[i] = audio[i] + gain_db/12 * filtered
          return audio
    
    def _apply_presence(self, audio, sr, presence):
        """Apply high-frequency presence boost."""
        gain_db = presence * 6  # 0-6dB boost around 12kHz
        nyquist = sr // 2
        freq = min(12000, nyquist * 0.9)
        sos = signal.butter(2, freq, btype='highpass', fs=sr, output='sos')
        
        for i in range(audio.shape[0]):
            filtered = signal.sosfilt(sos, audio[i])
            audio[i] = audio[i] + (gain_db/12) * filtered
        
        return audio
    
    def _apply_reverb(self, audio, sr, reverb_send):
        """Apply simple algorithmic reverb."""
        # Simple reverb using multiple delays
        reverb_times = [0.03, 0.07, 0.12, 0.18]  # seconds
        reverb_gains = [0.3, 0.25, 0.2, 0.15]
        
        reverb_audio = np.zeros_like(audio)
        
        for delay_time, gain in zip(reverb_times, reverb_gains):
            delay_samples = int(delay_time * sr)
            if delay_samples < audio.shape[1]:
                for i in range(audio.shape[0]):
                    delayed = np.zeros_like(audio[i])
                    delayed[delay_samples:] = audio[i][:-delay_samples]
                    reverb_audio[i] += delayed * gain * reverb_send
        
        return audio + reverb_audio
    
    def _apply_delay(self, audio, sr, delay_send):
        """Apply echo/delay effect."""
        delay_time = 0.25  # 250ms delay
        delay_samples = int(delay_time * sr)
        delay_gain = 0.4 * delay_send
        
        if delay_samples < audio.shape[1]:
            for i in range(audio.shape[0]):
                delayed = np.zeros_like(audio[i])
                delayed[delay_samples:] = audio[i][:-delay_samples] * delay_gain
                audio[i] = audio[i] + delayed
        
        return audio
    
    def _apply_stereo_width(self, audio, width):
        """Adjust stereo width using M/S processing."""
        if audio.shape[0] == 2:
            # Convert to Mid/Side
            mid = (audio[0] + audio[1]) * 0.5
            side = (audio[0] - audio[1]) * 0.5
            
            # Adjust side signal based on width parameter
            side_gain = 0.5 + width * 1.0  # 0.5x to 1.5x width
            side *= side_gain
            
            # Convert back to L/R
            audio[0] = mid + side
            audio[1] = mid - side
        
        return audio
    
    def create_comparison_chart(self, predictions, output_path):
        """Create a comparison chart of all model predictions."""
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Prepare data
        model_names = list(predictions.keys())
        x = np.arange(len(self.param_names))
        width = 0.8 / len(model_names)
        
        # Colors for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # Plot bars for each model
        for i, (model_name, params) in enumerate(predictions.items()):
            offset = (i - len(model_names)/2) * width
            bars = ax.bar(x + offset, params, width, label=model_name, color=colors[i % len(colors)], alpha=0.8)
            
            # Add value labels on bars
            for bar, param in zip(bars, params):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{param:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Mixing Parameters')
        ax.set_ylabel('Parameter Values (0-1)')
        ax.set_title('AI Mixing Model Comparison - Parameter Predictions')
        ax.set_xticks(x)
        ax.set_xticklabels(self.param_names, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison chart saved: {output_path.name}")
    
    def comprehensive_mix_comparison(self, audio_file_path):
        """Mix a song using all models and create comprehensive comparison."""
        audio_file_path = Path(audio_file_path)
        
        print(f"🎵 COMPREHENSIVE MIXING COMPARISON")
        print(f"🎵 Song: {audio_file_path.name}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load original audio
        original_audio, sr = librosa.load(audio_file_path, sr=self.sr, mono=False)
        if original_audio.ndim == 1:
            original_audio = np.stack([original_audio, original_audio])
        
        # Get predictions from all models
        predictions = self.predict_mixing_parameters(audio_file_path)
        
        # Save original for comparison
        original_output = self.output_dir / f"{audio_file_path.stem}_original.wav"
        sf.write(original_output, original_audio.T, sr)
        print(f"💾 Original: {original_output.name}")
        
        # Create comparison chart
        chart_path = self.output_dir / f"{audio_file_path.stem}_comparison.png"
        self.create_comparison_chart(predictions, chart_path)
        
        # Mix with each model and save results
        mixed_files = []
        results_summary = {}
        
        for model_name, params in predictions.items():
            print(f"\n🤖 {model_name}")
            print("-" * 40)
            
            # Display predictions
            for i, (param_name, value) in enumerate(zip(self.param_names, params)):
                print(f"  {param_name:15}: {value:.3f}")
            
            # Apply mixing
            mixed_audio = self.apply_mixing_parameters(original_audio.copy(), sr, params)
              # Calculate some audio metrics
            rms_original = np.sqrt(np.mean(original_audio**2))
            rms_mixed = np.sqrt(np.mean(mixed_audio**2))
            peak_original = np.max(np.abs(original_audio))
            peak_mixed = np.max(np.abs(mixed_audio))
            
            results_summary[model_name] = {
                'rms_change': float(rms_mixed / rms_original),
                'peak_change': float(peak_mixed / peak_original),
                'parameters': [float(p) for p in params]
            }
            
            print(f"  📊 RMS Change: {rms_mixed/rms_original:.2f}x")
            print(f"  📊 Peak Change: {peak_mixed/peak_original:.2f}x")
            
            # Save mixed version
            safe_model_name = model_name.replace(" ", "_").replace("/", "_").lower()
            output_file = self.output_dir / f"{audio_file_path.stem}_{safe_model_name}_mixed.wav"
            sf.write(output_file, mixed_audio.T, sr)
            mixed_files.append(output_file)
            print(f"  💾 Saved: {output_file.name}")
        
        # Save detailed comparison results
        results_file = self.output_dir / f"{audio_file_path.stem}_mixing_comparison.json"
        with open(results_file, 'w') as f:
            json.dump({
                'song': audio_file_path.name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'models_used': list(predictions.keys()),
                'results': results_summary,
                'processing_time': time.time() - start_time
            }, f, indent=2)
        
        # Final summary
        print(f"\n" + "=" * 60)
        print(f"🎉 MIXING COMPARISON COMPLETE")
        print(f"⏱️  Processing time: {time.time() - start_time:.1f} seconds")
        print(f"🤖 Models compared: {len(predictions)}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"\n📁 Files created:")
        print(f"  🎵 {original_output.name}")
        for file in mixed_files:
            print(f"  🎵 {file.name}")
        print(f"  📊 {chart_path.name}")
        print(f"  📝 {results_file.name}")
        
        print(f"\n🎧 LISTENING GUIDE:")
        print(f"1. Start with: {original_output.name} (original)")
        print(f"2. Compare each model's mix to hear the differences")
        print(f"3. Check the comparison chart to see parameter differences")
        
        return {
            'predictions': predictions,
            'mixed_files': mixed_files,
            'results_summary': results_summary,
            'output_dir': self.output_dir
        }

def main():
    """Demo the comprehensive mixer with available test songs."""
    mixer = ComprehensiveAudioMixer()
    
    # Find test songs from the dataset
    test_dirs = [
        Path(__file__).resolve().parent.parent / "data" / "test",
        Path(__file__).resolve().parent.parent / "data" / "train",
        Path(__file__).resolve().parent.parent / "data" / "val"
    ]
    
    test_files = []
    for test_dir in test_dirs:
        if test_dir.exists():
            test_files.extend(list(test_dir.glob("*.stem.mp4")))
            test_files.extend(list(test_dir.glob("*.wav")))
            test_files.extend(list(test_dir.glob("*.mp3")))
    
    if not test_files:
        print("❌ No test audio files found!")
        print("📁 Please place audio files in:")
        for test_dir in test_dirs:
            print(f"   - {test_dir}")
        return
    
    # Use the first available test file
    test_file = test_files[0]
    print(f"🎵 Demo comprehensive mixing with: {test_file.name}")
    
    # Run comprehensive comparison
    results = mixer.comprehensive_mix_comparison(test_file)
    
    print(f"\n🏆 MODEL PERFORMANCE NOTES:")
    print(f"📈 Enhanced CNN: Improved architecture vs baseline")
    print(f"🔧 Improved Enhanced CNN: Additional optimizations")
    print(f"🎯 Retrained Enhanced CNN: Retrained on augmented data")
    print(f"🎭 Weighted Ensemble: Combination of multiple models")
    print(f"🎺 AST Regressor: Feature-based audio analysis")
    
    return results

if __name__ == "__main__":
    main()
