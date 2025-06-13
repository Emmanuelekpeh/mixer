#!/usr/bin/env python3
"""
🎛️ Production-Grade AI Mixer with Enhanced Musicality
====================================================

Advanced AI mixing system that understands musical context and adapts
mixing decisions based on genre, tempo, key, and musical characteristics.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced components
from enhanced_musical_intelligence import AdvancedMusicalAnalyzer, AdvancedMusicalContext
from baseline_cnn import BaselineCNN, EnhancedCNN, N_OUTPUTS, DEVICE
from ast_regressor import ASTRegressor

class ProductionAIMixer:
    """Production-grade AI mixer with musical intelligence"""
    
    def __init__(self):
        self.sr = 44100  # Higher quality sample rate
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.musical_analyzer = AdvancedMusicalAnalyzer()
        
        # Load AI models
        self.load_models()
        
        # Enhanced parameter names with more detail
        self.param_names = [
            "Input Gain", "Compression Ratio", "Compression Attack", "Compression Release",
            "Low Shelf (80Hz)", "Low Mid (200Hz)", "Mid (1kHz)", "High Mid (4kHz)", 
            "High Shelf (12kHz)", "Presence (8kHz)", "Reverb Send", "Reverb Type",
            "Delay Send", "Delay Time", "Stereo Width", "Bass Mono", "Output Level"
        ]
        
        # Musical mixing processors
        self.processors = {
            'vintage_compressor': self._vintage_compressor,
            'modern_compressor': self._modern_compressor,
            'musical_eq': self._musical_eq,
            'harmonic_exciter': self._harmonic_exciter,
            'stereo_enhancer': self._stereo_enhancer,
            'musical_reverb': self._musical_reverb,
            'tempo_sync_delay': self._tempo_sync_delay
        }
    
    def load_models(self):
        """Load all available trained models"""
        print("🎵 Loading AI models with musical intelligence...")
        
        self.models = {}
        
        # Load baseline models
        try:
            self.baseline_model = BaselineCNN(n_outputs=N_OUTPUTS, dropout=0.3, n_conv_layers=3)
            if (self.models_dir / "baseline_cnn.pth").exists():
                self.baseline_model.load_state_dict(torch.load(self.models_dir / "baseline_cnn.pth", map_location=DEVICE))
                self.baseline_model.eval()
                self.models['Baseline CNN'] = self.baseline_model
                print("✅ Baseline CNN loaded")
        except Exception as e:
            print(f"⚠️ Baseline CNN: {e}")
        
        # Load enhanced models
        try:
            self.enhanced_model = EnhancedCNN(n_outputs=N_OUTPUTS, dropout=0.3)
            if (self.models_dir / "enhanced_cnn.pth").exists():
                self.enhanced_model.load_state_dict(torch.load(self.models_dir / "enhanced_cnn.pth", map_location=DEVICE))
                self.enhanced_model.eval()
                self.models['Enhanced CNN'] = self.enhanced_model
                print("✅ Enhanced CNN loaded")
        except Exception as e:
            print(f"⚠️ Enhanced CNN: {e}")
        
        # AST Regressor (feature-based)
        self.models['AST Regressor (Musical)'] = 'musical_feature_based'
        print("✅ Musical AST Regressor ready")
        
        print(f"🎛️ Loaded {len(self.models)} models for musical AI mixing")
    
    def analyze_and_mix(self, audio_path: str, output_dir: Optional[str] = None) -> Dict:
        """Complete musical analysis and mixing pipeline"""
        audio_path = Path(audio_path)
        
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🎵 Analyzing: {audio_path.name}")
        print("=" * 60)
        
        # Step 1: Musical Analysis
        print("🎼 Performing musical analysis...")
        musical_context = self.musical_analyzer.analyze_track(str(audio_path))
        mixing_strategy = self.musical_analyzer.generate_mixing_strategy(musical_context)
        
        self._print_musical_analysis(musical_context, mixing_strategy)
        
        # Step 2: Load and prepare audio
        print("\n🎧 Loading audio...")
        original_audio, sr = librosa.load(audio_path, sr=self.sr, mono=False)
        if original_audio.ndim == 1:
            original_audio = np.stack([original_audio, original_audio])
        
        # Step 3: Generate AI predictions with musical context
        print("\n🤖 Generating AI predictions with musical context...")
        predictions = self.predict_with_musical_context(audio_path, musical_context)
        
        # Step 4: Apply musical mixing strategy
        print("\n🎛️ Applying musical mixing strategy...")
        musically_mixed = self.apply_musical_mixing(original_audio, sr, mixing_strategy, musical_context)
        
        # Step 5: Apply AI predictions with musical adjustments
        mixed_versions = {}
        for model_name, params in predictions.items():
            print(f"\n🎵 {model_name} mixing...")
            
            # Adjust AI parameters based on musical context
            adjusted_params = self._adjust_params_for_musicality(params, musical_context, mixing_strategy)
            
            # Apply mixing
            mixed_audio = self.apply_enhanced_mixing(original_audio.copy(), sr, adjusted_params, musical_context)
            mixed_versions[model_name] = mixed_audio
        
        # Step 6: Save results
        results = self._save_all_versions(audio_path, original_audio, musically_mixed, mixed_versions, 
                                        musical_context, mixing_strategy, output_dir)
        
        print(f"\n✅ Musical AI mixing complete!")
        print(f"📁 Files saved to: {output_dir}")
        
        return results
    
    def predict_with_musical_context(self, audio_path: str, musical_context: AdvancedMusicalContext) -> Dict:
        """Generate AI predictions enhanced with musical context"""
        predictions = {}
        
        # Get base predictions from each model
        if 'Baseline CNN' in self.models:
            base_pred = self._predict_from_spectrogram(audio_path, self.models['Baseline CNN'])
            predictions['Baseline CNN (Musical)'] = self._enhance_prediction_with_musicality(base_pred, musical_context)
        
        if 'Enhanced CNN' in self.models:
            enhanced_pred = self._predict_from_spectrogram(audio_path, self.models['Enhanced CNN'])
            predictions['Enhanced CNN (Musical)'] = self._enhance_prediction_with_musicality(enhanced_pred, musical_context)
        
        # Musical feature-based prediction
        musical_pred = self._predict_from_musical_features(audio_path, musical_context)
        predictions['Musical AI'] = musical_pred
        
        return predictions
    
    def _predict_from_spectrogram(self, audio_path: str, model) -> np.ndarray:
        """Generate prediction from spectrogram using trained model"""
        audio, sr = librosa.load(audio_path, sr=22050, mono=True)  # Model training resolution
        
        # Create mel spectrogram
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
        
        # Ensure consistent dimensions
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
    
    def _predict_from_musical_features(self, audio_path: str, musical_context: AdvancedMusicalContext) -> np.ndarray:
        """Generate mixing parameters based on musical analysis"""
        # Genre-based base parameters
        genre_params = {
            'pop': [0.7, 0.6, 0.7, 0.5, 0.6, 0.8, 0.4, 0.3, 0.5],
            'rock': [0.8, 0.7, 0.6, 0.7, 0.8, 0.6, 0.2, 0.4, 0.7],
            'electronic': [0.6, 0.5, 0.8, 0.4, 0.9, 0.7, 0.6, 0.5, 0.6],
            'jazz': [0.4, 0.3, 0.5, 0.6, 0.5, 0.4, 0.7, 0.2, 0.5],
            'ballad': [0.5, 0.4, 0.6, 0.7, 0.4, 0.5, 0.8, 0.3, 0.6]
        }
        
        base = genre_params.get(musical_context.genre, genre_params['pop'])
        
        # Tempo adjustments
        tempo_factor = musical_context.tempo / 120.0  # Normalize around 120 BPM
        
        # Energy adjustments
        energy_factor = musical_context.energy_level
        
        # Valence adjustments (bright vs dark)
        valence_factor = musical_context.valence
        
        # Create musical parameters
        musical_params = []
        for i, param in enumerate(base):
            adjusted = param
            
            # Apply musical adjustments
            if i == 1:  # Compression
                adjusted *= (1.0 + energy_factor * 0.3)
            elif i in [2, 3, 4]:  # EQ
                if i == 4:  # High frequencies
                    adjusted *= (0.7 + valence_factor * 0.6)
                elif i == 2:  # Low frequencies
                    adjusted *= (0.8 + energy_factor * 0.4)
            elif i == 6:  # Reverb
                adjusted *= (2.0 - energy_factor)  # Less reverb for high energy
            elif i == 8:  # Output level
                adjusted *= (0.8 + energy_factor * 0.4)
            
            # Tempo-based adjustments
            if musical_context.tempo > 140:  # Fast tempo
                if i == 6:  # Reduce reverb
                    adjusted *= 0.7
                elif i == 7:  # Increase delay
                    adjusted *= 1.3
            elif musical_context.tempo < 80:  # Slow tempo
                if i == 6:  # Increase reverb
                    adjusted *= 1.4
                elif i == 1:  # Gentle compression
                    adjusted *= 0.8
            
            musical_params.append(np.clip(adjusted, 0.0, 1.0))
        
        return np.array(musical_params)
    
    def _enhance_prediction_with_musicality(self, prediction: np.ndarray, 
                                          musical_context: AdvancedMusicalContext) -> np.ndarray:
        """Enhance AI predictions with musical intelligence"""
        enhanced = prediction.copy()
        
        # Genre-specific adjustments
        if musical_context.genre == 'jazz':
            enhanced[1] *= 0.7  # Gentler compression
            enhanced[6] *= 1.3  # More reverb
        elif musical_context.genre == 'rock':
            enhanced[1] *= 1.2  # More compression
            enhanced[4] *= 1.1  # Boost presence
        elif musical_context.genre == 'electronic':
            enhanced[2] *= 1.2  # Boost highs
            enhanced[8] *= 1.1  # Wider stereo
        
        # Tempo-based adjustments
        if musical_context.tempo > 140:
            enhanced[6] *= 0.8  # Less reverb for fast tracks
            enhanced[7] *= 1.2  # More delay for rhythm
        elif musical_context.tempo < 80:
            enhanced[6] *= 1.4  # More reverb for slow tracks
            enhanced[1] *= 0.8  # Gentler compression
        
        # Energy-based adjustments
        if musical_context.energy_level > 0.8:
            enhanced[1] *= 1.2  # More compression for high energy
            enhanced[4] *= 1.1  # Boost presence
        elif musical_context.energy_level < 0.3:
            enhanced[1] *= 0.8  # Gentle compression for low energy
            enhanced[6] *= 1.3  # More reverb for atmosphere
        
        # Key/mode adjustments
        if musical_context.mode == 'minor':
            enhanced[3] *= 0.9  # Slightly warmer mids
            enhanced[6] *= 1.2  # More reverb for emotional depth
        
        return np.clip(enhanced, 0.0, 1.0)
    
    def apply_musical_mixing(self, audio: np.ndarray, sr: int, 
                           mixing_strategy: Dict, musical_context: AdvancedMusicalContext) -> np.ndarray:
        """Apply musical mixing strategy"""
        processed = audio.copy()
        
        # Apply processors based on musical context
        if musical_context.compression_style == 'vintage':
            processed = self.processors['vintage_compressor'](processed, sr, mixing_strategy['compression'])
        else:
            processed = self.processors['modern_compressor'](processed, sr, mixing_strategy['compression'])
        
        # Musical EQ
        processed = self.processors['musical_eq'](processed, sr, mixing_strategy['eq'], musical_context)
        
        # Harmonic enhancement for certain genres
        if musical_context.genre in ['rock', 'pop']:
            processed = self.processors['harmonic_exciter'](processed, sr, mixing_strategy.get('effects', {}))
        
        # Reverb with musical context
        processed = self.processors['musical_reverb'](processed, sr, mixing_strategy['reverb'], musical_context)
        
        # Tempo-synced delay
        if mixing_strategy.get('effects', {}).get('delay', 0) > 0.1:
            processed = self.processors['tempo_sync_delay'](processed, sr, musical_context.tempo, mixing_strategy['effects'])
        
        # Stereo enhancement
        processed = self.processors['stereo_enhancer'](processed, sr, mixing_strategy['stereo'], musical_context)
        
        # Final level management
        processed = self._apply_musical_dynamics(processed, mixing_strategy['dynamics'], musical_context)
        
        return processed
    
    def _print_musical_analysis(self, context: AdvancedMusicalContext, strategy: Dict):
        """Print detailed musical analysis"""
        print(f"\n🎼 Musical Analysis:")
        print(f"   Genre: {context.genre.title()} ({context.subgenre})")
        print(f"   Tempo: {context.tempo:.1f} BPM")
        print(f"   Key: {context.key} {context.mode}")
        print(f"   Energy: {context.energy_level:.2f}")
        print(f"   Danceability: {context.danceability:.2f}")
        print(f"   Valence: {context.valence:.2f} ({'positive' if context.valence > 0.6 else 'neutral' if context.valence > 0.4 else 'melancholic'})")
        print(f"   Vocal Presence: {context.vocal_presence:.2f}")
        print(f"   Dynamic Range: {context.dynamic_range:.1f} dB")
        
        print(f"\n🎛️ Mixing Strategy:")
        print(f"   Compression: {strategy['compression']['ratio']:.1f}:1 ({strategy['compression']['knee']} knee)")
        print(f"   EQ Character: {strategy['eq']['character']}")
        print(f"   Reverb: {strategy['reverb']['type']} style")
        print(f"   Target LUFS: {strategy['dynamics']['target_lufs']} dB")
        print(f"   Stereo Width: {strategy['stereo']['width']}")
      # Audio processing methods would continue here...
    # (vintage_compressor, modern_compressor, musical_eq, etc.)
    
    def _vintage_compressor(self, audio, sr, comp_settings):
        """Vintage-style compression with musical character"""
        ratio = comp_settings.get('ratio', 3.0)
        attack = comp_settings.get('attack', 0.003)  # 3ms
        release = comp_settings.get('release', 0.1)  # 100ms
        threshold = comp_settings.get('threshold', -12)  # dB
        
        # Convert to mono for processing if stereo
        if audio.ndim > 1:
            mono_audio = np.mean(audio, axis=0)
            is_stereo = True
        else:
            mono_audio = audio
            is_stereo = False
        
        # Simple peak detection and compression
        compressed = mono_audio.copy()
        envelope = 0.0
        
        for i in range(len(mono_audio)):
            # Peak detection
            peak = abs(mono_audio[i])
            
            # Envelope follower
            if peak > envelope:
                envelope += (peak - envelope) * attack
            else:
                envelope -= envelope * release
            
            # Apply compression
            if envelope > 10**(threshold/20):
                gain_reduction = 1.0 / (1.0 + (envelope / 10**(threshold/20) - 1.0) * (ratio - 1.0) / ratio)
                compressed[i] *= gain_reduction
        
        # Add vintage character (subtle harmonic distortion)
        vintage_character = 0.02  # 2% harmonic content
        compressed = np.tanh(compressed * (1 + vintage_character))
        
        # Restore stereo if needed
        if is_stereo:
            return np.stack([compressed, compressed])
        return compressed
    
    def _modern_compressor(self, audio, sr, comp_settings):
        """Modern transparent compression"""
        ratio = comp_settings.get('ratio', 2.0)
        attack = comp_settings.get('attack', 0.001)  # 1ms
        release = comp_settings.get('release', 0.05)  # 50ms
        threshold = comp_settings.get('threshold', -18)  # dB
        
        # More transparent compression with lookahead
        if audio.ndim > 1:
            processed = []
            for channel in audio:
                processed.append(self._apply_compression(channel, ratio, attack, release, threshold))
            return np.array(processed)
        else:
            return self._apply_compression(audio, ratio, attack, release, threshold)
    
    def _apply_compression(self, audio, ratio, attack, release, threshold):
        """Apply compression to single channel"""
        compressed = audio.copy()
        envelope = 0.0
        
        for i in range(len(audio)):
            peak = abs(audio[i])
            
            # Smooth envelope detection
            if peak > envelope:
                envelope += (peak - envelope) * attack
            else:
                envelope -= envelope * release * 0.1
            
            # Soft knee compression
            thresh_linear = 10**(threshold/20)
            if envelope > thresh_linear:
                over_thresh = envelope / thresh_linear
                gain_reduction = over_thresh ** (1.0/ratio - 1.0)
                compressed[i] *= gain_reduction
        
        return compressed
    
    def _musical_eq(self, audio, sr, eq_settings, musical_context):
        """Musically-aware EQ processing"""
        from scipy.signal import butter, filtfilt
        
        # EQ bands based on musical context
        processed = audio.copy()
        
        # Apply genre-specific EQ
        if musical_context.genre == 'rock':
            # Boost presence and cut muddiness
            processed = self._apply_eq_band(processed, sr, 200, -2, 1.0)  # Cut mud
            processed = self._apply_eq_band(processed, sr, 3000, 2, 1.2)  # Boost presence
        elif musical_context.genre == 'pop':
            # Bright and clear
            processed = self._apply_eq_band(processed, sr, 100, 1, 0.8)   # Gentle low boost
            processed = self._apply_eq_band(processed, sr, 8000, 2, 1.5)  # Air boost
        elif musical_context.genre == 'jazz':
            # Natural and warm
            processed = self._apply_eq_band(processed, sr, 1000, -1, 0.7) # Slight mid cut
            processed = self._apply_eq_band(processed, sr, 200, 1, 0.9)   # Warmth
        
        # Valence-based adjustments
        if musical_context.valence > 0.7:  # Happy music
            processed = self._apply_eq_band(processed, sr, 5000, 1, 1.0)  # Brighter
        elif musical_context.valence < 0.3:  # Sad music
            processed = self._apply_eq_band(processed, sr, 2000, -1, 0.8) # Warmer/darker
        
        return processed
    
    def _apply_eq_band(self, audio, sr, freq, gain_db, q):
        """Apply parametric EQ band"""
        from scipy.signal import butter, filtfilt
        
        if audio.ndim > 1:
            processed = []
            for channel in audio:
                processed.append(self._eq_single_band(channel, sr, freq, gain_db, q))
            return np.array(processed)
        else:
            return self._eq_single_band(audio, sr, freq, gain_db, q)
    
    def _eq_single_band(self, audio, sr, freq, gain_db, q):
        """Apply EQ to single channel"""
        # Simple shelving/peaking filter implementation
        if gain_db == 0:
            return audio
        
        nyquist = sr / 2
        normalized_freq = freq / nyquist
        
        if normalized_freq >= 1.0:
            return audio
        
        # Create a simple filter approximation
        gain_linear = 10**(gain_db/20)
        
        # High shelf for frequencies above 1kHz
        if freq > 1000:
            b, a = butter(2, normalized_freq, btype='high')
            filtered = filtfilt(b, a, audio)
            return audio + (filtered - audio) * (gain_linear - 1) * 0.5
        # Low shelf for frequencies below 500Hz
        elif freq < 500:
            b, a = butter(2, normalized_freq, btype='low')
            filtered = filtfilt(b, a, audio)
            return audio + (filtered - audio) * (gain_linear - 1) * 0.5
        else:
            # Peaking filter approximation
            return audio * gain_linear
    
    def _harmonic_exciter(self, audio, sr, fx_settings):
        """Harmonic excitation for warmth and presence"""
        exciter_amount = fx_settings.get('exciter', 0.1)
        
        if exciter_amount <= 0:
            return audio
        
        # Add subtle harmonic distortion
        if audio.ndim > 1:
            processed = []
            for channel in audio:
                # Add second and third harmonics
                harmonics = np.tanh(channel * 2) * 0.1 + np.tanh(channel * 3) * 0.05
                enhanced = channel + harmonics * exciter_amount
                processed.append(enhanced)
            return np.array(processed)
        else:
            harmonics = np.tanh(audio * 2) * 0.1 + np.tanh(audio * 3) * 0.05
            return audio + harmonics * exciter_amount
    
    def _stereo_enhancer(self, audio, sr, stereo_settings, musical_context):
        """Musical stereo enhancement"""
        if audio.ndim == 1:
            return audio
        
        width = stereo_settings.get('width', 1.0)
        bass_mono = stereo_settings.get('bass_management', True)
        
        left, right = audio[0], audio[1]
        
        # Mid/Side processing
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        
        # Apply stereo width
        side *= width
        
        # Bass management - keep low frequencies mono
        if bass_mono:
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 150 / (sr/2), btype='low')
            low_mid = filtfilt(b, a, mid)
            
            # High frequencies with stereo enhancement
            b, a = butter(2, 150 / (sr/2), btype='high')
            high_mid = filtfilt(b, a, mid)
            high_side = filtfilt(b, a, side)
            
            # Recombine
            enhanced_mid = low_mid + high_mid
            enhanced_side = high_side
        else:
            enhanced_mid = mid
            enhanced_side = side
        
        # Convert back to L/R
        left_out = enhanced_mid + enhanced_side
        right_out = enhanced_mid - enhanced_side
        
        return np.array([left_out, right_out])
    
    def _musical_reverb(self, audio, sr, reverb_settings, musical_context):
        """Context-aware reverb processing"""
        reverb_send = reverb_settings.get('send', 0.2)
        reverb_type = reverb_settings.get('type', 'hall')
        
        if reverb_send <= 0:
            return audio
        
        # Simple reverb using multiple delays
        delay_times = {
            'intimate': [0.02, 0.04, 0.08],
            'hall': [0.03, 0.07, 0.15, 0.31],
            'plate': [0.015, 0.025, 0.045, 0.085],
            'spacious': [0.05, 0.12, 0.25, 0.5]
        }.get(reverb_type, [0.03, 0.07, 0.15])
        
        # Apply musical context adjustments
        if musical_context.tempo > 140:  # Fast tempo
            delay_times = [t * 0.7 for t in delay_times]  # Shorter reverb
            reverb_send *= 0.8
        elif musical_context.tempo < 80:  # Slow tempo
            delay_times = [t * 1.3 for t in delay_times]  # Longer reverb
            reverb_send *= 1.2
        
        reverb_signal = np.zeros_like(audio)
        
        if audio.ndim > 1:
            for i, delay_time in enumerate(delay_times):
                delay_samples = int(delay_time * sr)
                if delay_samples < len(audio[0]):
                    decay = 0.7 ** (i + 1)
                    for ch in range(audio.shape[0]):
                        delayed = np.concatenate([np.zeros(delay_samples), audio[ch][:-delay_samples]])
                        reverb_signal[ch] += delayed * decay
        else:
            for i, delay_time in enumerate(delay_times):
                delay_samples = int(delay_time * sr)
                if delay_samples < len(audio):
                    decay = 0.7 ** (i + 1)
                    delayed = np.concatenate([np.zeros(delay_samples), audio[:-delay_samples]])
                    reverb_signal += delayed * decay
          # Mix dry and wet signal
        return audio + reverb_signal * reverb_send
    
    def _tempo_sync_delay(self, audio, sr, tempo, fx_settings):
        """Tempo-synchronized delay effects"""
        delay_amount = fx_settings.get('delay', 0.1)
        
        if delay_amount <= 0:
            return audio
        
        # Calculate delay time based on tempo
        beat_duration = 60.0 / tempo  # seconds per beat
        delay_time = beat_duration / 4  # 16th note delay
        
        delay_samples = int(delay_time * sr)
        
        if delay_samples >= len(audio[0] if audio.ndim > 1 else audio):
            return audio
        
        if audio.ndim > 1:
            delayed = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                delayed[ch] = np.concatenate([np.zeros(delay_samples), 
                                            audio[ch][:-delay_samples]]) * 0.4
            return audio + delayed * delay_amount
        else:
            delayed = np.concatenate([np.zeros(delay_samples), 
                                    audio[:-delay_samples]]) * 0.4
            return audio + delayed * delay_amount
    
    def _adjust_params_for_musicality(self, params: np.ndarray, 
                                     musical_context: AdvancedMusicalContext, 
                                     mixing_strategy: Dict) -> np.ndarray:
        """Adjust AI parameters based on musical context and strategy"""
        adjusted = params.copy()
        
        # Apply mixing strategy adjustments
        compression_ratio = mixing_strategy['compression']['ratio']
        eq_character = mixing_strategy['eq']['character']
        reverb_send = mixing_strategy['reverb']['send']
        
        # Map AI parameters to musical mixing strategy
        if len(adjusted) >= 10:
            # Compression adjustments
            adjusted[1] *= (compression_ratio / 3.0)  # Normalize to AI scale
            
            # EQ adjustments based on character
            if eq_character == 'bright':
                adjusted[4] *= 1.2  # Boost high frequencies
                adjusted[3] *= 0.9  # Slight mid cut
            elif eq_character == 'warm':
                adjusted[2] *= 1.1  # Boost low mids
                adjusted[4] *= 0.9  # Reduce highs
            elif eq_character == 'scooped':
                adjusted[3] *= 0.8  # Cut mids
                adjusted[2] *= 1.1  # Boost lows
                adjusted[4] *= 1.1  # Boost highs
            
            # Reverb adjustments
            if len(adjusted) > 6:
                adjusted[6] = reverb_send if isinstance(reverb_send, (int, float)) else reverb_send[0]
        
        # Ensure parameters stay in valid range
        return np.clip(adjusted, 0.0, 1.0)
    
    def apply_enhanced_mixing(self, audio: np.ndarray, sr: int, 
                            params: np.ndarray, musical_context: AdvancedMusicalContext) -> np.ndarray:
        """Apply enhanced mixing with both AI parameters and musical processing"""
        # First apply the standard AI mixing (from existing mixer)
        try:
            from ai_mixer import AIMixer
            # Create temporary AI mixer instance
            temp_mixer = AIMixer()
            ai_mixed = temp_mixer.apply_mixing_parameters(audio, sr, params)
        except ImportError:
            print("⚠️ ai_mixer not found, using basic processing")
            ai_mixed = audio.copy()
        
        # Then apply musical enhancements
        enhanced = self._apply_musical_enhancements(ai_mixed, sr, musical_context)
        
        return enhanced
    
    def _apply_musical_enhancements(self, audio: np.ndarray, sr: int, 
                                  musical_context: AdvancedMusicalContext) -> np.ndarray:
        """Apply musical intelligence enhancements on top of AI mixing"""
        enhanced = audio.copy()
        
        # Genre-specific processing
        if musical_context.genre == 'jazz':
            # Add warmth and space
            enhanced = self._harmonic_exciter(enhanced, sr, {'exciter': 0.05})
            enhanced = self._musical_reverb(enhanced, sr, 
                                          {'send': 0.15, 'type': 'hall'}, musical_context)
        
        elif musical_context.genre == 'rock':
            # Add presence and punch
            enhanced = self._harmonic_exciter(enhanced, sr, {'exciter': 0.1})
            enhanced = self._musical_reverb(enhanced, sr, 
                                          {'send': 0.08, 'type': 'plate'}, musical_context)
        
        elif musical_context.genre == 'electronic':
            # Wide stereo and spatial effects
            if enhanced.ndim > 1:
                enhanced = self._stereo_enhancer(enhanced, sr, 
                                               {'width': 1.3, 'bass_management': True}, musical_context)
            enhanced = self._tempo_sync_delay(enhanced, sr, musical_context.tempo, {'delay': 0.1})
        
        elif musical_context.genre == 'pop':
            # Bright and clear
            enhanced = self._harmonic_exciter(enhanced, sr, {'exciter': 0.07})
            enhanced = self._musical_reverb(enhanced, sr, 
                                          {'send': 0.12, 'type': 'intimate'}, musical_context)
        
        # Energy-based adjustments
        if musical_context.energy_level > 0.8:
            # High energy - add excitement
            enhanced = self._harmonic_exciter(enhanced, sr, {'exciter': 0.05})
        elif musical_context.energy_level < 0.3:
            # Low energy - add space and warmth
            enhanced = self._musical_reverb(enhanced, sr, 
                                          {'send': 0.2, 'type': 'hall'}, musical_context)
        
        return enhanced
    
    def _apply_musical_dynamics(self, audio: np.ndarray, dynamics_settings: Dict, 
                              musical_context: AdvancedMusicalContext) -> np.ndarray:
        """Apply musical dynamics processing"""
        target_lufs = dynamics_settings.get('target_lufs', -14)
        dynamic_range = dynamics_settings.get('dynamic_range', 12)
        
        # Simple peak limiting for now
        peak_level = np.max(np.abs(audio))
        if peak_level > 0.95:  # Prevent clipping
            audio = audio * (0.95 / peak_level)
        
        return audio
    
    def _save_all_versions(self, input_path: str, original_audio: np.ndarray, 
                          musically_mixed: np.ndarray, mixed_versions: Dict,
                          musical_context: AdvancedMusicalContext, mixing_strategy: Dict,
                          output_dir: str) -> Dict:
        """Save all versions and generate analysis report"""
        results = {
            'input_file': str(input_path),
            'musical_analysis': {
                'genre': f"{musical_context.genre} ({musical_context.subgenre})",
                'tempo': musical_context.tempo,
                'key': f"{musical_context.key} {musical_context.mode}",
                'energy': musical_context.energy_level,
                'danceability': musical_context.danceability,
                'valence': musical_context.valence,
                'vocal_presence': musical_context.vocal_presence,
                'dynamic_range': musical_context.dynamic_range
            },
            'mixing_strategy': mixing_strategy,
            'output_files': {}
        }
        
        output_dir = Path(output_dir)
        base_name = Path(input_path).stem
        
        # Save original for comparison
        original_path = output_dir / f"{base_name}_original.wav"
        sf.write(str(original_path), original_audio.T, self.sr)
        results['output_files']['original'] = str(original_path)
        
        # Save musical mixing version
        musical_path = output_dir / f"{base_name}_musical_strategy.wav"
        sf.write(str(musical_path), musically_mixed.T, self.sr)
        results['output_files']['musical_strategy'] = str(musical_path)
        
        # Save AI enhanced versions
        for model_name, mixed_audio in mixed_versions.items():
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            output_path = output_dir / f"{base_name}_{safe_name}.wav"
            sf.write(str(output_path), mixed_audio.T, self.sr)
            results['output_files'][safe_name] = str(output_path)
        
        # Save analysis report
        report_path = output_dir / f"{base_name}_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

if __name__ == "__main__":
    # Test the production AI mixer
    mixer = ProductionAIMixer()
    
    # Example usage
    test_file = Path(__file__).parent.parent / "data" / "test" / "sample.wav"
    if test_file.exists():
        results = mixer.analyze_and_mix(str(test_file))
        print(f"\n🎉 Musical mixing complete! Check the mixed_outputs folder.")
    else:
        print("🎵 Production AI Mixer initialized and ready!")
        print("Usage: mixer.analyze_and_mix('path/to/audio.wav')")
