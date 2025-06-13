#!/usr/bin/env python3
"""
Musical Intelligence Training - Making AI Mixing Musically Aware
This focuses on training the AI to understand musical context, genre conventions,
and human preferences rather than just technical metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler

@dataclass
class MusicalContext:
    """Musical context information for training"""
    genre: str
    tempo: float
    key: str
    time_signature: str
    energy_level: float  # 0-1
    danceability: float  # 0-1
    valence: float      # 0-1 (musical positivity)
    instrumentalness: float  # 0-1
    
class MusicalFeatureExtractor:
    """Extract musical features for context-aware mixing"""
    
    def __init__(self):
        self.tempo_range = (60, 200)  # BPM
        self.key_signatures = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
    def extract_musical_features(self, audio_path: str, sr: int = 44100) -> MusicalContext:
        """Extract comprehensive musical context from audio"""
        audio, _ = librosa.load(audio_path, sr=sr)
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Key detection (simplified - real implementation would use more sophisticated methods)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        key = self.key_signatures[key_idx]
        
        # Energy level (RMS energy)
        rms = librosa.feature.rms(y=audio)[0]
        energy_level = np.mean(rms)
        
        # Spectral features for genre classification
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
        
        # Estimate musical characteristics
        # These would typically come from a trained classifier or manual labels
        danceability = self._estimate_danceability(tempo, beats, energy_level)
        valence = self._estimate_valence(spectral_centroid, key, mfccs)
        instrumentalness = self._estimate_instrumentalness(audio, sr)
        
        # Simple genre classification based on spectral features
        genre = self._classify_genre(spectral_centroid, spectral_rolloff, tempo, energy_level)
        
        return MusicalContext(
            genre=genre,
            tempo=float(tempo),
            key=key,
            time_signature="4/4",  # Simplified - would need beat analysis
            energy_level=float(energy_level),
            danceability=danceability,
            valence=valence,
            instrumentalness=instrumentalness
        )
    
    def _estimate_danceability(self, tempo: float, beats: np.ndarray, energy: float) -> float:
        """Estimate how danceable the track is"""
        # Danceability correlates with steady tempo, strong beat, and moderate energy
        tempo_score = 1.0 if 120 <= tempo <= 130 else max(0, 1 - abs(tempo - 125) / 50)
        beat_consistency = 1.0 - np.std(np.diff(beats)) / np.mean(np.diff(beats)) if len(beats) > 1 else 0.5
        energy_score = min(1.0, energy * 3)  # Higher energy = more danceable
        
        return (tempo_score + beat_consistency + energy_score) / 3
    
    def _estimate_valence(self, spectral_centroid: float, key: str, mfccs: np.ndarray) -> float:
        """Estimate musical positivity/happiness"""
        # Major keys and brighter sounds tend to be more positive
        major_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        key_valence = 0.7 if key in major_keys else 0.3
        
        # Higher spectral centroid often correlates with brighter, happier sound
        brightness_valence = min(1.0, spectral_centroid / 3000)
        
        # MFCC patterns can indicate emotional content
        mfcc_valence = 0.5 + (mfccs[1] - mfccs[2]) / 20  # Simplified heuristic
        mfcc_valence = np.clip(mfcc_valence, 0, 1)
        
        return (key_valence + brightness_valence + mfcc_valence) / 3
    
    def _estimate_instrumentalness(self, audio: np.ndarray, sr: int) -> float:
        """Estimate how instrumental (vs vocal) the track is"""
        # Look for vocal frequency ranges and harmonic patterns
        # This is simplified - real implementation would use vocal detection
        
        # Vocals typically have energy in 85-255 Hz (fundamental) and harmonics
        vocal_freqs = [85, 255, 500, 1000, 2000]
        
        # Get power spectral density
        f, psd = signal.welch(audio, fs=sr, nperseg=4096)
        
        vocal_energy = 0
        total_energy = np.sum(psd)
        
        for vocal_freq in vocal_freqs:
            idx = np.argmin(np.abs(f - vocal_freq))
            vocal_energy += psd[idx]
        
        # Higher vocal energy = less instrumental
        instrumentalness = 1.0 - min(1.0, vocal_energy / total_energy * 10)
        return float(instrumentalness)
    
    def _classify_genre(self, centroid: float, rolloff: float, tempo: float, energy: float) -> str:
        """Simple genre classification based on audio features"""
        # This is very simplified - real implementation would use trained classifier
        
        if tempo > 140 and energy > 0.3 and centroid > 2000:
            return "electronic"
        elif tempo < 100 and centroid < 1500:
            return "ballad"
        elif 120 <= tempo <= 140 and energy > 0.25:
            return "pop"
        elif tempo > 160 and energy > 0.4:
            return "rock"
        elif tempo < 80:
            return "ambient"
        else:
            return "general"

class GenreAwareMixingParameters:
    """Define mixing conventions for different genres"""
    
    def __init__(self):
        self.genre_templates = {
            "pop": {
                "compression_ratio": (1.5, 3.0),
                "eq_low": (-1, 1),      # Controlled low end
                "eq_mid": (-0.5, 1.5),  # Vocal clarity
                "eq_high": (0, 2),      # Brightness for radio
                "reverb": (0.1, 0.3),   # Moderate reverb
                "stereo_width": (1.0, 1.1),  # Slight widening
                "target_lufs": -14,     # Streaming standard
                "dynamic_range": (8, 12)  # Moderate compression
            },
            "rock": {
                "compression_ratio": (2.0, 4.0),
                "eq_low": (0, 2),       # Strong low end
                "eq_mid": (-1, 0.5),    # Guitar clarity
                "eq_high": (0, 1.5),    # Presence without harshness
                "reverb": (0.05, 0.2),  # Tighter reverb
                "stereo_width": (1.0, 1.2),  # Wide guitars
                "target_lufs": -12,     # Louder for energy
                "dynamic_range": (6, 10)  # More compression
            },
            "electronic": {
                "compression_ratio": (1.0, 2.5),
                "eq_low": (0, 3),       # Strong sub bass
                "eq_mid": (-1, 0),      # Clean mids
                "eq_high": (0, 2),      # Sparkle and air
                "reverb": (0.2, 0.5),   # Spacious reverb
                "stereo_width": (1.1, 1.3),  # Wide stereo field
                "target_lufs": -16,     # Preserve dynamics
                "dynamic_range": (10, 16)  # Less compression
            },
            "ballad": {
                "compression_ratio": (1.0, 2.0),
                "eq_low": (-1, 0),      # Gentle low end
                "eq_mid": (0, 1),       # Vocal intimacy
                "eq_high": (-0.5, 1),   # Warmth over brightness
                "reverb": (0.3, 0.6),   # Lush reverb
                "stereo_width": (0.9, 1.1),  # Natural width
                "target_lufs": -18,     # Preserve dynamics
                "dynamic_range": (12, 20)  # Minimal compression
            },
            "ambient": {
                "compression_ratio": (1.0, 1.5),
                "eq_low": (-2, 1),      # Controlled low end
                "eq_mid": (-1, 0),      # Clean, uncolored
                "eq_high": (-1, 1),     # Natural highs
                "reverb": (0.4, 0.8),   # Very spacious
                "stereo_width": (1.2, 1.5),  # Very wide
                "target_lufs": -20,     # Very dynamic
                "dynamic_range": (15, 25)  # Minimal compression
            }
        }
        
    def get_genre_parameters(self, genre: str) -> Dict:
        """Get mixing parameter ranges for a specific genre"""
        return self.genre_templates.get(genre, self.genre_templates["pop"])

class MusicallyAwareLoss(nn.Module):
    """Loss function that considers musical context and human preferences"""
    
    def __init__(self):
        super().__init__()
        self.parameter_weight = 0.3      # Technical parameter accuracy
        self.perceptual_weight = 0.3     # Perceptual audio quality
        self.musical_weight = 0.25       # Musical appropriateness
        self.preference_weight = 0.15    # Human preference alignment
        
        self.genre_params = GenreAwareMixingParameters()
        
    def musical_appropriateness_loss(self, predicted_params, musical_context: MusicalContext):
        """Penalty for parameters that don't fit the musical genre/style"""
        genre_template = self.genre_params.get_genre_parameters(musical_context.genre)
        
        losses = []
        param_names = ['compression_ratio', 'eq_low', 'eq_mid', 'eq_high', 'reverb', 'stereo_width']
        
        for i, param_name in enumerate(param_names):
            if i < predicted_params.shape[-1] and param_name in genre_template:
                param_value = predicted_params[..., i]
                min_val, max_val = genre_template[param_name]
                
                # Penalty for being outside genre-appropriate range
                below_penalty = F.relu(min_val - param_value)
                above_penalty = F.relu(param_value - max_val)
                losses.append(below_penalty + above_penalty)
        
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)
    
    def perceptual_quality_loss(self, processed_audio, target_audio):
        """Perceptual audio quality based on psychoacoustic principles"""
        # Multi-resolution STFT for perceptual analysis
        stft_losses = []
        
        # Different window sizes capture different aspects
        window_sizes = [512, 1024, 2048]
        
        for win_size in window_sizes:
            hop_length = win_size // 4
            
            # STFT of both signals
            pred_stft = torch.stft(processed_audio, n_fft=win_size, hop_length=hop_length, 
                                 window=torch.hann_window(win_size), return_complex=True)
            target_stft = torch.stft(target_audio, n_fft=win_size, hop_length=hop_length,
                                   window=torch.hann_window(win_size), return_complex=True)
            
            # Magnitude difference (more perceptually relevant than complex difference)
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            
            # Log magnitude for perceptual scaling
            pred_log = torch.log(pred_mag + 1e-8)
            target_log = torch.log(target_mag + 1e-8)
            
            stft_loss = F.mse_loss(pred_log, target_log)
            stft_losses.append(stft_loss)
        
        return torch.mean(torch.stack(stft_losses))
    
    def preference_alignment_loss(self, predicted_params, user_preferences):
        """Align with user's demonstrated preferences"""
        # This would learn from user feedback over time
        # For now, implement basic preference constraints
        
        # Example: User prefers less compression
        compression_preference = user_preferences.get('compression_style', 'moderate')
        
        if compression_preference == 'gentle':
            # Penalty for high compression ratios
            compression_penalty = F.relu(predicted_params[..., 0] - 2.0)  # Assuming compression is first param
            return compression_penalty.mean()
        
        return torch.tensor(0.0)
    
    def forward(self, predicted_params, target_params, processed_audio, target_audio, 
                musical_context: MusicalContext, user_preferences: Dict = None):
        """Combined musical loss function"""
        
        # Technical parameter accuracy
        param_loss = F.mse_loss(predicted_params, target_params)
        
        # Perceptual audio quality
        perceptual_loss = self.perceptual_quality_loss(processed_audio, target_audio)
        
        # Musical appropriateness for genre
        musical_loss = self.musical_appropriateness_loss(predicted_params, musical_context)
        
        # User preference alignment
        preference_loss = self.preference_alignment_loss(predicted_params, user_preferences or {})
        
        total_loss = (self.parameter_weight * param_loss +
                     self.perceptual_weight * perceptual_loss +
                     self.musical_weight * musical_loss +
                     self.preference_weight * preference_loss)
        
        return total_loss, {
            'parameter': param_loss.item(),
            'perceptual': perceptual_loss.item(),
            'musical': musical_loss.item(),
            'preference': preference_loss.item()
        }

class MusicallyAwareCNN(nn.Module):
    """CNN that incorporates musical context for better mixing decisions"""
    
    def __init__(self, input_size=512, musical_context_size=8):
        super().__init__()
        
        # Audio feature extraction (multi-scale like before)
        self.audio_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)
        )
        
        # Musical context processing
        self.context_processor = nn.Sequential(
            nn.Linear(musical_context_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Combined feature processing
        self.feature_combine = nn.Linear(128 + 64, 128)  # Audio + context
        
        # Genre-aware parameter prediction
        self.parameter_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)  # 10 mixing parameters
        )
        
        self.genre_params = GenreAwareMixingParameters()
    
    def encode_musical_context(self, context: MusicalContext) -> torch.Tensor:
        """Convert musical context to tensor for neural network"""
        # Encode categorical and numerical features
        features = [
            context.tempo / 200.0,  # Normalize tempo
            context.energy_level,
            context.danceability,
            context.valence,
            context.instrumentalness,
            # One-hot encode genre (simplified to 3 main categories)
            1.0 if context.genre in ['pop', 'rock'] else 0.0,
            1.0 if context.genre in ['electronic', 'ambient'] else 0.0,
            1.0 if context.genre == 'ballad' else 0.0
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def apply_genre_constraints(self, raw_params: torch.Tensor, genre: str) -> torch.Tensor:
        """Apply genre-specific parameter constraints"""
        genre_template = self.genre_params.get_genre_parameters(genre)
        constrained = torch.zeros_like(raw_params)
        
        param_names = ['compression_ratio', 'eq_low', 'eq_mid', 'eq_high', 'reverb', 'stereo_width']
        
        for i, param_name in enumerate(param_names):
            if i < raw_params.shape[-1] and param_name in genre_template:
                min_val, max_val = genre_template[param_name]
                
                # Apply sigmoid and scale to genre-appropriate range
                normalized = torch.sigmoid(raw_params[..., i])
                constrained[..., i] = min_val + normalized * (max_val - min_val)
            else:
                # Default constraint for unspecified parameters
                constrained[..., i] = torch.tanh(raw_params[..., i])
        
        return constrained
    
    def forward(self, audio_features: torch.Tensor, musical_context: MusicalContext) -> torch.Tensor:
        # Process audio features
        audio_feats = self.audio_conv(audio_features.unsqueeze(1))
        audio_feats = audio_feats.view(audio_feats.size(0), -1)
        
        # Process musical context
        context_tensor = self.encode_musical_context(musical_context)
        if audio_features.dim() > 1:  # Batch processing
            context_tensor = context_tensor.unsqueeze(0).repeat(audio_features.size(0), 1)
        context_feats = self.context_processor(context_tensor)
        
        # Combine features
        combined = torch.cat([audio_feats, context_feats], dim=-1)
        features = torch.relu(self.feature_combine(combined))
        
        # Predict parameters
        raw_params = self.parameter_predictor(features)
        
        # Apply genre-specific constraints
        constrained_params = self.apply_genre_constraints(raw_params, musical_context.genre)
        
        return constrained_params

def create_musical_training_system():
    """Create the complete musical intelligence training system"""
    
    print("🎵 Creating Musically Aware AI Mixing System")
    print("=" * 60)
    
    # Initialize components
    feature_extractor = MusicalFeatureExtractor()
    model = MusicallyAwareCNN()
    criterion = MusicallyAwareLoss()
    
    print("✅ Musical Feature Extraction:")
    print("   - Genre classification from audio")
    print("   - Tempo, key, and mood detection")
    print("   - Energy and danceability analysis")
    print("   - Instrumentalness detection")
    
    print("\n✅ Genre-Aware Parameter Constraints:")
    genre_params = GenreAwareMixingParameters()
    for genre, params in list(genre_params.genre_templates.items())[:3]:
        print(f"   - {genre.title()}: LUFS {params['target_lufs']}, "
              f"Compression {params['compression_ratio']}, "
              f"Dynamic Range {params['dynamic_range']} dB")
    
    print("\n✅ Musical Loss Function:")
    print("   - Parameter accuracy (30%)")
    print("   - Perceptual quality (30%)")
    print("   - Musical appropriateness (25%)")
    print("   - User preference alignment (15%)")
    
    print("\n✅ Context-Aware Architecture:")
    print("   - Audio feature extraction")
    print("   - Musical context processing")
    print("   - Genre-specific parameter constraints")
    print("   - Combined decision making")
    
    return {
        'feature_extractor': feature_extractor,
        'model': model,
        'criterion': criterion,
        'genre_params': genre_params
    }

if __name__ == "__main__":
    # Create the musical training system
    system = create_musical_training_system()
    
    print("\n🎯 This system makes AI mixing musically aware by:")
    print("1. Understanding genre conventions and musical context")
    print("2. Applying appropriate mixing styles for each genre")
    print("3. Considering perceptual quality over technical metrics")
    print("4. Learning from user preferences and feedback")
    print("5. Constraining parameters to musically sensible ranges")
    
    print("\n🚀 Next steps:")
    print("1. Train with genre-labeled audio data")
    print("2. Collect user preference feedback")
    print("3. Implement A/B testing for musical quality")
    print("4. Add more sophisticated musical feature analysis")
