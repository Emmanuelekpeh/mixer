"""
Base Audio Mixing Model Framework
Provides common functionality for all audio mixing architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import soundfile as sf
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class BaseAudioMixer(ABC):
    """Base class for all audio mixing models"""
    
    def __init__(self, model_id: str, config: Dict):
        self.model_id = model_id
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.loss_history = []
        self.performance_metrics = {
            'spectral_loss': [],
            'perceptual_loss': [],
            'temporal_loss': [],
            'total_loss': []
        }
        
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the neural network architecture"""
        pass
    
    @abstractmethod
    def forward(self, input_audio: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def get_mixing_strategy(self) -> str:
        """Return description of mixing strategy"""
        pass
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 44100) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=False)
            
            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            elif audio.shape[0] > 2:
                audio = audio[:2]  # Take first 2 channels
                
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Convert to tensor
            return torch.FloatTensor(audio).to(self.device)
            
        except Exception as e:
            print(f"Error preprocessing audio {audio_path}: {e}")
            # Return silence if error
            return torch.zeros(2, target_sr * 3).to(self.device)
    
    def spectral_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract spectral features for processing"""
        # Convert to numpy for librosa
        audio_np = audio.cpu().numpy()
        
        features = []
        for channel in range(audio_np.shape[0]):
            # STFT
            stft = librosa.stft(audio_np[channel], n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_np[channel], sr=44100, n_mels=128
            )
            
            # Combine features
            channel_features = np.concatenate([
                magnitude[:128],  # First 128 frequency bins
                mel_spec
            ], axis=0)
            
            features.append(channel_features)
        
        return torch.FloatTensor(np.array(features)).to(self.device)
    
    def calculate_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate comprehensive loss"""
        losses = {}
        
        # Time domain loss
        losses['temporal'] = nn.MSELoss()(prediction, target)
        
        # Spectral loss
        pred_spec = self.spectral_features(prediction)
        target_spec = self.spectral_features(target)
        losses['spectral'] = nn.MSELoss()(pred_spec, target_spec)
        
        # Perceptual loss (simplified)
        losses['perceptual'] = 0.5 * losses['temporal'] + 0.5 * losses['spectral']
        
        # Total loss
        losses['total'] = (
            0.3 * losses['temporal'] + 
            0.4 * losses['spectral'] + 
            0.3 * losses['perceptual']
        )
        
        return losses
    
    def train_step(self, input_audio: torch.Tensor, target_audio: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        prediction = self.forward(input_audio)
        
        # Calculate losses
        losses = self.calculate_loss(prediction, target_audio)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Return loss values
        return {k: v.item() for k, v in losses.items()}
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'loss_history': self.loss_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.performance_metrics = checkpoint.get('performance_metrics', self.performance_metrics)
            self.loss_history = checkpoint.get('loss_history', [])
    
    def mix_audio(self, audio_path: str, output_path: str) -> bool:
        """Mix audio file and save result"""
        try:
            self.model.eval()
            with torch.no_grad():
                # Load and preprocess
                input_audio = self.preprocess_audio(audio_path)
                
                # Forward pass
                mixed_audio = self.forward(input_audio)
                
                # Post-process
                mixed_audio = self.postprocess_audio(mixed_audio)
                
                # Save
                sf.write(output_path, mixed_audio.T, 44100)
                return True
                
        except Exception as e:
            print(f"Error mixing audio: {e}")
            return False
    
    def postprocess_audio(self, audio: torch.Tensor) -> np.ndarray:
        """Post-process mixed audio"""
        audio_np = audio.cpu().numpy()
        
        # Normalize
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.95
        
        # Apply subtle compression
        audio_np = np.tanh(audio_np * 1.2) * 0.8
        
        return audio_np
    
    def update_from_tournament_result(self, won: bool, opponent_id: str, confidence: float):
        """Update model based on tournament result"""
        # This will be implemented in each specific model
        pass
    
    def get_model_info(self) -> Dict:
        """Get current model information"""
        return {
            'model_id': self.model_id,
            'architecture': self.__class__.__name__,
            'mixing_strategy': self.get_mixing_strategy(),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'device': str(self.device),
            'performance_metrics': self.performance_metrics
        }
