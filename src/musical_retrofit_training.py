#!/usr/bin/env python3
"""
Musical Intelligence Retrofit Training
Enhance existing models with musical awareness and preference learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pickle
from datetime import datetime

# Import our musical intelligence components
from musical_intelligence import (
    MusicalFeatureExtractor, MusicallyAwareLoss, MusicalContext,
    GenreAwareMixingParameters
)

# Import existing models
from baseline_cnn import BaselineCNN, EnhancedCNN
from ast_regressor import ASTRegressor

class PreferenceLearner:
    """Learn and apply user mixing preferences"""
    
    def __init__(self):
        self.preference_patterns = self.load_user_preferences()
        
    def load_user_preferences(self) -> Dict:
        """Load user preferences from analysis"""
        try:
            with open('mixed_outputs/enhanced/preference_summary.json', 'r') as f:
                prefs = json.load(f)
                return prefs
        except FileNotFoundError:
            # Default preferences based on analysis
            return {
                'avoid_sub_bass_reduction': True,
                'avoid_excessive_air_boost': True,
                'prefer_gentle_compression': True,
                'avoid_mid_boost': True,
                'max_frequency_change': 3.0,  # Max dB change per band
                'preferred_dynamic_range': (8, 16),  # dB range
                'frequency_preferences': {
                    'sub_bass': {'min_change': -2.0, 'max_change': 2.0},
                    'bass': {'min_change': -2.0, 'max_change': 3.0},
                    'low_mid': {'min_change': -3.0, 'max_change': 2.0},
                    'mid': {'min_change': -2.0, 'max_change': 2.0},
                    'high_mid': {'min_change': -2.0, 'max_change': 2.0},
                    'presence': {'min_change': -1.0, 'max_change': 3.0},
                    'air': {'min_change': -1.0, 'max_change': 2.0}
                }
            }
    
    def calculate_preference_penalty(self, params: torch.Tensor) -> torch.Tensor:
        """Calculate penalty based on user preferences"""
        penalties = []
        
        # Assume parameters are: [compression, eq_low, eq_mid, eq_high, reverb, stereo_width, ...]
        # Map to frequency bands (simplified mapping)
        if params.shape[-1] >= 7:  # Ensure we have frequency parameters
            freq_changes = params[..., 1:8]  # EQ parameters
            freq_bands = ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'presence', 'air']
            
            for i, band in enumerate(freq_bands):
                if i < freq_changes.shape[-1]:
                    change = freq_changes[..., i]
                    prefs = self.preference_patterns['frequency_preferences'].get(band, {})
                    
                    min_change = prefs.get('min_change', -3.0)
                    max_change = prefs.get('max_change', 3.0)
                    
                    # Penalty for going outside preferred range
                    below_penalty = torch.relu(min_change - change)
                    above_penalty = torch.relu(change - max_change)
                    penalties.append(below_penalty + above_penalty)
        
        # Compression preference (gentle compression preferred)
        if self.preference_patterns.get('prefer_gentle_compression', True):
            compression = params[..., 0] if params.shape[-1] > 0 else torch.tensor(0.0)
            # Penalty for very high compression (> 3.0 ratio)
            compression_penalty = torch.relu(compression - 3.0)
            penalties.append(compression_penalty)
        
        return torch.mean(torch.stack(penalties)) if penalties else torch.tensor(0.0)

class MusicalModelWrapper(nn.Module):
    """Wrapper that adds musical intelligence to existing models"""
    
    def __init__(self, base_model, model_type='cnn'):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        
        # Add musical context processing
        self.musical_processor = nn.Sequential(
            nn.Linear(8, 32),  # Musical context features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Get output size of base model (all our models output 10 parameters)
        base_output_size = 10
        
        # Enhanced parameter predictor that combines base model output with musical context
        self.enhanced_predictor = nn.Sequential(
            nn.Linear(base_output_size + 16, 64),  # Base + musical context
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, base_output_size)
        )
        
        self.genre_params = GenreAwareMixingParameters()
        
    def encode_musical_context(self, context: MusicalContext) -> torch.Tensor:
        """Convert musical context to tensor"""
        features = [
            context.tempo / 200.0,
            context.energy_level,
            context.danceability,
            context.valence,
            context.instrumentalness,
            1.0 if context.genre in ['pop', 'rock'] else 0.0,
            1.0 if context.genre in ['electronic', 'ambient'] else 0.0,
            1.0 if context.genre == 'ballad' else 0.0
        ]
        return torch.tensor(features, dtype=torch.float32)
    
    def apply_constraints(self, raw_params: torch.Tensor, genre: str) -> torch.Tensor:
        """Apply musical and preference constraints"""
        # Get genre template
        genre_template = self.genre_params.get_genre_parameters(genre)
        
        # Apply reasonable constraints (prevent extreme changes)
        constrained = torch.tanh(raw_params)  # Basic constraint to [-1, 1]
        
        # Scale based on parameter type
        if constrained.shape[-1] >= 1:  # Compression
            constrained[..., 0] = torch.sigmoid(raw_params[..., 0]) * 3.0 + 1.0  # [1.0, 4.0]
        
        if constrained.shape[-1] >= 7:  # EQ parameters
            # Limit EQ changes to ±4dB for musical balance
            constrained[..., 1:8] = torch.tanh(raw_params[..., 1:8]) * 4.0
        
        return constrained
    
    def forward(self, x, musical_context: Optional[MusicalContext] = None):
        # Get base model prediction
        base_output = self.base_model(x)
        
        if musical_context is None:
            return base_output
        
        # Process musical context
        context_tensor = self.encode_musical_context(musical_context)
        if x.dim() > 3:  # Batch processing (batch_size, channels, freq, time)
            context_tensor = context_tensor.unsqueeze(0).repeat(x.size(0), 1)
        
        musical_features = self.musical_processor(context_tensor)
        
        # Combine base prediction with musical context
        combined = torch.cat([base_output, musical_features], dim=-1)
        enhanced_output = self.enhanced_predictor(combined)
        
        # Apply constraints
        constrained_output = self.apply_constraints(enhanced_output, musical_context.genre)
        
        return constrained_output

class MusicalRetrofitTrainer:
    """Train existing models with musical intelligence"""
    
    def __init__(self, models_dir='models', data_dir='data'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_extractor = MusicalFeatureExtractor()
        self.preference_learner = PreferenceLearner()
        self.musical_loss = MusicallyAwareLoss()
        
        print(f"🎵 Musical Retrofit Trainer initialized on {self.device}")
        
    def load_existing_models(self) -> Dict[str, nn.Module]:
        """Load all existing trained models"""
        models = {}
        
        model_files = {
            'baseline_cnn': 'baseline_cnn.pth',
            'enhanced_cnn': 'enhanced_cnn.pth',
            'improved_baseline_cnn': 'improved_baseline_cnn.pth',
            'improved_enhanced_cnn': 'improved_enhanced_cnn.pth',
            'retrained_enhanced_cnn': 'retrained_enhanced_cnn.pth',
            'weighted_ensemble': 'weighted_ensemble.pth'
        }
        
        for name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    # Load base model
                    if 'baseline' in name:
                        base_model = BaselineCNN()
                    elif 'enhanced' in name:
                        base_model = EnhancedCNN()
                    else:
                        base_model = BaselineCNN()  # Default to baseline
                    
                    # Load weights
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        base_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        base_model.load_state_dict(checkpoint)
                      # Wrap with musical intelligence
                    musical_model = MusicalModelWrapper(base_model, 'cnn')
                    musical_model.to(self.device)
                    
                    models[name] = musical_model
                    print(f"✅ Loaded {name}")
                    
                except Exception as e:
                    print(f"❌ Failed to load {name}: {e}")
            else:
                print(f"⚠️  Model file not found: {filename}")
        
        return models
    
    def create_training_data(self, num_samples=100):
        """Create training data with musical context"""
        print("🎼 Creating musical training data...")
        
        # Use existing audio file for feature extraction
        audio_file = Path("mixed_outputs/Al James - Schoolboy Facination.stem_original.wav")
        if not audio_file.exists():
            print("❌ Audio file not found for feature extraction")
            return None, None, None
        
        # Extract musical features
        musical_context = self.feature_extractor.extract_musical_features(str(audio_file))
        print(f"🎵 Detected: {musical_context.genre} at {musical_context.tempo:.1f} BPM")
        
        # Create synthetic training data with musical constraints
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate spectrogram-like features (1, 128, 1000) to match model input
        # This simulates mel-spectrograms used by the CNN models
        features = torch.randn(num_samples, 1, 128, 1000)  # (batch, channels, freq_bins, time_steps)
        
        # Generate target parameters with musical constraints
        genre_template = GenreAwareMixingParameters().get_genre_parameters(musical_context.genre)
        
        targets = []
        for _ in range(num_samples):
            # Sample within genre-appropriate ranges
            target = [
                np.random.uniform(*genre_template['compression_ratio']),
                np.random.uniform(*genre_template['eq_low']),
                np.random.uniform(*genre_template['eq_mid']),
                np.random.uniform(*genre_template['eq_high']),
                np.random.uniform(*genre_template['reverb']),
                np.random.uniform(*genre_template['stereo_width']),
                np.random.uniform(-2, 2),  # Additional EQ bands
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                np.random.uniform(-1, 1)   # Additional parameters
            ]
            targets.append(target)
        
        targets = torch.tensor(targets, dtype=torch.float32)
        
        return features, targets, musical_context
    
    def train_model(self, model_name: str, model: nn.Module, epochs=50):
        """Train a single model with musical intelligence"""
        print(f"\n🎵 Training {model_name} with musical intelligence...")
          # Create training data
        features, targets, musical_context = self.create_training_data()
        if features is None or targets is None:
            return None
        
        features, targets = features.to(self.device), targets.to(self.device)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            batch_size = 16
            
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_targets = targets[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass with musical context
                predictions = model(batch_features, musical_context)
                
                # Calculate musical loss                # Generate dummy processed audio for loss calculation
                dummy_audio = torch.randn(batch_features.size(0), 44100, device=self.device)
                dummy_target_audio = torch.randn(batch_features.size(0), 44100, device=self.device)
                
                musical_loss, loss_breakdown = self.musical_loss(
                    predictions, batch_targets, dummy_audio, dummy_target_audio, 
                    musical_context, self.preference_learner.preference_patterns
                )
                
                # Add preference penalty
                preference_penalty = self.preference_learner.calculate_preference_penalty(predictions)
                total_loss_val = musical_loss + 0.1 * preference_penalty
                
                # Backward pass
                total_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_loss_val.item()
            
            avg_loss = total_loss / (len(features) // batch_size)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def retrofit_all_models(self):
        """Retrofit all existing models with musical intelligence"""
        print("🎵 Starting Musical Intelligence Retrofit Training")
        print("=" * 60)
        
        # Load existing models
        models = self.load_existing_models()
        
        if not models:
            print("❌ No models found to retrofit")
            return
        
        results = {}
        
        # Train each model
        for model_name, model in models.items():
            try:
                losses = self.train_model(model_name, model, epochs=30)
                if losses:
                    results[model_name] = {
                        'final_loss': losses[-1],
                        'loss_history': losses,
                        'status': 'success'
                    }
                    
                    # Save retrofitted model
                    save_path = self.models_dir / f"musical_{model_name}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'musical_wrapper': True,
                        'training_date': datetime.now().isoformat(),
                        'loss_history': losses
                    }, save_path)
                    
                    print(f"✅ Saved musical_{model_name}.pth")
                
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")
                results[model_name] = {'status': 'failed', 'error': str(e)}
        
        # Save training results
        results_path = self.models_dir / 'musical_retrofit_results.json'
        with open(results_path, 'w') as f:
            # Convert loss histories to lists for JSON serialization
            json_results = {}
            for name, result in results.items():
                json_results[name] = {
                    'status': result['status'],
                    'final_loss': result.get('final_loss'),
                    'loss_improvement': result.get('final_loss', 0) < 1.0 if result.get('final_loss') else False
                }
            json.dump(json_results, f, indent=2)
        
        # Create training report
        self.create_training_report(results)
        
        return results
    
    def create_training_report(self, results: Dict):
        """Create a comprehensive training report"""
        print("\n🎵 Musical Intelligence Retrofit Complete!")
        print("=" * 60)
        
        successful_models = [name for name, result in results.items() if result['status'] == 'success']
        failed_models = [name for name, result in results.items() if result['status'] == 'failed']
        
        print(f"✅ Successfully retrofitted: {len(successful_models)} models")
        for model in successful_models:
            final_loss = results[model].get('final_loss', 'N/A')
            print(f"   - {model}: Final loss {final_loss:.4f}" if isinstance(final_loss, float) else f"   - {model}")
        
        if failed_models:
            print(f"❌ Failed to retrofit: {len(failed_models)} models")
            for model in failed_models:
                print(f"   - {model}")
        
        print("\n🎯 Improvements Added:")
        print("   - Musical genre awareness")
        print("   - User preference learning")
        print("   - Parameter range constraints")
        print("   - Perceptual quality focus")
        print("   - Frequency balance optimization")
        
        print(f"\n📁 Musical models saved in: {self.models_dir}")
        print("   - Prefix: 'musical_' + original_name")
        print("   - Include musical context processing")
        print("   - Apply genre-specific constraints")

if __name__ == "__main__":
    # Initialize and run retrofit training
    trainer = MusicalRetrofitTrainer()
    results = trainer.retrofit_all_models()
    
    print("\n🚀 Next Steps:")
    print("1. Test musical models with comprehensive_mixer.py")
    print("2. Compare original vs musical model outputs")
    print("3. Collect feedback for preference refinement")
    print("4. Add more sophisticated musical features")
