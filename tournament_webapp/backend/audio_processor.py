#!/usr/bin/env python3
"""
🎵 Tournament Audio Processor
============================

Audio processing functions for the tournament system.
Provides spectrogram conversion, model inference, and battle execution.
"""

import sys
import os
import numpy as np
import shutil
import uuid
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

# Configure logging
import logging
logger = logging.getLogger(__name__)

# Dictionary to map model IDs to readable names
MODEL_NAMES = {
    "baseline_cnn": "Baseline CNN",
    "enhanced_cnn": "Enhanced CNN",
    "improved_baseline_cnn": "Improved Baseline CNN",
    "improved_enhanced_cnn": "Improved Enhanced CNN",
    "retrained_enhanced_cnn": "Retrained Enhanced CNN",
    "spectrogram_mixer": "Spectrogram Mixer",
    "weighted_ensemble": "Weighted Ensemble"
}

def get_model_name(model_id: str) -> str:
    """Get a readable name for a model ID"""
    return MODEL_NAMES.get(model_id, f"Model {model_id}")


def process_audio(
    audio_path: str,
    output_dir: str,
    model_id: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process an audio file with a model
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save outputs
        model_id: ID of the model to use
        progress_callback: Callback for progress updates
        
    Returns:
        Dictionary with processing results
    """
    if progress_callback:
        progress_callback(0.1, "Starting audio processing")
    
    result = {
        "audio_path": audio_path,
        "model_id": model_id,
        "model_name": get_model_name(model_id) if model_id else "None",
        "output_path": None,
        "processing_time": 0,
        "spectrogram_path": None,
        "parameters": {}
    }
    
    try:
        start_time = time.time()
        
        # Create output directories
        audio_dir = Path(output_dir) / "processed"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # In a production system, we would:
        # 1. Convert audio to spectrogram
        # 2. Load the model
        # 3. Process the spectrogram with the model
        # 4. Apply the processing parameters to the audio
        
        # For now, we'll simulate the process
        if progress_callback:
            progress_callback(0.3, "Converting to spectrogram")
        
        # Simulate spectrogram conversion
        spec_dir = Path(output_dir) / "spectrograms"
        spec_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy spectrogram file
        spec_path = spec_dir / f"{Path(audio_path).stem}.npy"
        result["spectrogram_path"] = str(spec_path)
        
        # Simulate model processing
        if model_id:
            if progress_callback:
                progress_callback(0.5, f"Processing with model {get_model_name(model_id)}")
            
            # Simulate parameter generation
            result["parameters"] = {
                "eq": {"low": 0.5, "mid": 0.7, "high": 0.6},
                "compression": {"threshold": -20, "ratio": 4, "attack": 10, "release": 100},
                "reverb": {"mix": 0.2, "room_size": 0.8, "damping": 0.5}
            }
            
            # In production, we would apply these parameters to the audio
            # For now, just copy the file to the output directory
            if progress_callback:
                progress_callback(0.8, "Applying parameters to audio")
            
            output_path = audio_dir / f"{Path(audio_path).stem}_processed_{model_id}.wav"
            shutil.copy(audio_path, output_path)
            result["output_path"] = str(output_path)
        else:
            # Just copy the file
            output_path = audio_dir / f"{Path(audio_path).stem}_original.wav"
            shutil.copy(audio_path, output_path)
            result["output_path"] = str(output_path)
        
        # Finalize
        result["processing_time"] = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, "Processing complete")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        
        result["error"] = str(e)
        return result


def execute_battle(
    audio_path: str,
    output_dir: str,
    model_a_id: str,
    model_b_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Execute a battle between two models
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save outputs
        model_a_id: ID of the first model
        model_b_id: ID of the second model
        progress_callback: Callback for progress updates
        
    Returns:
        Dictionary with battle results
    """
    battle_id = str(uuid.uuid4())
    
    if progress_callback:
        progress_callback(0.1, "Starting battle execution")
    
    result = {
        "battle_id": battle_id,
        "audio_path": audio_path,
        "model_a": {
            "id": model_a_id,
            "name": get_model_name(model_a_id),
            "output_path": None,
            "parameters": None
        },
        "model_b": {
            "id": model_b_id,
            "name": get_model_name(model_b_id),
            "output_path": None,
            "parameters": None
        },
        "processing_time": 0,
        "status": "pending"
    }
    
    try:
        start_time = time.time()
        
        # Create battle directory
        battle_dir = Path(output_dir) / "battles" / battle_id
        battle_dir.mkdir(parents=True, exist_ok=True)
        
        # Process with model A
        if progress_callback:
            progress_callback(0.2, f"Processing with {get_model_name(model_a_id)}")
        
        model_a_result = process_audio(
            audio_path,
            str(battle_dir / "model_a"),
            model_a_id,
            lambda p, m: progress_callback(0.2 + p * 0.3, m) if progress_callback else None
        )
        
        result["model_a"]["output_path"] = model_a_result.get("output_path")
        result["model_a"]["parameters"] = model_a_result.get("parameters")
        
        # Process with model B
        if progress_callback:
            progress_callback(0.5, f"Processing with {get_model_name(model_b_id)}")
        
        model_b_result = process_audio(
            audio_path,
            str(battle_dir / "model_b"),
            model_b_id,
            lambda p, m: progress_callback(0.5 + p * 0.3, m) if progress_callback else None
        )
        
        result["model_b"]["output_path"] = model_b_result.get("output_path")
        result["model_b"]["parameters"] = model_b_result.get("parameters")
        
        # Create comparison info
        if progress_callback:
            progress_callback(0.9, "Generating comparison data")
        
        # In production, we would generate waveform images, spectrograms, etc.
        # For now, just note that the battle is ready
        result["status"] = "ready_for_vote"
        
        # Finalize
        result["processing_time"] = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, "Battle execution complete")
        
        return result
    
    except Exception as e:
        logger.error(f"Error executing battle: {str(e)}", exc_info=True)
        
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        
        result["error"] = str(e)
        result["status"] = "failed"
        return result


def create_final_mix(
    audio_path: str,
    output_dir: str,
    model_id: str,
    tournament_id: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Create a final mix for a tournament
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save outputs
        model_id: ID of the model to use
        tournament_id: ID of the tournament
        progress_callback: Callback for progress updates
        
    Returns:
        Dictionary with final mix results
    """
    if progress_callback:
        progress_callback(0.1, "Starting final mix creation")
    
    result = {
        "tournament_id": tournament_id,
        "audio_path": audio_path,
        "model_id": model_id,
        "model_name": get_model_name(model_id),
        "output_path": None,
        "processing_time": 0,
        "parameters": {}
    }
    
    try:
        start_time = time.time()
        
        # Create output directory
        final_dir = Path(output_dir) / "tournaments" / tournament_id / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Process with the model
        model_result = process_audio(
            audio_path,
            str(final_dir),
            model_id,
            progress_callback
        )
        
        # Update result with model processing results
        result["output_path"] = model_result.get("output_path")
        result["parameters"] = model_result.get("parameters")
        result["spectrogram_path"] = model_result.get("spectrogram_path")
        
        # Finalize
        result["processing_time"] = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, "Final mix creation complete")
        
        return result
    
    except Exception as e:
        logger.error(f"Error creating final mix: {str(e)}", exc_info=True)
        
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        
        result["error"] = str(e)
        return result
    
    def _analyze_audio(self, audio_file: str) -> Dict[str, Any]:
        """Analyze audio for musical characteristics"""
        
        if self.musical_analyzer is None:
            # Fallback basic analysis
            return self._basic_audio_analysis(audio_file)
        
        try:
            # Use advanced musical intelligence
            musical_context = self.musical_analyzer.analyze_track(audio_file)
            
            return {
                'genre': musical_context.genre,
                'subgenre': musical_context.subgenre,
                'tempo': float(musical_context.tempo),
                'key': musical_context.key,
                'mode': musical_context.mode,
                'energy_level': float(musical_context.energy_level),
                'danceability': float(musical_context.danceability),
                'valence': float(musical_context.valence),
                'vocal_presence': float(musical_context.vocal_presence),
                'dynamic_range': float(musical_context.dynamic_range),
                'duration': self._get_audio_duration(audio_file)
            }
            
        except Exception as e:
            print(f"⚠️ Advanced analysis failed: {e}")
            return self._basic_audio_analysis(audio_file)
    
    def _basic_audio_analysis(self, audio_file: str) -> Dict[str, Any]:
        """Basic audio analysis fallback"""
        try:
            audio, sr = librosa.load(audio_file, sr=22050)
            
            # Basic features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            rms_energy = np.mean(librosa.feature.rms(y=audio))
            
            return {
                'genre': 'unknown',
                'tempo': float(tempo),
                'spectral_centroid': float(spectral_centroid),
                'energy_level': float(rms_energy),
                'duration': len(audio) / sr
            }
            
        except Exception as e:
            return {'error': str(e), 'duration': 0}
    
    def _process_with_model(self, audio_file: str, model_info: Dict, 
                           output_suffix: str) -> Dict[str, Any]:
        """Process audio with a specific model"""
        
        try:
            model_path = model_info.get('model_path')
            model_architecture = model_info.get('architecture', 'cnn')
            
            # Generate output filename
            output_filename = f"battle_{output_suffix}_{uuid.uuid4().hex[:8]}.wav"
            output_path = self.temp_dir / output_filename
            
            if model_architecture == 'cnn' and self._is_valid_model_path(model_path):
                # Use CNN model with production mixer
                result = self._process_with_cnn_model(audio_file, model_path, str(output_path))
                
            elif self.production_mixer is not None:
                # Use production mixer as fallback
                result = self._process_with_production_mixer(audio_file, str(output_path))
                
            else:
                # Basic processing fallback
                result = self._basic_audio_processing(audio_file, str(output_path))
            
            result.update({
                'model_info': model_info,
                'output_path': str(output_path),
                'output_url': f"/static/processed/{output_filename}"
            })
            
            return result
            
        except Exception as e:
            return {
                'model_info': model_info,
                'error': str(e),
                'output_path': None
            }
    
    def _is_valid_model_path(self, model_path: str) -> bool:
        """Check if model path exists and is valid"""
        if not model_path:
            return False
        
        path = Path(model_path)
        return path.exists() and path.suffix == '.pth'
    
    def _process_with_cnn_model(self, audio_file: str, model_path: str, 
                               output_path: str) -> Dict[str, Any]:
        """Process audio with CNN model"""
        
        if self.basic_mixer is None:
            raise Exception("AI mixer not available")
        
        try:
            # Load model
            from baseline_cnn import BaselineCNN, EnhancedCNN, N_OUTPUTS, DEVICE
            
            # Determine model type from path
            if 'enhanced' in model_path.lower():
                model = EnhancedCNN(n_outputs=N_OUTPUTS, dropout=0.3)
            else:
                model = BaselineCNN(n_outputs=N_OUTPUTS, dropout=0.3, n_conv_layers=3)
            
            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            # Generate prediction
            prediction = self._predict_with_model(audio_file, model)
            
            # Apply mixing
            audio, sr = librosa.load(audio_file, sr=44100, mono=False)
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            mixed_audio = self.basic_mixer.apply_mixing_parameters(audio, sr, prediction)
            
            # Save result
            sf.write(output_path, mixed_audio.T, sr)
            
            return {
                'success': True,
                'prediction_parameters': prediction.tolist(),
                'processing_method': 'cnn_model',
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            raise Exception(f"CNN processing failed: {e}")
    
    def _predict_with_model(self, audio_file: str, model) -> np.ndarray:
        """Generate prediction using trained model"""
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_file, sr=22050, mono=True)
        
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
            spec_tensor = torch.tensor(spec, dtype=torch.float32)
            prediction = model(spec_tensor).cpu().numpy()[0]
        
        return prediction
    
    def _process_with_production_mixer(self, audio_file: str, output_path: str) -> Dict[str, Any]:
        """Process with production mixer"""
        
        try:
            # Use production mixer's musical intelligence
            results = self.production_mixer.analyze_and_mix(audio_file, str(self.temp_dir))
            
            # Find the best mixed version
            best_version = None
            for name, path in results.get('output_files', {}).items():
                if 'musical' in name.lower() or 'baseline' in name.lower():
                    best_version = path
                    break
            
            if best_version and Path(best_version).exists():
                # Copy to output path
                import shutil
                shutil.copy(best_version, output_path)
                
                return {
                    'success': True,
                    'processing_method': 'production_mixer',
                    'musical_analysis': results.get('musical_analysis', {}),
                    'mixing_strategy': results.get('mixing_strategy', {})
                }
            else:
                raise Exception("No valid output from production mixer")
                
        except Exception as e:
            raise Exception(f"Production mixer failed: {e}")
    
    def _basic_audio_processing(self, audio_file: str, output_path: str) -> Dict[str, Any]:
        """Basic audio processing fallback"""
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=44100, mono=False)
            
            # Apply basic processing
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            # Simple normalization and light compression
            processed = audio * 0.8  # Simple level adjustment
            
            # Ensure no clipping
            peak = np.max(np.abs(processed))
            if peak > 0.95:
                processed = processed * (0.95 / peak)
            
            # Save
            sf.write(output_path, processed.T, sr)
            
            return {
                'success': True,
                'processing_method': 'basic_fallback',
                'note': 'Basic processing applied'
            }
            
        except Exception as e:
            raise Exception(f"Basic processing failed: {e}")
    
    def _generate_comparison(self, result_a: Dict, result_b: Dict, 
                           audio_analysis: Dict) -> Dict[str, Any]:
        """Generate comparison metrics between two processed versions"""
        
        comparison = {
            'models_compared': {
                'model_a': result_a.get('model_info', {}).get('name', 'Unknown'),
                'model_b': result_b.get('model_info', {}).get('name', 'Unknown')
            },
            'processing_methods': {
                'model_a': result_a.get('processing_method', 'unknown'),
                'model_b': result_b.get('processing_method', 'unknown')
            },
            'audio_context': {
                'genre': audio_analysis.get('genre', 'unknown'),
                'tempo': audio_analysis.get('tempo', 0),
                'energy': audio_analysis.get('energy_level', 0)
            }
        }
        
        # Add technical comparison if both processed successfully
        if result_a.get('success') and result_b.get('success'):
            comparison['technical_analysis'] = self._compare_audio_files(
                result_a.get('output_path'),
                result_b.get('output_path')
            )
        
        return comparison
    
    def _compare_audio_files(self, path_a: str, path_b: str) -> Dict[str, Any]:
        """Compare two audio files technically"""
        
        try:
            if not path_a or not path_b:
                return {'error': 'Missing audio files'}
            
            # Load both files
            audio_a, sr_a = librosa.load(path_a, sr=44100)
            audio_b, sr_b = librosa.load(path_b, sr=44100)
            
            # Basic metrics
            metrics = {
                'rms_energy': {
                    'model_a': float(np.sqrt(np.mean(audio_a**2))),
                    'model_b': float(np.sqrt(np.mean(audio_b**2)))
                },
                'peak_level': {
                    'model_a': float(np.max(np.abs(audio_a))),
                    'model_b': float(np.max(np.abs(audio_b)))
                },
                'spectral_centroid': {
                    'model_a': float(np.mean(librosa.feature.spectral_centroid(y=audio_a, sr=sr_a))),
                    'model_b': float(np.mean(librosa.feature.spectral_centroid(y=audio_b, sr=sr_b)))
                }
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_audio_duration(self, audio_file: str) -> float:
        """Get audio file duration"""
        try:
            audio, sr = librosa.load(audio_file, sr=None)
            return len(audio) / sr
        except:
            return 0.0
    
    def cleanup_temp_files(self, battle_id: str):
        """Clean up temporary files for a battle"""
        for file_path in self.temp_dir.glob(f"battle_{battle_id}_*"):
            try:
                file_path.unlink()
            except:
                pass


if __name__ == "__main__":
    # Test the audio processor
    processor = TournamentAudioProcessor()
    
    print("🎵 Tournament Audio Processor Ready!")
    print(f"   Production Mixer: {'✅' if processor.production_mixer else '❌'}")
    print(f"   Musical Analyzer: {'✅' if processor.musical_analyzer else '❌'}")
    print(f"   Basic Mixer: {'✅' if processor.basic_mixer else '❌'}")
    print(f"   Temp Directory: {processor.temp_dir}")
    
    # Test with a sample file if available
    test_files = list(Path("mixed_outputs").glob("*.wav"))
    if test_files:
        test_file = str(test_files[0])
        print(f"\n🧪 Testing with: {Path(test_file).name}")
        
        analysis = processor._analyze_audio(test_file)
        print(f"   Analysis: {analysis}")
    else:
        print("\n⚠️ No test audio files found in mixed_outputs/")
