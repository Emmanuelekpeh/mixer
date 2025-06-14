#!/usr/bin/env python3
"""
🎵 Audio Processing Tasks
=======================

Asynchronous tasks for audio processing in the AI mixer application.
Provides functions for spectrogram generation, model inference, and mixing.
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import shutil
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import spectrogram conversion utility if available
try:
    from src.audio_to_spectrogram import SpectrogramConverter
    spectrogram_converter = SpectrogramConverter()
    logger.info("Initialized spectrogram converter")
except ImportError:
    logger.warning("Could not import spectrogram converter")
    spectrogram_converter = None

# Import model inference if available
try:
    from src.spectrogram_mixing_model import SpectrogramMixerInference
    models_path = Path(parent_dir) / "models"
    model_inference = SpectrogramMixerInference(model_path=str(models_path))
    logger.info("Initialized spectrogram model inference")
except ImportError:
    logger.warning("Could not import spectrogram model inference")
    model_inference = None

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


def process_audio_file(
    audio_file_path: str,
    output_dir: str,
    model_id: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Process an audio file asynchronously
    
    Args:
        audio_file_path: Path to the audio file
        output_dir: Directory to save outputs
        model_id: Optional model ID to use for processing
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with processing results
    """
    result = {
        "audio_file": audio_file_path,
        "spectrogram_path": None,
        "processed_audio": None,
        "processing_time": 0,
        "model_id": model_id,
        "parameters": {}
    }
    
    try:
        start_time = time.time()
        
        # Update progress
        if progress_callback:
            progress_callback(0.1, "Converting audio to spectrogram")
            
        # Convert audio to spectrogram
        if spectrogram_converter:
            spec_dir = Path(output_dir) / "spectrograms"
            spec_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate spectrogram
            spec_path, meta_path = spectrogram_converter.audio_to_spectrogram(
                audio_file_path, str(spec_dir)
            )
            
            result["spectrogram_path"] = spec_path
            result["spectrogram_meta"] = meta_path
            
            logger.info(f"Generated spectrogram: {spec_path}")
            
            # Update progress
            if progress_callback:
                progress_callback(0.3, "Running model inference")
            
            # Run model inference if a model is specified
            if model_id and model_inference:
                # Load the specific model
                model_inference.load_model(model_id)
                
                # Get parameters from model
                parameters = model_inference.process_spectrogram(spec_path)
                result["parameters"] = parameters
                
                # Update progress
                if progress_callback:
                    progress_callback(0.6, "Applying mix parameters")
                
                # Apply parameters to audio
                # For now, this is a placeholder for the actual implementation
                output_audio_path = Path(output_dir) / "mixed" / f"{Path(audio_file_path).stem}_mixed.wav"
                output_audio_path.parent.mkdir(exist_ok=True, parents=True)
                
                # In a real implementation, this would apply the parameters to the audio
                # For now, just copy the original file
                shutil.copy(audio_file_path, output_audio_path)
                
                result["processed_audio"] = str(output_audio_path)
                
                logger.info(f"Applied mix parameters to: {output_audio_path}")
        else:
            # Fallback if no spectrogram converter
            logger.warning("Spectrogram converter not available")
            
            # Just copy the file to the output directory
            output_path = Path(output_dir) / "uploads" / Path(audio_file_path).name
            output_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(audio_file_path, output_path)
            
            result["processed_audio"] = str(output_path)
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, "Processing complete")
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
            
        result["error"] = str(e)
        return result


def execute_model_battle(
    audio_file_path: str,
    output_dir: str,
    model_a_id: str,
    model_b_id: str,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Execute a battle between two models
    
    Args:
        audio_file_path: Path to the audio file
        output_dir: Directory to save outputs
        model_a_id: ID of the first model
        model_b_id: ID of the second model
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with battle results
    """
    battle_id = str(uuid.uuid4())
    
    battle_result = {
        "battle_id": battle_id,
        "audio_file": audio_file_path,
        "model_a": {"id": model_a_id},
        "model_b": {"id": model_b_id},
        "spectrogram_path": None,
        "model_a_result": None,
        "model_b_result": None,
        "processing_time": 0
    }
    
    try:
        start_time = time.time()
        
        # Update progress
        if progress_callback:
            progress_callback(0.1, "Converting audio to spectrogram")
            
        # Convert audio to spectrogram
        if spectrogram_converter:
            spec_dir = Path(output_dir) / "spectrograms"
            spec_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate spectrogram
            spec_path, meta_path = spectrogram_converter.audio_to_spectrogram(
                audio_file_path, str(spec_dir)
            )
            
            battle_result["spectrogram_path"] = spec_path
            
            logger.info(f"Generated spectrogram for battle: {spec_path}")
            
            # Update progress
            if progress_callback:
                progress_callback(0.3, f"Processing with model {model_a_id}")
            
            # Process with model A
            if model_inference:
                # Get parameters from model A
                model_a_params = model_inference.process_spectrogram(spec_path, model_a_id)
                battle_result["model_a"]["parameters"] = model_a_params
                
                # Create output path for model A
                output_a_path = Path(output_dir) / "battles" / battle_id / f"{Path(audio_file_path).stem}_model_a.wav"
                output_a_path.parent.mkdir(exist_ok=True, parents=True)
                
                # In a real implementation, this would apply the parameters to the audio
                # For now, just copy the original file
                shutil.copy(audio_file_path, output_a_path)
                
                battle_result["model_a"]["output_path"] = str(output_a_path)
                
                # Add model details
                battle_result["model_a"]["name"] = model_inference.get_model_name(model_a_id)
                
                logger.info(f"Processed with model A: {model_a_id}")
                
                # Update progress
                if progress_callback:
                    progress_callback(0.6, f"Processing with model {model_b_id}")
                
                # Process with model B
                model_b_params = model_inference.process_spectrogram(spec_path, model_b_id)
                battle_result["model_b"]["parameters"] = model_b_params
                
                # Create output path for model B
                output_b_path = Path(output_dir) / "battles" / battle_id / f"{Path(audio_file_path).stem}_model_b.wav"
                
                # In a real implementation, this would apply the parameters to the audio
                # For now, just copy the original file
                shutil.copy(audio_file_path, output_b_path)
                
                battle_result["model_b"]["output_path"] = str(output_b_path)
                
                # Add model details
                battle_result["model_b"]["name"] = model_inference.get_model_name(model_b_id)
                
                logger.info(f"Processed with model B: {model_b_id}")
            else:
                logger.warning("Model inference not available, using dummy battle")
                
                # Add dummy model details
                battle_result["model_a"]["name"] = "Model A"
                battle_result["model_b"]["name"] = "Model B"
                
                # Create dummy output paths
                output_dir_path = Path(output_dir) / "battles" / battle_id
                output_dir_path.mkdir(exist_ok=True, parents=True)
                
                output_a_path = output_dir_path / f"{Path(audio_file_path).stem}_model_a.wav"
                output_b_path = output_dir_path / f"{Path(audio_file_path).stem}_model_b.wav"
                
                # Copy original file for both models
                shutil.copy(audio_file_path, output_a_path)
                shutil.copy(audio_file_path, output_b_path)
                
                battle_result["model_a"]["output_path"] = str(output_a_path)
                battle_result["model_b"]["output_path"] = str(output_b_path)
        else:
            # Fallback if no spectrogram converter
            logger.warning("Spectrogram converter not available, using dummy battle")
            
            # Add dummy model details
            battle_result["model_a"]["name"] = "Model A"
            battle_result["model_b"]["name"] = "Model B"
            
            # Create dummy output paths
            output_dir_path = Path(output_dir) / "battles" / battle_id
            output_dir_path.mkdir(exist_ok=True, parents=True)
            
            output_a_path = output_dir_path / f"{Path(audio_file_path).stem}_model_a.wav"
            output_b_path = output_dir_path / f"{Path(audio_file_path).stem}_model_b.wav"
            
            # Copy original file for both models
            shutil.copy(audio_file_path, output_a_path)
            shutil.copy(audio_file_path, output_b_path)
            
            battle_result["model_a"]["output_path"] = str(output_a_path)
            battle_result["model_b"]["output_path"] = str(output_b_path)
        
        # Add status
        battle_result["status"] = "ready_for_vote"
        
        # Calculate processing time
        battle_result["processing_time"] = time.time() - start_time
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, "Battle processing complete")
            
        return battle_result
        
    except Exception as e:
        logger.error(f"Error executing model battle: {str(e)}", exc_info=True)
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
            
        battle_result["error"] = str(e)
        battle_result["status"] = "failed"
        return battle_result


def create_final_mix(
    audio_file_path: str,
    output_dir: str,
    model_id: str,
    tournament_id: str,
    progress_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Create a final mix for a tournament
    
    Args:
        audio_file_path: Path to the audio file
        output_dir: Directory to save outputs
        model_id: ID of the model to use
        tournament_id: Tournament ID
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with final mix results
    """
    result = {
        "tournament_id": tournament_id,
        "audio_file": audio_file_path,
        "model_id": model_id,
        "final_mix_path": None,
        "parameters": {},
        "processing_time": 0
    }
    
    try:
        start_time = time.time()
        
        # Update progress
        if progress_callback:
            progress_callback(0.1, "Converting audio to spectrogram")
            
        # Convert audio to spectrogram
        if spectrogram_converter:
            spec_dir = Path(output_dir) / "spectrograms"
            spec_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate spectrogram
            spec_path, meta_path = spectrogram_converter.audio_to_spectrogram(
                audio_file_path, str(spec_dir)
            )
            
            result["spectrogram_path"] = spec_path
            
            logger.info(f"Generated spectrogram for final mix: {spec_path}")
            
            # Update progress
            if progress_callback:
                progress_callback(0.3, f"Processing with model {model_id}")
            
            # Process with the model
            if model_inference:
                # Get parameters from model
                parameters = model_inference.process_spectrogram(spec_path, model_id)
                result["parameters"] = parameters
                
                # Create output path for final mix
                output_path = Path(output_dir) / "tournaments" / tournament_id / "final_mix.wav"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                # In a real implementation, this would apply the parameters to the audio
                # For now, just copy the original file
                shutil.copy(audio_file_path, output_path)
                
                result["final_mix_path"] = str(output_path)
                
                # Add model details
                result["model_name"] = model_inference.get_model_name(model_id)
                
                logger.info(f"Created final mix with model {model_id}")
            else:
                logger.warning("Model inference not available, using dummy final mix")
                
                # Create dummy output path
                output_path = Path(output_dir) / "tournaments" / tournament_id / "final_mix.wav"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                # Copy original file
                shutil.copy(audio_file_path, output_path)
                
                result["final_mix_path"] = str(output_path)
                result["model_name"] = "Unknown Model"
        else:
            # Fallback if no spectrogram converter
            logger.warning("Spectrogram converter not available, using dummy final mix")
            
            # Create dummy output path
            output_path = Path(output_dir) / "tournaments" / tournament_id / "final_mix.wav"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Copy original file
            shutil.copy(audio_file_path, output_path)
            
            result["final_mix_path"] = str(output_path)
            result["model_name"] = "Unknown Model"
        
        # Calculate processing time
        result["processing_time"] = time.time() - start_time
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, "Final mix complete")
            
        return result
        
    except Exception as e:
        logger.error(f"Error creating final mix: {str(e)}", exc_info=True)
        
        # Update progress
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
            
        result["error"] = str(e)
        return result


# Test code
if __name__ == "__main__":
    # Test audio processing
    def print_progress(progress, message):
        print(f"Progress: {progress:.0%} - {message}")
    
    # Test with a dummy audio file
    test_file = "path/to/test/audio.wav"
    if os.path.exists(test_file):
        result = process_audio_file(test_file, "output", progress_callback=print_progress)
        print(json.dumps(result, indent=2))
    else:
        print(f"Test file not found: {test_file}")
