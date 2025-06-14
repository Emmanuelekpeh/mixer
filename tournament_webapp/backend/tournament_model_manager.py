#!/usr/bin/env python3
"""
🎮 Tournament Model Manager
=========================

Manages spectrogram-based models for the tournament system.
Integrates spectrogram conversion, model inference, and result processing
for efficient audio mixing in tournament battles.

Features:
- Automatic spectrogram conversion
- Efficient model loading and inference
- Caching for improved performance
- Asynchronous processing support
"""

import os
import sys
import asyncio
import logging
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import torch

# Add the parent directory to the path to import spectrogram and model modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import spectrogram and model modules
from src.audio_to_spectrogram import SpectrogramConverter
from src.spectrogram_mixing_model import SpectrogramMixerInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TournamentModelManager:
    """
    Manages models for tournament battles, handling spectrogram conversion
    and model inference.
    """
    def __init__(self, models_dir: Path):
        """
        Initialize the model manager
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.model_instances = {}  # Cache of loaded model instances
        self.spectrogram_converter = SpectrogramConverter()
        self.processing_cache = {}  # Cache of processed audio results
        
        # Load available models
        self.available_models = self._load_available_models()
        logger.info(f"Initialized TournamentModelManager with {len(self.available_models)} available models")
    
    def _load_available_models(self) -> List[Dict[str, Any]]:
        """
        Load information about available models
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        try:
            # Look for .pth model files
            model_files = list(self.models_dir.glob("**/*.pth"))
            
            for model_file in model_files:
                # Extract basic information
                model_id = model_file.stem
                relative_path = model_file.relative_to(self.models_dir)
                
                # Check for metadata file
                metadata_file = model_file.with_suffix('.json')
                metadata = {}
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {model_id}: {str(e)}")
                
                # Create model info
                model_info = {
                    "id": model_id,
                    "name": metadata.get("name", model_id),
                    "architecture": metadata.get("architecture", "cnn"),
                    "path": str(model_file),
                    "relative_path": str(relative_path),
                    "size_mb": model_file.stat().st_size / (1024 * 1024),
                    "created_at": metadata.get("created_at", ""),
                    "description": metadata.get("description", ""),
                    "performance_metrics": metadata.get("performance_metrics", {}),
                    "specializations": metadata.get("specializations", []),
                    "parameters": metadata.get("parameters", {})
                }
                
                models.append(model_info)
                
            # Sort by creation date if available
            models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Error loading available models: {str(e)}")
        
        return models
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with basic information
        
        Returns:
            List of model information dictionaries
        """
        return self.available_models
    
    def _get_model_instance(self, model_id: str) -> Optional[SpectrogramMixerInference]:
        """
        Get or create a model inference instance
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Model inference instance or None if model not found
        """
        # Return cached instance if available
        if model_id in self.model_instances:
            return self.model_instances[model_id]
        
        # Find model info
        model_info = next((m for m in self.available_models if m["id"] == model_id), None)
        if not model_info:
            logger.error(f"Model not found: {model_id}")
            return None
        
        try:
            # Create inference instance
            model_path = model_info["path"]
            
            # Determine if we should use GPU
            use_gpu = torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"
            
            # Create inference instance with appropriate settings
            inference = SpectrogramMixerInference(
                model_path=model_path,
                device=device,
                use_half_precision=use_gpu,  # Use half precision on GPU
                cache_capacity=100  # Configurable cache size
            )
            
            # Cache the instance
            self.model_instances[model_id] = inference
            logger.info(f"Loaded model {model_id} on {device}")
            
            return inference
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return None
    
    def process_audio(self, audio_path: str, model_id: str) -> Dict[str, Any]:
        """
        Process an audio file with a specific model
        
        Args:
            audio_path: Path to the audio file
            model_id: ID of the model to use
            
        Returns:
            Dictionary with processing results
        """
        # Check cache first
        cache_key = f"{audio_path}:{model_id}"
        if cache_key in self.processing_cache:
            logger.info(f"Using cached result for {cache_key}")
            return self.processing_cache[cache_key]
        
        start_time = time.time()
        
        try:
            # Convert audio to spectrogram if needed
            spec_dir = Path(audio_path).parent / "spectrograms"
            spec_dir.mkdir(exist_ok=True)
            
            # Check if we already have a spectrogram
            audio_file = Path(audio_path)
            potential_spec_path = spec_dir / f"{audio_file.stem}_mel_spec.npy"
            
            if potential_spec_path.exists():
                # Use existing spectrogram
                spec_path = potential_spec_path
                logger.info(f"Using existing spectrogram: {spec_path}")
            else:
                # Convert audio to spectrogram
                spec_path, _ = self.spectrogram_converter.audio_to_spectrogram(audio_path, str(spec_dir))
                logger.info(f"Converted audio to spectrogram: {spec_path}")
            
            # Load spectrogram
            spectrogram = np.load(spec_path)
            
            # Get model instance
            model_instance = self._get_model_instance(model_id)
            if not model_instance:
                return {"error": f"Could not load model {model_id}"}
            
            # Process the spectrogram
            mixing_params = model_instance.process_spectrogram(spectrogram)
            
            # Create result
            result = {
                "model_id": model_id,
                "audio_path": audio_path,
                "spectrogram_path": str(spec_path),
                "mixing_parameters": mixing_params,
                "processing_time": time.time() - start_time
            }
            
            # Cache the result
            self.processing_cache[cache_key] = result
            
            logger.info(f"Processed audio with model {model_id} in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio with model {model_id}: {str(e)}")
            return {"error": str(e)}
    
    async def process_audio_async(self, audio_path: str, model_id: str) -> Dict[str, Any]:
        """
        Process an audio file asynchronously
        
        Args:
            audio_path: Path to the audio file
            model_id: ID of the model to use
            
        Returns:
            Dictionary with processing results
        """
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.process_audio, audio_path, model_id
        )
    
    async def execute_battle(self, audio_path: str, model_a_id: str, model_b_id: str) -> Dict[str, Any]:
        """
        Execute a battle between two models
        
        Args:
            audio_path: Path to the audio file
            model_a_id: ID of the first model
            model_b_id: ID of the second model
            
        Returns:
            Battle result dictionary
        """
        start_time = time.time()
        
        try:
            # Process audio with both models concurrently
            result_a_task = asyncio.create_task(self.process_audio_async(audio_path, model_a_id))
            result_b_task = asyncio.create_task(self.process_audio_async(audio_path, model_b_id))
            
            # Wait for both to complete
            result_a = await result_a_task
            result_b = await result_b_task
            
            # Check for errors
            if "error" in result_a:
                return {"error": f"Error processing with model A: {result_a['error']}"}
            if "error" in result_b:
                return {"error": f"Error processing with model B: {result_b['error']}"}
            
            # Create battle result
            battle_id = f"battle_{int(time.time())}"
            
            # In a real implementation, we would apply the mixing parameters to create
            # mixed audio files here, but for now we'll just return the parameters
            
            battle_result = {
                "battle_id": battle_id,
                "audio_path": audio_path,
                "model_a": {
                    "id": model_a_id,
                    "name": next((m["name"] for m in self.available_models if m["id"] == model_a_id), model_a_id),
                    "mixing_parameters": result_a["mixing_parameters"]
                },
                "model_b": {
                    "id": model_b_id,
                    "name": next((m["name"] for m in self.available_models if m["id"] == model_b_id), model_b_id),
                    "mixing_parameters": result_b["mixing_parameters"]
                },
                "spectrogram_path": result_a["spectrogram_path"],  # Both should have the same path
                "processing_time": time.time() - start_time,
                "status": "ready_for_vote"
            }
            
            logger.info(f"Executed battle {battle_id} in {battle_result['processing_time']:.2f}s")
            return battle_result
            
        except Exception as e:
            logger.error(f"Error executing battle: {str(e)}")
            return {"error": str(e)}
    
    def clear_caches(self):
        """Clear all caches"""
        # Clear processing cache
        self.processing_cache = {}
        
        # Clear model instance caches
        for model_instance in self.model_instances.values():
            model_instance.clear_cache()
        
        logger.info("All caches cleared")


# Example usage
if __name__ == "__main__":
    # Set up the models directory
    models_dir = Path("models")
    
    # Create the model manager
    manager = TournamentModelManager(models_dir)
    
    # Print available models
    print("Available models:")
    for model in manager.get_model_list():
        print(f"  - {model['name']} ({model['id']})")
    
    # Example audio file
    audio_file = "mixed_outputs/Al James - Schoolboy Facination.stem_original.wav"
    if Path(audio_file).exists():
        # Process with a model
        if manager.get_model_list():
            model_id = manager.get_model_list()[0]["id"]
            print(f"Processing {audio_file} with model {model_id}...")
            result = manager.process_audio(audio_file, model_id)
            print(f"Result: {result}")
        else:
            print("No models available.")
    else:
        print(f"Audio file not found: {audio_file}")
