#!/usr/bin/env python3
"""
🧠 Spectrogram-Based Audio Mixing Model
=======================================

Neural network model optimized for processing mel spectrograms for audio mixing.
This implementation is specifically designed to work with pre-processed
spectrogram data instead of raw audio, providing significant performance
improvements for inference.

Features:
- Optimized CNN architecture for spectrogram inputs
- Efficient batch processing
- Quantization-ready design
- Support for model evolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectrogramMixingModel(nn.Module):
    """
    CNN model architecture optimized for spectrogram-based audio mixing
    """
    def __init__(
        self, 
        input_channels: int = 1, 
        num_freq_bins: int = 128,
        num_output_params: int = 10,
        model_complexity: str = "medium"
    ):
        """
        Initialize the model with configurable complexity
        
        Args:
            input_channels: Number of input channels (1 for mono, 2 for stereo spectrograms)
            num_freq_bins: Number of frequency bins in the mel spectrogram
            num_output_params: Number of mixing parameters to predict
            model_complexity: Model size/complexity ("small", "medium", "large")
        """
        super().__init__()
        
        # Store config
        self.input_channels = input_channels
        self.num_freq_bins = num_freq_bins
        self.num_output_params = num_output_params
        self.model_complexity = model_complexity
        
        # Define model sizes based on complexity
        if model_complexity == "small":
            filters = [16, 32, 64, 64]
            self.fc_size = 256
        elif model_complexity == "medium":
            filters = [32, 64, 128, 256]
            self.fc_size = 512
        elif model_complexity == "large":
            filters = [64, 128, 256, 512]
            self.fc_size = 1024
        else:
            raise ValueError(f"Invalid model complexity: {model_complexity}")
        
        # Convolutional layers for spectrogram processing
        self.conv1 = nn.Conv2d(input_channels, filters[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the size of the flattened convolutional output
        # This depends on the input spectrogram size and pooling operations
        # For a typical 128x128 spectrogram with 4 max pooling layers (2x2),
        # the output would be reduced by a factor of 16 in each dimension
        conv_output_size = (num_freq_bins // 16) * (num_freq_bins // 16) * filters[3]
        
        # Fully connected layers for mixing parameter prediction
        self.fc1 = nn.Linear(conv_output_size, self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, self.fc_size // 2)
        self.fc3 = nn.Linear(self.fc_size // 2, num_output_params)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        logger.info(f"Initialized {model_complexity} SpectrogramMixingModel with {self._count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape [batch_size, channels, freq_bins, time_frames]
               Typically [batch_size, 1, 128, 128] for mono mel spectrograms
        
        Returns:
            Tensor of shape [batch_size, num_output_params] containing mixing parameters
        """
        # Check input shape
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
        
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpectrogramMixerInference:
    """
    Inference manager for spectrogram-based mixing models with optimization
    and caching capabilities.
    """
    def __init__(
        self, 
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_half_precision: bool = False,
        cache_capacity: int = 100
    ):
        """
        Initialize the inference manager
        
        Args:
            model_path: Path to the saved model file
            device: Device to run inference on ("cpu", "cuda", "cuda:0", etc.)
            use_half_precision: Whether to use half precision (float16) for faster inference
            cache_capacity: Maximum number of results to cache
        """
        self.device = device
        self.use_half_precision = use_half_precision
        self.cache = {}  # Simple cache implementation
        self.cache_capacity = cache_capacity
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        if use_half_precision and device.startswith("cuda"):
            self.model = self.model.half()
            logger.info("Using half precision (float16) for inference")
        
        logger.info(f"Loaded model on {device} with{'out' if not use_half_precision else ''} half precision")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load a saved model from disk"""
        try:
            model_data = torch.load(model_path, map_location=self.device)
            
            # Handle different save formats
            if isinstance(model_data, dict) and "model_state_dict" in model_data:
                # Create model with saved hyperparameters if available
                if "model_config" in model_data:
                    config = model_data["model_config"]
                    model = SpectrogramMixingModel(**config)
                else:
                    model = SpectrogramMixingModel()
                
                # Load state dict
                model.load_state_dict(model_data["model_state_dict"])
            else:
                # Assume it's a directly saved model
                model = model_data
            
            return model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise
    
    def _create_cache_key(self, spectrogram: np.ndarray) -> str:
        """Create a unique key for the spectrogram for caching"""
        # Simple hash of the spectrogram array
        # In production, would use a more sophisticated fingerprinting method
        return str(hash(spectrogram.tobytes()))
    
    def _manage_cache(self, key: str, result: Dict[str, float]):
        """Add a result to the cache, managing capacity"""
        if len(self.cache) >= self.cache_capacity:
            # Remove a random item when full
            # In production, would use LRU or another eviction policy
            self.cache.pop(next(iter(self.cache.keys())))
        
        self.cache[key] = result
    
    def process_spectrogram(self, spectrogram: np.ndarray) -> Dict[str, float]:
        """
        Process a single spectrogram and return mixing parameters
        
        Args:
            spectrogram: Numpy array of shape [freq_bins, time_frames] or 
                         [channels, freq_bins, time_frames]
        
        Returns:
            Dictionary of mixing parameters
        """
        # Check cache first
        cache_key = self._create_cache_key(spectrogram)
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Prepare input tensor
        start_time = time.time()
        
        # Handle different input shapes
        if len(spectrogram.shape) == 2:
            # Add channel dimension for 2D spectrograms
            spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Convert to tensor and add batch dimension
        x = torch.from_numpy(spectrogram).float().unsqueeze(0)
        
        # Convert to half precision if enabled
        if self.use_half_precision and self.device.startswith("cuda"):
            x = x.half()
        
        # Move to device
        x = x.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(x)
        
        # Convert output tensor to numpy and then to dict
        output_np = output.cpu().numpy()[0]  # Remove batch dimension
        
        # Map outputs to named parameters (example mapping)
        # In a real implementation, this would map to actual parameter names
        param_names = [
            "gain_bass", "gain_mid", "gain_treble",
            "compression_threshold", "compression_ratio",
            "reverb_amount", "delay_amount",
            "stereo_width", "eq_presence", "eq_air"
        ]
        
        # Create result dictionary with parameter names and values
        result = {}
        for i, name in enumerate(param_names[:len(output_np)]):
            result[name] = float(output_np[i])
        
        # Add to cache
        self._manage_cache(cache_key, result)
        
        # Log performance
        inference_time = time.time() - start_time
        logger.debug(f"Inference completed in {inference_time:.3f}s")
        
        return result
    
    def batch_process(self, spectrograms: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Process a batch of spectrograms
        
        Args:
            spectrograms: List of spectrogram arrays
            
        Returns:
            List of dictionaries containing mixing parameters
        """
        results = []
        batch_size = 16  # Could be made configurable
        
        # Process in batches for efficiency
        for i in range(0, len(spectrograms), batch_size):
            batch = spectrograms[i:i+batch_size]
            batch_results = []
            
            # Check cache for each item first
            uncached_indices = []
            for j, spec in enumerate(batch):
                cache_key = self._create_cache_key(spec)
                if cache_key in self.cache:
                    self.cache_hits += 1
                    batch_results.append(self.cache[cache_key])
                else:
                    self.cache_misses += 1
                    batch_results.append(None)  # Placeholder
                    uncached_indices.append(j)
            
            # If there are any uncached items, process them
            if uncached_indices:
                # Prepare uncached spectrograms
                uncached_specs = [batch[j] for j in uncached_indices]
                processed_batch = self._process_batch(uncached_specs)
                
                # Insert results back into the right positions
                for idx, result in zip(uncached_indices, processed_batch):
                    batch_results[idx] = result
                    
                    # Add to cache
                    cache_key = self._create_cache_key(batch[idx])
                    self._manage_cache(cache_key, result)
            
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, spectrograms: List[np.ndarray]) -> List[Dict[str, float]]:
        """Process a batch of uncached spectrograms"""
        if not spectrograms:
            return []
        
        start_time = time.time()
        
        # Prepare input tensors
        tensors = []
        for spec in spectrograms:
            # Handle different input shapes
            if len(spec.shape) == 2:
                # Add channel dimension for 2D spectrograms
                spec = np.expand_dims(spec, axis=0)
            tensors.append(torch.from_numpy(spec).float())
        
        # Stack into a batch
        batch = torch.stack(tensors)
        
        # Convert to half precision if enabled
        if self.use_half_precision and self.device.startswith("cuda"):
            batch = batch.half()
        
        # Move to device
        batch = batch.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(batch)
        
        # Convert output tensor to numpy
        output_np = output.cpu().numpy()
        
        # Map outputs to named parameters
        param_names = [
            "gain_bass", "gain_mid", "gain_treble",
            "compression_threshold", "compression_ratio",
            "reverb_amount", "delay_amount",
            "stereo_width", "eq_presence", "eq_air"
        ]
        
        # Create result dictionaries
        results = []
        for i in range(len(output_np)):
            result = {}
            for j, name in enumerate(param_names[:output_np.shape[1]]):
                result[name] = float(output_np[i, j])
            results.append(result)
        
        inference_time = time.time() - start_time
        logger.debug(f"Batch inference completed in {inference_time:.3f}s for {len(spectrograms)} items")
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_capacity": self.cache_capacity,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """Clear the inference cache"""
        self.cache = {}
        logger.info("Inference cache cleared")


# Example usage
if __name__ == "__main__":
    # Create a random spectrogram for testing
    test_spectrogram = np.random.rand(1, 128, 128).astype(np.float32)
    
    # Create model and save
    model = SpectrogramMixingModel(
        input_channels=1,
        num_freq_bins=128,
        num_output_params=10,
        model_complexity="medium"
    )
    
    # Save model
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / "spectrogram_mixer.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_channels": 1,
            "num_freq_bins": 128,
            "num_output_params": 10,
            "model_complexity": "medium"
        }
    }, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Test inference
    inference = SpectrogramMixerInference(
        model_path=str(model_path),
        device="cpu",  # Use CPU for testing
        use_half_precision=False
    )
    
    # Process a single spectrogram
    result = inference.process_spectrogram(test_spectrogram)
    print("Single spectrogram result:", result)
    
    # Process a batch
    batch_results = inference.batch_process([test_spectrogram] * 5)
    print(f"Batch processing results: {len(batch_results)} items")
    
    # Check cache stats
    print("Cache stats:", inference.get_cache_stats())
