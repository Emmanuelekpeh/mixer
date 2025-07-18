#!/usr/bin/env python3
"""
🎵 Model Shape Adapter for Mixing Models
=======================================

This module provides adapter functions to reshape input data
to match the expected input formats for different model architectures.
"""

import torch
import logging

logger = logging.getLogger(__name__)

def adapt_input_for_model(model_name, input_data):
    """
    Adapt input data to the correct shape for different model architectures.
    
    Args:
        model_name (str): The name of the model
        input_data (torch.Tensor): Input data with shape [batch, features, time]
    
    Returns:
        torch.Tensor: Reshaped input data suitable for the specified model
    """
    # Get original shape
    batch_size = input_data.shape[0]
    
    if model_name == 'LSTM Audio Mixer':
        # LSTM expects [batch, features, time]
        # Our data is already [batch, features, time], so no reshaping needed
        # Just ensure it's 3D
        if len(input_data.shape) == 4:  # If it's [batch, channels, features, time]
            return input_data.squeeze(1)  # Remove channel dimension
        return input_data
        
    elif model_name in ['Audio GAN Mixer', 'VAE Audio Mixer', 'ResNet Audio Mixer']:
        # 2D CNNs expect [batch, channels, height, width]
        # Convert [batch, features, time] to [batch, 1, features, time]
        if len(input_data.shape) == 3:
            return input_data.unsqueeze(1)
        return input_data
        
    elif model_name == 'Advanced Transformer Mixer':
        # Transformer can handle various input formats
        # Just ensure it has a batch dimension
        return input_data
    
    else:
        # Default handling
        logger.warning(f"No specific adapter for {model_name}, using as-is")
        return input_data

def check_model_compatibility(model_name, input_shape):
    """
    Check if the input shape is compatible with the model requirements.
    
    Args:
        model_name (str): The name of the model
        input_shape (tuple): The shape of the input tensor
    
    Returns:
        bool: True if compatible, False otherwise
    """
    if model_name == 'LSTM Audio Mixer':
        # LSTM expects 3D input [batch, features, time]
        return len(input_shape) == 3
        
    elif model_name in ['Audio GAN Mixer', 'VAE Audio Mixer', 'ResNet Audio Mixer']:
        # 2D CNNs expect 4D input [batch, channels, height, width]
        return len(input_shape) == 4
        
    elif model_name == 'Advanced Transformer Mixer':
        # Transformer is flexible with input shapes
        return True
    
    else:
        # Default
        return True
