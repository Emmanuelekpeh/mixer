"""
Model Metadata Extractor for Tournament Web Application

This module provides functionality to extract and generate metadata for AI mixing models,
including parsing existing JSON metadata files and generating default metadata.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Standardized model metadata structure"""
    id: str
    name: str
    architecture: str
    file_path: str
    size_mb: float
    created_at: str
    description: str
    performance_metrics: Dict[str, Any]
    specializations: List[str]
    capabilities: Dict[str, float]
    parameters: Dict[str, Any]
    validation_status: str = "pending"
    compatibility_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.compatibility_info is None:
            self.compatibility_info = {}

class ModelMetadataExtractor:
    """
    Extracts and generates metadata for discovered models.
    
    This class provides functionality to:
    1. Parse existing JSON metadata files
    2. Generate default metadata for models without metadata
    3. Infer model capabilities from naming patterns
    4. Determine generation numbers and parent relationships
    """
    
    def __init__(self):
        """Initialize the ModelMetadataExtractor."""
        self.architecture_patterns = self._initialize_architecture_patterns()
        self.capability_mappings = self._initialize_capability_mappings()
        self.specialization_mappings = self._initialize_specialization_mappings()
        
        logger.info("ModelMetadataExtractor initialized")
    
    def extract_from_json(self, json_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a JSON file.
        
        Args:
            json_path: Path to the JSON metadata file
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            if not json_path.exists():
                logger.warning(f"JSON metadata file does not exist: {json_path}")
                return {}
            
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate and clean the metadata
            cleaned_metadata = self._validate_and_clean_metadata(metadata)
            
            logger.info(f"Successfully extracted metadata from {json_path}")
            return cleaned_metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata file {json_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error reading metadata file {json_path}: {e}")
            return {}
    
    def generate_default_metadata(self, model_path: Path) -> ModelMetadata:
        """
        Generate default metadata for a model file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            ModelMetadata object with generated metadata
        """
        try:
            # Get basic file information
            file_stats = model_path.stat()
            file_size_mb = file_stats.st_size / (1024 * 1024)
            created_at = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            
            # Extract model name and clean it
            model_name = model_path.stem
            clean_name = self._clean_model_name(model_name)
            
            # Infer architecture from name
            architecture = self._infer_architecture_from_name(model_name)
            
            # Generate capabilities based on architecture and name
            capabilities = self.infer_capabilities_from_name(model_name)
            
            # Generate specializations
            specializations = self._infer_specializations_from_name(model_name)
            
            # Generate description
            description = self._generate_description(model_name, architecture, specializations)
            
            # Create default performance metrics
            performance_metrics = self._generate_default_performance_metrics(architecture)
            
            # Generate parameters dictionary
            parameters = self._generate_default_parameters(model_name, architecture)
            
            # Create metadata object
            metadata = ModelMetadata(
                id=model_name,
                name=clean_name,
                architecture=architecture,
                file_path=str(model_path),
                size_mb=file_size_mb,
                created_at=created_at,
                description=description,
                performance_metrics=performance_metrics,
                specializations=specializations,
                capabilities=capabilities,
                parameters=parameters,
                validation_status="pending"
            )
            
            logger.info(f"Generated default metadata for {model_name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating default metadata for {model_path}: {e}")
            # Return minimal metadata
            return ModelMetadata(
                id=model_path.stem,
                name=model_path.stem.replace('_', ' ').title(),
                architecture="unknown",
                file_path=str(model_path),
                size_mb=0.0,
                created_at=datetime.now().isoformat(),
                description="Model metadata could not be generated",
                performance_metrics={},
                specializations=[],
                capabilities={},
                parameters={}
            )
    
    def infer_capabilities_from_name(self, model_name: str) -> Dict[str, float]:
        """
        Infer model capabilities from the model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of capabilities with confidence scores
        """
        capabilities = {}
        model_name_lower = model_name.lower()
        
        # Check for capability patterns in the name
        for capability, patterns in self.capability_mappings.items():
            confidence = 0.0
            
            for pattern in patterns:
                if pattern in model_name_lower:
                    confidence = max(confidence, 0.8)  # High confidence for direct matches
            
            # Check for related terms
            if capability == "spectral_analysis":
                if any(term in model_name_lower for term in ["spectrogram", "fft", "frequency"]):
                    confidence = max(confidence, 0.7)
            elif capability == "temporal_modeling":
                if any(term in model_name_lower for term in ["lstm", "rnn", "temporal", "sequence"]):
                    confidence = max(confidence, 0.7)
            elif capability == "feature_extraction":
                if any(term in model_name_lower for term in ["feature", "extract", "encoder"]):
                    confidence = max(confidence, 0.6)
            
            if confidence > 0:
                capabilities[capability] = confidence
        
        # Add default capabilities if none found
        if not capabilities:
            capabilities = {
                "audio_processing": 0.5,
                "feature_extraction": 0.4,
                "pattern_recognition": 0.3
            }
        
        return capabilities
    
    def determine_model_generation(self, model_name: str, existing_models: List[str]) -> int:
        """
        Determine generation number based on model name and existing models.
        
        Args:
            model_name: Name of the model
            existing_models: List of existing model names
            
        Returns:
            Generation number
        """
        # Extract base name without version indicators
        base_name = self._extract_base_model_name(model_name)
        
        # Find all models with the same base name
        related_models = []
        for existing_model in existing_models:
            if self._extract_base_model_name(existing_model) == base_name:
                related_models.append(existing_model)
        
        # If no related models, this is generation 1
        if not related_models:
            return 1
        
        # Extract version numbers from related models
        versions = []
        for model in related_models:
            version = self._extract_version_number(model)
            if version > 0:
                versions.append(version)
        
        # If no version numbers found, count models + 1
        if not versions:
            return len(related_models) + 1
        
        # Return next version number
        return max(versions) + 1
    
    def save_metadata(self, metadata: ModelMetadata, output_path: Path) -> bool:
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: ModelMetadata object to save
            output_path: Path to save the metadata file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert metadata to dictionary
            metadata_dict = asdict(metadata)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Metadata saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata to {output_path}: {e}")
            return False
    
    def _validate_and_clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        cleaned = {}
        
        # Required fields with defaults
        required_fields = {
            'id': '',
            'name': '',
            'architecture': 'unknown',
            'description': '',
            'performance_metrics': {},
            'specializations': [],
            'capabilities': {},
            'parameters': {}
        }
        
        # Copy and validate required fields
        for field, default in required_fields.items():
            if field in metadata:
                cleaned[field] = metadata[field]
            else:
                cleaned[field] = default
        
        # Validate specific field types
        if not isinstance(cleaned['specializations'], list):
            cleaned['specializations'] = []
        
        if not isinstance(cleaned['capabilities'], dict):
            cleaned['capabilities'] = {}
        
        if not isinstance(cleaned['performance_metrics'], dict):
            cleaned['performance_metrics'] = {}
        
        if not isinstance(cleaned['parameters'], dict):
            cleaned['parameters'] = {}
        
        # Copy optional fields
        optional_fields = ['created_at', 'updated_at', 'version', 'author', 'tags']
        for field in optional_fields:
            if field in metadata:
                cleaned[field] = metadata[field]
        
        return cleaned
    
    def _initialize_architecture_patterns(self) -> Dict[str, List[str]]:
        """Initialize architecture detection patterns."""
        return {
            'CNN': ['cnn', 'conv', 'convolution'],
            'Transformer': ['transformer', 'attention', 'bert', 'gpt'],
            'LSTM': ['lstm', 'long_short'],
            'RNN': ['rnn', 'recurrent'],
            'ResNet': ['resnet', 'residual'],
            'VAE': ['vae', 'variational'],
            'GAN': ['gan', 'generative'],
            'Ensemble': ['ensemble', 'hybrid', 'combined'],
            'AST': ['ast', 'audio_spectrogram'],
            'Diffusion': ['diffusion', 'ddpm', 'ddim']
        }
    
    def _initialize_capability_mappings(self) -> Dict[str, List[str]]:
        """Initialize capability detection patterns."""
        return {
            'spectral_analysis': ['spectral', 'spectrum', 'frequency'],
            'temporal_modeling': ['temporal', 'time', 'sequence'],
            'feature_extraction': ['feature', 'extract', 'encode'],
            'audio_enhancement': ['enhance', 'improve', 'quality'],
            'noise_reduction': ['denoise', 'clean', 'noise'],
            'mixing': ['mix', 'blend', 'combine'],
            'mastering': ['master', 'final', 'polish'],
            'creative_effects': ['creative', 'artistic', 'style'],
            'real_time': ['real_time', 'live', 'streaming']
        }
    
    def _initialize_specialization_mappings(self) -> Dict[str, List[str]]:
        """Initialize specialization detection patterns."""
        return {
            'Audio Processing': ['audio', 'sound', 'acoustic'],
            'Music Production': ['music', 'musical', 'song'],
            'Speech Processing': ['speech', 'voice', 'vocal'],
            'Sound Design': ['sound_design', 'fx', 'effects'],
            'Mixing': ['mix', 'mixing', 'blend'],
            'Mastering': ['master', 'mastering', 'final'],
            'Real-time Processing': ['real_time', 'live', 'streaming'],
            'Creative AI': ['creative', 'artistic', 'generative']
        }
    
    def _clean_model_name(self, model_name: str) -> str:
        """Clean and format model name for display."""
        # Replace underscores with spaces
        clean_name = model_name.replace('_', ' ')
        
        # Remove common suffixes
        suffixes = ['best', 'final', 'latest', 'v1', 'v2', 'v3']
        for suffix in suffixes:
            if clean_name.lower().endswith(f' {suffix}'):
                clean_name = clean_name[:-len(suffix)-1]
        
        # Title case
        clean_name = clean_name.title()
        
        return clean_name
    
    def _infer_architecture_from_name(self, model_name: str) -> str:
        """Infer architecture from model name."""
        model_name_lower = model_name.lower()
        
        for architecture, patterns in self.architecture_patterns.items():
            for pattern in patterns:
                if pattern in model_name_lower:
                    return architecture.lower()
        
        return "neural_network"
    
    def _infer_specializations_from_name(self, model_name: str) -> List[str]:
        """Infer specializations from model name."""
        specializations = []
        model_name_lower = model_name.lower()
        
        for specialization, patterns in self.specialization_mappings.items():
            for pattern in patterns:
                if pattern in model_name_lower:
                    if specialization not in specializations:
                        specializations.append(specialization)
        
        # Add default if none found
        if not specializations:
            specializations = ["Audio Processing"]
        
        return specializations
    
    def _generate_description(self, model_name: str, architecture: str, specializations: List[str]) -> str:
        """Generate a description for the model."""
        spec_str = ", ".join(specializations) if specializations else "audio processing"
        return f"{architecture} model specialized in {spec_str.lower()}"
    
    def _generate_default_performance_metrics(self, architecture: str) -> Dict[str, Any]:
        """Generate default performance metrics based on architecture."""
        base_metrics = {
            "training_epochs": 0,
            "validation_loss": 0.0,
            "training_time_hours": 0.0,
            "model_size_mb": 0.0
        }
        
        # Architecture-specific metrics
        if architecture in ["CNN", "ResNet"]:
            base_metrics.update({
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            })
        elif architecture in ["Transformer", "LSTM", "RNN"]:
            base_metrics.update({
                "perplexity": 0.0,
                "bleu_score": 0.0
            })
        elif architecture in ["VAE", "GAN"]:
            base_metrics.update({
                "reconstruction_loss": 0.0,
                "kl_divergence": 0.0
            })
        
        return base_metrics
    
    def _generate_default_parameters(self, model_name: str, architecture: str) -> Dict[str, Any]:
        """Generate default parameters dictionary."""
        return {
            "architecture": architecture,
            "input_shape": "unknown",
            "output_shape": "unknown",
            "trainable_parameters": 0,
            "inference_time_ms": 0.0
        }
    
    def _extract_base_model_name(self, model_name: str) -> str:
        """Extract base model name without version indicators."""
        # Remove version patterns
        base_name = re.sub(r'_v\d+$', '', model_name)
        base_name = re.sub(r'_version\d+$', '', base_name)
        
        # Remove common suffixes
        suffixes = ['_best', '_final', '_latest']
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
        
        return base_name
    
    def _extract_version_number(self, model_name: str) -> int:
        """Extract version number from model name."""
        # Look for version patterns
        patterns = [r'_v(\d+)$', r'_version(\d+)$', r'v(\d+)$']
        
        for pattern in patterns:
            match = re.search(pattern, model_name)
            if match:
                return int(match.group(1))
        
        return 0