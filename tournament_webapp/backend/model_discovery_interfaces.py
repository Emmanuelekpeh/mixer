"""
Model Discovery System Interfaces
================================

Core interfaces for the model integration system that work with the existing
TournamentModelManager and database infrastructure.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

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

@dataclass
class ValidationResult:
    """Model validation result"""
    is_valid: bool
    confidence: float
    issues: List[str]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelDiscoveryInterface(ABC):
    """Interface for discovering models in various locations"""
    
    @abstractmethod
    def discover_models(self, search_paths: List[Path]) -> List[ModelMetadata]:
        """
        Discover models in the specified paths
        
        Args:
            search_paths: List of paths to search for models
            
        Returns:
            List of discovered model metadata
        """
        pass
    
    @abstractmethod
    def scan_directory(self, directory: Path, recursive: bool = True) -> List[ModelMetadata]:
        """
        Scan a directory for models
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            
        Returns:
            List of discovered model metadata
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported model file formats
        
        Returns:
            List of supported file extensions
        """
        pass

class ModelMetadataInterface(ABC):
    """Interface for extracting and managing model metadata"""
    
    @abstractmethod
    def extract_metadata(self, model_path: Path) -> ModelMetadata:
        """
        Extract metadata from a model file
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Extracted model metadata
        """
        pass
    
    @abstractmethod
    def infer_architecture(self, model_path: Path) -> str:
        """
        Infer model architecture from file structure
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Inferred architecture name
        """
        pass
    
    @abstractmethod
    def extract_capabilities(self, model_path: Path) -> Dict[str, float]:
        """
        Extract or infer model capabilities
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary of capabilities with confidence scores
        """
        pass
    
    @abstractmethod
    def save_metadata(self, metadata: ModelMetadata, output_path: Path) -> bool:
        """
        Save metadata to a file
        
        Args:
            metadata: Model metadata to save
            output_path: Path to save metadata file
            
        Returns:
            True if successful, False otherwise
        """
        pass

class ModelValidatorInterface(ABC):
    """Interface for validating discovered models"""
    
    @abstractmethod
    def validate_model(self, model_path: Path, metadata: ModelMetadata) -> ValidationResult:
        """
        Validate a model file and its metadata
        
        Args:
            model_path: Path to the model file
            metadata: Model metadata
            
        Returns:
            Validation result
        """
        pass
    
    @abstractmethod
    def check_compatibility(self, model_path: Path) -> Dict[str, Any]:
        """
        Check model compatibility with the tournament system
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Compatibility information
        """
        pass
    
    @abstractmethod
    def test_inference(self, model_path: Path) -> Tuple[bool, str]:
        """
        Test if model can perform inference
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Tuple of (success, error_message)
        """
        pass

class ModelRegistryInterface(ABC):
    """Interface for managing the model registry"""
    
    @abstractmethod
    def register_model(self, metadata: ModelMetadata, validation_result: ValidationResult) -> bool:
        """
        Register a validated model in the system
        
        Args:
            metadata: Model metadata
            validation_result: Validation result
            
        Returns:
            True if registration successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update_model_status(self, model_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update model status in the registry
        
        Args:
            model_id: Model identifier
            status: New status
            details: Additional status details
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_registered_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get list of registered models
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of registered model information
        """
        pass
    
    @abstractmethod
    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the registry
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if removal successful, False otherwise
        """
        pass

class ModelIntegrationInterface(ABC):
    """Interface for integrating models into the tournament system"""
    
    @abstractmethod
    def integrate_model(self, model_id: str) -> bool:
        """
        Integrate a registered model into the tournament system
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if integration successful, False otherwise
        """
        pass
    
    @abstractmethod
    def test_tournament_compatibility(self, model_id: str) -> ValidationResult:
        """
        Test model compatibility with tournament battles
        
        Args:
            model_id: Model identifier
            
        Returns:
            Validation result for tournament compatibility
        """
        pass
    
    @abstractmethod
    def sync_with_database(self) -> Dict[str, Any]:
        """
        Synchronize discovered models with the database
        
        Returns:
            Synchronization results and statistics
        """
        pass

# Event system for model discovery
class ModelDiscoveryEvent:
    """Event for model discovery notifications"""
    
    def __init__(self, event_type: str, model_id: str, data: Optional[Dict[str, Any]] = None):
        self.event_type = event_type  # 'discovered', 'validated', 'registered', 'integrated'
        self.model_id = model_id
        self.data = data or {}
        self.timestamp = datetime.now()

class ModelDiscoveryEventHandler(ABC):
    """Interface for handling model discovery events"""
    
    @abstractmethod
    def handle_event(self, event: ModelDiscoveryEvent) -> None:
        """
        Handle a model discovery event
        
        Args:
            event: The event to handle
        """
        pass

# Configuration for model discovery
@dataclass
class ModelDiscoveryConfig:
    """Configuration for model discovery system"""
    search_paths: List[Path]
    supported_formats: List[str]
    auto_validation: bool = True
    auto_registration: bool = False
    auto_integration: bool = False
    validation_timeout: int = 30
    metadata_cache_ttl: int = 3600
    enable_events: bool = True
    backup_metadata: bool = True
    
    @classmethod
    def default(cls) -> 'ModelDiscoveryConfig':
        """Create default configuration"""
        return cls(
            search_paths=[
                Path("models"),
                Path("tournament_webapp/backend/models"),
                Path("tournament_webapp/tournament_models")
            ],
            supported_formats=[".pth", ".pt", ".onnx", ".pkl", ".joblib"],
            auto_validation=True,
            auto_registration=False,
            auto_integration=False
        )