"""
Model Integration System
=======================

Main orchestrator for the model discovery and integration system.
Provides a unified interface for discovering, validating, and integrating
AI models into the tournament system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .model_discovery_interfaces import ModelDiscoveryConfig, ModelDiscoveryEvent
from enhanced_model_discovery import (
    EnhancedModelDiscovery, EnhancedModelValidator, 
    EnhancedModelRegistry, ModelIntegrationEngine
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelIntegrationSystem:
    """
    Main system for model discovery and integration
    
    Coordinates between discovery, validation, registration, and integration
    components to provide a seamless model integration experience.
    """
    
    def __init__(self, config: Optional[ModelDiscoveryConfig] = None):
        """
        Initialize the model integration system
        
        Args:
            config: Configuration for model discovery, uses default if None
        """
        self.config = config or ModelDiscoveryConfig.default()
        
        # Initialize components
        self.discovery_engine = EnhancedModelDiscovery(self.config)
        self.validator = EnhancedModelValidator()
        self.registry = EnhancedModelRegistry()
        self.integration_engine = ModelIntegrationEngine()
        
        # Event handlers
        self.event_handlers = []
        
        logger.info("Model Integration System initialized")
    
    def discover_and_integrate_all(self) -> Dict[str, Any]:
        """
        Discover and integrate all models in configured search paths
        
        Returns:
            Dictionary with integration results and statistics
        """
        logger.info("Starting full model discovery and integration process...")
        
        results = {
            'discovered': 0,
            'validated': 0,
            'registered': 0,
            'integrated': 0,
            'failed': 0,
            'errors': [],
            'models': []
        }
        
        try:
            # Step 1: Discover models
            logger.info("Step 1: Discovering models...")
            discovered_models = self.discovery_engine.discover_models(self.config.search_paths)
            results['discovered'] = len(discovered_models)
            
            if not discovered_models:
                logger.warning("No models discovered in search paths")
                return results
            
            # Step 2: Process each discovered model
            for metadata in discovered_models:
                model_result = self._process_single_model(metadata)
                results['models'].append(model_result)
                
                # Update counters
                if model_result['validated']:
                    results['validated'] += 1
                if model_result['registered']:
                    results['registered'] += 1
                if model_result['integrated']:
                    results['integrated'] += 1
                if model_result['error']:
                    results['failed'] += 1
                    results['errors'].append(model_result['error'])
            
            logger.info(f"Integration process complete: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Integration process failed: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
    
    def discover_models_only(self) -> List[Dict[str, Any]]:
        """
        Discover models without validation or integration
        
        Returns:
            List of discovered model metadata
        """
        logger.info("Discovering models...")
        
        try:
            discovered_models = self.discovery_engine.discover_models(self.config.search_paths)
            
            # Convert to dictionaries for easier handling
            model_list = []
            for metadata in discovered_models:
                model_dict = asdict(metadata)
                model_list.append(model_dict)
            
            logger.info(f"Discovered {len(model_list)} models")
            return model_list
            
        except Exception as e:
            logger.error(f"Model discovery failed: {str(e)}")
            return []
    
    def validate_model(self, model_path: str) -> Dict[str, Any]:
        """
        Validate a specific model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Validation results
        """
        try:
            path = Path(model_path)
            
            # Create basic metadata
            metadata = self.discovery_engine._create_basic_metadata(path)
            
            # Validate
            validation_result = self.validator.validate_model(path, metadata)
            
            return {
                'model_id': metadata.id,
                'is_valid': validation_result.is_valid,
                'confidence': validation_result.confidence,
                'issues': validation_result.issues,
                'warnings': validation_result.warnings,
                'recommendations': validation_result.recommendations
            }
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_path}: {str(e)}")
            return {
                'model_id': Path(model_path).stem,
                'is_valid': False,
                'confidence': 0.0,
                'issues': [str(e)],
                'warnings': [],
                'recommendations': []
            }
    
    def register_model(self, model_path: str) -> bool:
        """
        Register a specific model
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if registration successful
        """
        try:
            path = Path(model_path)
            
            # Create metadata and validate
            metadata = self.discovery_engine._create_basic_metadata(path)
            validation_result = self.validator.validate_model(path, metadata)
            
            if not validation_result.is_valid:
                logger.error(f"Cannot register invalid model: {model_path}")
                return False
            
            # Register
            return self.registry.register_model(metadata, validation_result)
            
        except Exception as e:
            logger.error(f"Model registration failed for {model_path}: {str(e)}")
            return False
    
    def integrate_model(self, model_id: str) -> bool:
        """
        Integrate a registered model into the tournament system
        
        Args:
            model_id: ID of the model to integrate
            
        Returns:
            True if integration successful
        """
        try:
            return self.integration_engine.integrate_model(model_id)
        except Exception as e:
            logger.error(f"Model integration failed for {model_id}: {str(e)}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics
        
        Returns:
            System status information
        """
        try:
            # Get registered models
            registered_models = self.registry.get_registered_models()
            
            # Count by status
            active_models = len([m for m in registered_models if m['is_active']])
            inactive_models = len([m for m in registered_models if not m['is_active']])
            
            # Count by architecture
            architectures = {}
            for model in registered_models:
                arch = model['architecture']
                architectures[arch] = architectures.get(arch, 0) + 1
            
            return {
                'total_registered': len(registered_models),
                'active_models': active_models,
                'inactive_models': inactive_models,
                'architectures': architectures,
                'search_paths': [str(p) for p in self.config.search_paths],
                'supported_formats': self.config.supported_formats
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {'error': str(e)}
    
    def sync_with_database(self) -> Dict[str, Any]:
        """
        Synchronize all discovered models with the database
        
        Returns:
            Synchronization results
        """
        return self.integration_engine.sync_with_database()
    
    def _process_single_model(self, metadata) -> Dict[str, Any]:
        """Process a single discovered model through the full pipeline"""
        model_result = {
            'model_id': metadata.id,
            'model_path': metadata.file_path,
            'validated': False,
            'registered': False,
            'integrated': False,
            'error': None
        }
        
        try:
            # Step 1: Validate
            validation_result = self.validator.validate_model(Path(metadata.file_path), metadata)
            
            if validation_result.is_valid:
                model_result['validated'] = True
                
                # Step 2: Register if auto-registration is enabled
                if self.config.auto_registration:
                    if self.registry.register_model(metadata, validation_result):
                        model_result['registered'] = True
                        
                        # Step 3: Integrate if auto-integration is enabled
                        if self.config.auto_integration:
                            if self.integration_engine.integrate_model(metadata.id):
                                model_result['integrated'] = True
            else:
                logger.warning(f"Model {metadata.id} failed validation: {validation_result.issues}")
            
        except Exception as e:
            error_msg = f"Failed to process model {metadata.id}: {str(e)}"
            model_result['error'] = error_msg
            logger.error(error_msg)
        
        return model_result

# Convenience functions for easy usage
def quick_discover_models(search_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Quick function to discover models
    
    Args:
        search_paths: Optional list of paths to search
        
    Returns:
        List of discovered models
    """
    config = ModelDiscoveryConfig.default()
    if search_paths:
        config.search_paths = [Path(p) for p in search_paths]
    
    system = ModelIntegrationSystem(config)
    return system.discover_models_only()

def quick_integrate_all(search_paths: Optional[List[str]] = None, auto_register: bool = True, 
                       auto_integrate: bool = False) -> Dict[str, Any]:
    """
    Quick function to discover and integrate all models
    
    Args:
        search_paths: Optional list of paths to search
        auto_register: Whether to automatically register valid models
        auto_integrate: Whether to automatically integrate registered models
        
    Returns:
        Integration results
    """
    config = ModelDiscoveryConfig.default()
    if search_paths:
        config.search_paths = [Path(p) for p in search_paths]
    
    config.auto_registration = auto_register
    config.auto_integration = auto_integrate
    
    system = ModelIntegrationSystem(config)
    return system.discover_and_integrate_all()

def get_integration_status() -> Dict[str, Any]:
    """
    Get current integration system status
    
    Returns:
        System status information
    """
    system = ModelIntegrationSystem()
    return system.get_system_status()

# Example usage and testing
if __name__ == "__main__":
    print("🔍 Model Integration System Test")
    print("=" * 50)
    
    # Test discovery
    print("\n1. Testing model discovery...")
    models = quick_discover_models()
    print(f"   Found {len(models)} models")
    
    # Test system status
    print("\n2. Getting system status...")
    status = get_integration_status()
    print(f"   Registered models: {status.get('total_registered', 0)}")
    print(f"   Active models: {status.get('active_models', 0)}")
    
    # Test full integration (with auto-register but not auto-integrate)
    print("\n3. Testing full integration process...")
    results = quick_integrate_all(auto_register=True, auto_integrate=False)
    print(f"   Discovered: {results['discovered']}")
    print(f"   Validated: {results['validated']}")
    print(f"   Registered: {results['registered']}")
    print(f"   Integrated: {results['integrated']}")
    
    if results['errors']:
        print(f"   Errors: {len(results['errors'])}")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"     - {error}")
    
    print("\n✅ Model Integration System test complete!")