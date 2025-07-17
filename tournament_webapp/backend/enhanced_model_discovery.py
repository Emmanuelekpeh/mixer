"""
Enhanced Model Discovery Engine
==============================

Integrates with existing TournamentModelManager and database infrastructure
to provide comprehensive model discovery and integration capabilities.
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import field
from datetime import datetime
import torch

# Import existing components
from .tournament_model_manager import TournamentModelManager
from .database_service import DatabaseService
from .database import AIModel
from model_discovery_interfaces import (
    ModelDiscoveryInterface, ModelMetadataInterface, ModelValidatorInterface,
    ModelRegistryInterface, ModelIntegrationInterface,
    ModelMetadata, ValidationResult, ModelDiscoveryConfig, ModelDiscoveryEvent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedModelDiscovery(ModelDiscoveryInterface):
    """Enhanced model discovery that works with existing infrastructure"""
    
    def __init__(self, config: ModelDiscoveryConfig):
        self.config = config
        self.supported_formats = config.supported_formats
        self._cache = {}
    
    def discover_models(self, search_paths: List[Path]) -> List[ModelMetadata]:
        """Discover models in specified paths"""
        discovered_models = []
        
        for path in search_paths:
            if path.exists():
                models = self.scan_directory(path, recursive=True)
                discovered_models.extend(models)
                logger.info(f"Discovered {len(models)} models in {path}")
            else:
                logger.warning(f"Search path does not exist: {path}")
        
        # Remove duplicates based on file hash
        unique_models = self._deduplicate_models(discovered_models)
        logger.info(f"Total unique models discovered: {len(unique_models)}")
        
        return unique_models
    
    def scan_directory(self, directory: Path, recursive: bool = True) -> List[ModelMetadata]:
        """Scan directory for model files"""
        models = []
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix in self.supported_formats:
                    try:
                        metadata = self._create_basic_metadata(file_path)
                        models.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {str(e)}")
        
        return models
    
    def get_supported_formats(self) -> List[str]:
        """Get supported model formats"""
        return self.supported_formats
    
    def _create_basic_metadata(self, file_path: Path) -> ModelMetadata:
        """Create basic metadata for a model file"""
        file_stats = file_path.stat()
        
        # Try to load existing metadata
        metadata_file = file_path.with_suffix('.json')
        existing_metadata = {}
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {file_path}: {str(e)}")
        
        return ModelMetadata(
            id=existing_metadata.get('id', file_path.stem),
            name=existing_metadata.get('name', file_path.stem.replace('_', ' ').title()),
            architecture=existing_metadata.get('architecture', self._infer_architecture(file_path)),
            file_path=str(file_path),
            size_mb=file_stats.st_size / (1024 * 1024),
            created_at=existing_metadata.get('created_at', datetime.fromtimestamp(file_stats.st_mtime).isoformat()),
            description=existing_metadata.get('description', ''),
            performance_metrics=existing_metadata.get('performance_metrics', {}),
            specializations=existing_metadata.get('specializations', []),
            capabilities=existing_metadata.get('capabilities', {}),
            parameters=existing_metadata.get('parameters', {}),
            validation_status='pending'
        )
    
    def _infer_architecture(self, file_path: Path) -> str:
        """Infer architecture from filename and path"""
        filename = file_path.stem.lower()
        
        # Architecture inference rules
        if 'cnn' in filename or 'conv' in filename:
            return 'CNN'
        elif 'transformer' in filename or 'attention' in filename:
            return 'Transformer'
        elif 'lstm' in filename or 'rnn' in filename:
            return 'LSTM/RNN'
        elif 'gan' in filename:
            return 'GAN'
        elif 'vae' in filename:
            return 'VAE'
        elif 'resnet' in filename:
            return 'ResNet'
        elif 'diffusion' in filename:
            return 'Diffusion'
        else:
            return 'Unknown'
    
    def _deduplicate_models(self, models: List[ModelMetadata]) -> List[ModelMetadata]:
        """Remove duplicate models based on file hash"""
        seen_hashes = set()
        unique_models = []
        
        for model in models:
            try:
                file_hash = self._calculate_file_hash(Path(model.file_path))
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    unique_models.append(model)
            except Exception as e:
                logger.warning(f"Failed to hash {model.file_path}: {str(e)}")
                # Include model anyway if we can't hash it
                unique_models.append(model)
        
        return unique_models
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

class EnhancedModelValidator(ModelValidatorInterface):
    """Enhanced model validator with tournament system integration"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def validate_model(self, model_path: Path, metadata: ModelMetadata) -> ValidationResult:
        """Validate model file and metadata"""
        issues = []
        warnings = []
        recommendations = []
        
        # Basic file validation
        if not model_path.exists():
            issues.append(f"Model file does not exist: {model_path}")
            return ValidationResult(False, 0.0, issues, warnings, recommendations, {})
        
        # Size validation
        if metadata.size_mb > 500:  # 500MB threshold
            warnings.append(f"Large model file ({metadata.size_mb:.1f}MB) may impact performance")
        
        # Try to load the model
        load_success, load_error = self._test_model_loading(model_path)
        if not load_success:
            issues.append(f"Failed to load model: {load_error}")
        
        # Test inference if loading succeeded
        inference_success = False
        if load_success:
            inference_success, inference_error = self.test_inference(model_path)
            if not inference_success:
                issues.append(f"Model inference failed: {inference_error}")
        
        # Check compatibility
        compatibility = self.check_compatibility(model_path)
        if not compatibility.get('compatible', False):
            issues.extend(compatibility.get('issues', []))
        
        # Calculate confidence score
        confidence = self._calculate_confidence(load_success, inference_success, len(issues), len(warnings))
        
        # Generate recommendations
        if metadata.size_mb > 100:
            recommendations.append("Consider model optimization for better performance")
        
        if not metadata.capabilities:
            recommendations.append("Add capability metadata for better tournament matching")
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                'load_success': load_success,
                'inference_success': inference_success,
                'compatibility': compatibility
            }
        )
    
    def check_compatibility(self, model_path: Path) -> Dict[str, Any]:
        """Check tournament system compatibility"""
        compatibility_info = {
            'compatible': True,
            'issues': [],
            'requirements': [],
            'supported_features': []
        }
        
        try:
            # Check if it's a PyTorch model
            if model_path.suffix == '.pth':
                compatibility_info['supported_features'].append('pytorch_native')
            
            # Check device compatibility
            if torch.cuda.is_available():
                compatibility_info['supported_features'].append('gpu_acceleration')
            
            compatibility_info['supported_features'].append('cpu_inference')
            
        except Exception as e:
            compatibility_info['issues'].append(f"Compatibility check failed: {str(e)}")
            compatibility_info['compatible'] = False
        
        return compatibility_info
    
    def test_inference(self, model_path: Path) -> Tuple[bool, str]:
        """Test model inference capability"""
        try:
            # Load model
            model = torch.load(model_path, map_location=self.device)
            
            # Try to put in eval mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Create dummy input (typical spectrogram shape)
            dummy_input = torch.randn(1, 1, 128, 128).to(self.device)
            
            # Test forward pass
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    output = model(dummy_input)
                elif callable(model):
                    output = model(dummy_input)
                else:
                    return False, "Model is not callable"
            
            return True, "Inference test successful"
            
        except Exception as e:
            return False, str(e)
    
    def _test_model_loading(self, model_path: Path) -> Tuple[bool, str]:
        """Test if model can be loaded"""
        try:
            torch.load(model_path, map_location='cpu')
            return True, "Model loaded successfully"
        except Exception as e:
            return False, str(e)
    
    def _calculate_confidence(self, load_success: bool, inference_success: bool, 
                            num_issues: int, num_warnings: int) -> float:
        """Calculate validation confidence score"""
        base_score = 1.0
        
        if not load_success:
            base_score -= 0.5
        
        if not inference_success:
            base_score -= 0.3
        
        # Reduce score for issues and warnings
        base_score -= (num_issues * 0.1)
        base_score -= (num_warnings * 0.05)
        
        return max(0.0, min(1.0, base_score))

class EnhancedModelRegistry(ModelRegistryInterface):
    """Enhanced model registry with database integration"""
    
    def __init__(self):
        self.db_service = DatabaseService()
    
    def register_model(self, metadata: ModelMetadata, validation_result: ValidationResult) -> bool:
        """Register validated model in database"""
        try:
            # Check if model already exists
            existing_model = self.db_service.get_model(metadata.id)
            if existing_model:
                logger.info(f"Model {metadata.id} already registered, updating...")
                return self._update_existing_model(existing_model, metadata, validation_result)
            
            # Create new AIModel instance
            ai_model = AIModel(
                id=metadata.id,
                name=metadata.name,
                architecture=metadata.architecture,
                description=metadata.description,
                model_file_path=metadata.file_path,
                specializations=metadata.specializations,
                capabilities=metadata.capabilities,
                is_active=validation_result.is_valid,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            # Add to database
            self.db_service.db.add(ai_model)
            self.db_service.db.commit()
            
            logger.info(f"Successfully registered model: {metadata.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {metadata.id}: {str(e)}")
            self.db_service.db.rollback()
            return False
    
    def update_model_status(self, model_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Update model status"""
        try:
            model = self.db_service.get_model(model_id)
            if not model:
                logger.error(f"Model not found: {model_id}")
                return False
            
            # Update status-related fields
            if status == 'active':
                model.is_active = True
            elif status == 'inactive':
                model.is_active = False
            
            # Update additional details if provided
            if details:
                if 'description' in details:
                    model.description = details['description']
                if 'capabilities' in details:
                    model.capabilities = details['capabilities']
            
            model.updated_at = datetime.now()
            self.db_service.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model status for {model_id}: {str(e)}")
            self.db_service.db.rollback()
            return False
    
    def get_registered_models(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get registered models with optional filters"""
        try:
            query = self.db_service.db.query(AIModel)
            
            # Apply filters
            if filters:
                if 'architecture' in filters:
                    query = query.filter(AIModel.architecture == filters['architecture'])
                if 'is_active' in filters:
                    query = query.filter(AIModel.is_active == filters['is_active'])
                if 'tier' in filters:
                    query = query.filter(AIModel.tier == filters['tier'])
            
            models = query.all()
            
            # Convert to dictionaries
            return [{
                'id': model.id,
                'name': model.name,
                'architecture': model.architecture,
                'tier': model.tier,
                'elo_rating': model.elo_rating,
                'is_active': model.is_active,
                'file_path': model.model_file_path,
                'capabilities': model.capabilities,
                'specializations': model.specializations
            } for model in models]
            
        except Exception as e:
            logger.error(f"Failed to get registered models: {str(e)}")
            return []
    
    def remove_model(self, model_id: str) -> bool:
        """Remove model from registry"""
        try:
            model = self.db_service.get_model(model_id)
            if not model:
                logger.error(f"Model not found: {model_id}")
                return False
            
            self.db_service.db.delete(model)
            self.db_service.db.commit()
            
            logger.info(f"Successfully removed model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model {model_id}: {str(e)}")
            self.db_service.db.rollback()
            return False
    
    def _update_existing_model(self, existing_model: AIModel, metadata: ModelMetadata, 
                             validation_result: ValidationResult) -> bool:
        """Update existing model with new metadata"""
        try:
            # Cast to Any to satisfy static type checkers that dislike SQLAlchemy column assignment
            from typing import cast, Any  # local import to avoid top-level dependency

            ex_model = cast(Any, existing_model)

            ex_model.name = metadata.name  # type: ignore[assignment]
            ex_model.architecture = metadata.architecture  # type: ignore[assignment]
            ex_model.description = metadata.description  # type: ignore[assignment]
            ex_model.model_file_path = metadata.file_path  # type: ignore[assignment]
            ex_model.specializations = metadata.specializations  # type: ignore[assignment]
            ex_model.capabilities = metadata.capabilities  # type: ignore[assignment]
            ex_model.is_active = validation_result.is_valid  # type: ignore[assignment]
            ex_model.updated_at = datetime.now()  # type: ignore[assignment]
            
            self.db_service.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update existing model: {str(e)}")
            self.db_service.db.rollback()
            return False

class ModelIntegrationEngine(ModelIntegrationInterface):
    """Engine for integrating models into tournament system"""
    
    def __init__(self):
        self.tournament_manager = None  # Will be initialized when needed
        self.db_service = DatabaseService()
    
    def integrate_model(self, model_id: str) -> bool:
        """Integrate model into tournament system"""
        try:
            # Get model from database
            model = self.db_service.get_model(model_id)
            if not model:
                logger.error(f"Model not found in database: {model_id}")
                return False
            
            # Test tournament compatibility
            validation_result = self.test_tournament_compatibility(model_id)
            if not validation_result.is_valid:
                logger.error(f"Model {model_id} failed tournament compatibility test")
                return False
            
            # Mark as active and tournament-ready
            model.is_active = True
            model.last_used = datetime.now()
            self.db_service.db.commit()
            
            logger.info(f"Successfully integrated model into tournament system: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate model {model_id}: {str(e)}")
            return False
    
    def test_tournament_compatibility(self, model_id: str) -> ValidationResult:
        """Test model compatibility with tournament battles"""
        issues = []
        warnings = []
        
        try:
            # Get model from database
            model = self.db_service.get_model(model_id)
            if not model:
                issues.append(f"Model not found in database: {model_id}")
                return ValidationResult(False, 0.0, issues, warnings, [], {})
            
            # Check if model file exists
            if model.model_file_path and not Path(model.model_file_path).exists():
                issues.append(f"Model file not found: {model.model_file_path}")
            
            # Check required capabilities
            required_capabilities = ['spectral_analysis']  # Minimum requirement
            model_capabilities = model.capabilities or {}
            
            for capability in required_capabilities:
                if capability not in model_capabilities:
                    warnings.append(f"Missing recommended capability: {capability}")
            
            # Test with tournament manager if available
            if model.model_file_path:
                try:
                    # Initialize tournament manager if needed
                    if not self.tournament_manager:
                        models_dir = Path(model.model_file_path).parent
                        self.tournament_manager = TournamentModelManager(models_dir)
                    
                    # Test model loading
                    model_instance = self.tournament_manager._get_model_instance(model_id)
                    if not model_instance:
                        issues.append("Failed to create model instance for tournament")
                
                except Exception as e:
                    issues.append(f"Tournament manager test failed: {str(e)}")
            
            is_valid = len(issues) == 0
            confidence = 1.0 - (len(issues) * 0.3) - (len(warnings) * 0.1)
            confidence = max(0.0, min(1.0, confidence))
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                issues=issues,
                warnings=warnings,
                recommendations=[],
                metadata={'tournament_ready': is_valid}
            )
            
        except Exception as e:
            issues.append(f"Compatibility test failed: {str(e)}")
            return ValidationResult(False, 0.0, issues, warnings, [], {})
    
    def sync_with_database(self) -> Dict[str, Any]:
        """Synchronize discovered models with database"""
        stats = {
            'models_found': 0,
            'models_registered': 0,
            'models_updated': 0,
            'models_integrated': 0,
            'errors': []
        }
        
        try:
            # Get configuration
            config = ModelDiscoveryConfig.default()
            
            # Discover models
            discovery_engine = EnhancedModelDiscovery(config)
            discovered_models = discovery_engine.discover_models(config.search_paths)
            stats['models_found'] = len(discovered_models)
            
            # Validate and register models
            validator = EnhancedModelValidator()
            registry = EnhancedModelRegistry()
            
            for metadata in discovered_models:
                try:
                    # Validate model
                    validation_result = validator.validate_model(Path(metadata.file_path), metadata)
                    
                    # Register if valid
                    if validation_result.is_valid:
                        if registry.register_model(metadata, validation_result):
                            stats['models_registered'] += 1
                            
                            # Try to integrate
                            if self.integrate_model(metadata.id):
                                stats['models_integrated'] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to process model {metadata.id}: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Synchronization complete: {stats}")
            return stats
            
        except Exception as e:
            error_msg = f"Synchronization failed: {str(e)}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
            return stats