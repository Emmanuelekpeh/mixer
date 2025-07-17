"""
Model Discovery Service for Tournament Web Application

This service handles the discovery and scanning of AI mixing models from the file system,
resolving conflicts between different model directories and prioritizing newer models.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import json
import time
import re
import fcntl
import errno

# Import the ModelValidator
try:
    from model_validator_fixed import ModelValidator, ValidationResult as ModelValidationResult
except ImportError:
    from model_validator import ModelValidator, ValidationResult as ModelValidationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """Represents a discovered model file candidate for integration."""
    file_path: Path
    model_name: str
    file_size: int
    last_modified: datetime
    source_directory: str  # "main" or "deployment"
    metadata_path: Optional[Path]
    priority_score: int


class ModelDiscoveryService:
    """
    Service for discovering and managing AI mixing models from the file system.
    
    Handles scanning both main models directory and deployment folder,
    resolving conflicts and prioritizing newer models.
    """
    
    def __init__(self, main_models_dir: str = "models", deployment_dir: str = "models/deployment", 
                validation_timeout: int = 60):
        """
        Initialize the ModelDiscoveryService.
        
        Args:
            main_models_dir: Path to the main models directory
            deployment_dir: Path to the deployment models directory
            validation_timeout: Timeout in seconds for model validation operations
        """
        self.main_models_dir = Path(main_models_dir)
        self.deployment_dir = Path(deployment_dir)
        self.supported_extensions = {'.pth'}
        self.metadata_extensions = {'.json'}
        self.validation_timeout = validation_timeout
        
        # Initialize model validator
        self.model_validator = ModelValidator(timeout_seconds=validation_timeout)
        
        # Ensure directories exist
        self.main_models_dir.mkdir(parents=True, exist_ok=True)
        if not self.deployment_dir.exists():
            logger.info(f"Deployment directory {self.deployment_dir} does not exist, skipping")
    
    def scan_for_models(self) -> List[ModelCandidate]:
        """
        Scan both main and deployment directories for model files.
        
        Returns:
            List of ModelCandidate objects representing discovered models
        """
        logger.info("Starting model discovery scan...")
        
        candidates = []
        
        # Scan main models directory
        main_candidates = self._scan_directory(self.main_models_dir, "main")
        candidates.extend(main_candidates)
        logger.info(f"Found {len(main_candidates)} models in main directory")
        
        # Scan deployment directory if it exists
        if self.deployment_dir.exists():
            deployment_candidates = self._scan_directory(self.deployment_dir, "deployment")
            candidates.extend(deployment_candidates)
            logger.info(f"Found {len(deployment_candidates)} models in deployment directory")
        
        # Resolve conflicts between directories
        resolved_candidates = self.resolve_model_conflicts(candidates)
        
        logger.info(f"Model discovery complete. Found {len(resolved_candidates)} unique models")
        return resolved_candidates
    
    def _scan_directory(self, directory: Path, source_type: str) -> List[ModelCandidate]:
        """
        Scan a specific directory for model files.
        
        Args:
            directory: Directory path to scan
            source_type: Type of source directory ("main" or "deployment")
            
        Returns:
            List of ModelCandidate objects from this directory
        """
        candidates = []
        
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return candidates
        
        try:
            # Find all model files
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix in self.supported_extensions:
                    # Skip if file is being written (check if file is locked or very recent)
                    if not self.is_model_file_complete(file_path):
                        logger.debug(f"Skipping incomplete file: {file_path}")
                        continue
                    
                    candidate = self._create_model_candidate(file_path, source_type)
                    if candidate:
                        candidates.append(candidate)
                        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return candidates
    
    def _create_model_candidate(self, file_path: Path, source_type: str) -> Optional[ModelCandidate]:
        """
        Create a ModelCandidate from a discovered model file.
        
        Args:
            file_path: Path to the model file
            source_type: Source directory type
            
        Returns:
            ModelCandidate object or None if creation fails
        """
        try:
            # Get file stats
            stat = file_path.stat()
            file_size = stat.st_size
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Extract model name (without extension)
            model_name = file_path.stem
            
            # Look for associated metadata file
            metadata_path = self._find_metadata_file(file_path)
            
            # Calculate priority score
            priority_score = self.get_model_priority(file_path, source_type)
            
            return ModelCandidate(
                file_path=file_path,
                model_name=model_name,
                file_size=file_size,
                last_modified=last_modified,
                source_directory=source_type,
                metadata_path=metadata_path,
                priority_score=priority_score
            )
            
        except Exception as e:
            logger.error(f"Error creating candidate for {file_path}: {e}")
            return None
    
    def _find_metadata_file(self, model_path: Path) -> Optional[Path]:
        """
        Find associated metadata file for a model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Path to metadata file or None if not found
        """
        # Look for JSON file with same name
        json_path = model_path.with_suffix('.json')
        if json_path.exists():
            return json_path
        
        # Look for JSON file with model name pattern
        model_name = model_path.stem
        for json_file in model_path.parent.glob(f"*{model_name}*.json"):
            if json_file.exists():
                return json_file
        
        return None
    
    def resolve_model_conflicts(self, candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        """
        Resolve conflicts between models with the same name from different directories.
        
        Args:
            candidates: List of all discovered model candidates
            
        Returns:
            List of resolved model candidates with conflicts removed
        """
        # Group candidates by model name
        model_groups: Dict[str, List[ModelCandidate]] = {}
        for candidate in candidates:
            # Extract base model name (remove version suffixes like _v2, _v3)
            base_name = self._extract_base_model_name(candidate.model_name)
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(candidate)
        
        resolved_candidates = []
        conflicts_resolved = 0
        
        for model_name, group in model_groups.items():
            if len(group) == 1:
                # No conflict, add the single candidate
                resolved_candidates.append(group[0])
            else:
                # Resolve conflict by priority and timestamp
                best_candidate = self._select_best_candidate(group)
                resolved_candidates.append(best_candidate)
                conflicts_resolved += 1
                
                # Log the conflict resolution
                other_candidates = [c for c in group if c != best_candidate]
                logger.info(f"Resolved conflict for model '{model_name}': "
                          f"Selected {best_candidate.source_directory} version "
                          f"({best_candidate.file_path.name}, priority={best_candidate.priority_score}) "
                          f"over {[(c.file_path.name, c.priority_score) for c in other_candidates]}")
        
        if conflicts_resolved > 0:
            logger.info(f"Resolved {conflicts_resolved} model conflicts during discovery")
        
        return resolved_candidates
    
    def _extract_base_model_name(self, model_name: str) -> str:
        """
        Extract the base model name by removing version indicators.
        
        Args:
            model_name: Original model name
            
        Returns:
            Base model name without version suffixes
        """
        # Remove common version suffixes (_v1, _v2, etc.)
        base_name = re.sub(r'_v\d+$', '', model_name)
        
        # Remove other common suffixes
        suffixes = ['_best', '_final', '_latest']
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
        
        return base_name
    
    def _select_best_candidate(self, candidates: List[ModelCandidate]) -> ModelCandidate:
        """
        Select the best candidate from a group of conflicting models.
        
        Args:
            candidates: List of candidates with the same model name
            
        Returns:
            The best candidate based on priority and timestamp
        """
        # First, check if any candidate has "best" in the name
        best_candidates = [c for c in candidates if "best" in c.file_path.name.lower()]
        if best_candidates:
            candidates = best_candidates
        
        # Next, prioritize by source directory (main over deployment)
        main_candidates = [c for c in candidates if c.source_directory == "main"]
        if main_candidates:
            candidates = main_candidates
        
        # Then, check for version numbers in filenames
        version_dict = {}
        for candidate in candidates:
            version = self._extract_version_number(candidate.file_path.name)
            version_dict[candidate] = version
        
        # If we have version numbers, prioritize by version
        if any(v > 0 for v in version_dict.values()):
            highest_version = max(version_dict.values())
            highest_version_candidates = [c for c, v in version_dict.items() if v == highest_version]
            if highest_version_candidates:
                candidates = highest_version_candidates
        
        # Finally, sort by priority score and modification time
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.priority_score, c.last_modified),
            reverse=True
        )
        
        return sorted_candidates[0]
        
    def _extract_version_number(self, filename: str) -> int:
        """
        Extract version number from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Version number or 0 if not found
        """
        # Look for patterns like _v2, v3, -v4, etc.
        match = re.search(r'[_-]v(\d+)', filename.lower())
        if match:
            return int(match.group(1))
        
        # Look for patterns like _version2, version3, etc.
        match = re.search(r'version(\d+)', filename.lower())
        if match:
            return int(match.group(1))
        
        return 0
    
    def get_model_priority(self, model_path: Path, source_type: str) -> int:
        """
        Calculate priority score for a model based on its location and characteristics.
        
        Args:
            model_path: Path to the model file
            source_type: Source directory type
            
        Returns:
            Priority score (higher is better)
        """
        priority = 0
        
        # Base priority by source directory
        if source_type == "main":
            priority += 100  # Main directory has higher priority
        elif source_type == "deployment":
            priority += 50   # Deployment directory has lower priority
        
        # Bonus for "best" models
        if "best" in model_path.name.lower():
            priority += 20
        
        # Bonus for newer architecture patterns
        model_name_lower = model_path.name.lower()
        if "transformer" in model_name_lower:
            priority += 15
        elif "ensemble" in model_name_lower:
            priority += 10
        elif "enhanced" in model_name_lower:
            priority += 5
        
        # Small bonus for larger files (assuming more complex models)
        try:
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:
                priority += 5
            elif file_size_mb > 50:
                priority += 2
        except:
            pass
        
        return priority
    
    def is_model_file_complete(self, model_path: Path) -> bool:
        """
        Check if a model file is complete and not being written.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if file appears complete, False otherwise
        """
        try:
            # Check if file was modified very recently (within last 30 seconds)
            stat = model_path.stat()
            time_since_modified = time.time() - stat.st_mtime
            
            if time_since_modified < 30:
                logger.debug(f"File {model_path} was modified recently, may still be writing")
                return False
            
            # Check if file size is reasonable (not empty, not too small)
            if stat.st_size < 1024:  # Less than 1KB is suspicious for a model file
                logger.debug(f"File {model_path} is too small ({stat.st_size} bytes)")
                return False
            
            # Try to open file to check if it's locked
            try:
                with open(model_path, 'rb') as f:
                    # Try to read a small amount to verify file is accessible
                    f.read(1024)
                    
                    # On Windows, try to check if file is locked by attempting to open it for writing
                    if os.name == 'nt':  # Windows
                        try:
                            with open(model_path, 'r+b') as fw:
                                pass
                        except IOError:
                            logger.debug(f"File {model_path} is locked for writing")
                            return False
                    else:  # Unix-like systems
                        try:
                            # Try to get an exclusive lock to check if file is being written
                            fd = os.open(str(model_path), os.O_RDWR)
                            try:
                                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                # We got the lock, so file is not being written
                                fcntl.flock(fd, fcntl.LOCK_UN)
                            except IOError as e:
                                if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
                                    logger.debug(f"File {model_path} is locked by another process")
                                    return False
                                raise
                            finally:
                                os.close(fd)
                        except (AttributeError, ImportError):
                            # fcntl might not be available on all platforms
                            pass
                
                # Check if file size is stable (not growing)
                time.sleep(0.5)  # Wait a bit to see if file size changes
                new_size = model_path.stat().st_size
                if new_size != stat.st_size:
                    logger.debug(f"File {model_path} size changed during check, still being written")
                    return False
                
                return True
            except (IOError, OSError) as e:
                logger.debug(f"File {model_path} appears to be locked or inaccessible: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking file completeness for {model_path}: {e}")
            return False
    
    def validate_model(self, model_path: Path) -> ModelValidationResult:
        """
        Validate a model file to ensure it can be loaded and used in the tournament system.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            ValidationResult object with validation details
        """
        logger.info(f"Validating model: {model_path}")
        
        # Use the ModelValidator to validate the model
        validation_result = self.model_validator.validate_model_file(model_path)
        
        # Log validation results
        if validation_result.is_valid:
            logger.info(f"Model validation successful: {model_path}")
            logger.info(f"Architecture: {validation_result.model_architecture}")
            logger.info(f"Compatibility score: {validation_result.compatibility_score:.2f}")
        else:
            logger.warning(f"Model validation failed: {model_path}")
            logger.warning(f"Error: {validation_result.error_message}")
            
            # Log warnings if any
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Warning: {warning}")
        
        return validation_result
    
    def validate_candidates(self, candidates: List[ModelCandidate]) -> Dict[str, ModelValidationResult]:
        """
        Validate a list of model candidates.
        
        Args:
            candidates: List of ModelCandidate objects to validate
            
        Returns:
            Dictionary mapping model names to validation results
        """
        validation_results = {}
        
        for candidate in candidates:
            try:
                # Validate the model
                result = self.validate_model(candidate.file_path)
                validation_results[candidate.model_name] = result
                
            except Exception as e:
                logger.error(f"Error validating model {candidate.model_name}: {e}")
                # Create a failed validation result
                validation_results[candidate.model_name] = ModelValidationResult(
                    is_valid=False,
                    model_architecture="unknown",
                    error_message=f"Validation error: {str(e)}",
                    can_load=False,
                    inference_test_passed=False,
                    compatibility_score=0.0
                )
        
        # Log summary
        valid_count = sum(1 for result in validation_results.values() if result.is_valid)
        logger.info(f"Validation complete: {valid_count}/{len(candidates)} models passed validation")
        
        return validation_results
    
    def get_discovery_summary(self, candidates: List[ModelCandidate]) -> Dict:
        """
        Generate a summary of the discovery process.
        
        Args:
            candidates: List of discovered model candidates
            
        Returns:
            Dictionary containing discovery statistics
        """
        summary = {
            'total_models': len(candidates),
            'main_directory_models': len([c for c in candidates if c.source_directory == "main"]),
            'deployment_directory_models': len([c for c in candidates if c.source_directory == "deployment"]),
            'models_with_metadata': len([c for c in candidates if c.metadata_path is not None]),
            'models_by_architecture': {},
            'newest_model': None,
            'largest_model': None
        }
        
        if candidates:
            # Find newest and largest models
            summary['newest_model'] = max(candidates, key=lambda c: c.last_modified).model_name
            summary['largest_model'] = max(candidates, key=lambda c: c.file_size).model_name
            
            # Count by architecture (inferred from name)
            for candidate in candidates:
                arch = self._infer_architecture_from_name(candidate.model_name)
                summary['models_by_architecture'][arch] = summary['models_by_architecture'].get(arch, 0) + 1
        
        return summary
    
    def _infer_architecture_from_name(self, model_name: str) -> str:
        """
        Infer model architecture from the model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Inferred architecture type
        """
        name_lower = model_name.lower()
        
        if 'transformer' in name_lower:
            return 'transformer'
        elif 'cnn' in name_lower:
            return 'cnn'
        elif 'lstm' in name_lower:
            return 'lstm'
        elif 'resnet' in name_lower:
            return 'resnet'
        elif 'vae' in name_lower:
            return 'vae'
        elif 'gan' in name_lower:
            return 'gan'
        elif 'ensemble' in name_lower or 'hybrid' in name_lower:
            return 'ensemble'
        else:
            return 'unknown'