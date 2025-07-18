"""
Unit tests for the ModelDiscoveryService.

These tests verify the functionality of the model discovery process,
including file scanning, conflict resolution, and priority scoring.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

from model_discovery_service import ModelDiscoveryService, ModelCandidate


class TestModelDiscoveryService(unittest.TestCase):
    """Test cases for ModelDiscoveryService."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.main_models_dir = Path(self.test_dir) / "models"
        self.deployment_dir = Path(self.test_dir) / "models" / "deployment"
        
        # Create directory structure
        self.main_models_dir.mkdir(parents=True, exist_ok=True)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize service with test directories
        self.discovery_service = ModelDiscoveryService(
            main_models_dir=str(self.main_models_dir),
            deployment_dir=str(self.deployment_dir)
        )
    
    def tearDown(self):
        """Clean up temporary directories after tests."""
        shutil.rmtree(self.test_dir)
    
    def _create_test_model_file(self, directory, filename, content="test model content", 
                               metadata=None, modified_time=None):
        """Helper to create a test model file with optional metadata."""
        # Create model file
        model_path = directory / filename
        with open(model_path, 'w') as f:
            f.write(content)
        
        # Create metadata file if provided
        if metadata:
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        # Set modified time if provided
        if modified_time:
            os.utime(model_path, (modified_time.timestamp(), modified_time.timestamp()))
        
        return model_path
    
    def test_scan_empty_directories(self):
        """Test scanning empty directories."""
        candidates = self.discovery_service.scan_for_models()
        self.assertEqual(len(candidates), 0, "Empty directories should return no candidates")
    
    def test_scan_main_directory(self):
        """Test scanning main directory with model files."""
        # Create test model files
        self._create_test_model_file(self.main_models_dir, "model1.pth")
        self._create_test_model_file(self.main_models_dir, "model2.pth")
        
        candidates = self.discovery_service.scan_for_models()
        self.assertEqual(len(candidates), 2, "Should find 2 models in main directory")
        self.assertEqual(set(c.model_name for c in candidates), {"model1", "model2"})
        self.assertTrue(all(c.source_directory == "main" for c in candidates))
    
    def test_scan_with_metadata(self):
        """Test scanning models with associated metadata."""
        # Create model with metadata
        metadata = {"architecture": "transformer", "accuracy": 0.95}
        self._create_test_model_file(self.main_models_dir, "model_with_metadata.pth", 
                                    metadata=metadata)
        
        candidates = self.discovery_service.scan_for_models()
        self.assertEqual(len(candidates), 1)
        self.assertIsNotNone(candidates[0].metadata_path)
    
    def test_resolve_conflicts_same_name(self):
        """Test resolving conflicts between models with the same name."""
        # Create model in main directory (newer)
        main_time = datetime.now() - timedelta(days=1)
        self._create_test_model_file(self.main_models_dir, "same_name.pth", 
                                    modified_time=main_time)
        
        # Create model in deployment directory (older)
        deploy_time = datetime.now() - timedelta(days=7)
        self._create_test_model_file(self.deployment_dir, "same_name.pth", 
                                    modified_time=deploy_time)
        
        candidates = self.discovery_service.scan_for_models()
        self.assertEqual(len(candidates), 1, "Should resolve to 1 model")
        self.assertEqual(candidates[0].source_directory, "main", 
                        "Should prioritize main directory model")
    
    def test_resolve_conflicts_by_timestamp(self):
        """Test resolving conflicts by timestamp when in same directory."""
        # Create older model
        older_time = datetime.now() - timedelta(days=7)
        self._create_test_model_file(self.main_models_dir, "version.pth", 
                                    modified_time=older_time)
        
        # Create newer model with different name but will be renamed
        newer_time = datetime.now() - timedelta(days=1)
        self._create_test_model_file(self.main_models_dir, "version_v2.pth", 
                                    modified_time=newer_time)
        
        # Manually create candidates to test conflict resolution
        older_candidate = ModelCandidate(
            file_path=self.main_models_dir / "version.pth",
            model_name="version",
            file_size=100,
            last_modified=older_time,
            source_directory="main",
            metadata_path=None,
            priority_score=100
        )
        
        newer_candidate = ModelCandidate(
            file_path=self.main_models_dir / "version_v2.pth",
            model_name="version_v2",  # Different name but will be treated as same base model
            file_size=100,
            last_modified=newer_time,
            source_directory="main",
            metadata_path=None,
            priority_score=100
        )
        
        resolved = self.discovery_service.resolve_model_conflicts([older_candidate, newer_candidate])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].model_name, "version_v2", 
                        "Should select newer version model")
    
    def test_extract_base_model_name(self):
        """Test extracting base model name from versioned names."""
        test_cases = [
            ("transformer_model", "transformer_model"),  # No change
            ("transformer_model_v2", "transformer_model"),  # Remove version
            ("cnn_best", "cnn"),  # Remove best suffix
            ("lstm_final", "lstm"),  # Remove final suffix
            ("resnet_v3_latest", "resnet_v3"),  # Remove latest suffix
            ("vae_mixer_best_v2", "vae_mixer"),  # Remove both
        ]
        
        for input_name, expected_output in test_cases:
            result = self.discovery_service._extract_base_model_name(input_name)
            self.assertEqual(result, expected_output, 
                           f"Base name extraction failed for {input_name}")
    
    def test_extract_version_number(self):
        """Test extracting version numbers from filenames."""
        test_cases = [
            ("model_v2.pth", 2),
            ("model-v3.pth", 3),
            ("model_version4.pth", 4),
            ("modelv5.pth", 0),  # No separator
            ("model.pth", 0),  # No version
        ]
        
        for filename, expected_version in test_cases:
            result = self.discovery_service._extract_version_number(filename)
            self.assertEqual(result, expected_version, 
                           f"Version extraction failed for {filename}")
    
    def test_resolve_conflicts_with_best_models(self):
        """Test resolving conflicts with 'best' models."""
        # Create regular model
        regular_model = ModelCandidate(
            file_path=Path("models/regular_model.pth"),
            model_name="regular_model",
            file_size=100,
            last_modified=datetime.now() - timedelta(days=1),
            source_directory="main",
            metadata_path=None,
            priority_score=100
        )
        
        # Create best model (older but marked as best)
        best_model = ModelCandidate(
            file_path=Path("models/regular_model_best.pth"),
            model_name="regular_model_best",
            file_size=100,
            last_modified=datetime.now() - timedelta(days=2),  # Older
            source_directory="main",
            metadata_path=None,
            priority_score=120  # Higher priority due to "best" in name
        )
        
        resolved = self.discovery_service.resolve_model_conflicts([regular_model, best_model])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].file_path.name, "regular_model_best.pth", 
                        "Should select best model even if older")
    
    def test_priority_scoring(self):
        """Test priority scoring for different model types."""
        # Create various model files
        transformer_path = self._create_test_model_file(self.main_models_dir, "transformer_model.pth")
        cnn_path = self._create_test_model_file(self.main_models_dir, "cnn_model.pth")
        best_path = self._create_test_model_file(self.main_models_dir, "best_model.pth")
        
        # Get priority scores
        transformer_score = self.discovery_service.get_model_priority(transformer_path, "main")
        cnn_score = self.discovery_service.get_model_priority(cnn_path, "main")
        best_score = self.discovery_service.get_model_priority(best_path, "main")
        
        # Check that transformer models get higher priority than basic CNN
        self.assertGreater(transformer_score, cnn_score)
        
        # Check that "best" models get bonus points
        self.assertGreater(best_score, cnn_score)
    
    def test_file_completeness_check(self):
        """Test checking if model files are complete and not being written."""
        # Create a normal file
        model_path = self._create_test_model_file(self.main_models_dir, "complete_model.pth", 
                                                content="x" * 2048)  # 2KB file
        
        # Create a very small file (should be considered incomplete)
        small_path = self._create_test_model_file(self.main_models_dir, "small_model.pth", 
                                                content="x")  # 1 byte file
        
        # Create a very recent file (should be considered incomplete)
        recent_path = self._create_test_model_file(self.main_models_dir, "recent_model.pth", 
                                                 content="x" * 2048)
        # Manually set the modified time to now
        os.utime(recent_path, (time.time(), time.time()))
        
        # Check completeness
        self.assertTrue(self.discovery_service.is_model_file_complete(model_path))
        self.assertFalse(self.discovery_service.is_model_file_complete(small_path))
        self.assertFalse(self.discovery_service.is_model_file_complete(recent_path))
    
    def test_file_size_stability(self):
        """Test file size stability check."""
        # This test simulates a file that's still being written by changing its size
        
        # Create a test file
        test_path = self.main_models_dir / "growing_file.pth"
        with open(test_path, 'w') as f:
            f.write("x" * 2048)  # Initial content
        
        # Mock the stat and sleep methods to simulate a growing file
        original_stat = Path.stat
        original_sleep = time.sleep
        
        try:
            # First call to stat returns initial size
            mock_stat_result = test_path.stat()
            
            # Mock stat to return different sizes on subsequent calls
            def mock_stat(self):
                if self == test_path:
                    # Return a stat_result with a larger size on second call
                    if hasattr(mock_stat, 'called'):
                        # Create a new stat_result with modified size
                        return type('MockStatResult', (), {
                            'st_size': mock_stat_result.st_size + 1024,  # Larger size
                            'st_mtime': mock_stat_result.st_mtime - 60  # Not too recent
                        })
                    mock_stat.called = True
                    return mock_stat_result
                return original_stat(self)
            
            # Mock sleep to do nothing (speed up test)
            def mock_sleep(seconds):
                pass
            
            # Apply mocks
            Path.stat = mock_stat
            time.sleep = mock_sleep
            
            # Test the method
            self.assertFalse(self.discovery_service.is_model_file_complete(test_path))
            
        finally:
            # Restore original methods
            Path.stat = original_stat
            time.sleep = original_sleep
    
    def test_scan_with_incomplete_files(self):
        """Test scanning directory with incomplete files."""
        # Create complete model file
        self._create_test_model_file(self.main_models_dir, "complete_model.pth", 
                                    content="x" * 2048)
        
        # Create incomplete model file (too small)
        self._create_test_model_file(self.main_models_dir, "incomplete_model.pth", 
                                    content="x")
        
        # Mock is_model_file_complete to control behavior
        original_is_complete = self.discovery_service.is_model_file_complete
        
        try:
            def mock_is_complete(path):
                # Only the complete model should pass
                return "complete_model" in str(path)
            
            self.discovery_service.is_model_file_complete = mock_is_complete
            
            # Scan for models
            candidates = self.discovery_service.scan_for_models()
            
            # Should only find the complete model
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].model_name, "complete_model")
            
        finally:
            # Restore original method
            self.discovery_service.is_model_file_complete = original_is_complete
    
    def test_discovery_summary(self):
        """Test generating discovery summary."""
        # Create various model files
        self._create_test_model_file(self.main_models_dir, "transformer_model.pth")
        self._create_test_model_file(self.main_models_dir, "cnn_model.pth")
        self._create_test_model_file(self.deployment_dir, "lstm_model.pth")
        
        # Scan for models
        candidates = self.discovery_service.scan_for_models()
        
        # Get summary
        summary = self.discovery_service.get_discovery_summary(candidates)
        
        # Check summary contents
        self.assertEqual(summary['total_models'], 3)
        self.assertEqual(summary['main_directory_models'], 2)
        self.assertEqual(summary['deployment_directory_models'], 1)
        self.assertIn('transformer', summary['models_by_architecture'])
        self.assertIn('cnn', summary['models_by_architecture'])
        self.assertIn('lstm', summary['models_by_architecture'])


if __name__ == '__main__':
    unittest.main()