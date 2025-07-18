"""
Test script for ModelMetadataExtractor

This script tests the metadata extraction and generation functionality.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model_metadata_extractor import ModelMetadataExtractor, ModelMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_metadata_files():
    """Create test metadata files for testing"""
    test_dir = Path("test_metadata")
    test_dir.mkdir(exist_ok=True)
    
    # Create a complete metadata file
    complete_metadata = {
        "id": "advanced_cnn_mixer",
        "name": "Advanced CNN Audio Mixer",
        "architecture": "CNN",
        "description": "Advanced convolutional neural network for audio mixing",
        "performance_metrics": {
            "mae": 0.0554,
            "training_epochs": 100,
            "validation_loss": 0.0234
        },
        "specializations": ["Audio Processing", "Mixing"],
        "capabilities": {
            "spectral_analysis": 0.9,
            "mixing": 0.8,
            "real_time": 0.7
        },
        "parameters": {
            "input_shape": [1, 128, 128],
            "output_shape": [17],
            "trainable_parameters": 1234567
        },
        "created_at": "2024-01-15T10:30:00",
        "version": "2.1",
        "author": "AI Research Team"
    }
    
    with open(test_dir / "complete_metadata.json", 'w') as f:
        json.dump(complete_metadata, f, indent=2)
    
    # Create a minimal metadata file
    minimal_metadata = {
        "name": "Simple Model",
        "architecture": "LSTM"
    }
    
    with open(test_dir / "minimal_metadata.json", 'w') as f:
        json.dump(minimal_metadata, f, indent=2)
    
    # Create a malformed metadata file
    with open(test_dir / "malformed_metadata.json", 'w') as f:
        f.write('{"name": "Broken", "architecture":}')  # Invalid JSON
    
    # Create test model files
    test_models_dir = Path("test_models")
    test_models_dir.mkdir(exist_ok=True)
    
    # Create dummy model files with different naming patterns
    model_files = [
        "baseline_cnn_best.pth",
        "advanced_transformer_v2.pth",
        "lstm_mixer_final.pth",
        "resnet_audio_processor.pth",
        "vae_creative_mixer.pth",
        "gan_style_transfer.pth",
        "ensemble_hybrid_model.pth",
        "ast_regressor_best.pth",
        "unknown_model.pth"
    ]
    
    for model_file in model_files:
        model_path = test_models_dir / model_file
        model_path.touch()  # Create empty file
    
    return test_dir, test_models_dir

def test_json_extraction():
    """Test JSON metadata extraction"""
    logger.info("=== Testing JSON Metadata Extraction ===")
    
    extractor = ModelMetadataExtractor()
    test_dir, _ = create_test_metadata_files()
    
    # Test complete metadata
    complete_metadata = extractor.extract_from_json(test_dir / "complete_metadata.json")
    if complete_metadata:
        logger.info("✅ Complete metadata extraction successful")
        logger.info(f"   Name: {complete_metadata.get('name')}")
        logger.info(f"   Architecture: {complete_metadata.get('architecture')}")
        logger.info(f"   Capabilities: {len(complete_metadata.get('capabilities', {}))}")
    else:
        logger.error("❌ Complete metadata extraction failed")
    
    # Test minimal metadata
    minimal_metadata = extractor.extract_from_json(test_dir / "minimal_metadata.json")
    if minimal_metadata:
        logger.info("✅ Minimal metadata extraction successful")
        logger.info(f"   Name: {minimal_metadata.get('name')}")
        logger.info(f"   Architecture: {minimal_metadata.get('architecture')}")
    else:
        logger.error("❌ Minimal metadata extraction failed")
    
    # Test malformed metadata
    malformed_metadata = extractor.extract_from_json(test_dir / "malformed_metadata.json")
    if not malformed_metadata:
        logger.info("✅ Malformed metadata handled gracefully (returned empty dict)")
    else:
        logger.warning("⚠️ Malformed metadata should return empty dict")
    
    # Test nonexistent file
    nonexistent_metadata = extractor.extract_from_json(test_dir / "nonexistent.json")
    if not nonexistent_metadata:
        logger.info("✅ Nonexistent file handled gracefully (returned empty dict)")
    else:
        logger.warning("⚠️ Nonexistent file should return empty dict")

def test_default_metadata_generation():
    """Test default metadata generation"""
    logger.info("\n=== Testing Default Metadata Generation ===")
    
    extractor = ModelMetadataExtractor()
    _, test_models_dir = create_test_metadata_files()
    
    # Test different model types
    test_cases = [
        ("baseline_cnn_best.pth", "CNN"),
        ("advanced_transformer_v2.pth", "Transformer"),
        ("lstm_mixer_final.pth", "LSTM"),
        ("resnet_audio_processor.pth", "ResNet"),
        ("vae_creative_mixer.pth", "VAE"),
        ("gan_style_transfer.pth", "GAN"),
        ("ensemble_hybrid_model.pth", "Ensemble"),
        ("ast_regressor_best.pth", "AST"),
        ("unknown_model.pth", "Neural Network")
    ]
    
    for model_file, expected_arch in test_cases:
        model_path = test_models_dir / model_file
        metadata = extractor.generate_default_metadata(model_path)
        
        if metadata.architecture == expected_arch:
            logger.info(f"✅ {model_file}: Architecture correctly identified as {expected_arch}")
        else:
            logger.warning(f"⚠️ {model_file}: Expected {expected_arch}, got {metadata.architecture}")
        
        # Check capabilities
        if metadata.capabilities:
            logger.info(f"   Capabilities: {list(metadata.capabilities.keys())}")
        
        # Check specializations
        if metadata.specializations:
            logger.info(f"   Specializations: {metadata.specializations}")

def test_capability_inference():
    """Test capability inference from model names"""
    logger.info("\n=== Testing Capability Inference ===")
    
    extractor = ModelMetadataExtractor()
    
    test_cases = [
        ("spectral_cnn_mixer", ["spectral_analysis"]),
        ("temporal_lstm_processor", ["temporal_modeling"]),
        ("feature_extraction_model", ["feature_extraction"]),
        ("real_time_audio_enhancer", ["audio_enhancement", "real_time"]),
        ("noise_reduction_gan", ["noise_reduction"]),
        ("creative_style_transfer", ["creative_effects"]),
        ("mixing_master_model", ["mixing", "mastering"])
    ]
    
    for model_name, expected_capabilities in test_cases:
        capabilities = extractor.infer_capabilities_from_name(model_name)
        
        found_expected = any(cap in capabilities for cap in expected_capabilities)
        if found_expected:
            logger.info(f"✅ {model_name}: Found expected capabilities")
            logger.info(f"   Inferred: {list(capabilities.keys())}")
        else:
            logger.warning(f"⚠️ {model_name}: Expected capabilities not found")
            logger.warning(f"   Expected: {expected_capabilities}")
            logger.warning(f"   Got: {list(capabilities.keys())}")

def test_generation_determination():
    """Test model generation determination"""
    logger.info("\n=== Testing Generation Determination ===")
    
    extractor = ModelMetadataExtractor()
    
    # Test with existing models
    existing_models = [
        "baseline_cnn.pth",
        "baseline_cnn_v2.pth",
        "baseline_cnn_v3.pth",
        "transformer_model.pth",
        "transformer_model_v2.pth"
    ]
    
    test_cases = [
        ("baseline_cnn_v4", 4),  # Should be generation 4
        ("transformer_model_v3", 3),  # Should be generation 3
        ("new_model", 1),  # Should be generation 1 (no existing models)
        ("baseline_cnn_improved", 4)  # Should be generation 4 (same base name)
    ]
    
    for model_name, expected_gen in test_cases:
        generation = extractor.determine_model_generation(model_name, existing_models)
        
        if generation == expected_gen:
            logger.info(f"✅ {model_name}: Correctly determined as generation {generation}")
        else:
            logger.warning(f"⚠️ {model_name}: Expected generation {expected_gen}, got {generation}")

def test_metadata_saving():
    """Test metadata saving functionality"""
    logger.info("\n=== Testing Metadata Saving ===")
    
    extractor = ModelMetadataExtractor()
    
    # Create test metadata
    test_metadata = ModelMetadata(
        id="test_model",
        name="Test Model",
        architecture="CNN",
        file_path="test_model.pth",
        size_mb=10.5,
        created_at="2024-01-15T10:30:00",
        description="Test model for metadata saving",
        performance_metrics={"mae": 0.05},
        specializations=["Audio Processing"],
        capabilities={"mixing": 0.8},
        parameters={"input_shape": [1, 128, 128]}
    )
    
    # Save metadata
    output_path = Path("test_output") / "test_metadata.json"
    success = extractor.save_metadata(test_metadata, output_path)
    
    if success and output_path.exists():
        logger.info("✅ Metadata saving successful")
        
        # Verify saved content
        try:
            with open(output_path, 'r') as f:
                saved_data = json.load(f)
            
            if saved_data.get("name") == "Test Model":
                logger.info("✅ Saved metadata content verified")
            else:
                logger.warning("⚠️ Saved metadata content incorrect")
        except Exception as e:
            logger.error(f"❌ Error verifying saved metadata: {e}")
    else:
        logger.error("❌ Metadata saving failed")

if __name__ == "__main__":
    logger.info("Starting ModelMetadataExtractor tests")
    
    # Run tests
    test_json_extraction()
    test_default_metadata_generation()
    test_capability_inference()
    test_generation_determination()
    test_metadata_saving()
    
    logger.info("\nTests completed")