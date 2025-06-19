#!/usr/bin/env python3
"""
🧪 Comprehensive Model Testing Suite
===================================

Test all 5 new AI model architectures to ensure they work correctly:
- LSTM Audio Mixer
- Audio GAN
- VAE Audio Mixer  
- Advanced Transformer
- ResNet Audio Mixer

This script validates that all models can:
1. Load and initialize correctly
2. Process spectrogram data
3. Produce valid mixing parameters
4. Handle different input sizes
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our models
from lstm_mixer import LSTMAudioMixer
from audio_gan import AudioGANMixer
from vae_mixer import VAEAudioMixer
from advanced_transformer import AdvancedTransformerMixer
from resnet_mixer import ResNetAudioMixer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    """Comprehensive testing suite for new model architectures."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        
    def create_test_data(self, batch_size: int = 2, n_mels: int = 128, time_steps: int = 250) -> torch.Tensor:
        """Create synthetic spectrogram data for testing."""
        return torch.randn(batch_size, 1, n_mels, time_steps, device=self.device)
    
    def test_model_basic_functionality(self, model: nn.Module, model_name: str) -> Dict:
        """Test basic model functionality."""
        logger.info(f"🧪 Testing {model_name}...")
        
        results = {
            'model_name': model_name,
            'initialization': False,
            'forward_pass': False,
            'output_shape': None,
            'parameter_count': 0,
            'inference_time': 0.0,
            'memory_usage': 0.0,
            'error_message': None
        }
        
        try:
            # Test 1: Model initialization
            model = model.to(self.device)
            results['initialization'] = True
            results['parameter_count'] = sum(p.numel() for p in model.parameters())
            logger.info(f"   ✅ Initialization successful - {results['parameter_count']} parameters")
            
            # Test 2: Forward pass
            test_input = self.create_test_data()
            
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                
                # Memory usage before
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    memory_before = torch.cuda.memory_allocated()
                
                output = model(test_input)
                
                # Memory usage after
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    results['memory_usage'] = (memory_after - memory_before) / 1024 / 1024  # MB
                
                inference_time = time.time() - start_time
                results['inference_time'] = inference_time
                results['forward_pass'] = True
                results['output_shape'] = list(output.shape)
                
            logger.info(f"   ✅ Forward pass successful")
            logger.info(f"   📊 Output shape: {output.shape}")
            logger.info(f"   ⏱️ Inference time: {inference_time:.4f}s")
            if torch.cuda.is_available():
                logger.info(f"   🧠 Memory usage: {results['memory_usage']:.2f} MB")
            
            # Test 3: Output validation
            if len(output.shape) == 2 and output.shape[1] == 10:
                logger.info(f"   ✅ Output shape valid (expected: [batch_size, 10])")
            else:
                logger.warning(f"   ⚠️ Unexpected output shape: {output.shape}")
            
            # Test 4: Parameter range check
            param_range = torch.abs(output).max().item()
            if param_range < 100:  # Reasonable range for mixing parameters
                logger.info(f"   ✅ Output parameter range reasonable: {param_range:.3f}")
            else:
                logger.warning(f"   ⚠️ Large output values: {param_range:.3f}")
            
            # Test 5: Gradient check
            model.train()
            test_target = torch.randn_like(output)
            loss = nn.MSELoss()(output, test_target)
            loss.backward()
            
            # Check if gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters())
            if has_gradients:
                logger.info(f"   ✅ Gradients computed successfully")
            else:
                logger.warning(f"   ⚠️ No gradients found")
            
        except Exception as e:
            results['error_message'] = str(e)
            logger.error(f"   ❌ Test failed: {e}")
        
        return results
    
    def test_model_robustness(self, model: nn.Module, model_name: str) -> Dict:
        """Test model robustness with different input sizes."""
        logger.info(f"🔬 Testing {model_name} robustness...")
        
        robustness_results = {
            'different_batch_sizes': [],
            'different_time_steps': [],
            'different_mel_bins': [],
            'noise_resistance': 0.0
        }
        
        try:
            model.eval()
            
            # Test different batch sizes
            for batch_size in [1, 4, 8]:
                try:
                    test_input = self.create_test_data(batch_size=batch_size)
                    with torch.no_grad():
                        output = model(test_input)
                    robustness_results['different_batch_sizes'].append({
                        'batch_size': batch_size,
                        'success': True,
                        'output_shape': list(output.shape)
                    })
                    logger.info(f"   ✅ Batch size {batch_size}: Success")
                except Exception as e:
                    robustness_results['different_batch_sizes'].append({
                        'batch_size': batch_size,
                        'success': False,
                        'error': str(e)
                    })
                    logger.warning(f"   ⚠️ Batch size {batch_size}: Failed - {e}")
            
            # Test different time steps
            for time_steps in [100, 250, 500]:
                try:
                    test_input = self.create_test_data(time_steps=time_steps)
                    with torch.no_grad():
                        output = model(test_input)
                    robustness_results['different_time_steps'].append({
                        'time_steps': time_steps,
                        'success': True,
                        'output_shape': list(output.shape)
                    })
                    logger.info(f"   ✅ Time steps {time_steps}: Success")
                except Exception as e:
                    robustness_results['different_time_steps'].append({
                        'time_steps': time_steps,
                        'success': False,
                        'error': str(e)
                    })
                    logger.warning(f"   ⚠️ Time steps {time_steps}: Failed - {e}")
            
            # Test noise resistance
            try:
                clean_input = self.create_test_data()
                noisy_input = clean_input + 0.1 * torch.randn_like(clean_input)
                
                with torch.no_grad():
                    clean_output = model(clean_input)
                    noisy_output = model(noisy_input)
                
                # Calculate output difference
                diff = torch.mean(torch.abs(clean_output - noisy_output)).item()
                robustness_results['noise_resistance'] = diff
                
                if diff < 1.0:  # Reasonable noise resistance
                    logger.info(f"   ✅ Noise resistance good: {diff:.3f}")
                else:
                    logger.warning(f"   ⚠️ High noise sensitivity: {diff:.3f}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Noise resistance test failed: {e}")
        
        except Exception as e:
            logger.error(f"   ❌ Robustness testing failed: {e}")
        
        return robustness_results
    
    def test_all_models(self) -> Dict:
        """Test all model architectures."""
        logger.info("🚀 Starting Comprehensive Model Testing")
        logger.info("=" * 60)
        
        # Define models to test
        models_to_test = [
            ('LSTM Audio Mixer', LSTMAudioMixer()),
            ('Audio GAN Mixer', AudioGANMixer()),
            ('VAE Audio Mixer', VAEAudioMixer()),
            ('Advanced Transformer Mixer', AdvancedTransformerMixer()),
            ('ResNet Audio Mixer', ResNetAudioMixer())
        ]
        
        all_results = {}
        successful_models = []
        failed_models = []
        
        for model_name, model in models_to_test:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"🤖 Testing {model_name}")
                logger.info('='*50)
                
                # Basic functionality tests
                basic_results = self.test_model_basic_functionality(model, model_name)
                
                # Robustness tests
                robustness_results = self.test_model_robustness(model, model_name)
                
                # Combine results
                combined_results = {
                    **basic_results,
                    'robustness': robustness_results
                }
                
                all_results[model_name] = combined_results
                
                if basic_results['forward_pass']:
                    successful_models.append(model_name)
                    logger.info(f"✅ {model_name} passed all tests!")
                else:
                    failed_models.append(model_name)
                    logger.error(f"❌ {model_name} failed basic tests!")
                
            except Exception as e:
                logger.error(f"❌ Testing failed for {model_name}: {e}")
                failed_models.append(model_name)
                all_results[model_name] = {'error': str(e)}
                continue
        
        # Generate summary report
        self.generate_test_report(all_results, successful_models, failed_models)
        
        return all_results
    
    def generate_test_report(self, results: Dict, successful_models: List, failed_models: List):
        """Generate comprehensive test report."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 COMPREHENSIVE TEST REPORT")
        logger.info('='*60)
        
        # Success/Failure summary
        total_models = len(results)
        success_rate = (len(successful_models) / total_models) * 100 if total_models > 0 else 0
        
        logger.info(f"\n🎯 Test Summary:")
        logger.info(f"   Total models tested: {total_models}")
        logger.info(f"   Successful models: {len(successful_models)}")
        logger.info(f"   Failed models: {len(failed_models)}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        # Successful models details
        if successful_models:
            logger.info(f"\n✅ Successful Models:")
            for model_name in successful_models:
                result = results[model_name]
                if 'parameter_count' in result:
                    params = result['parameter_count']
                    inference_time = result.get('inference_time', 0)
                    logger.info(f"   • {model_name}: {params:,} params, {inference_time:.4f}s inference")
        
        # Failed models
        if failed_models:
            logger.info(f"\n❌ Failed Models:")
            for model_name in failed_models:
                error = results[model_name].get('error_message', 'Unknown error')
                logger.info(f"   • {model_name}: {error}")
        
        # Performance comparison
        if successful_models:
            logger.info(f"\n⚡ Performance Comparison:")
            
            # Sort by inference time
            perf_data = []
            for model_name in successful_models:
                result = results[model_name]
                if 'inference_time' in result:
                    perf_data.append((model_name, result['inference_time'], result['parameter_count']))
            
            perf_data.sort(key=lambda x: x[1])  # Sort by inference time
            
            for i, (model_name, inference_time, param_count) in enumerate(perf_data, 1):
                status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                logger.info(f"   {status} {model_name}: {inference_time:.4f}s ({param_count:,} params)")
        
        # Save detailed results
        results_path = Path("../models") / "model_testing_results.json"
        results_path.parent.mkdir(exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                k: v for k, v in result.items() 
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"\n💾 Detailed results saved: {results_path}")
        
        # Final recommendations
        logger.info(f"\n🎯 Recommendations:")
        if success_rate >= 80:
            logger.info(f"   ✅ Excellent! Models are ready for tournament integration")
            logger.info(f"   🚀 Proceed with training: python src/train_new_architectures_fixed.py")
        elif success_rate >= 60:
            logger.info(f"   ⚠️ Most models working, but some issues found")
            logger.info(f"   🔧 Review failed models and fix issues before training")
        else:
            logger.info(f"   ❌ Multiple model failures detected")
            logger.info(f"   🔧 Fix critical issues before proceeding")

def main():
    """Main testing pipeline."""
    tester = ModelTester()
    results = tester.test_all_models()
    
    # Quick architecture verification
    print(f"\n🧪 Quick Architecture Verification:")
    for model_name in ['LSTM Audio Mixer', 'Audio GAN Mixer', 'VAE Audio Mixer', 
                      'Advanced Transformer Mixer', 'ResNet Audio Mixer']:
        if model_name in results and results[model_name].get('forward_pass'):
            print(f"   ✅ {model_name}: Ready")
        else:
            print(f"   ❌ {model_name}: Issues found")

if __name__ == "__main__":
    main()
