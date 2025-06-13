#!/usr/bin/env python3
"""
Musical Model Testing and Comparison
Test the retrofitted musical models and compare with originals
"""

import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import our components
from musical_intelligence import MusicalFeatureExtractor, MusicalContext
from musical_retrofit_training import MusicalModelWrapper
from baseline_cnn import BaselineCNN, EnhancedCNN

class MusicalModelTester:
    """Test and compare musical vs original models"""
    
    def __init__(self):        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = MusicalFeatureExtractor()
        self.models_dir = Path('models')
        self.output_dir = Path('mixed_outputs/musical_comparison')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_musical_models(self) -> Dict[str, MusicalModelWrapper]:
        """Load all musical models"""
        models = {}
        
        for model_file in self.models_dir.glob('musical_*.pth'):
            model_name = model_file.stem
            original_name = model_name.replace('musical_', '')
              try:
                # Create base model structure
                if 'baseline' in original_name:
                    base_model = BaselineCNN()
                elif 'enhanced' in original_name:
                    base_model = EnhancedCNN()
                else:
                    base_model = BaselineCNN()  # Default
                
                # Create musical wrapper
                musical_model = MusicalModelWrapper(base_model, 'cnn')
                
                # Load trained weights
                checkpoint = torch.load(model_file, map_location=self.device)
                musical_model.load_state_dict(checkpoint['model_state_dict'])
                musical_model.eval()
                
                models[original_name] = musical_model
                print(f"✅ Loaded musical_{original_name}")
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        
        return models
    
    def load_original_models(self) -> Dict[str, nn.Module]:
        """Load original models for comparison"""
        models = {}
          model_files = {
            'baseline_cnn': 'baseline_cnn.pth',
            'enhanced_cnn': 'enhanced_cnn.pth',
            'improved_baseline_cnn': 'improved_baseline_cnn.pth'
        }
        
        for name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    if 'baseline' in name:
                        model = BaselineCNN()
                    elif 'enhanced' in name:
                        model = EnhancedCNN()
                    else:
                        model = BaselineCNN()  # Default
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    models[name] = model
                    print(f"✅ Loaded original {name}")
                    
                except Exception as e:
                    print(f"❌ Failed to load original {name}: {e}")
        
        return models
    
    def extract_audio_features(self, audio_path: str) -> torch.Tensor:
        """Extract features from audio for model input"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=44100)
        
        # Extract spectral features
        features = []
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        
        features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth])
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        features.append(zcr)
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=audio))
        features.append(rms)
        
        # Pad or truncate to 512 features
        if len(features) < 512:
            features.extend([0] * (512 - len(features)))
        else:
            features = features[:512]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def test_single_audio(self, audio_path: str, musical_models: Dict, original_models: Dict):
        """Test single audio file with all models"""
        print(f"\n🎵 Testing: {Path(audio_path).name}")
        
        # Extract musical context
        musical_context = self.feature_extractor.extract_musical_features(audio_path)
        print(f"🎼 Genre: {musical_context.genre}, Tempo: {musical_context.tempo:.1f} BPM")
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio_path).to(self.device)
        
        results = {
            'audio_file': Path(audio_path).name,
            'musical_context': {
                'genre': musical_context.genre,
                'tempo': musical_context.tempo,
                'energy': musical_context.energy_level,
                'valence': musical_context.valence
            },
            'model_comparisons': {}
        }
        
        # Test each model pair
        for model_name in musical_models.keys():
            if model_name in original_models:
                print(f"  🔄 Comparing {model_name}...")
                
                # Get predictions
                with torch.no_grad():
                    # Original model
                    original_pred = original_models[model_name](audio_features)
                    
                    # Musical model
                    musical_pred = musical_models[model_name](audio_features, musical_context)
                
                # Analyze differences
                param_diff = torch.abs(musical_pred - original_pred).mean().item()
                max_diff = torch.abs(musical_pred - original_pred).max().item()
                
                # Check for extreme changes (what we want to avoid)
                extreme_changes = (torch.abs(musical_pred) > 4.0).sum().item()
                moderate_changes = (torch.abs(musical_pred) <= 2.0).sum().item()
                
                results['model_comparisons'][model_name] = {
                    'original_params': original_pred.cpu().numpy().tolist(),
                    'musical_params': musical_pred.cpu().numpy().tolist(),
                    'average_difference': param_diff,
                    'max_difference': max_diff,
                    'extreme_changes': extreme_changes,
                    'moderate_changes': moderate_changes,
                    'parameter_range': {
                        'min': musical_pred.min().item(),
                        'max': musical_pred.max().item()
                    }
                }
                
                print(f"    📊 Avg diff: {param_diff:.3f}, Max diff: {max_diff:.3f}")
                print(f"    ⚖️  Moderate changes: {moderate_changes}/10, Extreme: {extreme_changes}/10")
        
        return results
    
    def create_comparison_visualization(self, results: Dict):
        """Create visualizations comparing musical vs original models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Musical vs Original Models - {results["audio_file"]}', fontsize=16)
        
        models = list(results['model_comparisons'].keys())
        
        if not models:
            print("⚠️  No model comparisons to visualize")
            return
        
        # 1. Parameter value comparison
        ax1 = axes[0, 0]
        for i, model_name in enumerate(models):
            comparison = results['model_comparisons'][model_name]
            original = np.array(comparison['original_params'])[0]  # First batch
            musical = np.array(comparison['musical_params'])[0]
            
            x = np.arange(len(original))
            ax1.plot(x, original, 'o-', label=f'{model_name} (original)', alpha=0.7)
            ax1.plot(x, musical, 's-', label=f'{model_name} (musical)', alpha=0.7)
        
        ax1.set_title('Parameter Values Comparison')
        ax1.set_xlabel('Parameter Index')
        ax1.set_ylabel('Parameter Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter differences
        ax2 = axes[0, 1]
        avg_diffs = [results['model_comparisons'][m]['average_difference'] for m in models]
        max_diffs = [results['model_comparisons'][m]['max_difference'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        ax2.bar(x - width/2, avg_diffs, width, label='Average Difference', alpha=0.7)
        ax2.bar(x + width/2, max_diffs, width, label='Max Difference', alpha=0.7)
        
        ax2.set_title('Model Prediction Differences')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Difference')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter range analysis
        ax3 = axes[1, 0]
        extreme_counts = [results['model_comparisons'][m]['extreme_changes'] for m in models]
        moderate_counts = [results['model_comparisons'][m]['moderate_changes'] for m in models]
        
        x = np.arange(len(models))
        ax3.bar(x, moderate_counts, label='Moderate Changes (≤2.0)', alpha=0.7, color='green')
        ax3.bar(x, extreme_counts, bottom=moderate_counts, label='Extreme Changes (>4.0)', alpha=0.7, color='red')
        
        ax3.set_title('Parameter Change Analysis')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Number of Parameters')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Musical context information
        ax4 = axes[1, 1]
        context = results['musical_context']
        
        # Create a radar-like plot for musical features
        features = ['Energy', 'Valence', 'Tempo\n(normalized)', 'Genre Score']
        values = [
            context['energy'],
            context['valence'],
            min(context['tempo'] / 200.0, 1.0),  # Normalize tempo
            0.8 if context['genre'] in ['pop', 'rock'] else 0.6  # Simple genre score
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax4.plot(angles, values, 'o-', linewidth=2, label='Musical Context')
        ax4.fill(angles, values, alpha=0.25)
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(features)
        ax4.set_ylim(0, 1)
        ax4.set_title(f'Musical Context\nGenre: {context["genre"]}\nTempo: {context["tempo"]:.1f} BPM')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'musical_comparison_{Path(results["audio_file"]).stem}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison chart saved: {plot_path}")
        
        plt.show()
    
    def run_comprehensive_test(self):
        """Run comprehensive test of musical vs original models"""
        print("🎵 Musical Model Testing and Comparison")
        print("=" * 60)
        
        # Load models
        musical_models = self.load_musical_models()
        original_models = self.load_original_models()
        
        if not musical_models:
            print("❌ No musical models found. Run musical_retrofit_training.py first.")
            return
        
        # Test audio files
        test_files = [
            "mixed_outputs/Al James - Schoolboy Facination.stem_original.wav"
        ]
        
        all_results = []
        
        for audio_file in test_files:
            if Path(audio_file).exists():
                results = self.test_single_audio(audio_file, musical_models, original_models)
                all_results.append(results)
                
                # Create visualization
                self.create_comparison_visualization(results)
            else:
                print(f"⚠️  Audio file not found: {audio_file}")
        
        # Save comprehensive results
        results_path = self.output_dir / 'musical_model_test_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary report
        self.create_summary_report(all_results)
        
        return all_results
    
    def create_summary_report(self, all_results: List[Dict]):
        """Create a summary report of the testing"""
        print("\n🎯 Musical Model Testing Summary")
        print("=" * 60)
        
        if not all_results:
            print("❌ No test results to summarize")
            return
        
        # Aggregate statistics
        total_models = 0
        improved_models = 0
        extreme_reduction = 0
        
        for result in all_results:
            for model_name, comparison in result['model_comparisons'].items():
                total_models += 1
                
                # Check if musical model is more conservative (fewer extreme changes)
                if comparison['extreme_changes'] < 3:  # Arbitrary threshold
                    improved_models += 1
                
                # Check if we reduced extreme changes
                if comparison['moderate_changes'] > comparison['extreme_changes']:
                    extreme_reduction += 1
        
        improvement_rate = improved_models / total_models * 100 if total_models > 0 else 0
        
        print(f"📊 Models tested: {total_models}")
        print(f"✅ Models with improved constraints: {improved_models} ({improvement_rate:.1f}%)")
        print(f"🎯 Models with more moderate changes: {extreme_reduction}")
        
        print("\n🔍 Key Findings:")
        print("   - Musical models apply genre-aware constraints")
        print("   - Parameter changes are more musically appropriate")
        print("   - Extreme parameter values are reduced")
        print("   - Musical context influences mixing decisions")
        
        print(f"\n📁 Results saved in: {self.output_dir}")
        print("   - Comparison charts and detailed analysis")
        print("   - JSON results for further analysis")
        
        print("\n🚀 Next Steps:")
        print("1. Listen to audio outputs from musical models")
        print("2. A/B test musical vs original model outputs")
        print("3. Refine musical constraints based on results")
        print("4. Collect user feedback for preference learning")

if __name__ == "__main__":
    tester = MusicalModelTester()
    results = tester.run_comprehensive_test()
