#!/usr/bin/env python3
"""
🎵 Model Comparison Summary Report
=================================

Generate a detailed report comparing all AI mixing models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

def create_comprehensive_report():
    """Create a comprehensive model comparison report."""
    mixed_outputs_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
    results_file = mixed_outputs_dir / "Al James - Schoolboy Facination.stem_mixing_comparison.json"
    
    if not results_file.exists():
        print("❌ No comparison results found!")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("🎵 AI MIXING MODEL COMPARISON REPORT")
    print("=" * 60)
    print(f"📅 Generated: {data['timestamp']}")
    print(f"🎵 Test Song: {data['song']}")
    print(f"⏱️  Processing Time: {data['processing_time']:.1f} seconds")
    print(f"🤖 Models Tested: {len(data['models_used'])}")
    
    results = data['results']
    
    # Create detailed analysis table
    print(f"\n📊 DETAILED MODEL COMPARISON")
    print("-" * 60)
    
    param_names = [
        "Input Gain", "Compression", "High EQ", "Mid EQ", 
        "Low EQ", "Presence", "Reverb", "Delay", 
        "Width", "Output"
    ]
    
    # Create a comprehensive table
    print(f"{'Model':<25} {'RMS':<6} {'Peak':<6} {'Style':<12} {'Notable Features'}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        rms_change = metrics['rms_change']
        peak_change = metrics['peak_change']
        params = metrics['parameters']
        
        # Determine style
        if 0.8 <= rms_change <= 1.2:
            style = "Conservative"
        elif 0.5 <= rms_change < 0.8 or 1.2 < rms_change <= 1.5:
            style = "Moderate"
        else:
            style = "Aggressive"
        
        # Find notable features (extreme parameters)
        notable = []
        for i, (param_name, value) in enumerate(zip(param_names, params)):
            if value > 0.8:
                notable.append(f"High {param_name}")
            elif value < 0.2:
                notable.append(f"Low {param_name}")
        
        notable_str = ", ".join(notable[:2]) if notable else "Balanced"
        
        print(f"{model_name:<25} {rms_change:<6.2f} {peak_change:<6.2f} {style:<12} {notable_str}")
    
    # Performance insights
    print(f"\n🔍 PERFORMANCE INSIGHTS")
    print("-" * 60)
    
    # Find best performers for different use cases
    rms_changes = [(name, metrics['rms_change']) for name, metrics in results.items()]
    
    # Most natural (closest to 1.0x)
    most_natural = min(rms_changes, key=lambda x: abs(x[1] - 1.0))
    
    # Most volume boost
    most_boost = max(rms_changes, key=lambda x: x[1])
    
    # Most reduction
    most_reduction = min(rms_changes, key=lambda x: x[1])
    
    print(f"🎯 Most Natural Processing: {most_natural[0]} ({most_natural[1]:.2f}x)")
    print(f"📈 Highest Volume Boost: {most_boost[0]} ({most_boost[1]:.2f}x)")
    print(f"📉 Most Volume Reduction: {most_reduction[0]} ({most_reduction[1]:.2f}x)")
    
    # Parameter analysis
    print(f"\n🎛️ PARAMETER INSIGHTS")
    print("-" * 60)
    
    param_stats = {}
    for i, param_name in enumerate(param_names):
        values = [metrics['parameters'][i] for metrics in results.values()]
        param_stats[param_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print(f"{'Parameter':<15} {'Mean':<8} {'StdDev':<8} {'Range':<15} {'Consensus'}")
    print("-" * 65)
    
    for param_name, stats in param_stats.items():
        range_str = f"{stats['min']:.2f}-{stats['max']:.2f}"
        
        # Determine consensus
        if stats['std'] < 0.1:
            consensus = "High"
        elif stats['std'] < 0.3:
            consensus = "Moderate"
        else:
            consensus = "Low"
        
        print(f"{param_name:<15} {stats['mean']:<8.3f} {stats['std']:<8.3f} {range_str:<15} {consensus}")
    
    # Use case recommendations
    print(f"\n🎧 USE CASE RECOMMENDATIONS")
    print("-" * 60)
    
    recommendations = {
        "🎵 Gentle Enhancement": most_natural[0],
        "🔊 Loudness Maximizing": most_boost[0],
        "🎚️ Dynamic Control": most_reduction[0],
        "🎭 Creative Processing": "Enhanced CNN",  # Based on high reverb/effects
        "📻 Radio Ready": "Improved Enhanced CNN",  # Based on compression and EQ
        "🎸 Live Sound": "AST Regressor"  # Based on balanced feature analysis
    }
    
    for use_case, recommended_model in recommendations.items():
        if recommended_model in results:
            rms = results[recommended_model]['rms_change']
            print(f"{use_case:<25}: {recommended_model} ({rms:.2f}x)")
    
    # Technical model comparison
    print(f"\n🤖 TECHNICAL MODEL COMPARISON")
    print("-" * 60)
    
    model_types = {
        "Baseline CNN": "Original CNN architecture",
        "Enhanced CNN": "Improved CNN with attention",
        "Improved Baseline CNN": "Optimized baseline model",
        "Improved Enhanced CNN": "Advanced architecture",
        "Retrained Enhanced CNN": "Enhanced model on augmented data",
        "AST Regressor": "Feature-based audio analysis"
    }
    
    for model_name, description in model_types.items():
        if model_name in results:
            print(f"{model_name:<25}: {description}")
    
    # Create detailed parameter comparison chart
    create_detailed_parameter_chart(results, param_names, mixed_outputs_dir)
    
    return data

def create_detailed_parameter_chart(results, param_names, output_dir):
    """Create a detailed parameter comparison chart."""
    
    # Create a more detailed visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    model_names = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    # Create individual parameter charts
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        
        values = [results[model]['parameters'][i] for model in model_names]
        short_names = [name.replace(' CNN', '').replace(' Regressor', '') for name in model_names]
        
        bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.8)
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(param_name, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add reference line at 0.5
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Detailed Parameter Comparison Across All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    chart_path = output_dir / "detailed_parameter_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Detailed parameter chart saved: {chart_path.name}")

def create_audio_waveform_comparison():
    """Create waveform comparison of different model outputs."""
    import librosa
    import soundfile as sf
    
    mixed_outputs_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
    
    # Find all mixed audio files
    audio_files = {}
    for file in mixed_outputs_dir.glob("*.wav"):
        if "original" in file.name:
            audio_files["Original"] = file
        elif "baseline_cnn" in file.name:
            audio_files["Baseline CNN"] = file
        elif "enhanced_cnn" in file.name:
            audio_files["Enhanced CNN"] = file
        elif "improved_enhanced" in file.name:
            audio_files["Improved Enhanced"] = file
        elif "ast_regressor" in file.name:
            audio_files["AST Regressor"] = file
    
    if len(audio_files) < 2:
        print("❌ Not enough audio files for waveform comparison")
        return
    
    # Create waveform comparison
    fig, axes = plt.subplots(len(audio_files), 1, figsize=(15, 2*len(audio_files)))
    if len(audio_files) == 1:
        axes = [axes]
    
    for i, (name, file_path) in enumerate(audio_files.items()):
        try:
            audio, sr = librosa.load(file_path, sr=22050, mono=True)
            
            # Show first 5 seconds
            duration = min(5.0, len(audio) / sr)
            samples = int(duration * sr)
            time_axis = np.linspace(0, duration, samples)
            
            axes[i].plot(time_axis, audio[:samples], linewidth=0.5)
            axes[i].set_title(f"{name} (RMS: {np.sqrt(np.mean(audio**2)):.3f})", fontweight='bold')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
            
            if i == len(audio_files) - 1:
                axes[i].set_xlabel('Time (seconds)')
                
        except Exception as e:
            print(f"❌ Could not load {name}: {e}")
    
    plt.suptitle('Waveform Comparison - First 5 Seconds', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    waveform_path = mixed_outputs_dir / "waveform_comparison.png"
    plt.savefig(waveform_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Waveform comparison saved: {waveform_path.name}")

if __name__ == "__main__":
    print("🎵 Creating Comprehensive Model Comparison Report")
    print("=" * 60)
    
    # Generate main report
    data = create_comprehensive_report()
    
    if data:
        # Create additional visualizations
        print(f"\n📊 Creating additional visualizations...")
        create_audio_waveform_comparison()
        
        print(f"\n✅ Comprehensive report complete!")
        print(f"📁 All files saved to: mixed_outputs/")
        print(f"\n🎧 NEXT STEPS:")
        print(f"1. Listen to the different mixed versions")
        print(f"2. Compare waveforms and parameter charts")
        print(f"3. Choose your preferred model for your use case")
        print(f"4. Consider ensemble approaches for optimal results")
