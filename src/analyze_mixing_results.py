#!/usr/bin/env python3
"""
🎵 Mixing Results Analysis
=========================

Analyze and visualize the results from the comprehensive mixer comparison.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_mixing_results():
    """Analyze the mixing comparison results."""
    mixed_outputs_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
    
    # Load the comparison results
    results_file = mixed_outputs_dir / "Al James - Schoolboy Facination.stem_mixing_comparison.json"
    
    if not results_file.exists():
        print("❌ No comparison results found!")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("🎵 MIXING COMPARISON ANALYSIS")
    print("=" * 50)
    print(f"Song: {data['song']}")
    print(f"Models compared: {len(data['models_used'])}")
    print(f"Processing time: {data['processing_time']:.1f} seconds")
    
    # Analyze audio changes
    print(f"\n📊 AUDIO LEVEL CHANGES:")
    print("-" * 30)
    
    results = data['results']
    for model_name, metrics in results.items():
        rms_change = metrics['rms_change']
        peak_change = metrics['peak_change']
        
        rms_symbol = "🔊" if rms_change > 1 else "🔉" if rms_change > 0.5 else "🔇"
        peak_symbol = "📈" if peak_change > 1 else "📊" if peak_change > 0.5 else "📉"
        
        print(f"{model_name:20}: RMS {rms_symbol} {rms_change:.2f}x, Peak {peak_symbol} {peak_change:.2f}x")
    
    # Parameter analysis
    print(f"\n🎛️ PARAMETER ANALYSIS:")
    print("-" * 30)
    
    param_names = [
        "Input Gain", "Compression Ratio", "High-Freq EQ", "Mid-Freq EQ", 
        "Low-Freq EQ", "Presence/Air", "Reverb Send", "Delay Send", 
        "Stereo Width", "Output Level"
    ]
    
    # Find most extreme values for each parameter
    for i, param_name in enumerate(param_names):
        values = []
        model_names = []
        
        for model_name, metrics in results.items():
            values.append(metrics['parameters'][i])
            model_names.append(model_name)
        
        min_idx = np.argmin(values)
        max_idx = np.argmax(values)
        
        print(f"{param_name:15}: Min {values[min_idx]:.2f} ({model_names[min_idx]}) | "
              f"Max {values[max_idx]:.2f} ({model_names[max_idx]})")
    
    # Model recommendations
    print(f"\n🏆 MODEL RECOMMENDATIONS:")
    print("-" * 30)
    
    # Most conservative (smallest changes)
    rms_changes = [(name, metrics['rms_change']) for name, metrics in results.items()]
    most_conservative = min(rms_changes, key=lambda x: abs(x[1] - 1.0))
    
    # Most aggressive (largest changes)
    most_aggressive = max(rms_changes, key=lambda x: abs(x[1] - 1.0))
    
    # Best overall balance (considering multiple factors)
    balance_scores = []
    for name, metrics in results.items():
        # Score based on reasonable parameter ranges and audio changes
        params = metrics['parameters']
        rms_change = metrics['rms_change']
        
        # Prefer moderate changes and balanced parameters
        param_balance = 1.0 - np.std(params)  # Lower std = more balanced
        level_balance = 1.0 - abs(rms_change - 1.0)  # Closer to 1.0 = more balanced
        
        score = param_balance * 0.6 + level_balance * 0.4
        balance_scores.append((name, score))
    
    best_balanced = max(balance_scores, key=lambda x: x[1])
    
    print(f"🔇 Most Conservative: {most_conservative[0]} ({most_conservative[1]:.2f}x RMS)")
    print(f"🔊 Most Aggressive: {most_aggressive[0]} ({most_aggressive[1]:.2f}x RMS)")
    print(f"⚖️  Best Balanced: {best_balanced[0]} (score: {best_balanced[1]:.3f})")
    
    print(f"\n🎧 LISTENING GUIDE:")
    print("-" * 30)
    print(f"1. Start with 'original.wav' to hear the source")
    print(f"2. Try '{best_balanced[0].lower().replace(' ', '_')}_mixed.wav' for balanced processing")
    print(f"3. Compare with '{most_conservative[0].lower().replace(' ', '_')}_mixed.wav' for subtle changes")
    print(f"4. Try '{most_aggressive[0].lower().replace(' ', '_')}_mixed.wav' for dramatic effects")
    
    print(f"\n📁 All files are in: {mixed_outputs_dir}")
    
    return data

def create_model_performance_chart():
    """Create a chart showing model performance characteristics."""
    mixed_outputs_dir = Path(__file__).resolve().parent.parent / "mixed_outputs"
    results_file = mixed_outputs_dir / "Al James - Schoolboy Facination.stem_mixing_comparison.json"
    
    if not results_file.exists():
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Create performance comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = list(results.keys())
    short_names = [name.replace(' CNN', '').replace(' Regressor', '') for name in model_names]
    
    # 1. Audio Level Changes
    rms_changes = [results[name]['rms_change'] for name in model_names]
    peak_changes = [results[name]['peak_change'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, rms_changes, width, label='RMS Change', alpha=0.8)
    bars2 = ax1.bar(x + width/2, peak_changes, width, label='Peak Change', alpha=0.8)
    
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Amplitude Multiplier')
    ax1.set_title('Audio Level Changes by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter Variance (how much each model varies from defaults)
    param_variances = []
    for name in model_names:
        params = np.array(results[name]['parameters'])
        # Calculate variance from 0.5 (typical default)
        variance = np.mean(np.abs(params - 0.5))
        param_variances.append(variance)
    
    bars = ax2.bar(short_names, param_variances, alpha=0.8, color='orange')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Average Parameter Deviation')
    ax2.set_title('Parameter Aggressiveness by Model')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, variance in zip(bars, param_variances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{variance:.2f}', ha='center', va='bottom')
    
    # 3. Processing Type Distribution
    processing_types = ['Conservative', 'Moderate', 'Aggressive']
    conservative = sum(1 for rms in rms_changes if 0.8 <= rms <= 1.2)
    moderate = sum(1 for rms in rms_changes if 0.5 <= rms < 0.8 or 1.2 < rms <= 1.5)
    aggressive = sum(1 for rms in rms_changes if rms < 0.5 or rms > 1.5)
    
    counts = [conservative, moderate, aggressive]
    colors = ['green', 'orange', 'red']
    
    ax3.pie(counts, labels=processing_types, colors=colors, autopct='%1.0f%%', startangle=90)
    ax3.set_title('Distribution of Processing Styles')
    
    # 4. Parameter Heatmap
    param_names = [
        "Input Gain", "Compression", "High EQ", "Mid EQ", 
        "Low EQ", "Presence", "Reverb", "Delay", 
        "Width", "Output"
    ]
    
    # Create parameter matrix
    param_matrix = []
    for name in model_names:
        param_matrix.append(results[name]['parameters'])
    
    param_matrix = np.array(param_matrix)
    
    im = ax4.imshow(param_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(param_names)))
    ax4.set_xticklabels(param_names, rotation=45, ha='right')
    ax4.set_yticks(range(len(short_names)))
    ax4.set_yticklabels(short_names)
    ax4.set_title('Parameter Values Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Parameter Value (0-1)')
    
    plt.tight_layout()
    
    output_path = mixed_outputs_dir / "model_performance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Performance analysis chart saved: {output_path.name}")

if __name__ == "__main__":
    print("🎵 AI Mixing Results Analysis")
    print("=" * 50)
    
    # Analyze results
    data = analyze_mixing_results()
    
    if data:
        # Create performance chart
        create_model_performance_chart()
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Check the mixed_outputs folder for:")
        print(f"   - Audio files for each model")
        print(f"   - Comparison charts")
        print(f"   - Performance analysis")
