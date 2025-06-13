#!/usr/bin/env python3
"""
User Preference Analysis - Understanding What Sounds Better
Analyzes differences between original and enhanced mixes to understand user preferences.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy import signal

class PreferenceAnalyzer:
    """Analyze what makes one mix sound better than another"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_preference_differences(self, original_path, enhanced_path):
        """Compare original vs enhanced to understand what you prefer"""
        
        print(f"🎵 Analyzing: {Path(original_path).name}")
        
        # Load both versions
        original, sr = librosa.load(original_path, sr=44100)
        enhanced, sr = librosa.load(enhanced_path, sr=44100)
        
        # Make sure they're the same length
        min_len = min(len(original), len(enhanced))
        original = original[:min_len]
        enhanced = enhanced[:min_len]
        
        analysis = {}
        
        # 1. Dynamic Range Analysis
        orig_rms = np.sqrt(np.mean(original**2))
        orig_peak = np.max(np.abs(original))
        orig_crest = orig_peak / (orig_rms + 1e-8)
        
        enh_rms = np.sqrt(np.mean(enhanced**2))
        enh_peak = np.max(np.abs(enhanced))
        enh_crest = enh_peak / (enh_rms + 1e-8)
        
        analysis['dynamics'] = {
            'original': {
                'rms': float(orig_rms),
                'peak': float(orig_peak),
                'crest_factor_db': float(20 * np.log10(orig_crest)),
                'dynamic_range': 'High' if orig_crest > 4 else 'Medium' if orig_crest > 2 else 'Low'
            },
            'enhanced': {
                'rms': float(enh_rms),
                'peak': float(enh_peak),
                'crest_factor_db': float(20 * np.log10(enh_crest)),
                'dynamic_range': 'High' if enh_crest > 4 else 'Medium' if enh_crest > 2 else 'Low'
            },
            'change': {
                'rms_change_db': float(20 * np.log10(enh_rms / (orig_rms + 1e-8))),
                'peak_change_db': float(20 * np.log10(enh_peak / (orig_peak + 1e-8))),
                'crest_change_db': float(20 * np.log10(enh_crest / (orig_crest + 1e-8)))
            }
        }
        
        # 2. Frequency Content Analysis
        orig_freqs, orig_psd = signal.welch(original, fs=sr, nperseg=4096)
        enh_freqs, enh_psd = signal.welch(enhanced, fs=sr, nperseg=4096)
        
        # Key frequency bands
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 200),
            'low_mid': (200, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 6000),
            'presence': (6000, 12000),
            'air': (12000, 20000)
        }
        
        freq_analysis = {}
        for band_name, (low, high) in bands.items():
            # Find frequency indices
            low_idx = np.argmin(np.abs(orig_freqs - low))
            high_idx = np.argmin(np.abs(orig_freqs - high))
            
            # Calculate energy in band
            orig_energy = np.sum(orig_psd[low_idx:high_idx])
            enh_energy = np.sum(enh_psd[low_idx:high_idx])
            
            change_db = 10 * np.log10(enh_energy / (orig_energy + 1e-12))
            
            freq_analysis[band_name] = {
                'original_energy': float(orig_energy),
                'enhanced_energy': float(enh_energy),
                'change_db': float(change_db),
                'description': self._describe_change(change_db)
            }
        
        analysis['frequency'] = freq_analysis
        
        # 3. Spectral Characteristics
        orig_centroid = np.mean(librosa.feature.spectral_centroid(y=original, sr=sr))
        enh_centroid = np.mean(librosa.feature.spectral_centroid(y=enhanced, sr=sr))
        
        orig_rolloff = np.mean(librosa.feature.spectral_rolloff(y=original, sr=sr))
        enh_rolloff = np.mean(librosa.feature.spectral_rolloff(y=enhanced, sr=sr))
        
        orig_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=original, sr=sr))
        enh_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=enhanced, sr=sr))
        
        analysis['spectral'] = {
            'brightness_change': {
                'original_centroid_hz': float(orig_centroid),
                'enhanced_centroid_hz': float(enh_centroid),
                'change_percent': float((enh_centroid - orig_centroid) / orig_centroid * 100),
                'description': 'Brighter' if enh_centroid > orig_centroid else 'Warmer'
            },
            'energy_distribution': {
                'original_rolloff_hz': float(orig_rolloff),
                'enhanced_rolloff_hz': float(enh_rolloff),
                'change_percent': float((enh_rolloff - orig_rolloff) / orig_rolloff * 100)
            },
            'frequency_spread': {
                'original_bandwidth_hz': float(orig_bandwidth),
                'enhanced_bandwidth_hz': float(enh_bandwidth),
                'change_percent': float((enh_bandwidth - orig_bandwidth) / orig_bandwidth * 100)
            }
        }
        
        # 4. Perceived Loudness
        analysis['loudness'] = {
            'rms_comparison': 'Enhanced is louder' if enh_rms > orig_rms else 'Original is louder',
            'peak_comparison': 'Enhanced has higher peaks' if enh_peak > orig_peak else 'Original has higher peaks',
            'overall_level_change': f"{analysis['dynamics']['change']['rms_change_db']:.1f} dB"
        }
        
        # 5. What might sound better about the original
        preferences = self._analyze_why_original_sounds_better(analysis)
        analysis['preference_insights'] = preferences
        
        return analysis
    
    def _describe_change(self, change_db):
        """Describe frequency changes in musical terms"""
        if change_db > 3:
            return "Significantly boosted"
        elif change_db > 1:
            return "Boosted"
        elif change_db > -1:
            return "Similar"
        elif change_db > -3:
            return "Reduced"
        else:
            return "Significantly reduced"
    
    def _analyze_why_original_sounds_better(self, analysis):
        """Analyze what might make the original sound better"""
        insights = []
        
        # Dynamic range preferences
        orig_crest = analysis['dynamics']['original']['crest_factor_db']
        enh_crest = analysis['dynamics']['enhanced']['crest_factor_db']
        
        if orig_crest > enh_crest:
            insights.append("Original has better dynamic range - sounds more natural and less compressed")
        
        # Frequency balance
        freq_changes = analysis['frequency']
        for band, data in freq_changes.items():
            change = data['change_db']
            if abs(change) > 2:
                if change > 0:
                    insights.append(f"Enhancement boosted {band.replace('_', ' ')} by {change:.1f}dB - might sound unnatural")
                else:
                    insights.append(f"Enhancement reduced {band.replace('_', ' ')} by {abs(change):.1f}dB - might sound dull")
        
        # Brightness changes
        brightness = analysis['spectral']['brightness_change']
        if abs(brightness['change_percent']) > 10:
            if brightness['change_percent'] > 0:
                insights.append("Enhancement made the mix brighter - might sound harsh or clinical")
            else:
                insights.append("Enhancement made the mix warmer - might sound muffled")
        
        # Overall processing
        rms_change = analysis['dynamics']['change']['rms_change_db']
        if abs(rms_change) > 1:
            insights.append(f"Enhancement changed overall level by {rms_change:.1f}dB - level changes can affect perception")
        
        return insights
    
    def create_preference_visualization(self, analysis, output_path):
        """Create visual comparison of original vs enhanced"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Dynamic Range Comparison
        dynamics = analysis['dynamics']
        categories = ['RMS Level', 'Peak Level', 'Crest Factor']
        original_vals = [dynamics['original']['rms'], 
                        dynamics['original']['peak'], 
                        dynamics['original']['crest_factor_db']]
        enhanced_vals = [dynamics['enhanced']['rms'], 
                        dynamics['enhanced']['peak'], 
                        dynamics['enhanced']['crest_factor_db']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax1.bar(x - width/2, original_vals, width, label='Original', alpha=0.8, color='blue')
        ax1.bar(x + width/2, enhanced_vals, width, label='Enhanced', alpha=0.8, color='orange')
        ax1.set_ylabel('Level')
        ax1.set_title('Dynamic Range Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Frequency Band Changes
        freq_data = analysis['frequency']
        bands = list(freq_data.keys())
        changes = [freq_data[band]['change_db'] for band in bands]
        
        colors = ['red' if x < -1 else 'green' if x > 1 else 'gray' for x in changes]
        bars = ax2.bar(bands, changes, color=colors, alpha=0.7)
        ax2.set_ylabel('Change (dB)')
        ax2.set_title('Frequency Band Changes (Enhanced vs Original)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}dB', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. Spectral Characteristics
        spectral = analysis['spectral']
        characteristics = ['Brightness', 'Energy Distribution', 'Frequency Spread']
        changes_pct = [spectral['brightness_change']['change_percent'],
                      spectral['energy_distribution']['change_percent'],
                      spectral['frequency_spread']['change_percent']]
        
        colors3 = ['red' if abs(x) > 10 else 'orange' if abs(x) > 5 else 'green' for x in changes_pct]
        bars3 = ax3.bar(characteristics, changes_pct, color=colors3, alpha=0.7)
        ax3.set_ylabel('Change (%)')
        ax3.set_title('Spectral Characteristic Changes')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Preference Insights (text summary)
        insights = analysis['preference_insights']
        ax4.axis('off')
        ax4.text(0.05, 0.95, "Why Original Might Sound Better:", fontsize=14, fontweight='bold',
                transform=ax4.transAxes, verticalalignment='top')
        
        y_pos = 0.85
        for i, insight in enumerate(insights[:6]):  # Limit to 6 insights
            ax4.text(0.05, y_pos - i*0.12, f"• {insight}", fontsize=10,
                    transform=ax4.transAxes, verticalalignment='top', wrap=True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Preference analysis chart saved: {output_path}")

def analyze_all_preferences():
    """Analyze preferences for all original vs enhanced pairs"""
    
    analyzer = PreferenceAnalyzer()
    
    mixed_outputs_dir = Path("mixed_outputs")
    enhanced_dir = mixed_outputs_dir / "enhanced"
    
    if not enhanced_dir.exists():
        print("❌ Enhanced directory not found. Run enhancement first.")
        return
    
    print("🎵 User Preference Analysis")
    print("=" * 50)
    print("Understanding why original mixes might sound better...")
    
    # Find original and enhanced pairs
    enhanced_files = list(enhanced_dir.glob("enhanced_*.wav"))
    
    all_analyses = {}
    
    for enhanced_file in enhanced_files:
        # Find corresponding original
        original_name = enhanced_file.name.replace("enhanced_", "")
        original_path = mixed_outputs_dir / original_name
        
        if original_path.exists():
            print(f"\n🔍 Analyzing: {original_name}")
            
            try:
                analysis = analyzer.analyze_preference_differences(
                    str(original_path), 
                    str(enhanced_file)
                )
                
                all_analyses[original_name] = analysis
                
                # Print key insights
                insights = analysis['preference_insights']
                print(f"   🎯 Top insights:")
                for insight in insights[:3]:
                    print(f"      • {insight}")
                
                # Create visualization for this pair
                viz_path = enhanced_dir / f"preference_analysis_{original_name.replace('.wav', '.png')}"
                analyzer.create_preference_visualization(analysis, viz_path)
                
            except Exception as e:
                print(f"   ❌ Error analyzing {original_name}: {e}")
    
    # Save complete analysis
    report_path = enhanced_dir / "preference_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_analyses, f, indent=2)
    
    print(f"\n📊 Complete preference analysis saved to: {report_path}")
    
    # Create summary insights
    create_preference_summary(all_analyses, enhanced_dir)
    
    print("\n🎯 Key Findings:")
    print("This analysis helps understand what you prefer about the originals")
    print("so we can improve the AI training to match your preferences!")

def create_preference_summary(analyses, output_dir):
    """Create a summary of preference patterns across all models"""
    
    print("\n📋 Creating Preference Summary...")
    
    summary = {
        'common_patterns': [],
        'dynamic_range_preferences': [],
        'frequency_preferences': [],
        'recommendations': []
    }
    
    # Analyze patterns across all models
    dynamic_preferences = []
    frequency_preferences = []
    
    for model_name, analysis in analyses.items():
        # Dynamic range patterns
        orig_crest = analysis['dynamics']['original']['crest_factor_db']
        enh_crest = analysis['dynamics']['enhanced']['crest_factor_db']
        dynamic_preferences.append(orig_crest - enh_crest)
        
        # Frequency preferences
        freq_changes = analysis['frequency']
        for band, data in freq_changes.items():
            change = data['change_db']
            if abs(change) > 1:
                frequency_preferences.append((band, change, model_name))
    
    # Common patterns
    avg_dynamic_pref = np.mean(dynamic_preferences)
    if avg_dynamic_pref > 1:
        summary['common_patterns'].append("You prefer higher dynamic range (less compression)")
    
    # Frequency patterns
    freq_by_band = {}
    for band, change, model in frequency_preferences:
        if band not in freq_by_band:
            freq_by_band[band] = []
        freq_by_band[band].append(change)
    
    for band, changes in freq_by_band.items():
        avg_change = np.mean(changes)
        if avg_change > 1:
            summary['frequency_preferences'].append(f"Enhancement boosted {band} too much (avg +{avg_change:.1f}dB)")
        elif avg_change < -1:
            summary['frequency_preferences'].append(f"Enhancement reduced {band} too much (avg {avg_change:.1f}dB)")
    
    # Recommendations for better AI training
    summary['recommendations'] = [
        "Train with less aggressive parameter changes",
        "Preserve dynamic range during enhancement",
        "Use gentler frequency adjustments",
        "Focus on musical balance over technical metrics",
        "Implement user preference learning"
    ]
    
    # Save summary
    summary_path = output_dir / "preference_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Preference summary saved: {summary_path}")
    
    # Print key findings
    print("\n🎯 Key Preference Patterns Found:")
    for pattern in summary['common_patterns']:
        print(f"   • {pattern}")
    
    for pref in summary['frequency_preferences']:
        print(f"   • {pref}")

if __name__ == "__main__":
    analyze_all_preferences()
