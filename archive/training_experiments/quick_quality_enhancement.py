#!/usr/bin/env python3
"""
Quick Mix Quality Improvement - Immediate fixes for better sound
Applies post-processing to existing model outputs for more balanced results.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import json

class MixingQualityEnhancer:
    """Post-process existing AI mixing results for better balance"""
    
    def __init__(self):
        # Target frequency response for balanced mixes
        self.target_lufs = -14  # Standard streaming loudness
        self.max_peak = -1.0    # Peak limiting
        
        # Frequency balance targets (dB relative to 1kHz)
        self.frequency_targets = {
            100: -3,    # Sub bass: slightly reduced
            200: -1,    # Low bass: natural
            500: 0,     # Low mids: reference
            1000: 0,    # Mids: reference point
            2000: 1,    # Upper mids: slight presence
            4000: 0,    # Presence: natural
            8000: -1,   # High mids: slightly reduced
            16000: -2   # Highs: gentle rolloff
        }
      def analyze_mix_quality(self, audio_path):
        """Analyze current mix quality issues"""
        audio, sr = librosa.load(audio_path, sr=44100)
        
        analysis = {}
        
        # Loudness analysis
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        crest_factor = peak / (rms + 1e-8)
        
        analysis['loudness'] = {
            'rms': float(rms),
            'peak': float(peak),
            'crest_factor': float(crest_factor),
            'dynamic_range_db': float(20 * np.log10(crest_factor))
        }
        
        # Frequency analysis
        freqs = np.array([100, 200, 500, 1000, 2000, 4000, 8000, 16000])
        freq_response = []
        
        # Get power spectral density
        f, psd = signal.welch(audio, fs=sr, nperseg=4096)
        
        for target_freq in freqs:
            # Find closest frequency bin
            idx = np.argmin(np.abs(f - target_freq))
            power_db = 10 * np.log10(psd[idx] + 1e-12)
            freq_response.append(power_db)
        
        # Normalize to 1kHz (reference = 0dB)
        ref_idx = np.where(freqs == 1000)[0][0]
        freq_response = np.array(freq_response) - freq_response[ref_idx]
        
        analysis['frequency'] = {
            'frequencies': freqs.tolist(),
            'response_db': freq_response.tolist(),
            'balance_score': self.calculate_balance_score(freqs, freq_response)
        }
        
        # Spectral characteristics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        
        analysis['spectral'] = {
            'centroid_hz': float(spectral_centroid),
            'rolloff_hz': float(spectral_rolloff),
            'brightness': float(spectral_centroid / 2000)  # Relative to 2kHz
        }
        
        return analysis
    
    def calculate_balance_score(self, freqs, response):
        """Calculate how well-balanced the frequency response is"""
        targets = [self.frequency_targets.get(f, 0) for f in freqs]
        differences = np.abs(np.array(response) - np.array(targets))
        return float(np.exp(-np.mean(differences) / 3))  # Score 0-1, higher is better
    
    def enhance_mix_balance(self, audio_path, output_path, enhancement_level=0.5):
        """Apply gentle enhancement for better balance"""
        audio, sr = librosa.load(audio_path, sr=44100)
        enhanced = audio.copy()
        
        print(f"🎵 Enhancing: {Path(audio_path).name}")
        
        # 1. Gentle dynamic range restoration
        enhanced = self.restore_dynamics(enhanced, enhancement_level)
        
        # 2. Frequency balance correction
        enhanced = self.balance_frequency_response(enhanced, sr, enhancement_level)
          # 3. Stereo enhancement (if stereo and audio is 2D)
        if hasattr(enhanced, 'shape') and len(enhanced.shape) == 2:
            enhanced = self.enhance_stereo_image(enhanced, enhancement_level)
        
        # 4. Gentle limiting and loudness normalization
        enhanced = self.normalize_loudness(enhanced)
        
        # Save enhanced audio
        sf.write(output_path, enhanced, sr)
        
        # Return enhancement statistics
        original_analysis = self.analyze_mix_quality(audio_path)
        enhanced_analysis = self.analyze_mix_quality(output_path)
        
        return {
            'original': original_analysis,
            'enhanced': enhanced_analysis,
            'improvement': {
                'balance_score_change': (enhanced_analysis['frequency']['balance_score'] - 
                                       original_analysis['frequency']['balance_score']),
                'dynamic_range_change': (enhanced_analysis['loudness']['dynamic_range_db'] - 
                                       original_analysis['loudness']['dynamic_range_db'])
            }
        }
    
    def restore_dynamics(self, audio, level):
        """Gently restore dynamic range for over-compressed audio"""
        # Detect compressed regions (low dynamic range)
        window_size = int(0.1 * 44100)  # 100ms windows
        dynamic_range = []
        
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window**2))
            peak = np.max(np.abs(window))
            dr = peak / (rms + 1e-8)
            dynamic_range.append(dr)
        
        # If average dynamic range is low, apply gentle expansion
        avg_dr = np.mean(dynamic_range)
        if avg_dr < 3:  # Very compressed
            # Gentle upward expansion
            expansion_ratio = 1 + level * 0.3  # Max 30% expansion
            threshold = 0.3
            
            # Apply soft expansion above threshold
            mask = np.abs(audio) > threshold
            excess = np.abs(audio[mask]) - threshold
            expanded = threshold + excess * expansion_ratio
            audio[mask] = np.sign(audio[mask]) * expanded
        
        return audio
    
    def balance_frequency_response(self, audio, sr, level):
        """Apply gentle frequency balancing"""
        # Design subtle EQ curve
        freqs = np.array([100, 200, 500, 1000, 2000, 4000, 8000, 16000])
        
        # Get current response
        f, psd = signal.welch(audio, fs=sr, nperseg=4096)
        current_response = []
        
        for target_freq in freqs:
            idx = np.argmin(np.abs(f - target_freq))
            power_db = 10 * np.log10(psd[idx] + 1e-12)
            current_response.append(power_db)
        
        # Normalize to 1kHz
        ref_idx = np.where(freqs == 1000)[0][0]
        current_response = np.array(current_response) - current_response[ref_idx]
        
        # Calculate needed corrections
        targets = [self.frequency_targets.get(f, 0) for f in freqs]
        corrections = (np.array(targets) - current_response) * level * 0.5  # Gentle correction
        
        # Apply gentle filtering (simplified - real implementation would use proper EQ)
        if np.max(np.abs(corrections)) > 0.5:  # Only if correction is needed
            # Simple high-pass and low-pass for gentle shaping
            if corrections[0] < -1:  # Too much low end
                sos = signal.butter(2, 80, 'hp', fs=sr, output='sos')
                audio = signal.sosfilt(sos, audio)
            
            if corrections[-1] < -1:  # Too much high end
                sos = signal.butter(2, 12000, 'lp', fs=sr, output='sos')
                audio = signal.sosfilt(sos, audio)
        
        return audio
      def enhance_stereo_image(self, audio, level):
        """Gentle stereo image enhancement"""
        if not isinstance(audio, np.ndarray) or audio.ndim != 2:
            return audio
        
        left, right = audio[:, 0], audio[:, 1]
        
        # Mid-side processing
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Gentle stereo widening (up to 20%)
        width_factor = 1 + level * 0.2
        side_enhanced = side * width_factor
        
        # Convert back to L/R
        left_enhanced = mid + side_enhanced
        right_enhanced = mid - side_enhanced
        
        # Ensure no clipping
        max_val = max(np.max(np.abs(left_enhanced)), np.max(np.abs(right_enhanced)))
        if max_val > 0.95:
            scale = 0.95 / max_val
            left_enhanced *= scale
            right_enhanced *= scale
        
        return np.column_stack([left_enhanced, right_enhanced])
    
    def normalize_loudness(self, audio):
        """Gentle loudness normalization"""
        # Target RMS for consistent playback
        current_rms = np.sqrt(np.mean(audio**2))
        target_rms = 0.15  # Moderate level
        
        if current_rms > 0:
            gain = target_rms / current_rms
            # Limit gain changes to ±6dB
            gain = np.clip(gain, 0.5, 2.0)
            audio = audio * gain
        
        # Gentle peak limiting
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio * (0.95 / peak)
        
        return audio

def enhance_all_mixed_outputs():
    """Enhance all existing AI mixing outputs for better balance"""
    
    enhancer = MixingQualityEnhancer()
    mixed_outputs_dir = Path("mixed_outputs")
    enhanced_dir = mixed_outputs_dir / "enhanced"
    enhanced_dir.mkdir(exist_ok=True)
    
    print("🎵 AI Mixing Quality Enhancement")
    print("=" * 50)
    print("Applying gentle post-processing for better balance and sound quality...")
    
    # Find all mixed audio files (skip original)
    mixed_files = [f for f in mixed_outputs_dir.glob("*.wav") 
                   if "original" not in f.name and "enhanced" not in f.name]
    
    enhancement_results = {}
    
    for audio_file in mixed_files:
        print(f"\n🔧 Processing: {audio_file.name}")
        
        # Create enhanced version
        enhanced_path = enhanced_dir / f"enhanced_{audio_file.name}"
        
        try:
            results = enhancer.enhance_mix_balance(
                str(audio_file), 
                str(enhanced_path),
                enhancement_level=0.6  # Moderate enhancement
            )
            
            enhancement_results[audio_file.name] = results
            
            # Print improvement summary
            balance_improvement = results['improvement']['balance_score_change']
            dr_improvement = results['improvement']['dynamic_range_change']
            
            print(f"   ✅ Balance score: {balance_improvement:+.3f}")
            print(f"   ✅ Dynamic range: {dr_improvement:+.1f} dB")
            
        except Exception as e:
            print(f"   ❌ Error processing {audio_file.name}: {e}")
    
    # Save enhancement report
    report_path = enhanced_dir / "enhancement_report.json"
    with open(report_path, 'w') as f:
        json.dump(enhancement_results, f, indent=2)
    
    print(f"\n📊 Enhancement report saved to: {report_path}")
    
    # Create comparison visualization
    create_enhancement_comparison(enhancement_results, enhanced_dir)
    
    print("\n🎯 Enhancement Complete!")
    print(f"Enhanced files saved in: {enhanced_dir}")
    print("\nRecommended listening order:")
    print("1. Original AI mix")
    print("2. Enhanced version")
    print("3. Compare balance and sound quality")

def create_enhancement_comparison(results, output_dir):
    """Create visual comparison of enhancements"""
    
    models = list(results.keys())
    balance_improvements = [results[m]['improvement']['balance_score_change'] for m in models]
    dr_improvements = [results[m]['improvement']['dynamic_range_change'] for m in models]
    
    # Clean up model names for display
    display_names = [m.replace('Al James - Schoolboy Facination.stem_', '').replace('_mixed.wav', '') 
                    for m in models]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Balance score improvements
    colors = ['green' if x > 0 else 'red' for x in balance_improvements]
    bars1 = ax1.bar(display_names, balance_improvements, color=colors, alpha=0.7)
    ax1.set_title('Frequency Balance Improvement', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Balance Score Change')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, balance_improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:+.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Dynamic range improvements
    colors2 = ['green' if x > 0 else 'red' for x in dr_improvements]
    bars2 = ax2.bar(display_names, dr_improvements, color=colors2, alpha=0.7)
    ax2.set_title('Dynamic Range Improvement', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Dynamic Range Change (dB)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, dr_improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:+.1f}dB', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhancement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Enhancement comparison chart saved: {output_dir / 'enhancement_comparison.png'}")

if __name__ == "__main__":
    enhance_all_mixed_outputs()
