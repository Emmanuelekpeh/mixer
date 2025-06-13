#!/usr/bin/env python3
"""
🎵 Production AI Mixer Demo
===========================

Demonstration of the production-grade AI mixer with musical intelligence.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from production_ai_mixer import ProductionAIMixer

def demo_production_mixer():
    """Demo the production AI mixer"""
    print("🎛️ Production AI Mixer Demo")
    print("=" * 50)
    
    # Initialize the mixer
    mixer = ProductionAIMixer()
    
    # Check for test audio files
    test_files = []
    data_dirs = [
        Path(__file__).parent / "data" / "test",
        Path(__file__).parent / "mixed_outputs",
        Path(__file__).parent / "data"
    ]
    
    for data_dir in data_dirs:
        if data_dir.exists():
            for ext in ['*.wav', '*.mp3', '*.m4a']:
                test_files.extend(data_dir.glob(ext))
    
    if test_files:
        print(f"\n🎵 Found {len(test_files)} audio files to test:")
        for i, file in enumerate(test_files[:5]):  # Limit to first 5
            print(f"   {i+1}. {file.name}")
        
        # Test the first file
        test_file = test_files[0]
        print(f"\n🎧 Testing with: {test_file.name}")
        
        try:
            results = mixer.analyze_and_mix(str(test_file))
            
            print("\n✅ Musical AI mixing complete!")
            print(f"📁 Output files:")
            for name, path in results['output_files'].items():
                print(f"   • {name}: {Path(path).name}")
            
            print(f"\n🎼 Musical Analysis:")
            analysis = results['musical_analysis']
            print(f"   Genre: {analysis['genre']}")
            print(f"   Tempo: {analysis['tempo']:.1f} BPM")
            print(f"   Key: {analysis['key']}")
            print(f"   Energy: {analysis['energy']:.2f}")
            print(f"   Valence: {analysis['valence']:.2f}")
            
        except Exception as e:
            print(f"⚠️ Error processing {test_file.name}: {e}")
    
    else:
        print("\n⚠️ No audio files found for testing.")
        print("Place some .wav, .mp3, or .m4a files in:")
        print("   • data/test/")
        print("   • mixed_outputs/")
        print("   • data/")
        print("\nOr provide a file path directly:")
        print("   mixer.analyze_and_mix('path/to/your/audio.wav')")
    
    print(f"\n🎛️ Production features ready:")
    print(f"   ✅ Musical intelligence analysis")
    print(f"   ✅ Genre-specific mixing strategies")
    print(f"   ✅ Vintage & modern audio processors")
    print(f"   ✅ Tempo-synchronized effects")
    print(f"   ✅ Musical context-aware parameter adjustment")
    print(f"   ✅ Multiple AI model integration")

if __name__ == "__main__":
    demo_production_mixer()
