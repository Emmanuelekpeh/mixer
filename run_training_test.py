import sys
from pathlib import Path

# Add src to path to allow imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

try:
    print("Attempting to import train_mixer_pipeline_fixed...")
    from train_mixer_pipeline_fixed import main
    print("Import successful. Running main function...")
    main()
    print("Main function finished.")
except Exception as e:
    import traceback
    print(f"An error occurred: {e}")
    traceback.print_exc() 