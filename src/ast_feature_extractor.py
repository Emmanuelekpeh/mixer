import os
import numpy as np
from pathlib import Path
from transformers import ASTFeatureExtractor, ASTModel
import torch
import soundfile as sf

SPECTROGRAMS_FOLDER = Path(__file__).resolve().parent.parent / "data" / "spectrograms"
AUDIO_FOLDER = Path(__file__).resolve().parent.parent / "data"
FEATURES_FOLDER = Path(__file__).resolve().parent.parent / "data" / "ast_features"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AST expects raw audio waveform (1D float32, mono, 16kHz recommended)
def load_audio(wav_path, target_sr=16000):
    y, sr = sf.read(wav_path)
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # Convert to mono if stereo
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32), target_sr

def extract_ast_features(split="train"):
    FEATURES_FOLDER.mkdir(parents=True, exist_ok=True)
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    split_dir = SPECTROGRAMS_FOLDER / split
    out_dir = FEATURES_FOLDER / split
    out_dir.mkdir(parents=True, exist_ok=True)
    for track_dir in split_dir.iterdir():
        if not track_dir.is_dir():
            continue
        for spec_file in track_dir.glob("*.npy"):
            # Find corresponding .wav file
            stem_name = spec_file.stem
            wav_path = AUDIO_FOLDER / split / track_dir.name / f"{stem_name}.wav"
            if not wav_path.exists():
                print(f"Missing audio file: {wav_path}")
                continue
            y, sr = load_audio(wav_path, target_sr=16000)
            inputs = feature_extractor(
                [y], sampling_rate=sr, return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token feature (global representation)
                cls_feature = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            # Save feature
            out_track_dir = out_dir / track_dir.name
            out_track_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_track_dir / f"{stem_name}_ast_cls.npy"
            np.save(out_path, cls_feature)
            print(f"Saved AST feature: {out_path}")

if __name__ == "__main__":
    extract_ast_features(split="train")
    extract_ast_features(split="val")
    extract_ast_features(split="test") 