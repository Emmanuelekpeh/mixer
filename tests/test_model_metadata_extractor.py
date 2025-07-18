#!/usr/bin/env python3
"""Unit tests for ModelMetadataExtractor (backend)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

import sys, pathlib, pytest

# Ensure import path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tournament_webapp.backend.model_metadata_extractor import ModelMetadataExtractor, ModelMetadata


def _write_json(tmp_dir: Path, name: str, data: Dict) -> Path:
    path = tmp_dir / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def test_extract_from_valid_json(tmp_path: Path):
    extractor = ModelMetadataExtractor()
    meta_in = {
        "id": "foo",
        "name": "Foo Model",
        "architecture": "cnn",
        "file_path": "models/foo.pth",
        "size_mb": 1.23,
        "created_at": "2024-01-01T00:00:00",
        "description": "Test model",
        "performance_metrics": {},
        "specializations": [],
        "capabilities": {},
        "parameters": {},
    }
    json_path = _write_json(tmp_path, "foo.json", meta_in)
    out = extractor.extract_from_json(json_path)
    assert out["id"] == "foo"
    assert out["architecture"] == "cnn"


def test_extract_from_missing_json_returns_empty(tmp_path: Path):
    extractor = ModelMetadataExtractor()
    result = extractor.extract_from_json(tmp_path / "does_not_exist.json")
    assert result == {}


def test_generate_default_metadata(tmp_path: Path):
    # Create a fake model file
    model_path = tmp_path / "my_cnn_model_v2.pth"
    model_path.write_bytes(b"fake")

    extractor = ModelMetadataExtractor()
    metadata: ModelMetadata = extractor.generate_default_metadata(model_path)
    assert metadata.id == "my_cnn_model_v2"
    assert metadata.architecture in {"cnn", "unknown"}
    assert metadata.size_mb > 0
    # Capabilities should contain at least one entry
    assert metadata.capabilities


def test_capability_inference_from_name():
    extractor = ModelMetadataExtractor()
    caps = extractor.infer_capabilities_from_name("spectrogram_feature_cnn")
    # Expect spectral_analysis & feature_extraction high confidence
    assert "spectral_analysis" in caps
    assert caps["spectral_analysis"] >= 0.7


def test_generation_determination():
    extractor = ModelMetadataExtractor()
    existing = [
        "awesome_model_v1",
        "awesome_model_v2",
        "awesome_model_v3",
    ]
    gen = extractor.determine_model_generation("awesome_model_v4", existing)
    assert gen == 4
    # New series should return 1
    assert extractor.determine_model_generation("brand_new_model", existing) == 1 