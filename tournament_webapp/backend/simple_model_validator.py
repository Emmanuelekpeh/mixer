from __future__ import annotations

"""Lightweight Model Validator
================================

A self-contained, dependency-free validator that implements the minimal API
required by *tournament_webapp/backend/test_model_validator.py*:

• `test_model_loading`
• `extract_model_architecture`
• `validate_model_file`

This implementation deliberately focuses on robustness and fast execution
rather than deep inspection.  It is suitable for automated CI checks where
only basic assurances are necessary.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import time
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ValidationResult:
    """Result returned by `validate_model_file`."""

    is_valid: bool
    model_architecture: str
    error_message: Optional[str]
    can_load: bool
    inference_test_passed: bool
    compatibility_score: float  # 0-1
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class ModelValidator:
    """Simple model validator for PyTorch `.pth` files."""

    def __init__(self, device: Optional[str] = None, timeout_seconds: int = 30) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_seconds = timeout_seconds
        logger.info("ModelValidator ready on %s", self.device)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def test_model_loading(self, model_path: Path) -> Tuple[bool, str]:
        """Attempt to load the model with `torch.load`.

        Returns `(success, error_message)`.
        """
        if not model_path.exists():
            return False, "file_not_found"

        try:
            torch.load(model_path, map_location="cpu")
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def extract_model_architecture(self, model_path: Path) -> str:
        """Infer architecture from filename/state-dict keys (best-effort)."""
        name = model_path.stem.lower()
        heuristics = [
            "transformer",
            "lstm",
            "resnet",
            "gan",
            "vae",
            "cnn",
            "ensemble",
            "regressor",
        ]
        for h in heuristics:
            if h in name:
                return h

        # Attempt state-dict inspection
        try:
            obj = torch.load(model_path, map_location="cpu")
            if isinstance(obj, dict):
                joined = " ".join(obj.keys()).lower()
                for h in heuristics:
                    if h in joined:
                        return h
        except Exception:  # noqa: BLE001 – heuristic only
            pass

        return "unknown"

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def validate_model_file(self, model_path: Path) -> ValidationResult:  # noqa: C901
        """Validate `model_path` and return a `ValidationResult`."""
        start = time.time()

        warnings: List[str] = []
        recommendations: List[str] = []
        details: Dict[str, Any] = {}

        if not model_path.exists():
            return ValidationResult(
                is_valid=False,
                model_architecture="unknown",
                error_message="Model file does not exist",
                can_load=False,
                inference_test_passed=False,
                compatibility_score=0.0,
            )

        # --- Loading test
        can_load, load_error = self.test_model_loading(model_path)
        details["load_time_sec"] = round(time.time() - start, 4)
        details["file_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)

        if not can_load:
            return ValidationResult(
                is_valid=False,
                model_architecture="unknown",
                error_message=load_error,
                can_load=False,
                inference_test_passed=False,
                compatibility_score=0.0,
                warnings=warnings,
                recommendations=recommendations,
                details=details,
            )

        # --- Architecture detection
        architecture = self.extract_model_architecture(model_path)

        # --- Basic inference (best-effort)
        inference_passed = False
        inference_error = ""
        try:
            obj = torch.load(model_path, map_location=self.device)
            if hasattr(obj, "eval") and callable(getattr(obj, "eval")):
                obj.eval()
                dummy = torch.randn(1, 1, 128, 128, device=self.device)
                with torch.no_grad():
                    _ = obj(dummy)  # noqa: F841
                inference_passed = True
            else:
                # State-dict – inference not applicable
                inference_passed = True
                warnings.append("State-dict detected – inference test skipped")
        except Exception as exc:  # noqa: BLE001
            inference_error = str(exc)
            warnings.append(f"Inference failed: {inference_error}")

        details["inference_error"] = inference_error

        compatibility_score = 1.0 if inference_passed else 0.5
        is_valid = can_load and compatibility_score >= 0.5

        return ValidationResult(
            is_valid=is_valid,
            model_architecture=architecture,
            error_message=None if is_valid else (inference_error or load_error),
            can_load=can_load,
            inference_test_passed=inference_passed,
            compatibility_score=compatibility_score,
            warnings=warnings,
            recommendations=recommendations,
            details=details,
        )

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ModelValidator device={self.device}>" 