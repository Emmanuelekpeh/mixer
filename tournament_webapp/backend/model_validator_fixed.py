"""Deprecation shim for backwards-compatibility.

`model_validator_fixed.py` previously attempted an alternate implementation of
`ModelValidator`. To avoid code & type-checking duplication, the canonical
implementation now lives exclusively in `model_validator.py`.  This module
re-exports the public symbols so that legacy import paths continue to work
without modification.
"""

from __future__ import annotations

# Public re-export
from .model_validator import ModelValidator, ValidationResult  # noqa: F401,E402

__all__: list[str] = ["ModelValidator", "ValidationResult"]