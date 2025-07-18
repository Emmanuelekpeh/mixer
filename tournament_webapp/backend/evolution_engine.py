from __future__ import annotations

"""EvolutionEngine – minimal real implementation
================================================
Maintains ELO scores, generates next battle pairs and records results.
This version is intentionally simple but fully functional so we can
eliminate DummyEvolutionEngine placeholders while we iterate toward a
more advanced evolutionary algorithm.
"""

from typing import List, Dict, Tuple, Optional, Any
import random
import math
from datetime import datetime

try:
    # Relative import to the ORM model (SQLAlchemy)
    from .database import AIModel
except ImportError:  # during static-analysis if database isn’t installed
    # Fallback lightweight dataclass for type-checking and unit tests
    from dataclasses import dataclass, field

    @dataclass
    class AIModel:  # type: ignore
        id: str
        name: str = "Unnamed"
        architecture: str = "unknown"
        elo_rating: float = 1200.0
        tier: str = "Amateur"
        generation: int = 1
        created_at: datetime = field(default_factory=datetime.utcnow)

__all__ = ["EvolutionEngine"]


class EvolutionEngine:
    """Simple ELO-based evolution/competition engine."""

    def __init__(self, models: Optional[List[AIModel]] = None, elo_k: float = 24) -> None:
        if models is None:
            models = []
        self.models: Dict[str, AIModel] = {m.id: m for m in models}
        self.elo_k = elo_k
        # Genealogy tree can be a simple dict for now
        self._genealogy: Dict[str, Any] = {
            "models": [self._simplify_model(m) for m in models],
            "statistics": {
                "total_evolved": len(models),
                "by_architecture": self._count_by_architecture(models),
            },
            "evolution_tree": {},  # placeholder for future tree structure
        }

    # ------------------------------------------------------------------
    # Model synchronization helpers
    # ------------------------------------------------------------------

    def set_models(self, models: List[AIModel]) -> None:
        """Replace current model list with a new list, updating stats."""
        self.models = {m.id: m for m in models}
        # refresh stats/genealogy snapshot
        self._genealogy["models"] = [self._simplify_model(m) for m in models]
        self._genealogy["statistics"]["total_evolved"] = len(models)
        self._genealogy["statistics"]["by_architecture"] = self._count_by_architecture(models)

    # ------------------------------------------------------------------
    # Public helpers expected by existing API code
    # ------------------------------------------------------------------

    @property
    def champion_models(self) -> List[Dict[str, Any]]:
        """Top-N models sorted by ELO."""
        top_sorted = sorted(self.models.values(), key=lambda m: m.elo_rating, reverse=True)
        return [self._simplify_model(m) for m in top_sorted[:5]]

    @property
    def genealogy(self) -> Dict[str, Any]:
        return self._genealogy

    # ------------------------------------------------------------------
    # Battle scheduling & recording
    # ------------------------------------------------------------------

    def next_pair(self) -> Tuple[AIModel, AIModel]:
        """Return two distinct models to battle – simple random for now."""
        if len(self.models) < 2:
            raise RuntimeError("Need at least 2 models to create a battle pair")
        return tuple(random.sample(list(self.models.values()), 2))  # type: ignore[return-value]

    def record_battle(self, winner: AIModel, loser: AIModel, confidence: float = 0.8) -> None:
        """Update ELO ratings based on the winner/loser result."""
        expected_w = self._expected_score(winner.elo_rating, loser.elo_rating)
        expected_l = 1 - expected_w

        k = self.elo_k * confidence  # scale by voter confidence
        winner.elo_rating += k * (1 - expected_w)
        loser.elo_rating += k * (0 - expected_l)

        # Update genealogy statistics
        self._genealogy["statistics"]["total_evolved"] = len(self.models)
        self._genealogy["statistics"]["by_architecture"] = self._count_by_architecture(self.models.values())

    def add_models(self, new_models: List[AIModel]) -> None:
        for m in new_models:
            if m.id not in self.models:
                self.models[m.id] = m
        # refresh stats
        self._genealogy["models"] = [self._simplify_model(m) for m in self.models.values()]

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    @staticmethod
    def _simplify_model(model: AIModel) -> Dict[str, Any]:
        return {
            "id": model.id,
            "name": getattr(model, "name", model.id),
            "architecture": getattr(model, "architecture", "unknown"),
            "elo_rating": round(float(model.elo_rating), 2),
            "tier": getattr(model, "tier", "Unknown"),
            "generation": getattr(model, "generation", 1),
        }

    @staticmethod
    def _count_by_architecture(models) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for m in models:
            arch = getattr(m, "architecture", "unknown")
            counts[arch] = counts.get(arch, 0) + 1
        return counts 