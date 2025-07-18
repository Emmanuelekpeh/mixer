#!/usr/bin/env python3
"""Unit tests for EvolutionEngine (backend).

These tests verify that the minimal EvolutionEngine implementation
behaves correctly and replaces previous DummyEvolutionEngine logic.
"""

from datetime import datetime
from dataclasses import dataclass
import pathlib
import sys

import pytest

# Ensure project root is on sys.path so we can import the backend package
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Import after path adjustment
from tournament_webapp.backend.evolution_engine import EvolutionEngine


@dataclass
class FakeModel:  # Lightweight stand-in for database AIModel
    id: str
    name: str
    architecture: str = "cnn"
    elo_rating: float = 1200.0
    tier: str = "Amateur"
    generation: int = 1
    created_at: datetime = datetime.utcnow()


def create_models(n: int = 4):
    return [FakeModel(id=f"m{i}", name=f"Model{i}") for i in range(1, n + 1)]


def test_champion_models_sorted():
    models = create_models()
    # Pre-assign different ELOs to check sorting
    models[0].elo_rating = 1400
    models[1].elo_rating = 1300
    models[2].elo_rating = 1250
    engine = EvolutionEngine(models)

    champions = engine.champion_models
    assert len(champions) == 4  # all models returned (less than 5)
    elo_list = [m["elo_rating"] for m in champions]
    assert elo_list == sorted(elo_list, reverse=True), "Champions not sorted by ELO desc"


def test_next_pair_gives_two_distinct_models():
    engine = EvolutionEngine(create_models(3))
    a, b = engine.next_pair()
    assert a.id != b.id, "next_pair returned duplicate models"


def test_elo_update_after_battle():
    m1, m2 = create_models(2)
    engine = EvolutionEngine([m1, m2])

    old_m1 = m1.elo_rating
    old_m2 = m2.elo_rating

    engine.record_battle(winner=m1, loser=m2, confidence=1.0)

    assert m1.elo_rating > old_m1, "Winner ELO did not increase"
    assert m2.elo_rating < old_m2, "Loser ELO did not decrease"


def test_genealogy_statistics_update():
    models = create_models(3)
    engine = EvolutionEngine(models)
    stats = engine.genealogy["statistics"]
    assert stats["total_evolved"] == 3
    assert stats["by_architecture"]["cnn"] == 3

    # Add a new model via set_models and ensure stats refresh
    new_model = FakeModel(id="m4", name="Model4", architecture="transformer")
    engine.set_models(models + [new_model])
    stats2 = engine.genealogy["statistics"]
    assert stats2["total_evolved"] == 4
    assert stats2["by_architecture"]["transformer"] == 1 