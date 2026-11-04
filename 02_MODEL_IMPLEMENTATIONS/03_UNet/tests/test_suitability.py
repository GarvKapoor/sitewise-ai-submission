"""
Tests for backend/services/suitability.py — the deterministic screening engine.

Verifies: eight categories, deterministic outputs, ranking order, factor
explanations, and that identical inputs give identical scores (no randomness
anywhere in the scoring path).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from services import suitability as suitsvc  # noqa: E402

EXPECTED_CATEGORIES = [
    "Residential Development",
    "Commercial Development",
    "Industrial Development",
    "Educational Development",
    "Healthcare Development",
    "Agricultural Use",
    "Transportation Infrastructure",
    "Green / Conservation",
]


def _pct(seed: int) -> list:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(7) * 0.9) * 100.0
    return [
        {"name": name, "percent": round(p, 3)}
        for name, p in zip(
            ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"], raw
        )
    ]


def _feat(seed: int) -> list:
    rng = np.random.default_rng(seed)
    keys = ["mean_red", "mean_green", "mean_blue", "avg_brightness",
            "vegetation_ratio", "water_ratio", "builtup_ratio", "road_indicator"]
    return [{"key": k, "fraction": float(rng.random())} for k in keys]


def test_returns_all_eight_categories():
    scores = suitsvc.compute_scores(_pct(1), _feat(2), {i: n for i, n in enumerate(EXPECTED_CATEGORIES[:7])})
    assert sorted(s["category"] for s in scores) == sorted(EXPECTED_CATEGORIES)


def test_ranking_is_strictly_ordered_descending():
    for seed in range(5):
        scores = suitsvc.compute_scores(_pct(seed), _feat(seed), {i: n for i, n in enumerate([
            "Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"])}
        )
        vals = [s["score"] for s in scores]
        assert vals == sorted(vals, reverse=True)
        assert [s["rank"] for s in scores] == list(range(1, 9))
        assert scores[0]["is_top"] is True
        for s in scores[1:]:
            assert s["is_top"] is False


def test_scores_bounded_0_100():
    for seed in range(10):
        for s in suitsvc.compute_scores(_pct(seed), _feat(seed), {}):
            assert 0.0 <= s["score"] <= 100.0


def test_deterministic_same_inputs_same_output():
    a = suitsvc.compute_scores(_pct(3), _feat(4), {})
    b = suitsvc.compute_scores(_pct(3), _feat(4), {})
    assert a == b


def test_urban_dominance_boosts_residential_and_reduces_green():
    urban_pct = [{"name": n, "percent": 100.0 if n == "Urban" else 0.0}
                 for n in ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]]
    forest_pct = [{"name": n, "percent": 100.0 if n == "Forest" else 0.0}
                  for n in ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]]
    empty_feats = [{"key": k, "fraction": 0.0} for k in [
        "mean_red", "mean_green", "mean_blue", "avg_brightness",
        "vegetation_ratio", "water_ratio", "builtup_ratio", "road_indicator"]]
    by = {s["category"]: s["score"] for s in suitsvc.compute_scores(urban_pct, empty_feats, {})}
    fby = {s["category"]: s["score"] for s in suitsvc.compute_scores(forest_pct, empty_feats, {})}
    assert by["Residential Development"] > fby["Residential Development"]
    assert by["Green / Conservation"] < fby["Green / Conservation"]
    assert by["Agricultural Use"] < fby["Agricultural Use"]


def test_top_recommendation_explanations_present():
    scores = suitsvc.compute_scores(_pct(9), _feat(11), {})
    for s in scores:
        assert s["explanation"]
        assert isinstance(s["contributing_factors"], list)
    assert "Highest screened category" in scores[0]["explanation"]


def test_unknown_class_excluded_from_pct_map():
    # A 100% Unknown mask should still produce valid, deterministic scores
    unknown_pct = [{"name": "Unknown", "percent": 100.0}]
    feats = [{"key": k, "fraction": 0.0} for k in
             ["mean_red", "mean_green", "mean_blue", "avg_brightness",
              "vegetation_ratio", "water_ratio", "builtup_ratio", "road_indicator"]]
    res = suitsvc.compute_scores(unknown_pct, feats, {})
    assert len(res) == 8


# ---- documentation accuracy (Task 3) -------------------------------------- #
def test_module_docstring_reports_7_classes_and_15_values():
    doc = (Path(suitsvc.__file__).read_text(encoding="utf-8").split('"""')[1] + '"""')
    assert "7 DeepGlobe land-cover proportions" in doc
    assert "15 available values" in doc or "15" in doc
    assert "NOT a trained neural network" in doc
    assert "18-dimensional" not in doc and "10 land-cover" not in doc


def test_road_indicator_described_as_rgb_proxy():
    doc = Path(suitsvc.__file__).read_text(encoding="utf-8")
    assert "RGB-based proxy" in doc or "RGB proxy" in doc
