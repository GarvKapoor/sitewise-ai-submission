"""
Tests for backend/services/preprocessing.py — U-Net tensor preparation and
the eight RGB visual features. Runs without TensorFlow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from config import model_config as cfg  # noqa: E402
from services import preprocessing as prep  # noqa: E402


def test_preprocess_shape_and_range():
    rgb = np.random.default_rng(1).integers(0, 256, size=(50, 30, 3), dtype=np.uint8)
    tensor, target = prep.preprocess_for_unet(rgb)
    h, w = cfg.INPUT_SIZE
    assert tensor.shape == (1, h, w, 3)
    assert target == (h, w)
    assert tensor.dtype == np.float32
    assert (tensor >= 0).all() and (tensor <= 1).all()


def test_preprocess_is_deterministic():
    rgb = np.random.default_rng(2).integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    a, _ = prep.preprocess_for_unet(rgb)
    b, _ = prep.preprocess_for_unet(rgb)
    assert np.array_equal(a, b)


FEATURE_KEYS = [
    "mean_red", "mean_green", "mean_blue", "avg_brightness",
    "vegetation_ratio", "water_ratio", "builtup_ratio", "road_indicator",
]


def test_extract_features_returns_all_eight():
    rgb = np.random.default_rng(5).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    features, refined = prep.extract_visual_features(rgb, None, cfg.SEGMENTATION_CLASSES, False)
    assert [f["key"] for f in features] == FEATURE_KEYS
    assert refined is False
    for f in features:
        assert 0.0 <= f["fraction"] <= 1.0
        f["display"]  # present for UI


def test_features_in_expected_value_ranges():
    rgb = np.full((10, 10, 3), 180, dtype=np.uint8)  # flat bright grey
    features, _ = prep.extract_visual_features(rgb, None, cfg.SEGMENTATION_CLASSES, False)
    by_key = {f["key"]: f for f in features}
    assert by_key["avg_brightness"]["fraction"] == pytest.approx(180 / 255, abs=1e-3)
    assert by_key["mean_red"]["fraction"] == pytest.approx(180 / 255, abs=1e-3)
    # flat grey has no vegetation/water/built-up colour dominance
    assert by_key["vegetation_ratio"]["fraction"] == pytest.approx(0.0, abs=1e-6)


def test_segmentation_refinement_uses_class_percentages():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    pct = [
        {"name": n, "percent": 100.0 if n == "Forest" else 0.0}
        for n in cfg.SEGMENTATION_CLASSES
    ]
    features, refined = prep.extract_visual_features(rgb, pct, cfg.SEGMENTATION_CLASSES, True)
    by_key = {f["key"]: f for f in features}
    assert refined is True
    assert by_key["vegetation_ratio"]["fraction"] == pytest.approx(1.0, abs=1e-6)
    assert by_key["builtup_ratio"]["fraction"] == pytest.approx(0.0, abs=1e-6)


def test_refinement_ignored_without_compatible_mapping():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    features, refined = prep.extract_visual_features(
        rgb, [{"name": "Weird", "percent": 100.0}], ["Weird"], True
    )
    assert refined is False  # incompatible class mapping ⇒ colour heuristics only
