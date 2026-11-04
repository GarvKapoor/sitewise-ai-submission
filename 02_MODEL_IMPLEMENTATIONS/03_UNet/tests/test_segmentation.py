"""
Tests for backend/services/segmentation.py — percentage, dominant-class,
confidence and validation logic. Runs without TensorFlow.

Edge cases covered:
  • dominant class is the ACTUAL maximum (the historic bug: inserting the
    current class before taking the max let the first class win)
  • tie handling keeps the lowest class_id
  • probabilities are read per-pixel through the (H*W, C)/(H*W, 1) flatten
  • malformed masks / NaNs / empty masks are rejected, not silently accepted
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from config import model_config as cfg  # noqa: E402
from services import segmentation as seg  # noqa: E402


def test_dominant_is_actual_maximum():
    # All pixels class 3 (Forest) → dominant must be Forest, not Urban(0)
    mask = np.full((4, 4), 3, dtype=np.uint8)
    res = seg.analyse(mask, None)
    assert res["dominant_class"]["class_id"] == 3
    assert res["dominant_class"]["name"] == "Forest"
    assert res["dominant_class"]["percent"] == 100.0


def test_dominant_with_mixed_counts():
    mask = np.full((4, 4), 3, dtype=np.uint8)  # start: all Forest (16 px)
    mask[2:, 2:] = 4                          # Water 4 px → Forest 12 px
    res = seg.analyse(mask, None)
    assert res["dominant_class"]["class_id"] == 3
    assert res["dominant_class"]["name"] == "Forest"
    forest = [c for c in res["class_percentages"] if c["class_id"] == 3][0]
    assert forest["percent"] == pytest.approx(75.0, abs=0.1)


def test_dominant_tie_keeps_lowest_class_id():
    # 8 Urban + 8 Forest → tie; configured order keeps Urban (id 0)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, :4] = 0
    mask[1:, 2:] = 3
    mask[1:, :2] = 0
    mask[0, 2:] = 3
    assert (mask == 0).sum() == 8 and (mask == 3).sum() == 8
    res = seg.analyse(mask, None)
    assert res["dominant_class"]["class_id"] == 0


def test_percentages_sum_to_100():
    rng = np.random.default_rng(7)
    mask = rng.integers(0, 7, size=(10, 10)).astype(np.uint8)
    res = seg.analyse(mask, None)
    assert pytest.approx(sum(e["percent"] for e in res["class_percentages"]), abs=0.3) == 100.0


def test_confidence_flatten_retrieves_assigned_class_probability():
    # probs[:, :, 0] all 0.9, probs[:, :, 1] all 0.1; mask all class 1 → mean 0.10
    probs = np.zeros((2, 2, 7), dtype=np.float32)
    probs[..., 0] = 0.9
    probs[..., 1] = 0.1
    mask = np.ones((2, 2), dtype=np.uint8)  # every pixel predicted class 1
    res = seg.analyse(mask, probs)
    assert res["confidence"]["mean"] == pytest.approx(10.0, abs=0.05)
    assert res["confidence"]["min"] == pytest.approx(10.0, abs=0.05)
    assert res["confidence"]["max"] == pytest.approx(10.0, abs=0.05)


def test_confidence_varies_per_pixel():
    probs = np.zeros((1, 2, 7), dtype=np.float32)
    probs[0, 0, 0] = 0.95
    probs[0, 1, 0] = 0.45
    other = (1.0 - probs) / 6.0
    probs = np.where(np.eye(1, 7, 0, dtype=bool), probs, other)
    mask = np.zeros((1, 2), dtype=np.uint8)
    res = seg.analyse(mask, probs)
    assert res["confidence"]["min"] < res["confidence"]["max"]
    assert res["confidence"]["mean"] == pytest.approx(70.0, abs=0.5)


def test_empty_mask_rejected():
    with pytest.raises(ValueError, match="empty"):
        seg.analyse(np.zeros((0, 0), dtype=np.uint8), None)


def test_bad_dimensions_rejected():
    with pytest.raises(ValueError, match="2-D"):
        seg.analyse(np.zeros((2, 2, 3), dtype=np.uint8), None)


def test_mask_out_of_range_indices_rejected():
    # An explicit out-of-range class index (255 > 6) must be rejected with a
    # clear message — not silently folded into percentages that sum ≠ 100%.
    with pytest.raises(ValueError, match="invalid class indices"):
        seg.analyse(np.full((2, 2), 255, dtype=np.uint8), None)


def test_nan_probs_rejected():
    probs = np.zeros((2, 2, 7), dtype=np.float32)
    probs[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        seg.analyse(np.zeros((2, 2), dtype=np.uint8), probs)


def test_prob_shape_mismatch_rejected():
    probs = np.zeros((3, 3, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match mask"):
        seg.analyse(np.zeros((2, 2), dtype=np.uint8), probs)


def test_rgb_mask_palette_matches_config():
    lut = np.asarray(seg._colors_for_classes(), dtype=np.uint8)
    assert lut.shape == (7, 3)
    assert lut[0].tolist() == [0xEF, 0x44, 0x44]  # Urban red
    assert lut[-1].tolist() == [0x6B, 0x72, 0x80]  # Unknown slate


def test_predict_validations_trigger_without_model():
    m = seg.SegmentationModel()
    assert not m.connected
    with pytest.raises(RuntimeError, match="not loaded"):
        m.predict(np.zeros((1, 4, 4, 3), dtype=np.float32))


def test_demo_mask_is_deterministic_and_in_range():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    m1 = seg.demo_heuristic_mask(img)
    m2 = seg.demo_heuristic_mask(img)
    assert np.array_equal(m1, m2)  # never randomised
    assert m1.min() >= 0 and m1.max() <= 6
