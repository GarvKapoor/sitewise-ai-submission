"""
Image preprocessing for the U-Net pipeline + RGB visual feature extraction.

Steps (per product spec):
1. RGB conversion
2. Resize to the input dimensions expected by the trained U-Net
3. Pixel normalisation (0-1)
4. Tensor conversion (HWC -> 1,H,W,C)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import model_config as cfg

PREPROCESSING_STEPS = [
    "RGB conversion",
    "Resize to U-Net input size",
    "Pixel normalisation (0-1)",
    "Tensor conversion (1, H, W, C)",
    "Model inference",
]


def load_uploaded_image(stream: Any) -> Tuple[np.ndarray, int, int]:
    """Decode an uploaded file to an RGB uint8 numpy array plus its original size."""
    img = Image.open(stream).convert("RGB")
    arr = np.asarray(img)
    return arr, int(img.width), int(img.height)


def preprocess_for_unet(rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return (tensor, target_size): normalised float32 tensor shaped (1,H,W,C)."""
    target_h, target_w = cfg.INPUT_SIZE
    img = Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR)
    norm = np.asarray(img, dtype=np.float32) / 255.0
    tensor = norm[None, ...]  # (1, H, W, C)
    return tensor, (target_h, target_w)


# ---------------------------------------------------------------------------
# Eight RGB-based visual features
# ---------------------------------------------------------------------------

def _as_percent_fraction(v: float) -> float:
    """pct(v, 255) -> 0-1."""
    return min(max(v / 255.0, 0.0), 1.0)


def extract_visual_features(
    rgb: np.ndarray,
    class_percentages: Optional[List[Dict[str, Any]]],
    class_names: List[str],
    seg_available: bool,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Extract the eight required features.

    Returns (features, refined_with_segmentation). Where the trained class
    mapping is compatible (a connected U-Net whose class list matches the
    configured segmentation classes), vegetation / water / built-up ratios are
    refined using segmentation output. Otherwise they are computed from raw
    colour heuristics and reported as such.
    """
    rgb_f = rgb.astype(np.float32)
    height, width, _ = rgb_f.shape
    total = float(height * width)

    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    mean_r = float(r.mean())
    mean_g = float(g.mean())
    mean_b = float(b.mean())
    brightness = float(rgb_f.mean())

    # Colour heuristics: dominant-parity masks
    veg_ratio = float(((g > r) & (g > b)).sum()) / total
    water_ratio = float(((b > r) & (b > g)).sum()) / total
    built_ratio = float(
        ((np.abs(r - g) < 18) & (np.abs(g - b) < 18) & (r < 200)).sum()
    ) / total
    # Road indicator proxy: share of bright, low-saturation (grayish) pixels
    road_indicator = float(
        ((np.abs(r - g) < 12) & (np.abs(g - b) < 12) & (r > 110)).sum()
    ) / total

    refined = False
    if seg_available and class_percentages is not None and class_names == cfg.SEGMENTATION_CLASSES:
        pct_by_name = {c["name"]: float(c["percent"]) / 100.0 for c in class_percentages}

        veg_names = ["Forest", "Rangeland", "Agriculture"]
        water_names = ["Water"]
        built_names = ["Urban"]
        veg_ratio = sum(pct_by_name.get(n, 0.0) for n in veg_names)
        water_ratio = sum(pct_by_name.get(n, 0.0) for n in water_names)
        built_ratio = sum(pct_by_name.get(n, 0.0) for n in built_names)
        road_indicator = max(road_indicator, pct_by_name.get("Urban", 0.0) * 0.6)
        refined = True

    def feature(key: str, name: str, value: float, intensity: bool) -> Dict[str, Any]:
        fraction = _as_percent_fraction(value) if intensity else float(value)
        return {
            "key": key,
            "name": name,
            "value": round(value, 2),
            "fraction": round(fraction, 4),
            "display": f"{fraction * 100:.1f}",
            "unit": "%",
        }

    return (
        [
            feature("mean_red", "Mean Red Intensity", mean_r, True),
            feature("mean_green", "Mean Green Intensity", mean_g, True),
            feature("mean_blue", "Mean Blue Intensity", mean_b, True),
            feature("avg_brightness", "Average Brightness", brightness, True),
            feature("vegetation_ratio", "Vegetation Ratio", veg_ratio, False),
            feature("water_ratio", "Water Ratio", water_ratio, False),
            feature("builtup_ratio", "Built-up Ratio", built_ratio, False),
            feature("road_indicator", "Road Indicator", road_indicator, False),
        ],
        refined,
    )
