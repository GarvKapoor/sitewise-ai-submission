"""
U-Net segmentation service.

Loads a trained TensorFlow/Keras U-Net (.keras or .h5), runs pixel-wise
inference and turns the softmax output into:
  - an indexed segmentation mask (argmax per pixel)
  - per-class pixel counts / percentages (actual model output, never random)
  - dominant class
  - mean/min/max per-pixel confidence (softmax probability of assigned class)
  - a colour-coded mask PNG and an alpha-blended overlay PNG (base64)

Magic, ground-truth or random values are never generated here: with no
connected model the service reports itself as unavailable and the Flask app
switches to clearly-labelled demonstration mode (deterministic colour
heuristics, always flagged as demonstration data — never presented as
U-Net output).
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import model_config as cfg


class SegmentationModel:
    """Thin wrapper around a trained Keras U-Net."""

    def __init__(self) -> None:
        self.model: Any = None
        self.source_path: str = ""
        self.error: str = ""

    @property
    def connected(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        if self.model is not None:
            return True
        try:
            import tensorflow as tf  # noqa: PLC0415 — deferred import
        except Exception as e:  # pragma: no cover - environment dependent
            self.error = f"TensorFlow unavailable: {e}"
            return False

        for cand in cfg.UNET_MODEL_CANDIDATES:
            if cand.exists():
                try:
                    self.model = tf.keras.models.load_model(str(cand))
                    self.source_path = str(cand)
                    self.error = ""
                    return True
                except Exception as e:  # pragma: no cover
                    self.error = f"Failed to load {cand.name}: {e}"
                    return False
        self.error = "No U-Net model file found (looked for unet_model.keras / unet_model.h5 in models/)"
        return False

    def predict(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        tensor: (1,H,W,C) float32 normalised.
        Returns (argmax_mask (H,W) uint8, probs (H,W,num_classes) float32).
        Validates the U-Net output shape and values; raises ValueError on
        malformed output instead of returning invalid artefacts.
        """
        if self.model is None:
            raise RuntimeError("U-Net model not loaded")
        raw = self.model.predict(tensor, verbose=0)
        probs = np.asarray(raw, dtype=np.float32)
        if probs.ndim != 4 or probs.shape[0] != 1:
            raise ValueError(
                "U-Net output must be (1, H, W, num_classes) — got shape "
                f"{probs.shape}"
            )
        num_classes = len(cfg.SEGMENTATION_CLASSES)
        if probs.shape[-1] != num_classes:
            raise ValueError(
                f"U-Net output must have exactly {num_classes} channels (one per "
                f"DeepGlobe class) — got {probs.shape[-1]}. The output channel"
                " order must match config/model_config.py CLASS_NAMES."
            )
        if probs.shape[1] != tensor.shape[1] or probs.shape[2] != tensor.shape[2]:
            raise ValueError(
                "U-Net output spatial dims" + f" {probs.shape[1:3]} do not match "
                f"the preprocessed input {tensor.shape[1:3]}."
            )
        if not np.isfinite(probs).all():
            raise ValueError("U-Net output contains NaN / inf values — check the trained model.")
        channel_sums = probs.sum(axis=-1)
        if not np.allclose(channel_sums, 1.0, atol=1e-2):
            raise ValueError(
                "U-Net output does not look softmax-normalised "
                f"(channel sums {channel_sums.min():.3f}-{channel_sums.max():.3f})."
            )
        mask = np.argmax(probs[0], axis=-1).astype(np.uint8)
        mask = mask if mask.shape == probs.shape[1:3] else mask.reshape(probs.shape[1:3])
        return mask, probs[0]


def _colors_for_classes() -> List[Tuple[int, int, int]]:
    def hex2rgb(hx: str) -> Tuple[int, int, int]:
        hx = hx.lstrip("#")
        return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    return [hex2rgb(c) for c in cfg.SEGMENTATION_COLORS]


def to_png_dataurl(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rgb_mask(mask: np.ndarray) -> np.ndarray:
    lut = np.array(_colors_for_classes(), dtype=np.uint8)
    return lut[mask]


def overlay(original_resized: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    ov = rgb_mask(mask).astype(np.float32)
    src = original_resized.astype(np.float32)
    blended = ov * alpha + src * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def analyse(mask: np.ndarray, probs: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Turn an argmax mask (+ softmax probs) into the JSON-ready structure.

    Percentages are computed from actual predicted pixel labels only:
        class % = (pixels assigned to the class / total valid pixels) × 100
    Confidence is the probability of each pixel's predicted class, retrieved
    by flattening the (H, W, C) tensor to (H*W, C) and the mask to (H*W, 1).
    No fabricated or default confidence values are generated.
    """
    mask = np.asarray(mask)
    num_classes = len(cfg.SEGMENTATION_CLASSES)
    if mask.ndim != 2:
        raise ValueError(f"Segmentation mask must be 2-D (H, W) — got shape {mask.shape}")
    if mask.size == 0:
        raise ValueError("Segmentation mask is empty (no pixels to analyse)")
    if mask.max() >= num_classes:
        raise ValueError(
            f"Segmentation mask contains invalid class indices (max {int(mask.max())} "
            f"> class range 0..{num_classes - 1})"
        )
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape[0] != mask.shape[0] or probs.shape[1] != mask.shape[1]:
            raise ValueError(
                f"Probability tensor {probs.shape[:2]} does not match mask {mask.shape}"
            )
        if not np.isfinite(probs).all():
            raise ValueError("Probability tensor contains NaN/inf values")

    total = int(mask.size)
    counts = np.bincount(mask.reshape(-1), minlength=num_classes)
    if counts.max() == 0:
        raise ValueError("Mask contains no valid class indices")
    max_px = int(counts.max())  # actual maximum pixel count across all classes

    percentages: List[Dict[str, Any]] = []
    dominant = None
    for class_id, name in enumerate(cfg.SEGMENTATION_CLASSES):
        px = int(counts[class_id])
        pct = px / max(total, 1)
        entry = {
            "class_id": class_id,
            "name": name,
            "color": cfg.SEGMENTATION_COLORS[class_id],
            "percent": round(pct * 100, 2),
            "pixel_count": px,
        }
        percentages.append(entry)
        # Dominant class = the class with the strictly largest pixel count;
        # ties keep the first class in configured order (lowest class_id first).
        if dominant is None and px == max_px:
            dominant = entry
        elif dominant is not None and px > max_px:  # defensive; max_px already global
            dominant = entry

    assert abs(sum(e["percent"] for e in percentages) - 100.0) < 0.5, (
        "class percentages must sum to ~100%"
    )

    confidence = None
    if probs is not None:
        # probs: (H, W, C) → (H*W, C); mask: (H, W) → (H*W, 1). take_along_axis
        # retrieves the probability each pixel's predicted class got.
        probs2d = probs.reshape(-1, probs.shape[-1])
        flat = mask.reshape(-1, 1).astype(np.int64)
        assigned_probs = np.take_along_axis(probs2d, flat, axis=1).reshape(-1)
        confidence = {
            "mean": round(float(assigned_probs.mean()) * 100, 2),
            "min": round(float(assigned_probs.min()) * 100, 2),
            "max": round(float(assigned_probs.max()) * 100, 2),
        }

    result: Dict[str, Any] = {
        "input_size": [int(mask.shape[0]), int(mask.shape[1])],
        "class_percentages": percentages,
        "dominant_class": dominant,
        "confidence": confidence,
        "total_pixels": total,
    }
    return result


# ---------------------------------------------------------------------------
# Demonstration-mode segmentation (labelled everywhere, never model output)
# ---------------------------------------------------------------------------

def demo_heuristic_mask(resized_rgb: np.ndarray) -> np.ndarray:
    """
    Deterministic colour heuristics on the resized image → DeepGlobe-class
    indices. Used ONLY to generate clearly-labelled demonstration data when
    the real U-Net is not connected. Never randomised.
    """
    a = resized_rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = np.full(a.shape[:2], cfg.SEGMENTATION_CLASSES.index("Unknown"), dtype=np.uint8)

    # Water: dominant blue/cyan
    water = (b > 110) & (b > r + 25) & (b > g + 10)
    # Forest: dark, dominant green
    forest = (~water) & (g > 70) & (g > r + 20) & (g > b + 20) & (g <= 150)
    # Agriculture: bright, dominant green
    agriculture = (~water & ~forest) & (g > 110) & (g > r + 15) & (g > b + 15)
    # Rangeland: yellow/tan/dry (red≥green>blue)
    rangeland = (~water & ~forest & ~agriculture) & (r > 110) & (r >= g) & (g > b + 10) & (r > b + 25)
    # Urban: mid-bright low-saturation grey
    urban = (~water & ~forest & ~agriculture & ~rangeland) & \
        (np.abs(r - g) < 22) & (np.abs(g - b) < 22) & (r > 90) & (r < 205)
    # Barren: bright sandy/rocky
    barren = (~water & ~forest & ~agriculture & ~rangeland & ~urban) & (r > 165) & (g > 140) & (b > 110) & (r >= g) & (g > b)

    order = [
        (water, "Water"),
        (forest, "Forest"),
        (agriculture, "Agriculture"),
        (rangeland, "Rangeland"),
        (urban, "Urban"),
        (barren, "Barren"),
    ]
    for cond, name in order:
        mask[cond] = cfg.SEGMENTATION_CLASSES.index(name)
    return mask
