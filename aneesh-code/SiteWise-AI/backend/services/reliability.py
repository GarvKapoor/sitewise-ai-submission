"""
Image reliability service (separate CNN — independent of the U-Net).

Loads reliability_model.keras / .h5 and scores how much the input resembles
clean vs degraded imagery on a 0-100 scale, checked against the project
threshold (55%, configurable in config/model_config.py).

When no trained reliability model is connected this module exposes a
transparent, labelled heuristic fallback (brightness/sharpness/entropy sketch)
that is never presented to the user as model output — the API marks such
results with model_used=False and the frontend labels them.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from config import model_config as cfg


class ReliabilityModel:
    def __init__(self) -> None:
        self.model = None
        self.source_path: str = ""
        self.error: str = ""

    @property
    def connected(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        if self.model is not None:
            return True
        try:
            import tensorflow as tf  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            self.error = f"TensorFlow unavailable: {e}"
            return False

        for cand in cfg.RELIABILITY_MODEL_CANDIDATES:
            if cand.exists():
                try:
                    self.model = tf.keras.models.load_model(str(cand))
                    self.source_path = str(cand)
                    self.error = ""
                    return True
                except Exception as e:  # pragma: no cover
                    self.error = f"Failed to load {cand.name}: {e}"
                    return False
        self.error = "No reliability model file found (models/reliability_model.keras or .h5)"
        return False

    @property
    def output_mode(self) -> str:
        """Resolved output mode: "sigmoid" or "softmax2" ("auto" discovered at load)."""
        mode = cfg.RELIABILITY_OUTPUT_MODE
        if mode in ("sigmoid", "softmax2"):
            return mode
        shape = getattr(self.model, "output_shape", None) if self.model else None
        if shape and len(shape) >= 2 and shape[-1] == 2:
            return "softmax2"
        return "sigmoid"

    def score(self, rgb: np.ndarray) -> int:
        """
        Model-backed reliability score, 0-100.

        Expected model output (documented contract, Task 5):
          sigmoid   — single neuron: P(clean). Score = P(clean) × 100.
          softmax2  — two neurons (C={degraded, clean} by default): the value at
                      RELIABILITY_CLEAN_INDEX is P(clean).
        The clean/degraded channel mapping is configurable via
        SITEWISE_RELIABILITY_OUTPUT / SITEWISE_RELIABILITY_CLEAN_INDEX.
        Input is resized/replicated to the model's declared input shape;
        malformed model output raises instead of fabricating a score.
        """
        if self.model is None:
            raise RuntimeError("Reliability model not loaded")

        arr = rgb
        input_shape = getattr(self.model, "input_shape", None)
        if input_shape and len(input_shape) == 4:
            _, h, w, c = input_shape
            if not h or not w:
                raise ValueError("Reliability model declares invalid input dims (H/W = 0)")
            img = Image.fromarray(arr)
            if c == 1:
                img = img.convert("L")
            arr = np.asarray(img.resize((w, h), Image.BILINEAR))
        x = arr.astype(np.float32) / 255.0
        if x.ndim == 2:
            x = x[..., None]
        if x.shape[-1] == 1 and (input_shape or (None, None, None, 3))[3] == 3:
            x = np.repeat(x, 3, axis=-1)
        x = x[None, ...]
        out = np.asarray(self.model.predict(x, verbose=0), dtype=np.float32)
        out = out.reshape(-1)
        if not np.isfinite(out).all():
            raise ValueError("Reliability model output contains NaN/inf values")

        mode = self.output_mode
        if mode == "softmax2":
            if out.size != 2:
                raise ValueError(
                    f"softmax2 output expected 2 channels — got {out.size} values"
                )
            idx = max(0, min(1, cfg.RELIABILITY_CLEAN_INDEX))
            p = float(out[idx])
        else:  # sigmoid
            if out.size != 1:
                raise ValueError(
                    f"sigmoid output expected 1 value — got {out.size} values; "
                    "set SITEWISE_RELIABILITY_OUTPUT=softmax2 if your model has 2 channels"
                )
            p = float(out[0])
        return int(round(min(max(p, 0.0), 1.0) * 100))


# ---------------------------------------------------------------------------
# Labelled heuristic fallback (NOT model output)
# ---------------------------------------------------------------------------

def heuristic_score(rgb: np.ndarray) -> int:
    """
    Transparent quality sketch: penalises extreme brightness, extreme
    darkness, and blur (low gradient energy). Used only when the trained
    reliability CNN is not connected; always reported as model_used=False.
    """
    arr = np.asarray(Image.fromarray(rgb).resize((256, 256), Image.BILINEAR)).astype(np.float32)
    brightness = arr.mean() / 255.0
    gray = arr @ np.array([0.299, 0.587, 0.114])

    gy, gx = np.gradient(gray)
    grad_energy = float(np.sqrt(gx * gx + gy * gy).mean())

    focus = min(grad_energy / 12.0, 1.0)          # sharpness component
    expo = 1.0 - abs(brightness - 0.5) * 2.0      # exposure component
    return int(round(np.clip(100 * (0.6 * expo + 0.4 * focus), 0, 100)))


def assess(rgb: np.ndarray, model: ReliabilityModel) -> dict[str, Any]:
    state: dict[str, Any] | None = None
    inference_failed = False
    if model.connected:
        try:
            s = model.score(rgb)
            state = {
                "score": s,
                "status": "clean" if s >= cfg.RELIABILITY_THRESHOLD else "degraded",
                "model_used": True,
                "assessed_by": "reliability_cnn",
                "threshold": cfg.RELIABILITY_THRESHOLD,
                "passed": s >= cfg.RELIABILITY_THRESHOLD,
            }
        except Exception as e:  # pragma: no cover — real models only
            model.error = f"Inference failed: {e}"
            state = None
            inference_failed = True

    s = heuristic_score(rgb)
    fallback = {
        "score": s,
        "status": "clean" if s >= cfg.RELIABILITY_THRESHOLD else "degraded",
        "model_used": False,
        "assessed_by": "heuristic_fallback",
        "threshold": cfg.RELIABILITY_THRESHOLD,
        "passed": s >= cfg.RELIABILITY_THRESHOLD,
        "note": (
            "The trained reliability CNN is not connected — this is a labelled "
            "heuristic fallback, NOT a CNN output."
        ),
    }
    if inference_failed:
        fallback["inference_error"] = model.error  # set inside the except above
    return state if state is not None else fallback
