"""
SiteWise AI — central model / pipeline configuration (DeepGlobe standard).

Everything that depends on *your* trained models and *your* segmentation
dataset is configurable here. Changing input size, class names, colours,
thresholds or dataset choices requires no frontend change: the frontend
discovers these through GET /health.

Active dataset: DeepGlobe Land Cover (7 land-cover classes with ground-truth
pixel-level masks). Class order MUST match the channel order of the trained
U-Net's output layer (argmax index -> class).
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Model files — drop your trained models here.
# .keras and .h5 are both supported; the loader checks every candidate path.
# ---------------------------------------------------------------------------
UNET_MODEL_CANDIDATES = [
    MODELS_DIR / "unet_model.keras",
    MODELS_DIR / "unet_model.h5",
]

RELIABILITY_MODEL_CANDIDATES = [
    MODELS_DIR / "reliability_model.keras",
    MODELS_DIR / "reliability_model.h5",
]

# Optional offline evaluation results file (Upside: nothing fabricated —
# metrics appear on the website only if this file exists after real training).
# Expected structure:
# {
#   "unet": {"training_accuracy": 0.94, "validation_loss": 0.31,
#            "mean_iou": 0.71, "dice": 0.78}
# }
EVALUATION_FILE = MODELS_DIR / "evaluation.json"

# Threshold for the reliability CNN: score >= threshold => clean / pass.
# 55% retained from the original SiteWise AI project.
RELIABILITY_THRESHOLD = float(os.environ.get("SITEWISE_RELIABILITY_THRESHOLD", "55"))

# ---------------------------------------------------------------------------
# Segmentation input size — MUST match the trained U-Net.
# Override with SITEWISE_INPUT_SIZE="H,W" (e.g. "128,128").
# ---------------------------------------------------------------------------
def _parse_input_size():
    raw = os.environ.get("SITEWISE_INPUT_SIZE", "128,128")
    try:
        h, w = (int(p) for p in raw.split(","))
        return (h, w)
    except ValueError:
        return (128, 128)


INPUT_SIZE = _parse_input_size()

# ---------------------------------------------------------------------------
# DeepGlobe land-cover classes — standardised 7-class mapping used across the
# backend, masks, legends and charts.
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Urban",
    "Agriculture",
    "Rangeland",
    "Forest",
    "Water",
    "Barren",
    "Unknown",
]

CLASS_COLORS = [
    "#ef4444",  # Urban       — red
    "#84cc16",  # Agriculture — lime green
    "#eab308",  # Rangeland   — gold
    "#16a34a",  # Forest      — green
    "#3b82f6",  # Water       — blue
    "#d6c39b",  # Barren      — sandy
    "#6b7280",  # Unknown     — slate grey
]

# ---------------------------------------------------------------------------
# DeepGlobe ANNOTATION mask palette — the raw dataset encoding, NOT the
# display palette above. Verified against the official DeepGlobe challenge
# label description (deepglobe.org / CVPRW 2018 paper, Table 1) and TorchGeo's
# reference implementation (torchgeo.datasets.deepglobelandcover).
#
#   Urban        (0, 255, 255)  cyan
#   Agriculture  (255, 255, 0)  yellow
#   Rangeland    (255, 0, 255)  magenta
#   Forest       (0, 255, 0)    green
#   Water        (0, 0, 255)    blue
#   Barren land  (255, 255, 255) white
#   Unknown      (0, 0, 0)      black   (excluded from challenge mIoU)
#
# IMPORTANT: mask values may drift slightly due to compression. The official
# instruction is to binarise each R/G/B channel at threshold 128 before
# decoding (see training/dataset_loader.py).
# ---------------------------------------------------------------------------
DATASET_MASK_RGB: tuple[tuple[int, int, int, str], ...] = (
    (0, 255, 255, "Urban"),
    (255, 255, 0, "Agriculture"),
    (255, 0, 255, "Rangeland"),
    (0, 255, 0, "Forest"),
    (0, 0, 255, "Water"),
    (255, 255, 255, "Barren"),
    (0, 0, 0, "Unknown"),
)
MASK_CHANNEL_THRESHOLD = 128

# ---------------------------------------------------------------------------
# Dataset configuration (training stays OUTSIDE the website).
# ---------------------------------------------------------------------------
DATASET_OPTIONS = {
    "DEEPGLOBE_LANDCOVER": {
        "name": "DeepGlobe Land Cover",
        "has_pixel_masks": True,
        "note": (
            "RGB satellite tiles with ground-truth pixel-level masks "
            "(7 land-cover classes) — suitable for supervised U-Net training."
        ),
    },
    "CUSTOM": {
        "name": os.environ.get("SITEWISE_DATASET_NAME", "Custom segmentation dataset"),
        "has_pixel_masks": True,
        "note": "User-supplied dataset with pixel-level masks and a compatible class mapping.",
    },
}

DATASET_KEY = os.environ.get("SITEWISE_DATASET", "DEEPGLOBE_LANDCOVER")
DATASET = DATASET_OPTIONS.get(DATASET_KEY, DATASET_OPTIONS["CUSTOM"])

# Classes used for the CONNECTED model. Replace if you retrain with a
# different class set. The trained U-Net's output channel order MUST match
# this list; the loader validates it at inference time.
SEGMENTATION_CLASSES = CLASS_NAMES
SEGMENTATION_COLORS = CLASS_COLORS

# ---------------------------------------------------------------------------
# Reliability CNN output configuration.
#   RELIABILITY_OUTPUT_MODE: auto | sigmoid | softmax2
#     auto    → infer from the model's output shape (discovered at load time)
#     sigmoid → single neuron; the value is P(clean)
#     softmax2→ two neurons; P(clean) = RELIABILITY_CLEAN_INDEX-th output
#   RELIABILITY_CLEAN_INDEX: which softmax channel means "clean"
# ---------------------------------------------------------------------------
RELIABILITY_OUTPUT_MODE = os.environ.get("SITEWISE_RELIABILITY_OUTPUT", "auto")
RELIABILITY_CLEAN_INDEX = int(os.environ.get("SITEWISE_RELIABILITY_CLEAN_INDEX", "0"))

# ---------------------------------------------------------------------------
# CORS — restrict in production by setting e.g.
#   SITEWISE_CORS_ORIGINS="https://myfrontend.app,https://www.myfrontend.app"
# Default "*" is intended for local development only.
# ---------------------------------------------------------------------------
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("SITEWISE_CORS_ORIGINS", "*").split(",")
    if o.strip()
] or ["*"]

# Guard against pathological uploads ( Pillow will happily decode a 50000px
# image into RAM ). Max allowed decompressed pixel count.
MAX_IMAGE_PIXELS = 40_000_000

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
MAX_CONTENT_LENGTH_MB = 16
