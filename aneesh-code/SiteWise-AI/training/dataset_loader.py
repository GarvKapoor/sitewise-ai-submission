"""
DeepGlobe Land Cover dataset loader.

Reads image/mask pairs (`*_sat.jpg`, `*_mask.png`) and yields
(uint8 RGB image, uint8 class-index mask) batches with a **verified**
RGB→class-index conversion.

Mask encodings (verified against the DeepGlobe challenge label description —
deepglobe.org / Demir et al., CVPRW 2018, Table 1 — and TorchGeo's reference
implementation `torchgeo.datasets.DeepGlobeLandCover`):

    Urban       (0, 255, 255)   cyan
    Agriculture (255, 255, 0)   yellow
    Rangeland   (255, 0, 255)   magenta
    Forest      (0, 255, 0)     green
    Water       (0, 0, 255)     blue
    Barren land (255, 255, 255) white
    Unknown     (0, 0, 0)       black

The official guidance is to BINARISE each R/G/B channel at threshold 128
before decoding, because JPEG/JXL compression of satellite tiles perturbs
exact mask pixel values. `_rgb_mask_to_indices` implements exactly that.

Class order MUST stay in sync with backend/config/model_config.py
(CLASS_NAMES) — the U-Net's softmax channel order depends on it.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# -----------------------------------------------------------------training-- #
# Local copy of the class configuration so training/ stays import-free from
# the backend package (the values must be identical; test files assert this).
DATASET_MASK_RGB: Tuple[Tuple[int, int, int, str], ...] = (
    (0, 255, 255, "Urban"),
    (255, 255, 0, "Agriculture"),
    (255, 0, 255, "Rangeland"),
    (0, 255, 0, "Forest"),
    (0, 0, 255, "Water"),
    (255, 255, 255, "Barren"),
    (0, 0, 0, "Unknown"),
)
CLASS_NAMES: Tuple[str, ...] = tuple(name for *_, name in DATASET_MASK_RGB)
MASK_CHANNEL_THRESHOLD = 128


def rgb_mask_to_indices(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB DeepGlobe annotation mask to a (H, W) uint8 class-index mask.

    Uses the verified palette + per-channel binarisation at threshold 128.
    Regions matching no class exactly (mixed annotations, compression noise)
    fall through to the 'Unknown' (black) channel rule.
    """
    m = np.asarray(mask_rgb).astype(np.int16)
    if m.ndim != 3 or m.shape[-1] != 3:
        raise ValueError(f"RGB mask expected (H, W, 3) — got shape {m.shape}")
    # Per-channel binarisation (official recommendation)
    bin = m >= MASK_CHANNEL_THRESHOLD
    indices = np.full(m.shape[:2], CLASS_NAMES.index("Unknown"), dtype=np.uint8)
    # Unknown/black must win only when nothing else fires, so iterate the
    # non-black classes first, then let Unknown remain as the default.
    for r, g, b, name in DATASET_MASK_RGB[:-1]:
        match = (bin[..., 0] == bool(r)) & (bin[..., 1] == bool(g)) & (bin[..., 2] == bool(b))
        indices[match] = CLASS_NAMES.index(name)
    return indices


def find_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Return sorted (satellite, mask) path pairs present in *data_dir*."""
    data_dir = Path(data_dir)
    sats = sorted(data_dir.glob("*_sat.jpg"))
    pairs: List[Tuple[Path, Path]] = []
    for sat in sats:
        stem = sat.name[: -len("_sat.jpg")]
        mask = data_dir / f"{stem}_mask.png"
        if mask.exists():
            pairs.append((sat, mask))
    if not pairs:
        raise FileNotFoundError(
            f"No *_sat.jpg/*_mask.png pairs found in {data_dir}. "
            "Download the DeepGlobe dataset separately (see training/README.md)."
        )
    return pairs


def make_split(
    pairs: List[Tuple[Path, Path]],
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Reproducible shuffled train/validation split."""
    pairs = sorted(pairs, key=lambda p: p[0].name)
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def load_pair(
    sat_path: Path,
    mask_path: Path,
    crop: Optional[int] = 256,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one (image, class-index mask) sample, both cropped consistently.

    With `crop` set, a random (crop × crop) window is taken from the 1024²
    tile (the same window for image and mask); rng controls reproducibility.
    """
    img = Image.open(sat_path).convert("RGB")
    raw_mask = Image.open(mask_path).convert("RGB")
    if img.size != raw_mask.size:
        raw_mask = raw_mask.resize(img.size, Image.NEAREST)
    w, h = img.size

    if crop is not None and crop < min(w, h) and rng is not None:
        left = rng.randint(0, w - crop)
        top = rng.randint(0, h - crop)
        img = img.crop((left, top, left + crop, top + crop))
        raw_mask = raw_mask.crop((left, top, left + crop, top + crop))

    arr = np.asarray(img, dtype=np.uint8)
    idx = rgb_mask_to_indices(np.asarray(raw_mask))
    return arr, idx


def augment(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    """Light deterministic augmentation: horizontal / vertical flips."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
        mask = mask[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
        mask = mask[::-1, :]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def batch_generator(
    pairs: List[Tuple[Path, Path]],
    batch_size: int,
    crop: int = 256,
    augment_data: bool = True,
    seed: int = 42,
    shuffle: bool = True,
):
    """Infinite python generator yielding (images float32 [0,1], one-hot masks)."""
    rng = random.Random(seed)
    while True:
        order = list(range(len(pairs)))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            img, mask = load_pair(*pairs[i], crop=crop, rng=rng)
            if augment_data:
                img, mask = augment(img, mask, rng)
            onehot = np.eye(len(CLASS_NAMES), dtype=np.float32)[mask.reshape(-1)].reshape(*mask.shape, -1)
            yield img.astype(np.float32) / 255.0, onehot


def class_distribution(indices: np.ndarray) -> Dict[str, int]:
    """Quick sanity check: per-class pixel counts in a decoded mask."""
    flat = indices.reshape(-1)
    return {CLASS_NAMES[i]: int((flat == i).sum()) for i in range(len(CLASS_NAMES))}
