"""
Offline evaluation of a trained SiteWise AI U-Net on DeepGlobe validation tiles.

Writes ONLY genuine numbers to backend/models/evaluation.json — the file the
website's /metrics endpoint reads. Without this file the site correctly shows
"Evaluation pending"; nothing here invents accuracy, IoU or Dice values.

Metrics (per DeepGlobe challenge conventions):
  • Pixel accuracy      — correctly labelled pixels / total labelled pixels
  • Mean IoU            — averaged over the 6 scored classes (Unknown excluded)
  • Dice coefficient    — 2|A∩B|/(|A|+|B|), computed per class, macro-averaged

The validation split is reproduced with the same --seed/--val-split used in
training, so evaluation is comparable to the reported validation history.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf

from dataset_loader import CLASS_NAMES, find_pairs, load_pair, make_split

NUM_CLASSES = len(CLASS_NAMES)
UNKNOWN_IDX = CLASS_NAMES.index("Unknown")
EVALUATION_OUTPUT = Path(__file__).resolve().parent.parent / "backend" / "models" / "evaluation.json"


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """C×C confusion matrix for flat class-index arrays."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true.reshape(-1), y_pred.reshape(-1)), 1)
    return cm


def per_class_from_cm(cm: np.ndarray):
    """(iou, dice, precision_values, recall_values) from a C×C confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return iou, dice, precision, recall


def evaluate_model(model: tf.keras.Model, pairs, crop: int, seed: int, max_tiles: int | None):
    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    evaluated = 0
    for sat_path, mask_path in pairs:
        img, true_idx = load_pair(sat_path, mask_path, crop=crop, rng=None)
        pred = model.predict(img[None, ...].astype(np.float32) / 255.0, verbose=0)
        pred_idx = np.argmax(pred[0], axis=-1).astype(np.uint8)
        cm_total += confusion(true_idx, pred_idx)
        evaluated += 1
        if max_tiles and evaluated >= max_tiles:
            break
    return cm_total, evaluated


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained U-Net; writes real metrics to models/evaluation.json.")
    ap.add_argument("--model", default="backend/models/unet_model.keras")
    ap.add_argument("--data", required=True, help="DeepGlobe train/ directory with *_sat.jpg + *_mask.png")
    ap.add_argument("--crop", type=int, default=1024, help="Tile side length fed to the model (training crop at eval time)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = evaluate the full validation split")
    ap.add_argument("--history", default="", help="Optional training_log CSV via callbacks.CSVLogger for training_accuracy")
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    pairs = find_pairs(Path(args.data))
    _, val_pairs = make_split(pairs, val_split=args.val_split, seed=args.split_seed)
    print(f"Evaluating {len(val_pairs)} held-out validation tiles from {args.data}")

    cm, evaluated = evaluate_model(model, val_pairs, args.crop, args.split_seed, args.max_tiles)
    acc_total = np.diag(cm).sum() / max(cm.sum(), 1)
    iou, dice, precision, recall = per_class_from_cm(cm)
    scored = [i for i, n in enumerate(CLASS_NAMES) if n != "Unknown"]  # Unknown excluded per challenge
    mean_iou = float(iou[scored].mean()) if evaluated else None
    mean_dice = float(dice[scored].mean()) if evaluated else None

    per_class = {
        CLASS_NAMES[i]: ({"precision": float(precision[i]), "recall": float(recall[i])} if evaluated else None)
        for i in range(NUM_CLASSES)
    }

    training_accuracy = validation_loss = None
    if args.history and Path(args.history).exists():
        # Real logged values from the actual training run — nothing synthesised.
        try:
            rows = [ln.strip().split(",") for ln in Path(args.history).read_text().splitlines() if "," in ln]
            cols = rows[0]
            li = cols.index("epoch")
            la = cols.index("accuracy") if "accuracy" in cols else cols.index("categorical_accuracy")
            lv = cols.index("val_loss")
            training_accuracy = float(rows[-1][la])
            validation_loss = float(min(float(r[lv]) for r in rows[1:] if r[lv]))
        except Exception as e:
            print(f"Could not parse history CSV {args.history}: {e}")

    payload = {
        "unet": {
            "training_accuracy": training_accuracy,     # None → omitted by /metrics check
            "validation_loss": validation_loss,
            "mean_iou": mean_iou,
            "dice": mean_dice,
        },
        "pixel_accuracy": float(acc_total) if evaluated else None,
        "unknown_excluded": True,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "tiles_evaluated": evaluated,
        "model": str(args.model),
        "source": f"offline evaluation of {Path(args.model).name} on {evaluated} DeepGlobe validation tiles",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    EVALUATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {EVALUATION_OUTPUT}")
    if evaluated:
        print(f"pixel_acc={acc_total:.4f}  mean_iou={mean_iou:.4f}  dice={mean_dice:.4f}")
    else:
        print("No tiles evaluated — evaluation.json will keep /metrics pending.")


if __name__ == "__main__":
    main()
