"""
Train the SiteWise AI U-Net on the DeepGlobe Land Cover dataset.

Produces backend/models/unet_model.keras with a 7-channel softmax head whose
channel order matches backend/config/model_config.py CLASS_NAMES (Urban,
Agriculture, Rangeland, Forest, Water, Barren, Unknown).

The dataset is NEVER downloaded or included by this script — point --data at
your own local copy. Run this offline (locally or in Google Colab); nothing
here trains inside the Flask app or the website. No metrics are fabricated:
actual training history is kept only in the saved model artefact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make dataset_loader importable when run as a script or from a notebook.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from dataset_loader import CLASS_NAMES, batch_generator, find_pairs, make_split

NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = (256, 256)  # H, W


def build_unet(input_size: tuple[int, int] = INPUT_SIZE, channels: int = 3) -> tf.keras.Model:
    """
    Classic U-Net: encoder (contracting path) → bottleneck → decoder
    (expanding path with skip connections) → 1×1 conv + softmax over all
    seven channels.
    """
    inputs = layers.Input((*input_size, channels))

    def enc(filters: int, name: str):
        def block(x):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}-conv1")(x)
            x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}-conv2")(x)
            return x
        return block

    c1 = enc(64, "enc1")(inputs)
    p1 = layers.MaxPooling2D()(c1)
    c2 = enc(128, "enc2")(p1)
    p2 = layers.MaxPooling2D()(c2)
    c3 = enc(256, "enc3")(p2)
    p3 = layers.MaxPooling2D()(c3)
    c4 = enc(512, "enc4")(p3)
    p4 = layers.MaxPooling2D()(c4)

    bn = enc(1024, "bottleneck")(p4)  # deep feature representation

    u3 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bn)
    u3 = layers.Concatenate()([u3, c4])
    c5 = enc(512, "dec4")(u3)
    u2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(c5)
    u2 = layers.Concatenate()([u2, c3])
    c6 = enc(256, "dec3")(u2)
    u1 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c6)
    u1 = layers.Concatenate()([u1, c2])
    c7 = enc(128, "dec2")(u1)
    u0 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c7)
    u0 = layers.Concatenate()([u0, c1])
    c8 = enc(64, "dec1")(u0)

    outputs = layers.Conv2D(NUM_CLASSES, 1, activation="softmax", name="class_head")(c8)
    return models.Model(inputs, outputs, name="sitewise_unet")


def categorical_focal_entropy_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Segmentation loss for the 7-way softmax: focal loss + categorical crossentropy."""

    def loss(y_true, y_pred):
        y_pred_c = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred_c)
        pt = tf.reduce_sum(y_true * y_pred_c, axis=-1)
        focal = alpha * tf.pow(1.0 - pt, gamma) * tf.reduce_sum(cross_entropy, axis=-1)
        ce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        return tf.reduce_mean(focal) + ce

    return loss


def sparse_to_onehot(masks: np.ndarray) -> np.ndarray:
    """(B, H, W) uint8 → (B, H, W, C) float32 one-hot."""
    return np.eye(NUM_CLASSES, dtype=np.float32)[masks.astype(np.int64)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the SiteWise AI U-Net on DeepGlobe tiles (offline).")
    ap.add_argument("--data", required=True, help="Directory containing *_sat.jpg / *_mask.png pairs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256, help="Random square crop side length per sample")
    ap.add_argument("--val-split", type=float, default=0.15, help="Validation fraction (reproducible)")
    ap.add_argument("--seed", type=int, default=42, help="Reproducibility seed for split + augmentation")
    ap.add_argument("--steps-per-epoch", type=int, default=200, help="Batches drawn per epoch")
    ap.add_argument("--output", default="backend/models/unet_model.keras")
    args = ap.parse_args()

    data_dir = Path(args.data)
    pairs = find_pairs(data_dir)
    train_pairs, val_pairs = make_split(pairs, val_split=args.val_split, seed=args.seed)
    print(f"Dataset tiles: {len(pairs)} · train {len(train_pairs)} · validation {len(val_pairs)}")

    model = build_unet((args.crop, args.crop))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=categorical_focal_entropy_loss(),
        metrics=[
            tf.keras.metrics.IoU(
                num_classes=NUM_CLASSES,
                target_class_ids=[i for i, n in enumerate(CLASS_NAMES) if n != "Unknown"],  # per challenge
                sparse_y_pred=False,
                name="mean_iou",
            ),
        ],
    )

    train_gen = batch_generator(train_pairs, batch_size=args.batch_size, crop=args.crop, seed=args.seed)
    val_gen = batch_generator(val_pairs, batch_size=args.batch_size, crop=args.crop, seed=args.seed + 1)

    cb = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output, save_best_only=True, monitor="val_loss", mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        tf.keras.callbacks.CSVLogger("unet_training_log.csv", append=True),
    ]

    model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=max(1, len(val_pairs) // args.batch_size),
        callbacks=cb,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained U-Net to {out}")
    model.save(out)
    print(
        "IMPORTANT — the website will only show REAL results after\n"
        "     also running training/evaluate.py, which writes backend/models/evaluation.json."
    )


if __name__ == "__main__":
    main()
