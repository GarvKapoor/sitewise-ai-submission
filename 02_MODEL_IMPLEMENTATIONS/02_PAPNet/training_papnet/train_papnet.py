#!/usr/bin/env python3
"""
PAPNet Full Training Pipeline for EuroSAT & SiteWise AI
======================================================
Trains:
  1. Model 1 — PAPNet (Pattern-Affinitive Propagation Network) for EuroSAT 10-class land classification
  2. Model 2 — Site Suitability Predictor
  3. Model 3 — 8-Class Multi-Output Site Recommender

Usage:
  python train_papnet.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--quick-demo] [--save-to-models]
"""

import os
import sys
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Ensure local imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from papnet_model import create_papnet_custom, create_papnet_transfer, PatternAffinityModule

# 10 EuroSAT Classes
CLASS_NAMES = [
    "Annual Crop Land",
    "Forest",
    "Brushland or Shrubland",
    "Highway or Road",
    "Industrial Buildings or Commercial Buildings",
    "Pasture Land",
    "Permanent Crop Land",
    "Residential Buildings or Homes or Apartments",
    "River",
    "Lake or Sea"
]

# 8 Model 3 Recommendation Classes
RECOMMENDATION_NAMES = [
    "Residential Development",
    "Commercial Development",
    "Industrial Development",
    "Educational Development",
    "Healthcare Development",
    "Agricultural Use",
    "Transportation Infrastructure",
    "Green / Conservation"
]


# ============================================================
# 8 VISUAL FEATURES (MATCHING SITEWISE-AI NOTEBOOK & APP.PY)
# ============================================================

def extract_image_features(images):
    """
    Extracts 8 visual/site features from RGB images:
      0: Mean Red
      1: Mean Green
      2: Mean Blue
      3: Brightness
      4: Vegetation
      5: Water
      6: Built-up
      7: Road Indicator
    """
    images_norm = images.astype("float32") / 255.0
    features = []

    for img in images_norm:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)
        brightness = np.mean(img)

        # Vegetation: green significantly stronger than red and blue
        vegetation = np.mean((g > r * 1.05) & (g > b * 1.05))

        # Water: blue dominant and low overall brightness or high blue contrast
        water = np.mean((b > r * 1.1) & (b > g * 0.95))

        # Built-up: high brightness with balanced RGB (concrete, rooftops)
        built_up = np.mean((brightness > 0.45) & (np.abs(r - g) < 0.1) & (np.abs(g - b) < 0.1))

        # Road indicator: linear high contrast edges / gray spectrum
        road = np.mean((np.abs(r - g) < 0.05) & (np.abs(g - b) < 0.05) & (brightness > 0.25) & (brightness < 0.6))

        features.append([
            mean_r, mean_g, mean_b, brightness,
            vegetation, water, built_up, road
        ])

    return np.array(features, dtype=np.float32)


# ============================================================
# DATASET LOADING (HUGGING FACE / FALLBACK GENERATOR)
# ============================================================

def load_or_generate_data(quick_demo=False, num_samples=1000):
    """
    Loads EuroSAT from Hugging Face if available, otherwise generates
    realistic synthetic satellite image batches for verification.
    """
    try:
        from datasets import load_dataset
        print("Loading EuroSAT dataset from Hugging Face ('tanganke/eurosat')...")
        dataset = load_dataset("tanganke/eurosat")
        
        train_data = dataset["train"]
        if quick_demo:
            indices = np.random.choice(len(train_data), size=min(1200, len(train_data)), replace=False)
            train_data = train_data.select(indices.tolist())

        images = np.array([
            np.array(img.convert("RGB")) for img in train_data["image"]
        ], dtype=np.uint8)
        labels = np.array(train_data["label"], dtype=np.int32)
        print(f"Loaded {len(images)} real EuroSAT images.")
        return images, labels

    except Exception as e:
        print(f"Notice: HuggingFace datasets load skipped ({e}).")
        print("Generating synthetic satellite dataset for PAPNet training verification...")
        count = 600 if quick_demo else num_samples
        images = []
        labels = []
        for i in range(count):
            lbl = i % 10
            # Generate characteristic color pattern per class
            base_color = np.array({
                0: [180, 160, 90],   # Annual Crop: yellowish green
                1: [30, 120, 40],    # Forest: dark green
                2: [100, 140, 60],   # Shrubland: olive green
                3: [120, 120, 120],  # Highway: gray
                4: [160, 150, 140],  # Industrial: concrete
                5: [80, 180, 70],    # Pasture: bright grass
                6: [140, 150, 80],   # Permanent Crop: patterned green
                7: [190, 130, 110],  # Residential: reddish rooftops
                8: [50, 90, 160],    # River: blue water
                9: [20, 60, 140]     # SeaLake: deep blue
            }[lbl], dtype=np.float32)

            noise = np.random.normal(0, 20, (64, 64, 3))
            img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            images.append(img)
            labels.append(lbl)

        return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


# ============================================================
# MODEL 2 & 3 SCORING GENERATORS
# ============================================================

def create_suitability_score(features, facility="bank"):
    """
    Project-defined suitability label generator (19 inputs -> 1 binary target).
    """
    brightness = features[:, 3]
    vegetation = features[:, 4]
    water = features[:, 5]
    built_up = features[:, 6]
    road = features[:, 7]
    probs = features[:, 8:18]

    forest = probs[:, 1]
    highway = probs[:, 3]
    industrial = probs[:, 4]
    residential = probs[:, 7]
    river = probs[:, 8]
    sea_lake = probs[:, 9]

    if facility == "bank":
        score = (
            residential * 35.0 +
            highway * 25.0 +
            built_up * 20.0 +
            industrial * 10.0 -
            (river + sea_lake + water) * 40.0 -
            forest * 20.0
        )
    else:
        score = (built_up * 30.0 + highway * 20.0 + residential * 20.0 - (water + river) * 30.0)

    # Scale to 0-100 range
    score = np.clip(score, 0.0, 100.0)
    return score


def create_model3_scores(features):
    """
    Computes 8 recommendation target profiles based on Model 1 class probabilities
    and 8 visual features (same formulation as training notebook).
    """
    brightness = features[:, 3]
    vegetation = features[:, 4]
    water = features[:, 5]
    built_up = features[:, 6]
    road = features[:, 7]

    probs = features[:, 8:18]
    annual_crop = probs[:, 0]
    forest = probs[:, 1]
    shrubland = probs[:, 2]
    highway = probs[:, 3]
    industrial = probs[:, 4]
    pasture = probs[:, 5]
    perm_crop = probs[:, 6]
    residential = probs[:, 7]
    river = probs[:, 8]
    sea_lake = probs[:, 9]

    scores = np.zeros((features.shape[0], 8), dtype=np.float32)

    # 0 Residential
    scores[:, 0] = residential * 0.45 + built_up * 0.25 + highway * 0.15 - (river + sea_lake) * 0.35
    # 1 Commercial
    scores[:, 1] = built_up * 0.35 + highway * 0.30 + residential * 0.20 - forest * 0.25
    # 2 Industrial
    scores[:, 2] = industrial * 0.50 + highway * 0.30 + built_up * 0.20 - residential * 0.30
    # 3 Educational
    scores[:, 3] = residential * 0.40 + vegetation * 0.25 + built_up * 0.20 - industrial * 0.30
    # 4 Healthcare
    scores[:, 4] = residential * 0.35 + highway * 0.25 + built_up * 0.25 - industrial * 0.35
    # 5 Agricultural
    scores[:, 5] = annual_crop * 0.40 + perm_crop * 0.30 + pasture * 0.20 + vegetation * 0.10 - built_up * 0.30
    # 6 Transportation
    scores[:, 6] = highway * 0.50 + road * 0.30 + built_up * 0.20 - (river + sea_lake) * 0.30
    # 7 Green / Conservation
    scores[:, 7] = forest * 0.45 + shrubland * 0.25 + vegetation * 0.20 + (river + sea_lake) * 0.10 - built_up * 0.40

    return np.clip(scores, 0.0, 1.0)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train PAPNet and SiteWise AI Models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for Model 1")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-type", choices=["transfer", "custom"], default="transfer", help="Model type: 'transfer' (Pretrained backbone + PAP Module, recommended, >92-95% acc) or 'custom' (from-scratch CNN + PAP blocks)")
    parser.add_argument("--quick-demo", action="store_true", help="Run quick demo on small subset")
    parser.add_argument("--save-to-models", action="store_true", default=True, help="Copy models to root models/ directory")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("    SITEWISE AI — PAPNET (PATTERN-AFFINITIVE PROPAGATION) TRAINING")
    print("=" * 70)

    # 1. Load Data
    images, labels = load_or_generate_data(quick_demo=args.quick_demo)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.18, random_state=42, stratify=y_train_val
    )

    print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 2. Build Model 1: PAPNet
    print(f"\n[Step 1/3] Building PAPNet Architecture (type: {args.model_type})...")
    if args.model_type == "transfer":
        papnet, backbone = create_papnet_transfer(input_shape=(64, 64, 3), num_classes=10)
    else:
        papnet = create_papnet_custom(input_shape=(64, 64, 3), num_classes=10)

    papnet.summary()

    papnet.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n[Step 1/3] Training PAPNet for {args.epochs} epochs...")
    history1 = papnet.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # Evaluate Model 1
    test_loss, test_acc = papnet.evaluate(X_test, y_test, verbose=0)
    print(f"✓ PAPNet Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    # Save Learning Curve Plot
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history1.history["accuracy"], label="Train Acc")
        plt.plot(history1.history["val_accuracy"], label="Val Acc")
        plt.title("PAPNet Accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history1.history["loss"], label="Train Loss")
        plt.plot(history1.history["val_loss"], label="Val Loss")
        plt.title("PAPNet Loss")
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(SCRIPT_DIR, "model1_papnet_learning_curve.png")
        plt.savefig(curve_path)
        plt.close()
        print(f"✓ Saved learning curve to {curve_path}")

    # Save Confusion Matrix Plot
    if HAS_MATPLOTLIB:
        y_pred_probs = papnet.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[c[:8] for c in CLASS_NAMES])
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        plt.title("PAPNet EuroSAT Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(SCRIPT_DIR, "model1_papnet_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"✓ Saved confusion matrix to {cm_path}")

    # Save PAPNet Model Artifact
    model1_path = os.path.join(SCRIPT_DIR, "model1_papnet.keras")
    papnet.save(model1_path)
    print(f"✓ Saved PAPNet Model 1 to {model1_path}")

    # 3. Extract 18 Features for Model 2 and Model 3
    print("\n[Step 2/3] Extracting 18 Features (8 Visual + 10 PAPNet Probabilities)...")
    train_visual = extract_image_features(X_train_val)
    train_probs = papnet.predict(X_train_val, batch_size=args.batch_size, verbose=0)
    X_base = np.hstack([train_visual, train_probs])  # (N, 18)

    # 4. Train Model 2 (Suitability Predictor)
    print("\n[Step 2/3] Training Model 2 (Suitability Predictor)...")
    # Generate suitability targets (e.g. for bank / commercial facility)
    facility_scores = create_suitability_score(X_base, facility="bank")
    threshold = np.percentile(facility_scores, 50)
    y2 = (facility_scores >= threshold).astype(np.float32)

    X2_train, X2_val, y2_train, y2_val = train_test_split(
        X_base, y2, test_size=0.2, random_state=42
    )

    scaler2 = StandardScaler()
    X2_train_scaled = scaler2.fit_transform(X2_train)
    X2_val_scaled = scaler2.transform(X2_val)

    model2 = keras.Sequential([
        layers.Input(shape=(18,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ], name="Model2_Suitability")

    model2.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model2.fit(
        X2_train_scaled, y2_train,
        validation_data=(X2_val_scaled, y2_val),
        epochs=min(args.epochs, 20),
        batch_size=args.batch_size,
        verbose=0
    )

    model2_path = os.path.join(SCRIPT_DIR, "model2_suitability.keras")
    model2_scaler_path = os.path.join(SCRIPT_DIR, "model2_scaler.pkl")
    model2.save(model2_path)
    joblib.dump(scaler2, model2_scaler_path)
    print(f"✓ Saved Model 2 to {model2_path}")

    # 5. Train Model 3 (Site Recommender)
    print("\n[Step 3/3] Training Model 3 (8-Class Site Recommender)...")
    y3 = create_model3_scores(X_base)  # (N, 8)

    X3_train, X3_val, y3_train, y3_val = train_test_split(
        X_base, y3, test_size=0.2, random_state=42
    )

    scaler3 = StandardScaler()
    X3_train_scaled = scaler3.fit_transform(X3_train)
    X3_val_scaled = scaler3.transform(X3_val)

    model3 = keras.Sequential([
        layers.Input(shape=(18,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(8, activation="sigmoid")
    ], name="Model3_Recommender")

    model3.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    model3.fit(
        X3_train_scaled, y3_train,
        validation_data=(X3_val_scaled, y3_val),
        epochs=min(args.epochs, 25),
        batch_size=args.batch_size,
        verbose=0
    )

    model3_path = os.path.join(SCRIPT_DIR, "model3_recommender.keras")
    model3_scaler_path = os.path.join(SCRIPT_DIR, "model3_scaler.pkl")
    model3.save(model3_path)
    joblib.dump(scaler3, model3_scaler_path)
    print(f"✓ Saved Model 3 to {model3_path}")

    # 6. Copy to models/ directory if requested
    if args.save_to_models:
        root_models_dir = os.path.join(SCRIPT_DIR, "..", "models")
        os.makedirs(root_models_dir, exist_ok=True)
        import shutil
        shutil.copy(model1_path, os.path.join(root_models_dir, "model1_papnet.keras"))
        print(f"✓ Copied 'model1_papnet.keras' to {root_models_dir}")

    print("\n" + "=" * 70)
    print("           PAPNET TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
