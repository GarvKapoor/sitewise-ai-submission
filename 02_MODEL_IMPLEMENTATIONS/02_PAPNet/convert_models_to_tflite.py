"""
Script to convert Keras models into optimized TensorFlow Lite (.tflite) models
for low-memory deployment (e.g. Render 512MB free tier).
"""

import os
import tensorflow as tf
from training_papnet.papnet_model import PatternAffinityModule, PAPResidualBlock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "models")
DST_DIR = os.path.join(BASE_DIR, "models_optimized")

os.makedirs(DST_DIR, exist_ok=True)

custom_objects = {
    "PatternAffinityModule": PatternAffinityModule,
    "PAPResidualBlock": PAPResidualBlock,
}

MODELS_TO_CONVERT = [
    ("model3_recommender.keras", "model3_recommender.tflite", None),
    ("model2_suitability.keras", "model2_suitability.tflite", None),
    ("model1_efficientnet.keras", "model1_efficientnet.tflite", None),
    ("model1_papnet.keras", "model1_papnet.tflite", custom_objects),
]

for src_name, dst_name, custom_obj in MODELS_TO_CONVERT:
    src_path = os.path.join(SRC_DIR, src_name)
    dst_path = os.path.join(DST_DIR, dst_name)

    if not os.path.exists(src_path):
        print(f"Skipping {src_name} (file not found)")
        continue

    print(f"Converting {src_name} -> {dst_name}...")
    model = tf.keras.models.load_model(src_path, custom_objects=custom_obj)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(dst_path, "wb") as f:
        f.write(tflite_model)
    print(f"✓ Created {dst_name} ({len(tflite_model) / (1024 * 1024):.2f} MB)")

# Also copy scalers
for scaler in ["model2_scaler.pkl", "model3_scaler.pkl"]:
    src_sc = os.path.join(SRC_DIR, scaler)
    dst_sc = os.path.join(DST_DIR, scaler)
    if os.path.exists(src_sc):
        import shutil
        shutil.copy2(src_sc, dst_sc)
        print(f"✓ Copied {scaler} to {DST_DIR}")

print("\nDone! All optimized models and scalers are ready in models_optimized/")

