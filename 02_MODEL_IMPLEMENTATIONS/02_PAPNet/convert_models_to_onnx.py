"""
SiteWise AI: Convert TFLite / Keras models to ONNX
==================================================
Produces ultra-lightweight ONNX models for sub-second, low-memory inference
on memory-constrained hosts (e.g. Render.com 512MB RAM free tier).
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPT_DIR = os.path.join(BASE_DIR, "models_optimized")
ONNX_DIR = os.path.join(BASE_DIR, "models_onnx")

os.makedirs(ONNX_DIR, exist_ok=True)

models = [
    ("model1_papnet.tflite", "model1_papnet.onnx"),
    ("model1_efficientnet.tflite", "model1_efficientnet.onnx"),
    ("model2_suitability.tflite", "model2_suitability.onnx"),
    ("model3_recommender.tflite", "model3_recommender.onnx")
]

for src_name, dst_name in models:
    src = os.path.join(OPT_DIR, src_name)
    dst = os.path.join(ONNX_DIR, dst_name)
    if not os.path.exists(src):
        print(f"Skipping {src_name} (not found in {OPT_DIR})")
        continue

    print(f"Converting {src_name} -> {dst_name}...")
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", src,
        "--output", dst
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created {dst} ({os.path.getsize(dst) / (1024 * 1024):.2f} MB)")

# Copy scalers
for scaler in ["model2_scaler.pkl", "model3_scaler.pkl"]:
    src_s = os.path.join(OPT_DIR, scaler)
    dst_s = os.path.join(ONNX_DIR, scaler)
    if os.path.exists(src_s):
        import shutil
        shutil.copy2(src_s, dst_s)
        print(f"✓ Copied {scaler} to {ONNX_DIR}")

print("\nAll models converted to ONNX successfully!")

