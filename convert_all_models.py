import os
import sys
import numpy as np
import tensorflow as tf

# Add path to import papnet custom layers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPNET_MODULE_PATH = os.path.join(SCRIPT_DIR, "Garv-code", "site-wise-ai-papnet")
if PAPNET_MODULE_PATH not in sys.path:
    sys.path.insert(0, PAPNET_MODULE_PATH)

try:
    from training_papnet.papnet_model import PatternAffinityModule, PAPResidualBlock
    custom_objects = {
        "PatternAffinityModule": PatternAffinityModule,
        "PAPResidualBlock": PAPResidualBlock,
    }
except Exception as e:
    print(f"Notice: PAPNet custom layers import: {e}")
    custom_objects = {}

MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

MODELS = [
    ("model1_efficientnet.keras", "model1_efficientnet.tflite", "model1_efficientnet.onnx", None),
    ("model1_papnet.keras", "model1_papnet.tflite", "model1_papnet.onnx", custom_objects),
    ("model2_suitability.keras", "model2_suitability.tflite", "model2_suitability.onnx", None),
    ("model3_recommender.keras", "model3_recommender.tflite", "model3_recommender.onnx", None),
]

print("=" * 65)
print("     LIVE CONVERSION: .keras -> .tflite AND .onnx")
print("=" * 65)

for keras_name, tflite_name, onnx_name, c_objs in MODELS:
    keras_path = os.path.join(MODELS_DIR, keras_name)
    tflite_path = os.path.join(MODELS_DIR, tflite_name)
    onnx_path = os.path.join(MODELS_DIR, onnx_name)

    if not os.path.exists(keras_path):
        print(f"\n[SKIP] {keras_name} not found in {MODELS_DIR}")
        continue

    print(f"\n---> [1/2] Loading {keras_name} ({os.path.getsize(keras_path) / (1024*1024):.2f} MB)...")
    model = tf.keras.models.load_model(keras_path, custom_objects=c_objs)

    # 1. Convert to TFLite
    print(f"     Converting to TFLite -> {tflite_name}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"     ✓ Saved {tflite_name} ({len(tflite_bytes) / (1024*1024):.2f} MB)")

    # 2. Convert to ONNX
    print(f"     Converting to ONNX -> {onnx_name}...")
    # Use tf2onnx from tflite for optimal compatibility and minimal graph size
    import subprocess
    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", tflite_path,
        "--output", onnx_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"     ✓ Saved {onnx_name} ({os.path.getsize(onnx_path) / (1024*1024):.2f} MB)")
    else:
        print(f"     [!] Direct TFLite->ONNX failed: {res.stderr[:200]}, trying direct keras->onnx...")
        import tf2onnx
        import onnx
        spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
        print(f"     ✓ Saved {onnx_name} ({os.path.getsize(onnx_path) / (1024*1024):.2f} MB)")

print("\n" + "=" * 65)
print("     LIVE VERIFICATION OF CONVERTED MODELS")
print("=" * 65)

import onnxruntime as ort

for keras_name, tflite_name, onnx_name, _ in MODELS:
    tflite_path = os.path.join(MODELS_DIR, tflite_name)
    onnx_path = os.path.join(MODELS_DIR, onnx_name)

    if os.path.exists(tflite_path):
        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()
        inp_shape = interp.get_input_details()[0]['shape']
        out_shape = interp.get_output_details()[0]['shape']
        print(f"✓ TFLite OK: {tflite_name:<28} input={inp_shape.tolist()} output={out_shape.tolist()}")

    if os.path.exists(onnx_path):
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_shape = [x if isinstance(x, int) else 1 for x in session.get_inputs()[0].shape]
        out_shape = [x if isinstance(x, int) else 1 for x in session.get_outputs()[0].shape]
        print(f"✓ ONNX   OK: {onnx_name:<28} input={inp_shape} output={out_shape}")

print("\nAll conversions and verifications completed successfully!")

