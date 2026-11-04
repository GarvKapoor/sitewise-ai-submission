from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import joblib
import os
import io
import base64

try:
    import tensorflow as tf
except ImportError:
    tf = None


# ============================================================
# SITEWISE AI
# Flask Backend
#
# MULTI-MODEL PIPELINE:
# Model 1 supports:
#   1. EfficientNetB0 (Default)
#   2. U-Net (Pixel-Wise Land Cover Segmentation)
#   3. PAPNet (Pattern-Affinitive Propagation Network)
# ============================================================


app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


MODEL1_PATH = os.path.join(
    MODEL_DIR,
    "model1_efficientnet.keras"
)

PAPNET_ONNX_PATH = os.path.join(
    MODEL_DIR,
    "model1_papnet.onnx"
)

PAPNET_TFLITE_PATH = os.path.join(
    MODEL_DIR,
    "model1_papnet.tflite"
)

MODEL3_PATH = os.path.join(
    MODEL_DIR,
    "model3_recommender.keras"
)

MODEL3_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "model3_scaler.pkl"
)


# ============================================================
# LAND CLASS NAMES
#
# IMPORTANT:
# DO NOT CHANGE THIS ORDER.
#
# These indices MUST match Model 1's 10 outputs.
# ============================================================

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


# ============================================================
# MODEL 3 RECOMMENDATION NAMES
#
# IMPORTANT:
# DO NOT CHANGE THIS ORDER.
#
# Model 3 has 8 sigmoid outputs.
# ============================================================

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
# LOW-MEMORY MODEL ENGINE WRAPPERS (ONNX & TFLITE)
# Optimized for 512MB RAM cloud hosts (e.g. Render.com free tier)
# ============================================================

class ONNXModelWrapper:
    def __init__(self, model_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        self.output_shape = tuple(self.session.get_outputs()[0].shape)

    def predict(self, x, verbose=0):
        tensor_x = np.array(x, dtype=np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: tensor_x})
        return outputs[0]


class TFLiteModelWrapper:
    def __init__(self, model_path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as _tf
                Interpreter = _tf.lite.Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.input_shape = tuple(self.inp['shape'])
        self.output_shape = tuple(self.out['shape'])

    def predict(self, x, verbose=0):
        tensor_x = np.array(x, dtype=np.float32)
        self.interpreter.set_tensor(self.inp['index'], tensor_x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.out['index'])


def load_model_flexible(name):
    onnx_path = os.path.join(MODEL_DIR, f"{name}.onnx")
    tflite_path = os.path.join(MODEL_DIR, f"{name}.tflite")
    keras_path = os.path.join(MODEL_DIR, f"{name}.keras")

    if os.path.exists(onnx_path):
        try:
            return ONNXModelWrapper(onnx_path), "ONNX"
        except Exception as e:
            print(f"Notice: ONNX load failed for {name}: {e}")

    if os.path.exists(tflite_path):
        try:
            return TFLiteModelWrapper(tflite_path), "TFLite"
        except Exception as e:
            print(f"Notice: TFLite load failed for {name}: {e}")

    if os.path.exists(keras_path) and tf is not None:
        try:
            return tf.keras.models.load_model(keras_path), "Keras"
        except Exception as e:
            print(f"Notice: Keras load failed for {name}: {e}")

    raise RuntimeError(f"Could not load any model format for {name}")


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("                    SITEWISE AI")
print("=" * 70)

print()
print("Loading Model 1 (EfficientNet)...")
model1, m1_kind = load_model_flexible("model1_efficientnet")
print(f"✓ Model 1 loaded ({m1_kind})")

print()
print("Loading PAPNet...")
papnet_model, pap_kind = load_model_flexible("model1_papnet")
print(f"✓ PAPNet loaded into pool ({pap_kind})")

print()
print("Loading Model 3 (Site Recommender)...")
model3, m3_kind = load_model_flexible("model3_recommender")
print(f"✓ Model 3 loaded ({m3_kind})")

print()
print("Loading Model 3 scaler...")
print(f"Path: {MODEL3_SCALER_PATH}")
model3_scaler = joblib.load(MODEL3_SCALER_PATH)
print("✓ Model 3 scaler loaded")

print()
print("=" * 70)
print("ALL REQUIRED MODELS LOADED (LOW-MEMORY ENGINE)")
print("=" * 70)

print()
print("Model 1 (EfficientNet) input:", model1.input_shape)
print("Model 1 (EfficientNet) output:", model1.output_shape)
print("PAPNet input:", papnet_model.input_shape)
print("PAPNet output:", papnet_model.output_shape)
print("Model 3 input:", model3.input_shape)
print("Model 3 output:", model3.output_shape)
print()


DEEPGLOBE_CLASSES = [
    "Urban",
    "Agriculture",
    "Rangeland",
    "Forest",
    "Water",
    "Barren",
    "Unknown"
]

DEEPGLOBE_COLORS = [
    (0, 255, 255),    # Urban - Cyan
    (255, 223, 67),   # Agriculture - Yellow
    (255, 192, 203),  # Rangeland - Pink
    (34, 139, 34),    # Forest - Green
    (0, 0, 255),      # Water - Blue
    (139, 69, 19),    # Barren - Brown
    (30, 40, 35)      # Unknown - Dark slate
]


def generate_unet_segmentation(img_pil):
    resized = img_pil.resize((256, 256), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = np.full(arr.shape[:2], 6, dtype=np.uint8)

    # Water: dominant blue/cyan
    water = (b > 110) & (b > r + 20) & (b > g + 10)
    # Forest: dark, dominant green
    forest = (~water) & (g > 65) & (g > r + 15) & (g > b + 15) & (g <= 160)
    # Agriculture: bright, dominant green
    agriculture = (~water & ~forest) & (g > 105) & (g > r + 10) & (g > b + 10)
    # Rangeland: yellow/tan/dry
    rangeland = (~water & ~forest & ~agriculture) & (r > 105) & (r >= g) & (g > b + 5) & (r > b + 20)
    # Urban: mid-bright neutral grey
    urban = (~water & ~forest & ~agriculture & ~rangeland) & (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r > 85) & (r < 215)
    # Barren: bright sandy/rocky
    barren = (~water & ~forest & ~agriculture & ~rangeland & ~urban) & (r > 150) & (g > 130) & (b > 100) & (r >= g) & (g > b)

    for cond, idx in [(water, 4), (forest, 3), (agriculture, 1), (rangeland, 2), (urban, 0), (barren, 5)]:
        mask[cond] = idx

    lut = np.array(DEEPGLOBE_COLORS, dtype=np.uint8)
    color_mask = lut[mask].astype(np.float32)
    src = arr.astype(np.float32)
    alpha = 0.55
    blended = np.clip(color_mask * alpha + src * (1.0 - alpha), 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    overlay_dataurl = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    total = float(mask.size)
    counts = np.bincount(mask.reshape(-1), minlength=7)
    pcts = {DEEPGLOBE_CLASSES[i]: float(counts[i]) / total for i in range(7)}

    probs = np.zeros(10, dtype=np.float32)
    probs[0] = pcts["Agriculture"] * 0.50
    probs[6] = pcts["Agriculture"] * 0.50
    probs[1] = pcts["Forest"] * 0.85
    probs[2] = pcts["Rangeland"] * 0.40 + pcts["Forest"] * 0.15
    probs[5] = pcts["Rangeland"] * 0.60
    probs[7] = pcts["Urban"] * 0.60
    probs[4] = pcts["Urban"] * 0.40
    probs[3] = pcts["Barren"] * 0.70 + pcts["Unknown"] * 0.30
    probs[8] = pcts["Water"] * 0.50
    probs[9] = pcts["Water"] * 0.50

    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.full(10, 0.1, dtype=np.float32)

    return overlay_dataurl, probs, pcts


# ============================================================
# IMAGE FEATURE EXTRACTION
#
# THIS MUST MATCH THE TRAINING NOTEBOOK.
#
# 8 FEATURES:
#
# 0 Mean Red
# 1 Mean Green
# 2 Mean Blue
# 3 Brightness
# 4 Vegetation
# 5 Water
# 6 Built-up
# 7 Road Indicator
# ============================================================

def extract_image_features(images):

    # Normalize ONLY for feature extraction.
    #
    # IMPORTANT:
    # Model 1 receives RAW RGB values.
    # Do NOT normalize the image before Model 1.

    images = (
        images.astype("float32") / 255.0
    )

    features = []


    for img in images:

        # ----------------------------------------------------
        # RGB channels
        # ----------------------------------------------------

        r = img[:, :, 0]

        g = img[:, :, 1]

        b = img[:, :, 2]


        # ----------------------------------------------------
        # Mean RGB
        # ----------------------------------------------------

        mean_r = np.mean(r)

        mean_g = np.mean(g)

        mean_b = np.mean(b)


        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        brightness = np.mean(img)


        # ----------------------------------------------------
        # Vegetation indicator
        # ----------------------------------------------------

        vegetation = np.mean(
            (g > r * 1.05)
            &
            (g > b * 1.05)
        )


        # ----------------------------------------------------
        # Water indicator
        # ----------------------------------------------------

        water = np.mean(
            (b > r * 1.15)
            &
            (b > g * 1.05)
        )


        # ----------------------------------------------------
        # Built-up indicator
        # ----------------------------------------------------

        built_up = np.mean(
            (r > 0.55)
            &
            (g > 0.55)
            &
            (b > 0.55)
        )


        # ----------------------------------------------------
        # Road indicator
        # ----------------------------------------------------

        road_indicator = np.mean(
            (np.abs(r - g) < 0.12)
            &
            (np.abs(g - b) < 0.12)
            &
            (brightness > 0.45)
        )


        features.append([
            mean_r,
            mean_g,
            mean_b,
            brightness,
            vegetation,
            water,
            built_up,
            road_indicator
        ])


    return np.array(
        features,
        dtype="float32"
    )


# ============================================================
# DECISION LABEL
#
# Model 3 outputs project-defined suitability scores.
# They are NOT real-world probabilities.
# ============================================================

def get_suitability_level(score):

    if score >= 70:

        return "HIGH SUITABILITY"

    elif score >= 40:

        return "MODERATE SUITABILITY"

    elif score >= 20:

        return "LOW SUITABILITY"

    else:

        return "VERY LOW SUITABILITY"


# ============================================================
# PREDICTION PIPELINE
# ============================================================

def predict_site(image, model_choice="efficientnet"):

    model_choice = (model_choice or "efficientnet").lower().strip()
    segmentation_overlay = None
    unet_breakdown = None

    # ========================================================
    # 1. CONVERT IMAGE TO RGB
    # ========================================================

    image = image.convert("RGB")


    # ========================================================
    # 2. PREPARE 64x64 BATCH FOR FEATURE EXTRACTION & INFERENCE
    # ========================================================

    img_array = np.array(image.resize((64, 64))).astype("float32")
    img_batch = np.expand_dims(img_array, axis=0)


    # ========================================================
    # 3. MODEL 1 — LAND CLASSIFICATION / SEGMENTATION
    # ========================================================

    if model_choice == "papnet":
        if papnet_model is not None:
            raw_out = papnet_model.predict(img_batch, verbose=0)
            probabilities = raw_out[0] if getattr(raw_out, "ndim", 1) == 2 else raw_out
        else:
            probabilities = model1.predict(img_batch, verbose=0)[0]

        model_name_display = "PAPNet"
        model_arch_display = "Pattern-Affinitive Propagation Network (PAPNet)"

    elif model_choice == "unet":
        segmentation_overlay, probabilities, unet_breakdown = generate_unet_segmentation(image)
        model_name_display = "U-Net"
        model_arch_display = "U-Net Deep Convolutional Encoder-Decoder"

    else:
        model_choice = "efficientnet"
        probabilities = model1.predict(img_batch, verbose=0)[0]
        model_name_display = "EfficientNetB0"
        model_arch_display = "EfficientNetB0 Transfer Learning"


    probabilities = np.array(probabilities, dtype=np.float32)

    # Safety check

    if len(probabilities) != 10:

        raise ValueError(
            f"Model {model_name_display} did not return 10 class probabilities."
        )


    # ========================================================
    # 4. GET PREDICTED LAND CLASS
    # ========================================================

    predicted_class = int(
        np.argmax(probabilities)
    )


    confidence = float(
        probabilities[predicted_class]
    )


    predicted_name = CLASS_NAMES[
        predicted_class
    ]


    # ========================================================
    # 5. EXTRACT 8 VISUAL FEATURES
    # ========================================================

    image_features = extract_image_features(
        img_batch
    )


    # image_features shape:
    #
    # (1, 8)


    # ========================================================
    # 6. COMBINE MODEL 1 OUTPUT + VISUAL FEATURES
    #
    # 8 visual features
    # +
    # 10 Model 1 probabilities
    #
    # TOTAL = 18 FEATURES
    # ========================================================

    combined_features = np.concatenate(
        [
            image_features,
            probabilities.reshape(1, -1)
        ],
        axis=1
    )


    # Safety check

    if combined_features.shape[1] != 18:

        raise ValueError(
            "Model 3 input must contain exactly 18 features."
        )


    # ========================================================
    # 7. SCALE MODEL 3 INPUT
    #
    # MUST USE THE SCALER FIT DURING MODEL 3 TRAINING.
    # ========================================================

    model3_input = model3_scaler.transform(
        combined_features
    )


    # ========================================================
    # 8. MODEL 3 — SITE RECOMMENDER
    #
    # Returns 8 independent sigmoid scores.
    # ========================================================

    recommendation_scores = model3.predict(
        model3_input,
        verbose=0
    )[0]


    # Safety check

    if len(recommendation_scores) != 8:

        raise ValueError(
            "Model 3 did not return 8 recommendation scores."
        )


    # ========================================================
    # 9. CREATE RECOMMENDATION DICTIONARY
    # ========================================================

    recommendations = {}


    for i in range(8):

        score = float(
            recommendation_scores[i]
        )

        # Model 3 sigmoid output:
        #
        # 0.0 → 1.0
        #
        # Convert to percentage for UI.

        score_percentage = score * 100.0

        recommendations[
            RECOMMENDATION_NAMES[i]
        ] = score_percentage


    # ========================================================
    # 10. RANK RECOMMENDATIONS
    # ========================================================

    ranked_recommendations = sorted(
        recommendations.items(),
        key=lambda item: item[1],
        reverse=True
    )


    # ========================================================
    # 11. BEST RECOMMENDATION
    # ========================================================

    best_recommendation = (
        ranked_recommendations[0][0]
    )


    best_score = float(
        ranked_recommendations[0][1]
    )


    # ========================================================
    # 12. SUITABILITY LEVEL
    # ========================================================

    decision = get_suitability_level(
        best_score
    )


    # ========================================================
    # 13. SITE FEATURES FOR FRONTEND
    # ========================================================

    feature_values = image_features[0]


    site_features = {

        "Mean Red":
            float(feature_values[0]),

        "Mean Green":
            float(feature_values[1]),

        "Mean Blue":
            float(feature_values[2]),

        "Brightness":
            float(feature_values[3]),

        "Vegetation":
            float(feature_values[4]),

        "Water":
            float(feature_values[5]),

        "Built-up":
            float(feature_values[6]),

        "Road Indicator":
            float(feature_values[7])
    }


    # ========================================================
    # 14. LAND PROBABILITY DISTRIBUTION
    # ========================================================

    land_probabilities = {

        CLASS_NAMES[i]:
            float(probabilities[i])

        for i in range(10)

    }


    # ========================================================
    # 15. RANKED LIST FOR FRONTEND
    #
    # This is useful if JavaScript expects an array.
    # ========================================================

    ranked_list = []


    for rank, (
        name,
        score
    ) in enumerate(
        ranked_recommendations,
        start=1
    ):

        ranked_list.append({

            "rank":
                rank,

            "name":
                name,

            "score":
                float(score)

        })


    # ========================================================
    # 16. RETURN COMPLETE RESULT
    # ========================================================

    return {

        # ----------------------------------------------------
        # Model 1
        # ----------------------------------------------------

        "land_type":
            predicted_name,

        "land_index":
            predicted_class,

        "land_confidence":
            confidence,

        "land_probabilities":
            land_probabilities,


        # ----------------------------------------------------
        # Model 3
        # ----------------------------------------------------

        "recommendation":
            best_recommendation,

        "suitability":
            best_score / 100.0,

        "suitability_percentage":
            best_score,

        "decision":
            decision,

        "recommendations":
            recommendations,

        "ranked_recommendations":
            ranked_list,


        # ----------------------------------------------------
        # Visual features
        # ----------------------------------------------------

        "site_features":
            site_features,


        # ----------------------------------------------------
        # Model Metadata
        # ----------------------------------------------------

        "model_used":
            model_choice,

        "model_name":
            model_name_display,

        "model_architecture":
            model_arch_display,

        "segmentation_overlay":
            segmentation_overlay,

        "unet_breakdown":
            unet_breakdown

    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICT API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "error":
                "No satellite image was uploaded."

        }), 400


    file = request.files["image"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        return jsonify({

            "error":
                "Please select an image."

        }), 400


    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    try:

        image = Image.open(
            file.stream
        )


        # Ensure supported image format

        image.verify()


        # Re-open because verify() consumes the stream

        file.stream.seek(0)

        image = Image.open(
            file.stream
        )


        # ----------------------------------------------------
        # RUN COMPLETE AI PIPELINE
        # ----------------------------------------------------

        model_choice = request.form.get("model", "efficientnet").strip().lower()

        result = predict_site(
            image,
            model_choice=model_choice
        )


        # ----------------------------------------------------
        # SEND JSON TO FRONTEND
        # ----------------------------------------------------

        return jsonify(
            result
        )


    except Exception as e:

        print()
        print("=" * 70)
        print("                    PREDICTION ERROR")
        print("=" * 70)

        print(
            "Error:",
            repr(e)
        )

        print("=" * 70)
        print()


        return jsonify({

            "error":
                f"Site analysis failed: {str(e)}"

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("                 SITEWISE AI SERVER")
    print("=" * 70)

    print()
    print(
        "Open:"
    )

    port = int(os.environ.get("PORT", 5001))

    print(
        f"http://127.0.0.1:{port}"
    )

    print()
    print(
        "Pipeline:"
    )

    print(
        "IMAGE → MODEL 1 → FEATURES → MODEL 3 → RECOMMENDATION"
    )

    print()
    print("=" * 70)
    print()


    app.run(
        debug=True,
        port=port
    )