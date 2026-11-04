from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import joblib
import os


# ============================================================
# SITEWISE AI
# Ultra-Low-Memory Backend for Cloud Deployment (Render.com 512MB RAM)
# Supports ONNX Runtime & LiteRT / TFLite (RAM < 120MB total)
# ============================================================


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


# ============================================================
# 1. ONNX RUNTIME WRAPPER (DEFAULT LIGHTWEIGHT ENGINE: ~100MB RAM)
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


# ============================================================
# 2. TFLITE / LITERT WRAPPER (FALLBACK LIGHTWEIGHT ENGINE: ~120MB RAM)
# ============================================================

class TFLiteModelWrapper:
    def __init__(self, model_path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                try:
                    import tensorflow as tf
                    Interpreter = tf.lite.Interpreter
                except ImportError:
                    raise ImportError("No LiteRT, TFLite-runtime, or TensorFlow interpreter found.")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = tuple(self.input_details[0]['shape'])
        self.output_shape = tuple(self.output_details[0]['shape'])

    def predict(self, x, verbose=0):
        tensor_x = np.array(x, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor_x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])


# ============================================================
# 3. FLEXIBLE MODEL LOADER
# Prioritizes ONNX -> TFLite -> Keras (lazy load)
# ============================================================

def load_model_flexible(onnx_path, tflite_path, keras_path, custom_objs_fn=None):
    # 1. Try ONNX Runtime (fastest CPU inference, memory ~100MB)
    if onnx_path and os.path.exists(onnx_path):
        try:
            wrapper = ONNXModelWrapper(onnx_path)
            return wrapper, "ONNX"
        except Exception as e:
            print(f"Notice: Failed to load ONNX model from {onnx_path}: {e}")

    # 2. Try LiteRT / TFLite (lightweight Google edge runtime, memory ~120MB)
    if tflite_path and os.path.exists(tflite_path):
        try:
            wrapper = TFLiteModelWrapper(tflite_path)
            return wrapper, "TFLite"
        except Exception as e:
            print(f"Notice: Failed to load TFLite model from {tflite_path}: {e}")

    # 3. Fallback to Keras / Full TensorFlow ONLY if lightweight runtimes are missing
    if keras_path and os.path.exists(keras_path):
        try:
            import tensorflow as tf
            custom_objs = custom_objs_fn() if callable(custom_objs_fn) else custom_objs_fn
            m = tf.keras.models.load_model(keras_path, custom_objects=custom_objs)
            return m, "Keras"
        except Exception as e:
            print(f"Notice: Failed to load Keras model from {keras_path}: {e}")

    return None, None


# ============================================================
# PATHS & MODEL SELECTION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ONNX_MODEL_DIR = os.path.join(BASE_DIR, "models_onnx")
OPT_MODEL_DIR = os.path.join(BASE_DIR, "models_optimized")
MODEL_DIR = os.path.join(BASE_DIR, "models")

_papnet_onnx = os.path.join(ONNX_MODEL_DIR, "model1_papnet.onnx")
_papnet_tflite = os.path.join(OPT_MODEL_DIR, "model1_papnet.tflite")
_papnet_keras = os.path.join(MODEL_DIR, "model1_papnet.keras")

_efficientnet_onnx = os.path.join(ONNX_MODEL_DIR, "model1_efficientnet.onnx")
_efficientnet_tflite = os.path.join(OPT_MODEL_DIR, "model1_efficientnet.tflite")
_efficientnet_keras = os.path.join(MODEL_DIR, "model1_efficientnet.keras")

_model2_onnx = os.path.join(ONNX_MODEL_DIR, "model2_suitability.onnx")
_model2_tflite = os.path.join(OPT_MODEL_DIR, "model2_suitability.tflite")
_model2_keras = os.path.join(MODEL_DIR, "model2_suitability.keras")

_model3_onnx = os.path.join(ONNX_MODEL_DIR, "model3_recommender.onnx")
_model3_tflite = os.path.join(OPT_MODEL_DIR, "model3_recommender.tflite")
_model3_keras = os.path.join(MODEL_DIR, "model3_recommender.keras")

def _find_scaler(filename):
    for d in [ONNX_MODEL_DIR, OPT_MODEL_DIR, MODEL_DIR]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return os.path.join(MODEL_DIR, filename)

_model2_scaler_path = _find_scaler("model2_scaler.pkl")
_model3_scaler_path = _find_scaler("model3_scaler.pkl")


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

FACILITY_TO_MODEL3_MAP = {
    "bank": "Commercial Development",
    "hospital": "Healthcare Development",
    "school": "Educational Development",
    "college": "Educational Development",
    "commercial development": "Commercial Development",
    "commercial": "Commercial Development",
    "residential development": "Residential Development",
    "residential": "Residential Development",
    "industrial plant": "Industrial Development",
    "industrial development": "Industrial Development",
    "industrial": "Industrial Development",
    "green conservation": "Green / Conservation",
    "green / conservation": "Green / Conservation",
    "agricultural use": "Agricultural Use",
    "transportation infrastructure": "Transportation Infrastructure"
}

PROFILE_MAX_SCORES = {
    "Residential Development": 70.0,
    "Commercial Development": 45.0,
    "Industrial Development": 78.0,
    "Educational Development": 42.0,
    "Healthcare Development": 45.0,
    "Agricultural Use": 45.0,
    "Transportation Infrastructure": 80.0,
    "Green / Conservation": 45.0
}

FACILITY_TO_CODE = {
    "bank": 0.0,
    "school": 1.0,
    "college": 1.0
}


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("                    SITEWISE AI")
print("=" * 70)

def _get_papnet_custom_objects():
    try:
        from training_papnet.papnet_model import PatternAffinityModule, PAPResidualBlock
        return {
            "PatternAffinityModule": PatternAffinityModule,
            "PAPResidualBlock": PAPResidualBlock
        }
    except Exception:
        return {}

MODELS_POOL = {}

# Load EfficientNetB0 (ONNX -> TFLite -> Keras)
eff_model, eff_kind = load_model_flexible(_efficientnet_onnx, _efficientnet_tflite, _efficientnet_keras)
if eff_model:
    MODELS_POOL["efficientnet"] = eff_model
    print(f"✓ EfficientNetB0 loaded into pool ({eff_kind})")

# Load PAPNet (ONNX -> TFLite -> Keras)
pap_model, pap_kind = load_model_flexible(_papnet_onnx, _papnet_tflite, _papnet_keras, custom_objs_fn=_get_papnet_custom_objects)
if pap_model:
    MODELS_POOL["papnet"] = pap_model
    print(f"✓ PAPNet loaded into pool ({pap_kind})")

# Select active default Model 1
_requested_type = os.environ.get("MODEL1_TYPE", "papnet").strip().lower()
if _requested_type == "efficientnet" and "efficientnet" in MODELS_POOL:
    MODEL1_ID = "efficientnet"
    MODEL1_NAME = "EfficientNetB0"
elif "papnet" in MODELS_POOL:
    MODEL1_ID = "papnet"
    MODEL1_NAME = "PAPNet (Pattern-Affinitive Propagation)"
else:
    MODEL1_ID = next(iter(MODELS_POOL.keys()), "unknown")
    MODEL1_NAME = "EfficientNetB0" if MODEL1_ID == "efficientnet" else "PAPNet"

model1 = MODELS_POOL.get(MODEL1_ID)
print(f"✓ Active default Model 1: {MODEL1_NAME}")


# Load Model 2 (Target Facility Suitability Predictor)
model2, m2_kind = load_model_flexible(_model2_onnx, _model2_tflite, _model2_keras)
model2_scaler = None
if model2 and os.path.exists(_model2_scaler_path):
    try:
        model2_scaler = joblib.load(_model2_scaler_path)
        print(f"✓ Model 2 (Target Facility Suitability) loaded ({m2_kind})")
    except Exception as e:
        print("Notice: Could not load Model 2 scaler:", e)
else:
    print("Notice: Model 2 not active")


print()
print("Loading Model 3...")
model3, m3_kind = load_model_flexible(_model3_onnx, _model3_tflite, _model3_keras)
if model3:
    print(f"✓ Model 3 loaded ({m3_kind})")
else:
    raise FileNotFoundError("Could not find or load Model 3 from models_onnx/, models_optimized/, or models/.")

print()
print("Loading Model 3 scaler...")
if os.path.exists(_model3_scaler_path):
    model3_scaler = joblib.load(_model3_scaler_path)
    print("✓ Model 3 scaler loaded")
else:
    raise FileNotFoundError(f"Model 3 scaler not found at {_model3_scaler_path}")


print()
print("=" * 70)
print("ALL REQUIRED MODELS LOADED")
print("=" * 70)

print()
print(
    "Model 1 input:",
    model1.input_shape
)

print(
    "Model 1 output:",
    model1.output_shape
)

print(
    "Model 3 input:",
    model3.input_shape
)

print(
    "Model 3 output:",
    model3.output_shape
)

print()


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

        water_pixels = (
            ((b > r * 1.15) & (b > g * 1.02))
            |
            ((b > 0.16) & (r < 0.16) & (g < 0.24) & (b > r))
        )

        water = float(np.mean(water_pixels))


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

def predict_site(image, model_choice=None, target_facility="Bank"):

    # Determine active model
    active_m1 = model1
    active_m1_name = MODEL1_NAME
    if model_choice and model_choice.lower() in MODELS_POOL:
        active_m1 = MODELS_POOL[model_choice.lower()]
        active_m1_name = "PAPNet (Pattern-Affinitive Propagation)" if model_choice.lower() == "papnet" else "EfficientNetB0"

    # ========================================================
    # 1. CONVERT IMAGE TO RGB
    # ========================================================

    image = image.convert("RGB")


    # ========================================================
    # 2. RESIZE
    #
    # EXACT SIZE USED BY MODEL 1
    # ========================================================

    img_array = np.array(
        image.resize((64, 64))
    )


    # Make sure datatype is suitable
    # for TensorFlow inference.

    img_array = img_array.astype(
        "float32"
    )


    # Add batch dimension:
    #
    # (64, 64, 3)
    #       ↓
    # (1, 64, 64, 3)

    img_batch = np.expand_dims(
        img_array,
        axis=0
    )


    # ========================================================
    # 3. MODEL 1 — LAND CLASSIFICATION
    #
    # IMPORTANT:
    # RAW RGB VALUES ARE SENT TO MODEL 1.
    # ========================================================

    probabilities = active_m1.predict(
        img_batch,
        verbose=0
    )[0]


    # Safety check

    if len(probabilities) != 10:

        raise ValueError(
            "Model 1 did not return 10 class probabilities."
        )


    # ========================================================
    # 4. EXTRACT 8 VISUAL FEATURES
    # ========================================================

    image_features = extract_image_features(
        img_batch
    )


    # ========================================================
    # 5. DUAL LAND & WATER RECOGNITION
    # ========================================================

    land_indices = list(range(8))
    water_indices = [8, 9]

    # Land breakdown
    top_land_idx = int(np.argmax(probabilities[land_indices]))
    top_land_score = float(probabilities[top_land_idx])
    top_land_name = CLASS_NAMES[top_land_idx]
    total_land_prob = float(np.sum(probabilities[:8]))

    # Water breakdown
    top_water_local = int(np.argmax(probabilities[water_indices]))
    top_water_idx = water_indices[top_water_local]
    top_water_score = float(probabilities[top_water_idx])
    top_water_name = CLASS_NAMES[top_water_idx]
    total_water_prob = float(probabilities[8] + probabilities[9])

    # Water visual fraction from image features
    water_pixel_fraction = float(image_features[0, 5])

    # Calibrated detection thresholds
    water_detected = (
        (total_water_prob >= 0.15)
        or (top_water_score >= 0.12)
        or (water_pixel_fraction >= 0.08)
    )

    land_detected = (
        (total_land_prob >= 0.15)
        or (top_land_score >= 0.12)
    )

    is_mixed_site = bool(water_detected and land_detected)

    if is_mixed_site:
        predicted_name = f"{top_land_name} with Water Body ({top_water_name})"
        predicted_class = top_land_idx
        confidence = float((top_land_score + top_water_score) / 2.0)
    elif water_detected and not land_detected:
        predicted_name = top_water_name
        predicted_class = top_water_idx
        confidence = top_water_score
    else:
        predicted_name = top_land_name
        predicted_class = top_land_idx
        confidence = top_land_score


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
    # 11. BEST OVERALL RECOMMENDATION (MODEL 3)
    # ========================================================

    best_recommendation = (
        ranked_recommendations[0][0]
    )

    best_score = float(
        ranked_recommendations[0][1]
    )

    best_decision = get_suitability_level(
        best_score
    )

    # ========================================================
    # 12. TARGET FACILITY SUITABILITY (MODEL 2 & MODEL 3)
    # ========================================================
    clean_facility = str(target_facility or "Bank").strip().lower()
    mapped_rec_name = FACILITY_TO_MODEL3_MAP.get(clean_facility, "Commercial Development")

    # Evaluate using Model 2 for Bank (code 0.0) or School/College (code 1.0)
    # Model 2 was specifically trained as a binary classifier on these two facilities.
    if clean_facility == "bank" and model2 is not None and model2_scaler is not None:
        try:
            X_19 = np.hstack([combined_features, np.array([[0.0]])]) # 0.0 = Bank
            X_19_scaled = model2_scaler.transform(X_19)
            m2_prob = float(model2.predict(X_19_scaled, verbose=0)[0][0])
            facility_suitability_score = round(m2_prob * 100.0, 1)
            evaluation_engine = "Model 2 (Bank Binary Predictor)"
        except Exception as e:
            raw_m3 = float(recommendations.get(mapped_rec_name, best_score))
            facility_suitability_score = round(min(100.0, (raw_m3 / 45.0) * 100.0), 1)
            evaluation_engine = "Model 3 (Fallback)"
    elif clean_facility in ["school", "college"] and model2 is not None and model2_scaler is not None:
        try:
            X_19 = np.hstack([combined_features, np.array([[1.0]])]) # 1.0 = College / School
            X_19_scaled = model2_scaler.transform(X_19)
            m2_prob = float(model2.predict(X_19_scaled, verbose=0)[0][0])
            facility_suitability_score = round(m2_prob * 100.0, 1)
            evaluation_engine = "Model 2 (School/College Binary Predictor)"
        except Exception as e:
            raw_m3 = float(recommendations.get(mapped_rec_name, best_score))
            facility_suitability_score = round(min(100.0, (raw_m3 / 42.0) * 100.0), 1)
            evaluation_engine = "Model 3 (Fallback)"
    else:
        # All other facility profiles (Hospital, Residential, Industrial, Green Conservation, Agricultural, etc.)
        # are accurately scored by Model 3's dedicated 8-profile heads, calibrated to profile ceiling:
        raw_m3 = float(recommendations.get(mapped_rec_name, best_score))
        ceiling = PROFILE_MAX_SCORES.get(mapped_rec_name, 50.0)
        facility_suitability_score = round(min(100.0, (raw_m3 / ceiling) * 100.0), 1)
        evaluation_engine = f"Model 3 ({mapped_rec_name})"

    facility_decision = get_suitability_level(facility_suitability_score)


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

        "model_name":
            active_m1_name,

        "land_type":
            predicted_name,

        "primary_land_type":
            top_land_name,

        "land_index":
            predicted_class,

        "land_confidence":
            confidence,

        "land_probabilities":
            land_probabilities,

        "water_detected":
            bool(water_detected),

        "water_type":
            top_water_name if water_detected else "None",

        "water_confidence":
            float(top_water_score) if water_detected else 0.0,

        "water_coverage_percentage":
            round(float(water_pixel_fraction * 100), 1),

        "is_mixed_site":
            bool(is_mixed_site),

        "scene_description":
            predicted_name,


        # ----------------------------------------------------
        # Target Facility (Model 2) & Best Recommendation (Model 3)
        # ----------------------------------------------------

        "target_facility":
            target_facility,

        "evaluation_engine":
            evaluation_engine,

        "target_facility_suitability":
            facility_suitability_score,

        "suitability_percentage":
            facility_suitability_score,

        "suitability":
            facility_suitability_score / 100.0,

        "decision":
            facility_decision,

        "facility_decision":
            facility_decision,

        "recommendation":
            best_recommendation,

        "optimal_recommendation":
            best_recommendation,

        "optimal_score":
            best_score,

        "recommendations":
            recommendations,

        "ranked_recommendations":
            ranked_list,


        # ----------------------------------------------------
        # Visual features & Model info
        # ----------------------------------------------------

        "site_features":
            site_features,

        "model_used": {
            "id": "papnet" if active_m1 == MODELS_POOL.get("papnet") else "efficientnet",
            "name": active_m1_name
        }

    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        model_name=MODEL1_NAME,
        model_id=MODEL1_ID
    )


@app.route("/api/models")
def get_models():

    return jsonify({
        "active_model": MODEL1_ID,
        "models": [
            {
                "id": "papnet",
                "name": "PAPNet",
                "tag": "Pattern-Affinitive Propagation",
                "accuracy": "91.67%",
                "available": "papnet" in MODELS_POOL
            },
            {
                "id": "efficientnet",
                "name": "EfficientNetB0",
                "tag": "CNN Backbone",
                "accuracy": "94.96%",
                "available": "efficientnet" in MODELS_POOL
            }
        ]
    })


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

        model_choice = request.form.get("model_type", request.args.get("model_type", None))
        facility = request.form.get("facility", request.args.get("facility", "Bank"))

        result = predict_site(
            image,
            model_choice=model_choice,
            target_facility=facility
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

    port = int(os.environ.get("PORT", 5000))

    print(
        f"http://127.0.0.1:{port}"
    )

    print()
    print("Pipeline:")
    print(f"IMAGE → MODEL 1 ({MODEL1_NAME}) → FEATURES → MODEL 3 → RECOMMENDATION")
    print()
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False
    )