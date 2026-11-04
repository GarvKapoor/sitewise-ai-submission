from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import os


# ============================================================
# SITEWISE AI
# Flask Backend
#
# FINAL PIPELINE:
#
# Satellite Image
#       ↓
# Model 1 — EfficientNetB0
#       ↓
# Land Classification + 10 Probabilities
#       +
# 8 Visual Features
#       ↓
# Model 3 — Site Recommender
#       ↓
# 8 Recommendation Scores
#       ↓
# Best Site Recommendation
#
# Model 2 is NOT used in the main website flow.
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
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("                    SITEWISE AI")
print("=" * 70)

print()
print("Loading Model 1...")
print(
    f"Path: {MODEL1_PATH}"
)

model1 = tf.keras.models.load_model(
    MODEL1_PATH
)

print("✓ Model 1 loaded")


print()
print("Loading Model 3...")
print(
    f"Path: {MODEL3_PATH}"
)

model3 = tf.keras.models.load_model(
    MODEL3_PATH
)

print("✓ Model 3 loaded")


print()
print("Loading Model 3 scaler...")
print(
    f"Path: {MODEL3_SCALER_PATH}"
)

model3_scaler = joblib.load(
    MODEL3_SCALER_PATH
)

print("✓ Model 3 scaler loaded")


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

def predict_site(image):

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

    probabilities = model1.predict(
        img_batch,
        verbose=0
    )[0]


    # Safety check

    if len(probabilities) != 10:

        raise ValueError(
            "Model 1 did not return 10 class probabilities."
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
            site_features

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

        result = predict_site(
            image
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

    print(
        "http://127.0.0.1:5000"
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
        debug=True
    )