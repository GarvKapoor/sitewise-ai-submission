"""
SiteWise AI — Flask backend.

Endpoints
  GET  /health    → backend + per-model connection status
  GET  /metrics   → genuine evaluation results (models/evaluation.json) or
                    "Evaluation pending"
  POST /predict   → preprocessing → U-Net segmentation → reliability CNN →
                    visual features → suitability screening

Per-model provenance (each model reports its own state, never one global flag):
  connected : trained model loaded and inference succeeded
  unavailable : no model file / TensorFlow missing (demonstration fallback)
  failed    : model connected but inference raised — NO silent heuristic
              substitution; the section is returned as null with an error

Upload hardening (Task 7):
  - JPG / JPEG / PNG only, validated by decoded image CONTENT (PIL format
    check), not by filename extension
  - 16 MB request limit (MAX_CONTENT_LENGTH)
  - decompressed-size guard (MAX_IMAGE_PIXELS)
  - corrupted or truncated images are rejected with 400 messages

Demonstration data is always explicitly labelled (source
"demonstration_heuristic", assessed_by "heuristic_fallback") and never mixed
with real model output.
"""
from __future__ import annotations

import base64
import io
import json
import traceback
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request
from flask_cors import CORS

from config import model_config as cfg
from services import preprocessing as prep
from services import reliability as relsvc
from services import segmentation as segsvc
from services import suitability as suitsvc

app = Flask(__name__)
CORS(app, origins=cfg.CORS_ORIGINS)
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_LENGTH_MB * 1024 * 1024

ALLOWED_FORMATS = {"JPEG", "PNG"}

unet = segsvc.SegmentationModel()
reliability_model = relsvc.ReliabilityModel()


def ensure_models() -> None:
    try:
        unet.load()
    except Exception:  # pragma: no cover — loading itself malformed
        traceback.print_exc()
        unet.error = unet.error or "U-Net loading crashed (see server log)."
    try:
        reliability_model.load()
    except Exception:  # pragma: no cover
        traceback.print_exc()
        reliability_model.error = (
            reliability_model.error or "Reliability model loading crashed (see server log)."
        )


def model_state(connected: bool, error: str, phase: str) -> str:
    """connected / unavailable (no model file or TF) / failed (load error)."""
    if connected:
        return "connected"
    hard_load_failure = error.startswith("Failed to load")
    return "failed" if hard_load_failure and phase == "load" else "unavailable"


# ---------------------------------------------------------------------------
# Structured error responses (Task 2)
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_e):
    return (
        jsonify(
            {
                "error": (
                    f"Image exceeds the {cfg.MAX_CONTENT_LENGTH_MB} MB upload limit. "
                    "Please upload a smaller file."
                )
            }
        ),
        413,
    )


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Unknown endpoint. Use /health, /metrics or POST /predict."}), 404


@app.errorhandler(405)
def wrong_method(_e):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


@app.errorhandler(Exception)
def unhandled(e):  # pragma: no cover — safety net
    traceback.print_exc()
    return jsonify({"error": f"Internal server error: {e}"}), 500


def validate_upload(data: bytes) -> Image.Image:
    """
    Validate the decoded CONTENT of an upload (Task 7):
      - must be a decodable image (corrupt / truncated → clear 400)
      - decoded format must be JPEG or PNG (not just the filename)
      - decompressed dimensions must stay within cfg.MAX_IMAGE_PIXELS
    Returns the PIL image on success, raises ValueError otherwise.
    """
    if not data:
        raise ValueError("Uploaded file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # structural validation of the encoded bytes
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as e:
        raise ValueError(f"Unsupported or corrupted image file: {e}") from e
    if img.format not in ALLOWED_FORMATS:
        raise ValueError(
            f"Only JPG / JPEG / PNG images are accepted — decoded content was {img.format}."
        )
    if img.width * img.height > cfg.MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image dimensions too large ({img.width}×{img.height}). "
            f"Maximum supported pixel count is {cfg.MAX_IMAGE_PIXELS:,}."
        )
    return img


@app.get("/health")
def health():
    ensure_models()
    unet_state = model_state(unet.connected, unet.error, "load")
    rel_state = model_state(reliability_model.connected, reliability_model.error, "load")
    mode = "real" if unet_state == "connected" else "demo"
    return jsonify(
        {
            "ok": True,
            "mode": mode,
            "models": {
                "unet_connected": unet.connected,
                "reliability_connected": reliability_model.connected,
            },
            "model_status": {
                "unet": unet_state,
                "reliability": rel_state,
            },
            "model_details": {
                "unet_path": unet.source_path,
                "unet_error": unet.error,
                "reliability_path": reliability_model.source_path,
                "reliability_error": reliability_model.error,
            },
            "segmentation": {
                "input_size": list(cfg.INPUT_SIZE),
                "num_classes": len(cfg.SEGMENTATION_CLASSES),
                "class_names": cfg.SEGMENTATION_CLASSES,
                "dataset": cfg.DATASET["name"],
                "dataset_note": cfg.DATASET["note"],
            },
            "reliability": {
                "threshold": cfg.RELIABILITY_THRESHOLD,
                "output_mode": cfg.RELIABILITY_OUTPUT_MODE,
            },
        }
    )


@app.get("/metrics")
def metrics():
    """
    Model performance — served only from backend/models/evaluation.json,
    produced after real offline training/evaluation (training/evaluate.py
    writes genuine values). No file ⇒ "Evaluation pending". Values are never
    fabricated or guessed.
    """
    path = cfg.EVALUATION_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            unet_res = data.get("unet", {})
            required = ["training_accuracy", "validation_loss", "mean_iou", "dice"]
            if all(isinstance(unet_res.get(k), (int, float)) for k in required):
                vals = {k: float(unet_res[k]) for k in required}
                if any(np.isnan(v) or v < 0 for v in vals.values()):
                    return jsonify({"available": False, "message": "Evaluation file malformed."})
                return jsonify(
                    {
                        "available": True,
                        "unet": vals,
                        "source": str(data.get("source", "offline evaluation")),
                    }
                )
            return jsonify({"available": False, "message": "Evaluation file malformed."})
        except Exception as e:
            return jsonify({"available": False, "message": f"Could not read evaluation file: {e}"})
    return jsonify(
        {
            "available": False,
            "message": "Evaluation pending — metrics appear only after real training and evaluation.",
        }
    )


@app.post("/predict")
def predict():
    ensure_models()

    # ---------------- Upload validation (Task 7) ----------------
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send multipart form field 'image'."}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    raw = f.read()
    try:
        img = validate_upload(raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        raw = None  # release the raw upload buffer before heavy work

    warnings: list[str] = []

    try:
        rgb = np.asarray(img.convert("RGB"))
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400
    orig_w, orig_h = int(img.width), int(img.height)

    try:
        tensor, target_size = prep.preprocess_for_unet(rgb)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 400
    resized_rgb = np.asarray(
        Image.fromarray(rgb).resize((target_size[1], target_size[0]), Image.BILINEAR)
    )

    # ---------------- Segmentation ----------------
    seg_unet_failed = False
    if unet.connected:
        try:
            mask, probs = unet.predict(tensor)
            analysed = segsvc.analyse(mask, probs)
            seg_result: dict | None = {
                **analysed,
                "mask_base64": segsvc.to_png_dataurl(segsvc.rgb_mask(mask)),
                "overlay_base64": segsvc.to_png_dataurl(
                    segsvc.overlay(resized_rgb, mask)
                ),
                "source": "unet",
            }
        except Exception as e:  # dependency of the trained model
            # NO heuristic substitution when a connected model failed.
            seg_result = None
            seg_unet_failed = True
            warnings.append(
                f"U-Net inference failed while the model IS connected: {e}. "
                "No segmentation output is returned (demonstration heuristics "
                "are deliberately NOT substituted)."
            )
            traceback.print_exc()
    else:
        # Clearly-labelled DEMONSTRATION segmentation: deterministic colour
        # heuristics — never presented as a U-Net prediction.
        mask = segsvc.demo_heuristic_mask(resized_rgb)
        analysed = segsvc.analyse(mask, None)
        seg_result = {
            **analysed,
            "mask_base64": segsvc.to_png_dataurl(segsvc.rgb_mask(mask)),
            "overlay_base64": segsvc.to_png_dataurl(
                segsvc.overlay(resized_rgb, mask)
            ),
            "source": "demonstration_heuristic",
        }
        warnings.append(
            "DEMONSTRATION DATA — segmentation mask from deterministic colour "
            "heuristics, not a U-Net prediction."
        )

    seg_status = "unavailable"
    if seg_unet_failed:
        seg_status = "failed"
    elif seg_result is not None:
        seg_status = "connected" if seg_result.get("source") == "unet" else "demonstration"

    # ---------------- Reliability ----------------
    reliability = relsvc.assess(rgb, reliability_model)
    rel_status = "connected" if reliability["model_used"] else (
        "failed" if "inference_error" in reliability else "unavailable"
    )
    if rel_status != "connected":
        warnings.append(
            "Reliability CNN "
            + ("inference failed — " if rel_status == "failed" else "not connected — ")
            + "score comes from the labelled heuristic fallback."
        )
    # Segmentation and reliability are separate models: a missing/failing
    # reliability CNN does NOT downgrade the U-Net results.

    # ---------------- Features ----------------
    features = prep.extract_visual_features(
        rgb,
        seg_result["class_percentages"] if seg_result else None,
        cfg.SEGMENTATION_CLASSES,
        bool(seg_result and seg_result.get("source") == "unet"),
    )
    if not features[1]:
        warnings.append(
            "Vegetation/water/built-up ratios use colour heuristics "
            "(segmentation-based refinement unavailable)."
        )

    # ---------------- Suitability ----------------
    name_map = {i: n for i, n in enumerate(cfg.SEGMENTATION_CLASSES)}
    suitability = suitsvc.compute_scores(
        seg_result["class_percentages"] if seg_result else None,
        features[0],
        name_map,
    )
    top = suitability[0] if suitability else None

    engine_status = "deterministic_engine"

    overall_mode = "real" if (seg_result and seg_result.get("source") == "unet") else "demo"
    if overall_mode == "demo":
        warnings.append(
            "MODEL NOT CONNECTED — segmentation on this page is DEMONSTRATION "
            "DATA, not real U-Net output."
        )
    if seg_unet_failed:
        warnings.append(
            "Overall mode flagged demo because segmentation is unavailable this "
            "request; any reliability output shown is still per-model labelled."
        )

    return jsonify(
        {
            "mode": overall_mode,
            "model_status": {
                "unet_connected": unet.connected,
                "reliability_connected": reliability_model.connected,
            },
            "sections_status": {
                "segmentation": seg_status,
                "reliability": rel_status,
                "suitability_engine": engine_status,
            },
            "image": {
                "filename": f.filename,
                "original_base64": encode_original(rgb),
                "width": orig_w,
                "height": orig_h,
                "request_id": uuid.uuid4().hex[:12],
            },
            "preprocessing": {
                "target_size": list(target_size),
                "normalization": "0-1 min-max on RGB",
                "steps": prep.PREPROCESSING_STEPS,
            },
            "segmentation": seg_result,
            "segmentation_error": seg_unet_failed,
            "reliability": reliability,
            "visual_features": features[0],
            "suitability": suitability,
            "top_recommendation": (
                {"category": top["category"], "score": top["score"]} if top else None
            ),
            "warnings": warnings,
        }
    )


def encode_original(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb)
    if max(img.size) > 1024:  # keep payload sane
        s = 1024 / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    print("Starting SiteWise AI backend…")
    ensure_models()
    print(
        f"U-Net: {'connected' if unet.connected else 'NOT connected'} · "
        f"Reliability CNN: {'connected' if reliability_model.connected else 'NOT connected'}"
    )
    app.run(host=cfg.HOST, port=cfg.PORT, debug=False)
