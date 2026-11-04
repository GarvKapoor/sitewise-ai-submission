"""
Tests for the Flask API (backend/app.py) with TensorFlow absent.

Verifies the "demonstration" / provenance / error-handling contract:
  • /health reports per-model status (unavailable when no model files)
  • /metrics reports "Evaluation pending" without evaluation.json
  • /predict returns structured JSON — 400 on bad uploads, tone-correct
    warnings, never heuristic substitution for a connected-but-failed model
  • segmentation and reliability are provenance-separate
Runs without TensorFlow or model binaries.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import app as app_module  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point model/evaluation paths at isolated temp dirs — never touch real ones.
    monkeypatch.setattr(app_module.cfg, "EVALUATION_FILE", tmp_path / "evaluation.json")
    monkeypatch.setattr(app_module, "unet", app_module.segsvc.SegmentationModel())
    monkeypatch.setattr(app_module, "reliability_model", app_module.relsvc.ReliabilityModel())
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def _png_init():
    img = Image.new("RGB", (64, 64), (40, 120, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _jpg_init():
    img = Image.new("RGB", (32, 32), (120, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_reports_per_model_status(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["models"] == {"unet_connected": False, "reliability_connected": False}
    assert body["model_status"]["unet"] == "unavailable"
    assert body["model_status"]["reliability"] == "unavailable"
    assert body["segmentation"]["class_names"] == [
        "Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"
    ]
    assert body["segmentation"]["num_classes"] == 7
    assert body["reliability"]["threshold"] == 55.0


# --------------------------------------------------------------------------- #
# /metrics
# --------------------------------------------------------------------------- #
def test_metrics_pending_without_file(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert body["available"] is False
    assert "pending" in body["message"].lower()


def test_metrics_accepts_wellformed_evaluation(client, tmp_path):
    data = {
        "unet": {
            "training_accuracy": 0.91,
            "validation_loss": 0.34,
            "mean_iou": 0.71,
            "dice": 0.75,
        },
        "source": "unit-test evaluation row",
    }
    app_module.cfg.EVALUATION_FILE.write_text(json.dumps(data))
    body = client.get("/metrics").get_json()
    assert body["available"] is True
    assert body["unet"]["mean_iou"] == pytest.approx(0.71)
    assert body["source"].startswith("unit-test")


# --------------------------------------------------------------------------- #
# /predict — success with labeled demonstration data
# --------------------------------------------------------------------------- #
def test_predict_demo_segmentation_and_labels(client):
    r = client.post("/predict", data={"image": (_png_init(), "tile.png")}, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["mode"] == "demo"
    assert body["sections_status"]["segmentation"] == "demonstration"
    assert body["sections_status"]["suitability_engine"] == "deterministic_engine"
    seg = body["segmentation"]
    assert seg["source"] == "demonstration_heuristic"
    assert len(seg["class_percentages"]) == 7
    assert pytest.approx(sum(c["percent"] for c in seg["class_percentages"]), abs=0.2) == 100
    assert seg["mask_base64"].startswith("data:image/png;base64,")
    # Confidence must stay absent for demonstration data (never fabricated)
    assert seg["confidence"] is None
    # Reliability is a labelled heuristic fallback (no trained CNN present)
    rel = body["reliability"]
    assert rel["model_used"] is False
    assert rel["assessed_by"] == "heuristic_fallback"
    assert "heuristic fallback" in rel["note"]
    # Visual features = eight cards
    assert len(body["visual_features"]) == 8
    assert len(body["suitability"]) == 8
    assert any("DEMONSTRATION DATA" in w for w in body["warnings"])


def test_predict_jpeg_accepted(client):
    r = client.post("/predict", data={"image": (_jpg_init(), "tile.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200


def test_predict_counts_exact_when_all_one_class(client):
    img = Image.new("RGB", (20, 20), (10, 10, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    body = client.post("/predict", data={"image": (buf, "flat.png")}, content_type="multipart/form-data").get_json()
    water = [c for c in body["segmentation"]["class_percentages"] if c["name"] == "Water"]
    assert water and water[0]["percent"] >= 50.0


# --------------------------------------------------------------------------- #
# /predict — upload validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n CORRUPTED"])
def test_predict_rejects_corrupt_or_empty(client, payload):
    r = client.post("/predict",
                    data={"image": (io.BytesIO(payload), "x.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_rejects_when_no_field(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "image" in r.get_json()["error"]


def test_predict_rejects_gif_content(client):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="GIF")
    buf.seek(0)
    r = client.post("/predict", data={"image": (buf, "still.gif")}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "JPG / JPEG / PNG" in r.get_json()["error"]


def test_predict_rejects_oversized_dimensions(client, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "MAX_IMAGE_PIXELS", 100)
    buf = _png_init()  # 64×64 = 4096 px > 100
    r = client.post("/predict", data={"image": (buf, "big.png")}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "dimensions too large" in r.get_json()["error"]


# --------------------------------------------------------------------------- #
# /predict — connected-model failure handling (Task 2 explicit guarantee)
# --------------------------------------------------------------------------- #
class _FailingUnet:
    """Simulates a connected U-Net (load OK) whose predict can raise."""

    def __init__(self):
        self.model = object()
        self.error = ""
        self.source_path = "/tmp/fake.keras"

    @property
    def connected(self):
        return True

    def load(self) -> bool:
        return True

    def predict(self, tensor):
        raise ValueError("bad softmax")


def _pristine_reliability(connected: bool = False):
    m = app_module.relsvc.ReliabilityModel()
    if connected:
        m.model = object()  # fake "loaded" — score() is monkeypatched by tests
    return m


def test_predict_fails_open_on_connected_unet_error(client, monkeypatch):
    fake = _FailingUnet()
    monkeypatch.setattr(app_module, "unet", fake)
    monkeypatch.setattr(app_module, "reliability_model", _pristine_reliability())
    r = client.post("/predict", data={"image": (_png_init(), "tile.png")}, content_type="multipart/form-data")
    assert r.status_code == 200  # structured, not a 500
    body = r.get_json()
    assert body["segmentation"] is None
    assert body["sections_status"]["segmentation"] == "failed"
    assert any("NOT substituted" in w or "failed while the model IS connected" in w for w in body["warnings"])
    # heuristics never fill in when a real model was present but failed
    assert body["mode"] != "real"


def test_predict_reliability_failure_retagged(client, monkeypatch):
    fake = _FailingUnet()
    monkeypatch.setattr(app_module, "unet", fake)
    failing_rel = _pristine_reliability(connected=True)
    monkeypatch.setattr(failing_rel, "score", lambda rgb: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(app_module, "reliability_model", failing_rel)
    body = client.post("/predict", data={"image": (_png_init(), "tile.png")}, content_type="multipart/form-data").get_json()
    assert body["sections_status"]["reliability"] == "failed"
    assert body["reliability"]["model_used"] is False


def test_predict_with_connected_unet_uses_real_path(client, monkeypatch):
    """When a working U-Net is present its output drives all downstream fields."""
    H, W = [int(v) for v in app_module.cfg.INPUT_SIZE]
    fake = _FailingUnet()
    probs = np.full((1, H, W, 7), 0.1, dtype=np.float32)
    probs[..., 5] = 0.4  # Barren wins → softmax sums to 1.0
    probs /= probs.sum(-1, keepdims=True)

    def predict(tensor):
        assert tensor.ndim == 4 and tensor.shape[0] == 1
        mask = np.full((H, W), 5, dtype=np.uint8)  # all Barren
        return mask, probs[0]

    fake.predict = predict
    monkeypatch.setattr(app_module, "unet", fake)
    body = client.post("/predict", data={"image": (_png_init(), "tile.png")}, content_type="multipart/form-data").get_json()
    assert body["mode"] == "real"
    assert body["sections_status"]["segmentation"] == "connected"
    assert body["segmentation"]["source"] == "unet"
    assert body["segmentation"]["confidence"] is not None
    assert body["segmentation"]["confidence"]["mean"] == pytest.approx(40.0, abs=0.5)
    barren = [c for c in body["segmentation"]["class_percentages"] if c["name"] == "Barren"][0]
    assert barren["percent"] == pytest.approx(100.0, abs=0.2)
    assert len(body["suitability"]) == 8
