# models/ — trained model drop-in directory

Place your trained models here:

| File | Purpose |
|---|---|
| `unet_model.keras` (or `.h5`) | U-Net for pixel-wise land-cover segmentation |
| `reliability_model.keras` (or `.h5`) | Reliability CNN (clean vs degraded imagery) |

Rules:

1. The U-Net output must be shaped `(batch, H, W, num_classes)` with a
   per-pixel softmax — the backend takes `argmax` over the last axis.
2. Class order must match `CLASS_NAMES` in `backend/config/model_config.py`.
   Change that list (and colours) if your model uses a different class set
   (e.g. DeepGlobe's 5 classes).
3. `SITEWISE_INPUT_SIZE="H,W"` (env var) must match the trained input size —
   default 128,128.
4. Both files are optional at startup: with either one missing the backend
   runs in **demo mode** and every produced value is flagged as
   DEMONSTRATION DATA; the frontend displays the same labelling.

No accuracy numbers, evaluation results, or sample placeholders are stored
here — do not commit fake model files.
