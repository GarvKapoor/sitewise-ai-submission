# SiteWise AI — Flask backend

REST backend for the U-Net segmentation pipeline. Keeps trained-model
inference strictly separate from the frontend and reported honestly:
real outputs vs explicitly-labelled demonstration data.

## Run

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python app.py
```

- `POST http://127.0.0.1:5000/health`
- `POST http://127.0.0.1:5000/predict` (multipart form field `image`)

Environment (all optional):
- `SITEWISE_INPUT_SIZE="128,128"` — must match the trained U-Net
- `SITEWISE_RELIABILITY_THRESHOLD="55"`
- `SITEWISE_DATASET="DEEPGLOBE_LANDCOVER"` (also `EUROSAT_RGB`, `CUSTOM`)
- `HOST` / `PORT` (default `0.0.0.0:5000`, Render sets PORT)

## Structure

```
backend/
├── app.py                  # Flask routes: /health, /predict
├── models/                 # drop-in directory for your trained models
│   └── README.md
├── services/
│   ├── segmentation.py     # U-Net loading + pixel-wise inference + analysis
│   ├── preprocessing.py    # RGB/resize/normalise/tensor + 8 RGB features
│   ├── reliability.py      # separate reliability CNN + labelled fallback
│   └── suitability.py      # deterministic 18-D screening engine
├── config
│   └── model_config.py     # classes, colours, sizes, dataset, thresholds
└── requirements.txt
```

## Honesty guarantees

- With no trained model connected, `/predict` returns `mode: "demo"` and a
  warning list; frontend renders a **DEMONSTRATION DATA** badge. Demo outputs
  are deterministic colour-heuristic composites, never random, never presented
  as model output.
- `class_percentages` come only from actual argmax pixel counts.
- No fabricated accuracy/evaluation metrics are served or displayed anywhere.
- EuroSAT (image-level labels) is documented in `/health` as *not* containing
  pixel masks; supervised segmentation training should use a dataset such as
  DeepGlobe Land Cover.
