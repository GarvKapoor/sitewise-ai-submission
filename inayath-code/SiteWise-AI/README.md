# SiteWise AI

A Flask web interface for the EuroSAT land-classification + site-suitability Deep Learning project.

## 1. Folder structure

```text
SiteWise-AI/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── models/
│   ├── model1_efficientnet.keras
│   ├── model2_suitability.keras
│   └── model2_scaler.pkl
└── uploads/
```

## 2. Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## 3. Add trained model files

Put these exact files inside `models/`:

- `model1_efficientnet.keras`
- `model2_suitability.keras`
- `model2_scaler.pkl`

## 4. Important model integration step

The notebook is the source of truth for the exact Model-2 feature engineering.

The web application intentionally does **not** invent the suitability feature vector. Open `app.py` and replace the `predict_site()` function with the exact prediction/feature-engineering logic from your notebook.

The frontend already expects this JSON:

```json
{
  "land_class": "Forest",
  "confidence": 0.94,
  "facility": "Bank",
  "suitability": 0.87,
  "suitable": true
}
```

## 5. Run

```bash
python app.py
```

Open:

http://127.0.0.1:5000

## Why this design?

The UI is separated from inference so your trained model remains the source of predictions. The app fails clearly if model artifacts or the exact Model-2 pipeline are missing instead of displaying fabricated/demo predictions as real results.
