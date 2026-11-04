# SiteWise AI

A Flask web interface for the EuroSAT satellite land-classification + site-suitability and multi-criteria site recommendation Deep Learning project.

---

## 1. Supported Model Architectures

SiteWise AI supports two primary Land Classification backbones (Model 1):
1. **PAPNet (Pattern-Affinitive Propagation Network)** *(New)*:
   - Uses non-local spatial pattern affinity matrix computations ($A = \text{Softmax}(Q K^T / \sqrt{d})$) and affinity propagation ($P = A \cdot V$) to capture disjoint structural features (canopy textures, agricultural crop geometries, river curves, and road networks).
   - Located in [`training_papnet/`](training_papnet/).
2. **EfficientNetB0**:
   - Pretrained ImageNet transfer learning CNN backbone.
   - Located in [`training/`](training/).

---

## 2. Folder Structure

```text
SiteWise-AI/
├── app.py                         # Flask Web Backend (auto-detects PAPNet or EfficientNet)
├── requirements.txt               # Dependencies
├── templates/
│   └── index.html                 # Frontend Web Interface
├── static/
│   ├── samples/                   # Real Sentinel-2 EuroSAT sample tiles
│   ├── style.css                  # HUD animations & styling
│   └── script.js                  # Frontend interactions
├── models_onnx/                   # Ultra-lightweight ONNX models (RAM ~100MB, Render-ready)
│   ├── model1_papnet.onnx
│   ├── model1_efficientnet.onnx
│   ├── model2_suitability.onnx
│   ├── model3_recommender.onnx
│   └── *.pkl
├── models_optimized/              # Quantized LiteRT / TFLite models (RAM ~120MB)
│   ├── model1_papnet.tflite
│   ├── model1_efficientnet.tflite
│   └── ...
├── models/                        # Full Keras model artifacts
│   ├── model1_papnet.keras
│   └── ...
├── training/                      # Original EfficientNet training notebook & artifacts
│   ├── project_notebook.ipynb
│   └── ...
└── training_papnet/               # PAPNet training folder
    ├── papnet_model.py            # Keras Pattern Affinity Layers & Model Builders
    ├── project_notebook_papnet.ipynb # Complete end-to-end PAPNet Jupyter notebook
    ├── train_papnet.py            # Standalone CLI training script
    ├── model1_papnet.keras        # Trained PAPNet model
    ├── model2_suitability.keras   # Trained Model 2
    ├── model3_recommender.keras   # Trained Model 3
    └── README.md                  # Detailed PAPNet documentation
```

---

## 3. Installation

```bash
python -m venv venv
```

Activate:

- macOS / Linux:
  ```bash
  source venv/bin/activate
  ```
- Windows:
  ```bash
  venv\Scripts\activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Training PAPNet

You can train PAPNet via either the CLI script or the interactive Jupyter notebook:

### CLI Script:
```bash
# Full training on EuroSAT
python training_papnet/train_papnet.py --epochs 20 --batch-size 32

# Quick verification demo
python training_papnet/train_papnet.py --quick-demo --epochs 2
```

### Jupyter Notebook:
Open and run [`training_papnet/project_notebook_papnet.ipynb`](training_papnet/project_notebook_papnet.ipynb) in Jupyter Lab, VS Code, or Google Colab.

---

## 5. Running the Web Application

To run with PAPNet (default when `models/model1_papnet.keras` exists):
```bash
python app.py
```

To explicitly force a specific architecture:
```bash
# Run with PAPNet
MODEL1_TYPE=papnet python app.py

# Run with EfficientNet
MODEL1_TYPE=efficientnet python app.py
```

Open:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 6. Deploying to Render.com (512MB RAM Free Tier)

This repository is optimized for memory-constrained platforms like Render.com:
- **Zero TensorFlow Overhead**: Runs on pure **ONNX Runtime** (`onnxruntime`) or **LiteRT** (`ai-edge-litert`).
- **RAM Footprint**: **~100MB – 120MB**, leaving over 380MB of headroom on Render's 512MB tier.
- **Preconfigured**: `Procfile` and `render.yaml` are included.

### Render Configuration:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 app:app`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.11.9`
  - `MODEL1_TYPE`: `papnet`

