# sitewise-ai-submission

**SiteWise AI** is an automated, AI-assisted Earth observation platform designed for satellite image land-cover analysis, physical terrain indexing, and multi-criteria site suitability screening. The system translates raw Sentinel-2 satellite tiles into 10 EuroSAT land categories, computes 8 optical terrain telemetry features, and predicts optimal zoning and development uses across 8 distinct sectors.

This project is submitted by:
- **Garv**
- **Inayath**
- **Aneesh**

**Registration Numbers:** `24BCA8067`, `24BCA7850`, `24BCA7708`, `24BCE7118`  
*School of Computer Science and Engineering, Vellore Institute of Technology (VIT-AP University)*

> **Project Report:** The complete academic and technical documentation for this project is detailed in [`Ai_project_report.docx`](./Ai_project_report.docx).

---

# Architecture

SiteWise AI utilizes a multi-stage hierarchical inference pipeline:

```
[Satellite Tile (64x64 RGB)]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 1: Land Classification Engine (Model 1)         │
│  • EfficientNetB0 (Compound Scaling Baseline)          │
│  • PAPNet (Pattern-Affinitive Propagation Network)     │
│  • U-Net (Pixel-Wise Terrain Semantic Segmentation)    │
└────────────────────────────────────────────────────────┘
           │
     10 Class Probs (or Segmentation Map)
           │
           ├────────────────────────┐
           │                        ▼
           │       ┌─────────────────────────────────────┐
           │       │ STAGE 2: 8 Visual Feature Indices   │
           │       │ • Mean RGB & Scene Luminance        │
           │       │ • Vegetation & Water Indices        │
           │       │ • Built-up Density & Road Contrast  │
           │       └─────────────────────────────────────┘
           │                        │
           ▼                        ▼
     ┌───────────────────────────────────────────────────┐
     │ 18-Dimensional Scaled Input Vector (Scaler .pkl)  │
     └───────────────────────────────────────────────────┘
                            │
                            ▼
     ┌───────────────────────────────────────────────────┐
     │ STAGE 3: Multi-Criteria Recommender (Model 3)     │
     │ 8-Head Sigmoid Neural Network                     │
     │ Outputs calibrated suitability scores for:        │
     │ Residential, Commercial, Industrial, Educational, │
     │ Healthcare, Agricultural, Transportation, Green   │
     └───────────────────────────────────────────────────┘
                            │
                            ▼
     [Ranked Development Recommendations & Suitability %]
```

### 1. EfficientNet (EfficientNetB0)
- **Design:** Pretrained convolutional architecture leveraging compound coefficient scaling (balancing network depth, width, and image resolution).
- **Function:** Serves as a reliable, high-accuracy baseline for satellite scene classification over EuroSAT land categories.
- **Weights:** Serialized in `.keras`, `.onnx` (15 MB), and `.tflite` (15 MB) formats for low-latency inference.

### 2. PAPNet (Pattern-Affinitive Propagation Network)
- **Design:** A custom deep architecture engineered specifically for remote sensing Earth observation.
- **Mechanism:** Integrates **Pattern Affinity Modules (PAM)** that compute non-local pairwise spatial affinity matrices:
  $$\text{Affinity}(Q, K) = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)$$
  Contextual features are aggregated across the entire spatial receptive field and combined with local representations using a learnable residual gating parameter ($\gamma$).
- **Advantage:** Overcomes the limitations of standard local convolutions by connecting disjoint spatial structures across a satellite tile (such as fragmented crop parcels, winding river channels, or discontinuous highway networks).
- **Weights:** Optimized in `.keras`, `.onnx` (29 MB), and `.tflite` (28 MB) formats for sub-second, ultra-low memory execution (<120MB RAM).

---

# File Structure

```text
dl-proj/
├── app.py                     # Unified production Flask backend with dynamic model switching
├── requirements.txt           # Lean deployment dependencies (Flask, gunicorn, onnxruntime)
├── Procfile                   # Process file for Render.com deployment (gunicorn 1 worker, 4 threads)
├── render.yaml                # Render.com blueprint configuration
├── convert_all_models.py      # Live conversion utility (.keras -> .tflite & .onnx)
├── Ai_project_report.docx     # Comprehensive project report
├── models/                    # Production model weights and scalers
│   ├── model1_efficientnet.keras / .onnx / .tflite
│   ├── model1_papnet.keras / .onnx / .tflite
│   ├── model2_suitability.keras / .onnx / .tflite
│   ├── model3_recommender.keras / .onnx / .tflite
│   ├── model2_scaler.pkl
│   └── model3_scaler.pkl
├── templates/
│   └── index.html             # UI with centered EfficientNet/UNet/PAPNet model selector
├── static/
│   ├── style.css              # Cyber-themed styling and layout
│   ├── script.js              # Frontend pipeline logic and dynamic telemetry rendering
│   └── samples/               # Sample satellite tiles for testing (forest, urban, crops, etc.)
│
├── inayath-code/              # Source code, notebooks, and models developed by Inayath
│   └── SiteWise-AI/           # Flask app, EfficientNetB0 implementation, EuroSAT training
│
├── aneesh-code/               # Source code, notebooks, and models developed by Aneesh
│   └── SiteWise-AI/           # U-Net semantic segmentation, DeepGlobe pipeline, evaluation scripts
│
└── Garv-code/                 # Source code, notebooks, and models developed by Garv
    └── site-wise-ai-papnet/   # PAPNet custom architecture, ONNX/TFLite low-memory optimizations
```

---

# Contributors

- **Garv**
- **Inayath**
- **Aneesh**

**Registration Numbers:** `24BCA8067`, `24BCA7850`, `24BCA7708`, `24BCE7118`  
*School of Computer Science and Engineering, Vellore Institute of Technology, Amaravathi*

---

# Steps to run locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.
