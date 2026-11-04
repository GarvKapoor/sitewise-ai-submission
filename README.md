# sitewise-ai-submission

**SiteWise AI** is an automated, AI-assisted Earth observation platform designed for satellite image land-cover analysis, physical terrain indexing, and multi-criteria site suitability screening. The system translates raw Sentinel-2 satellite tiles into 10 EuroSAT land categories, computes 8 optical terrain telemetry features, and predicts optimal zoning and development uses across 8 distinct sectors.

This project is submitted by:
- Chinta Aneesh Murthy 24BCA7708
- Shaik Inayatulla Ahemed 24BCA7850
- Komari Thanvika 24BCA8067
- Garv Kapoor 24BCE7118

# Quick Links
 - Live Demo : https://sitewise-ai-submission.onrender.com
 - Youtube video Demonstrating the app : 

> **Project Report:** The complete academic and technical documentation for this project is detailed in [`Ai_project_report.docx`](./04_FINAL_PROJECT_REPORT/SiteWise_AI_Complete_Project_Report%20(1).docx)

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
sitewise-ai-submission/
├── 01_REVIEW_PRESENTATIONS/       # Review presentations and PDF slide decks
├── 02_MODEL_IMPLEMENTATIONS/       # Standalone model pipelines, source code, and benchmarks
│   ├── 01_EfficientNetB0/         # EuroSAT classification pipeline (Inayath & Thanvika)
│   ├── 02_PAPNet/                 # Pattern-Affinitive Propagation Network (Garv)
│   └── 03_UNet/                   # DeepGlobe terrain semantic segmentation (Aneesh)
├── 03_APPLICATION/                # Production full-stack deployment
│   ├── app.py                     # Unified Flask backend with multi-model inference pipeline
│   ├── models/                    # Low-memory ONNX, TFLite, and Keras model pool
│   ├── templates/                 # Frontend UI templates with interactive model selector
│   ├── static/                    # Frontend styling (CSS), UI logic (JS), and test sample tiles
│   ├── requirements.txt           # Lean deployment dependencies (Flask, gunicorn, onnxruntime)
│   ├── Procfile & render.yaml     # Render.com cloud hosting configuration
│   └── benchmark_ram.py           # Runtime memory (RAM) & latency profiling benchmark
├── 04_FINAL_PROJECT_REPORT/       # Complete technical project report (DOCX)
├── 05_IEEE_RESEARCH_ARTICLE/      # Formatted IEEE research paper & 30+ academic citations
└── 06_PROJECT_LINKS/              # Project repository link (Links.txt)
```

---

# Contributors

- **Chinta Aneesh Murthy** 24BCA7708 contributed to Implementing the model in [UNET](./02_MODEL_IMPLEMENTATIONS/03_UNet/)
- **Shaik Inayatulla Ahemed** 24BCA7850 contributed to main UI of the Website and implementing the model in [EfficientNetB0](./02_MODEL_IMPLEMENTATIONS/01_EfficientNetB0/)
- **Komari Thanvika** 24BCA8067 contributed to implementing the model in [EfficientNetB0](./02_MODEL_IMPLEMENTATIONS/01_EfficientNetB0/)
- **Garv Kapoor** 24BCE7118 contributed to implementing the model in [PAPNet](./02_MODEL_IMPLEMENTATIONS/02_PAPNet/)
---

# Steps to run locally

```bash
# 1. Create and activate a virtual environment
git clone https://github.com/GarvKapoor/sitewise-ai-submission.git
cd /Users/garvkapoor/dl-proj/03_APPLICATION
python -m venv venv
source venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.
