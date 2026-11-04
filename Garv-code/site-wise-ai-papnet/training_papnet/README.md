# PAPNet (Pattern-Affinitive Propagation Network) Training Pipeline

This folder contains the complete, self-contained training environment for **PAPNet (Pattern-Affinitive Propagation Network)** for EuroSAT satellite land classification and downstream site suitability analysis.

---

## 1. Overview & Theoretical Motivation

### Why PAPNet over Traditional CNNs (like EfficientNet)?
Standard convolutional backbones (e.g., EfficientNet, ResNet) process images strictly using local receptive fields (typically $3\times 3$ convolutional filters). In remote sensing and satellite Earth observation imagery:
1. **Disjoint Spatial Coherence**: Agricultural crop fields, river paths, forest corridors, and road grids exhibit strong structural continuity and textural affinity across geographically dispersed patches of the image.
2. **Multi-Scale Pattern Relationships**: High-frequency textures (such as residential rooftops and road intersections) coexist with macro-environmental features (such as forest canopies and lakes).
3. **Pattern-Affinitive Propagation (PAP)**:
   - Computes a non-local **Pattern Affinity Matrix** across all spatial patches of the feature map:
     $$A_{i, j} = \text{Softmax}\left(\frac{Q_i \cdot K_j^T}{\sqrt{d}}\right)$$
   - Propagates feature context across the receptive field:
     $$P = A \cdot V$$
   - Enriches the representation via learnable residual gating:
     $$Y = X + \gamma \cdot \text{Proj}(P)$$
     where $\gamma$ is a learnable parameter initialized to $0$ to guarantee smooth, stable optimization from initialization.

---

## 2. Directory Structure

```text
training_papnet/
├── papnet_model.py               # Reusable Keras layers & PAPNet model definitions
├── project_notebook_papnet.ipynb # End-to-end interactive Jupyter notebook
├── train_papnet.py               # Headless CLI training script
├── create_papnet_notebook.py     # Generator script for the notebook
├── model1_papnet.keras           # Trained PAPNet model artifact
├── model2_suitability.keras      # Trained Model 2 suitability predictor
├── model2_scaler.pkl             # StandardScaler for Model 2
├── model3_recommender.keras      # Trained Model 3 site recommender
├── model3_scaler.pkl             # StandardScaler for Model 3
└── README.md                     # Documentation
```

---

## 3. How to Train PAPNet

### Option A: Using the CLI Script (`train_papnet.py`)
Run training directly from the command line:

```bash
# Full training on EuroSAT
python training_papnet/train_papnet.py --epochs 20 --batch-size 32

# Quick verification run (fast demo mode)
python training_papnet/train_papnet.py --quick-demo --epochs 2
```

Flags:
- `--epochs`: Number of training epochs (default: `15`).
- `--batch-size`: Batch size (default: `32`).
- `--quick-demo`: Fast verification mode on a subset.
- `--save-to-models`: Automatically copies the trained `model1_papnet.keras` to `models/` (enabled by default).

### Option B: Using the Jupyter Notebook (`project_notebook_papnet.ipynb`)
Open and run in Jupyter Lab or Google Colab:
1. Open `training_papnet/project_notebook_papnet.ipynb`.
2. Execute all cells sequentially.
3. The notebook will automatically:
   - Download the EuroSAT dataset via Hugging Face (`tanganke/eurosat`).
   - Run exploratory data visualization and train/val/test splits.
   - Build and train Model 1 (**PAPNet**).
   - Evaluate test accuracy, classification report, and confusion matrix.
   - Extract 18 base features (8 visual features + 10 PAPNet probabilities).
   - Train and save Model 2 (Suitability) and Model 3 (Site Recommender).
   - Save all artifacts to `models/model1_papnet.keras`.

---

## 4. Model Architecture Details

| Stage | Input Resolution | Channels | Operations |
|---|---|---|---|
| **Input & Augmentation** | $64\times 64\times 3$ | 3 | RandomFlip, RandomRotation, RandomZoom, Rescaling $(1/255)$ |
| **Stem** | $64\times 64$ | 32 | Conv2D ($3\times 3$), BatchNorm, Swish |
| **Stage 1** | $32\times 32$ | 64 | PAPResidualBlock (stride 2) + PAPResidualBlock with PAP (key_dim=16) |
| **Stage 2** | $16\times 16$ | 128 | PAPResidualBlock (stride 2) + PAPResidualBlock with PAP (key_dim=32) |
| **Stage 3** | $8\times 8$ | 256 | PAPResidualBlock (stride 2) + PAPResidualBlock with PAP (key_dim=64) |
| **Stage 4** | $8\times 8$ | 256 | Top Pattern-Affinitive Propagation Refinement |
| **Head** | Vector (512) | 128 $\to$ 10 | GAP + GMP Concatenation, Dense(128), BatchNorm, Dropout(0.3), Dense(10, Softmax) |

---

## 5. Integrating PAPNet with the Flask App (`app.py`)

The Flask backend in `app.py` has been updated to automatically support PAPNet:
- By default or via environment variable `MODEL1_TYPE=papnet`, `app.py` will load `models/model1_papnet.keras`.
- If `model1_papnet.keras` is not present, it gracefully falls back to `models/model1_efficientnet.keras`.
- Custom layers (`PatternAffinityModule`, `PAPResidualBlock`) are registered in Keras so models deserialize seamlessly.

