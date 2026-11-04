# SiteWise AI: Architectural Specification & Methodology

This document details the machine learning architecture, mathematical foundations, end-to-end methodology, dataset curation, and preprocessing pipeline powering **SiteWise AI** — an automated satellite Earth observation platform for land-use classification, physical terrain indexing, and multi-criteria site suitability prediction.

---

## 1. System Overview & Methodology

SiteWise AI operates on a multi-stage hierarchical inference pipeline that translates raw multi-spectral satellite imagery into actionable zoning and development recommendations.

```
+-------------------------------------------------------------------+
|               Input: Sentinel-2 Satellite Tile                    |
|                      (64 x 64 x 3 RGB)                            |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|  Stage 1: Land Classification Engine (PAPNet)                     |
|  - Pattern-Affinitive Propagation Network                         |
|  - Spatial Attention & Non-Local Texture Aggregation              |
|  - Output: 10-Class Softmax Probability Distribution              |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|  Stage 2: Multispectral & Visual Feature Extraction               |
|  - Computes 8 Heuristic Optical Indices                           |
|    (Mean R, G, B, Brightness, Vegetation, Water, Built-up, Road)  |
|  - Dual Land & Water Feature Disambiguation                       |
+-------------------------------------------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+---------------------------------+  +------------------------------+
| Stage 3: Facility Predictor     |  | Stage 4: Site Recommender    |
| (Model 2)                       |  | (Model 3)                    |
| - Target-specific suitability   |  | - 8-Profile Suitability      |
| - Binary neural scoring         |  | - Multi-Objective Sigmoid    |
| - Evaluates targeted proposal   |  | - Calibrated Ranking Matrix  |
+---------------------------------+  +------------------------------+
```

### The 4-Stage Methodology

1. **Stage 1 — Primary Land-Use Classification (PAPNet)**: The satellite patch ($64 \times 64 \times 3$) is evaluated by a deep Pattern-Affinitive Propagation Network (PAPNet). PAPNet computes spatial affinity matrices to discover disjoint structural coherence (such as fragmented crop fields, waterways, or discontinuous road segments) and produces a 10-dimensional posterior probability vector.
2. **Stage 2 — Visual Feature Extraction & Telemetry**: In parallel with deep embeddings, 8 physical surface telemetry indices are computed directly from pixel channel distributions (vegetation index, water absorption index, built-up structural density, linear road contrast, mean RGB, and overall scene luminance).
3. **Stage 3 — Facility-Targeted Suitability Analysis (Model 2)**: The 18-element composite representation (10 classification probabilities + 8 visual indices) is concatenated with a facility identifier code and normalized via a standard scaler. A dedicated neural classifier evaluates whether the terrain satisfies specific spatial constraints for targeted facilities.
4. **Stage 4 — Multi-Criteria Site Recommendation Matrix (Model 3)**: An 8-head multi-task network scores the site across eight distinct real-world development categories simultaneously (*Residential, Commercial, Industrial, Educational, Healthcare, Agricultural, Transportation, and Green / Conservation*), establishing the optimal development use for the parcel.

---

## 2. Model Architecture: PAPNet (Pattern-Affinitive Propagation Network)

### 2.1 Theoretical Motivation

Standard convolutional architectures process images strictly via local receptive fields (typically $3 \times 3$ or $5 \times 5$ spatial kernels). In remote sensing Earth observation imagery, standard convolutions encounter distinct challenges:

1. **Disjoint Spatial Coherence**: Agricultural parcels, river networks, road corridors, and forested patches often span discontinuous regions across a tile. Standard convolutions require excessively deep layer stacks to correlate spatially separated patches.
2. **Scale & Texture Invariance**: Micro-level textures (individual rooftops, road lines) must be dynamically reconciled with macro-level contexts (surrounding brushland, water bodies).
3. **Pattern Affinity**: In satellite imagery, distinct pixels sharing identical textural and spectral signatures should propagate contextual features to one another regardless of Euclidean spatial distance.

To solve this, **PAPNet** integrates non-local **Pattern-Affinitive Propagation (PAP) modules** into a deep residual convolutional backbone designed natively for $64 \times 64$ satellite tiles.

---

### 2.2 Mathematical Formulation of the PAP Module

The Pattern-Affinitive Propagation (PAP) layer computes pairwise affinity across flattened spatial feature patches and propagates non-local semantic context with learnable residual gating.

Given an input feature tensor $X \in \mathbb{R}^{B \times H \times W \times C}$:

1. **Linear Projections**:
   $$Q = \text{Conv}_{1\times 1}^{q}(X) \in \mathbb{R}^{B \times H \times W \times d_k}$$
   $$K = \text{Conv}_{1\times 1}^{k}(X) \in \mathbb{R}^{B \times H \times W \times d_k}$$
   $$V = \text{Conv}_{1\times 1}^{v}(X) \in \mathbb{R}^{B \times H \times W \times d_v}$$

2. **Spatial Flattening**: The spatial dimensions $H \times W$ are flattened into $N = H \cdot W$ spatial tokens:
   $$Q_{\text{flat}} \in \mathbb{R}^{B \times N \times d_k}, \quad K_{\text{flat}} \in \mathbb{R}^{B \times N \times d_k}, \quad V_{\text{flat}} \in \mathbb{R}^{B \times N \times d_v}$$

3. **Pattern Affinity Matrix**: Pairwise affinity between spatial locations $i$ and $j$ is computed via scaled dot-product and row-wise softmax normalization:
   $$A_{i, j} = \frac{\exp\left(\frac{Q_i \cdot K_j^T}{\sqrt{d_k}}\right)}{\sum_{m=1}^{N} \exp\left(\frac{Q_i \cdot K_m^T}{\sqrt{d_k}}\right)}, \quad A \in \mathbb{R}^{B \times N \times N}$$

4. **Contextual Propagation**: Contextual information is propagated across the entire spatial grid:
   $$P_{\text{flat}} = A \cdot V_{\text{flat}} \in \mathbb{R}^{B \times N \times d_v}$$
   $$P = \text{Reshape}(P_{\text{flat}}) \in \mathbb{R}^{B \times H \times W \times d_v}$$

5. **Projection & Residual Gating**:
   $$\tilde{P} = \text{BatchNorm}(\text{Conv}_{1\times 1}^{\text{proj}}(P))$$
   $$Y = X + \gamma \cdot \tilde{P}$$
   where $\gamma \in \mathbb{R}^1$ is a scalar gating coefficient initialized to $0.0$. This ensures that at the start of training, the layer acts as an identity mapping ($Y = X$), guaranteeing smooth gradient flow and stable optimization.

---

### 2.3 PAP Residual Block (`PAPResidualBlock`)

The core building block of PAPNet combines standard residual convolutions with optional PAP attention:

```
Input X (H x W x C_in)
  │
  ├──► [Shortcut Branch] ──► 1x1 Conv + BatchNorm (if stride > 1 or C_in != C_out) ──┐
  │                                                                                  │
  └──► [Residual Branch]                                                             │
         ├── 3x3 Conv2D (stride = s) + BatchNorm + Swish Activation                  │
         ├── 3x3 Conv2D (stride = 1) + BatchNorm                                     │
         ├── [Optional] Pattern-Affinitive Propagation (PAP) Module                  │
         │                                                                           │
         ▼                                                                           ▼
       ( y ) ───────────────────────────── (+) ◄─────────────────────────────────────┘
                                           │
                                           ▼
                                 [ Swish Activation ]
                                           │
                                           ▼
                                Output Y (H' x W' x C_out)
```

---

### 2.4 Complete PAPNet Layer Hierarchy

| Hierarchy Stage | Layer Name | Filter / Unit Dim | Output Shape $(H \times W \times C)$ | Operations & Mechanics |
|---|---|---|---|---|
| **Input** | `satellite_image` | — | $64 \times 64 \times 3$ | Sentinel-2 RGB patch |
| **Augmentation** | `aug_flip`, `aug_rot`, `aug_zoom` | — | $64 \times 64 \times 3$ | Random H/V flips, $\pm 20\%$ rotation, $\pm 20\%$ zoom |
| **Rescaling** | `rescaling` | — | $64 \times 64 \times 3$ | Pixel normalization $[0, 255] \to [0.0, 1.0]$ |
| **Stem** | `stem_conv`, `stem_bn`, `stem_act` | 32 filters ($3\times 3$) | $64 \times 64 \times 32$ | Conv2D, Batch Normalization, Swish |
| **Stage 1** | `stage1_block1` | 64 filters ($3\times 3$) | $32 \times 32 \times 64$ | Downsampling block (stride 2) |
| | `stage1_block2` | 64 filters ($3\times 3$) | $32 \times 32 \times 64$ | Identity residual block + PAP Module ($d_k=16$) |
| **Stage 2** | `stage2_block1` | 128 filters ($3\times 3$) | $16 \times 16 \times 128$ | Downsampling block (stride 2) |
| | `stage2_block2` | 128 filters ($3\times 3$) | $16 \times 16 \times 128$ | Identity residual block + PAP Module ($d_k=32$) |
| **Stage 3** | `stage3_block1` | 256 filters ($3\times 3$) | $8 \times 8 \times 256$ | Downsampling block (stride 2) |
| | `stage3_block2` | 256 filters ($3\times 3$) | $8 \times 8 \times 256$ | Identity residual block + PAP Module ($d_k=64$) |
| **Stage 4** | `pap_top_refinement` | $d_k=64, d_v=256$ | $8 \times 8 \times 256$ | Global non-local refinement over $8\times 8$ grid |
| **Pooling** | `global_avg_pool` | — | 256 | Global Average Pooling across spatial dimensions |
| | `global_max_pool` | — | 256 | Global Max Pooling across spatial dimensions |
| | `pool_concat` | — | 512 | Concatenation of average and peak salient features |
| **Head** | `head_dense` | 128 units | 128 | Dense projection with Swish activation |
| | `head_bn`, `head_dropout` | rate = 0.30 | 128 | Batch Normalization & Dropout regularization |
| | `land_class_probabilities` | 10 units | 10 | Softmax activation over 10 EuroSAT classes |

**Total Parameters**: $\approx 3,061,000$ trainable weights.

---

## 3. Heuristic Visual Indices & Dual Recognition

To provide transparent, interpretable telemetry alongside neural probabilities, the pipeline extracts eight physical optical properties:

```python
# 1. Mean Red, Green, Blue Reflectance
mean_r = np.mean(R)
mean_g = np.mean(G)
mean_b = np.mean(B)

# 2. Scene Luminance / Brightness
brightness = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

# 3. Chlorophyll / Vegetation Density
vegetation = np.mean((G > R * 1.05) & (G > B * 1.05))

# 4. Water Absorption Index
water = np.mean((B > R * 1.10) & (B > G * 0.95))

# 5. Built-up Impervious Surface Density
built_up = np.mean((brightness > 0.45) & (abs(R - G) < 0.10) & (abs(G - B) < 0.10))

# 6. Linear Road & Spectral Contrast Indicator
road = np.mean((abs(R - G) < 0.05) & (abs(G - B) < 0.05) & (brightness > 0.25) & (brightness < 0.60))
```

### Dual Land & Water Disambiguation
Satellite scenes frequently contain both terrestrial ground cover and water bodies (e.g., rivers traversing forests, lakes bordered by residential zones). The system computes:
- $\text{Total Water Prob} = P(\text{River}) + P(\text{Lake/Sea})$
- $\text{Total Land Prob} = \sum_{c \in \text{Land}} P(c)$
- If both probabilities exceed calibrated thresholds ($> 0.15$), the scene is flagged as a **Dual Recognition Mixed Site** (e.g., *"Forest with Water Body (River)"*), ensuring flood vulnerability is accounted for in downstream suitability calculations.

---

## 4. Downstream Suitability & Recommendation Networks

The 18-element baseline vector is defined as:
$$X_{\text{base}} = [\underbrace{v_1, v_2, \dots, v_8}_{\text{Visual Telemetry Indices}}, \; \underbrace{p_1, p_2, \dots, p_{10}}_{\text{PAPNet Class Probabilities}}] \in \mathbb{R}^{18}$$

### 4.1 Model 2: Facility Suitability Predictor
Model 2 predicts the feasibility of establishing a targeted facility on the scanned terrain:
- **Input Dimension**: 19 features ($X_{\text{base}}$ concatenated with the facility operational code).
- **Architecture**:
  - `Dense(64, activation='relu')`
  - `BatchNormalization()`
  - `Dropout(0.20)`
  - `Dense(32, activation='relu')`
  - `Dense(1, activation='sigmoid')`
- **Loss Function**: Binary Cross-Entropy (BCE).
- **Output**: Direct calibrated probability score $[0.0, 1.0] \to [0.0\%, 100.0\%]$.

### 4.2 Model 3: Multi-Criteria 8-Class Site Recommender
Model 3 evaluates the parcel across eight development sectors simultaneously:
- **Input Dimension**: 18 features ($X_{\text{base}}$ scaled via `StandardScaler`).
- **Architecture**:
  - `Dense(128, activation='relu')`
  - `Dropout(0.30)`
  - `Dense(64, activation='relu')`
  - `Dropout(0.20)`
  - `Dense(32, activation='relu')`
  - `Dense(8, activation='sigmoid')`
- **Output Categories**:
  1. *Residential Development*
  2. *Commercial Development*
  3. *Industrial Development*
  4. *Educational Development*
  5. *Healthcare Development*
  6. *Agricultural Use*
  7. *Transportation Infrastructure*
  8. *Green / Conservation*
- **Suitability Decision Boundaries**:
  - $\ge 70.0\%$: **HIGH SUITABILITY**
  - $40.0\% - 69.9\%$: **MODERATE SUITABILITY**
  - $20.0\% - 39.9\%$: **LOW SUITABILITY**
  - $< 20.0\%$: **VERY LOW SUITABILITY**

---

## 5. Dataset & Preprocessing Details

### 5.1 Dataset Description: EuroSAT (Sentinel-2)

The model is trained and benchmarked on the **EuroSAT Land Use and Land Cover** dataset, sourced from the European Space Agency's (ESA) **Sentinel-2 MSI (Multi-Spectral Instrument)** satellite constellation.

- **Total Samples**: 27,000 georeferenced satellite image patches.
- **Geographic Coverage**: 34 European nations across varied climatic and ecological zones.
- **Image Spatial Resolution**: 10 meters Ground Sample Distance (GSD) per pixel.
- **Patch Resolution**: $64 \times 64$ pixels (covering approximately $640 \text{ m} \times 640 \text{ m}$ per image).
- **Spectral Bands**: RGB (Band 4: 665 nm, Band 3: 560 nm, Band 2: 490 nm).

#### Class Distribution (2,000 – 3,000 images per class):
1. **Annual Crop Land**: Seasonally tilled fields and arable land.
2. **Forest**: Coniferous, broadleaf, and mixed forest canopies.
3. **Brushland or Shrubland**: Sclerophyllous vegetation, arid scrub, and moorland.
4. **Highway or Road**: Inter-city motorways, bypasses, and secondary roads.
5. **Industrial Buildings**: Factories, storage depots, and heavy commercial parks.
6. **Pasture Land**: Meadow pastures and natural grasslands.
7. **Permanent Crop Land**: Vineyards, fruit orchards, and berry plantations.
8. **Residential Buildings**: Low and high-density residential housing and urban settlements.
9. **River**: Flowing natural and artificial inland freshwater channels.
10. **Lake or Sea**: Open ocean, maritime coastal waters, and inland reservoirs.

---

### 5.2 Data Partitioning & Stratification

To eliminate sampling bias and preserve class frequency across splits, stratified random sampling is enforced:

```
Total Dataset: 27,000 Patches
├── Training Split:   18,900 samples (70%)
├── Validation Split:  4,050 samples (15%)
└── Testing Split:     4,050 samples (15%)
```

- **Stratification**: Guaranteed equal proportional class distribution across all three partitions ($2,700$ patches per class distributed $70/15/15$).
- **Random Seed**: Fixed pseudo-random seed (`42`) ensures deterministic reproducibility.

---

### 5.3 Data Preprocessing & Augmentation Pipeline

#### 1. Input Normalization & Scaling
- **Pixel Intensity Normalization**: Raw 8-bit integer channel values in range $[0, 255]$ are dynamically rescaled to continuous floating-point values in $[0.0, 1.0]$:
  $$x_{\text{normalized}} = \frac{x_{\text{raw}}}{255.0}$$
- **Tabular Feature Normalization**: The 18-dimensional vector fed into Models 2 and 3 undergoes Z-score standardization:
  $$z = \frac{x - \mu}{\sigma}$$
  where mean $\mu$ and standard deviation $\sigma$ are fit exclusively on the training split and stored in serialized scalers (`model2_scaler.pkl`, `model3_scaler.pkl`).

#### 2. Training-Time Data Augmentation
To prevent overfitting on static spatial orientations and improve invariance to satellite orbital headings:
- **Random Horizontal & Vertical Flips**: Probability $p = 0.5$ for both axes.
- **Random Rotations**: Uniform random rotational perturbation $\theta \in [-0.2\pi, +0.2\pi]$ (approx $\pm 72^\circ$), filling borders using reflective boundary conditions.
- **Random Zoom**: Inward and outward scaling within $\pm 20\%$ to simulate altitude variations.

---

## 6. Training Hyperparameters & Optimization Protocol

| Parameter | Configuration | Justification |
|---|---|---|
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adaptive moment estimation provides fast convergence across spatial attention weights |
| **Initial Learning Rate** | $1 \times 10^{-3}$ ($0.001$) | Standard initialization for residual networks with learnable gating |
| **Learning Rate Schedule** | `ReduceLROnPlateau` | Halves learning rate ($\text{factor}=0.5$) if validation loss plateaus for 2 consecutive epochs ($\text{min\_lr}=10^{-6}$) |
| **Early Stopping** | `EarlyStopping(patience=5)` | Restores best model weights according to minimum validation loss |
| **Batch Size** | 32 (or 64 for headless multi-GPU) | Balances stochastic gradient noise with GPU memory utilization |
| **Loss Function** | Sparse Categorical Cross-Entropy | Direct optimization over integer class targets |
| **Regularization** | Dropout ($0.30$) + Batch Normalization | Mitigates co-adaptation of features across deep layers |

---

## 7. Model Serialization & Low-Memory Edge Runtimes

For deployment on memory-constrained infrastructure (such as free-tier cloud containers with strict 512 MB RAM ceilings), the trained models are compiled into lightweight execution formats:

1. **ONNX Runtime (`.onnx`)**: Graph-optimized static models run using single-threaded CPU execution providers (`intra_op_num_threads = 1`), reducing runtime memory overhead to **$< 115\text{ MB}$ RAM**.
2. **LiteRT / TFLite (`.tflite`)**: Quantized flatbuffer models executed through Google's standalone LiteRT interpreter with XNNPACK acceleration, consuming **$\approx 120\text{ MB}$ RAM**.
3. **Lazy-Load Fallback**: Full deep learning frameworks are prevented from loading into process memory unless specifically required, ensuring rock-solid stability and zero Out-of-Memory (OOM) termination.

