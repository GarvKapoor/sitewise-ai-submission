# Training the SiteWise AI U-Net

Training runs **separately from the website** — nothing here trains inside the
frontend or the Flask app. Run these scripts locally or in **Google Colab**
(needs a GPU for reasonable times), then copy the artefacts into
`backend/models/`.

> No trained model ships in this repository and **no metrics here are
> pre-computed or fabricated**: the website shows evaluation results only from
> `backend/models/evaluation.json`, written by `evaluate.py` after a real run.

## 1. Get the dataset

Download the **DeepGlobe Land Cover Classification Challenge** dataset
([Kaggle mirror](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset),
[Official site](http://deepglobe.org/)). The dataset is **not** distributed
through this repository (licence restricts redistribution).

Expected layout:

```
data/deepglobe/
├── train/                       # 1,146 tiles
│   ├── 0a0b0c..._sat.jpg        # 1024×1024 RGB satellite tile
│   ├── 0a0b0c..._mask.png       # matching pixel-level colour mask
│   └── ...
└── class_dict.csv               # optional — dataset's own colour mapping
```

The environment never auto-downloads the dataset.

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

(Colab: `pip install -q tensorflow` is already present; just upload/notebook-mount `backend/` and `training/`.)

## 3. Train

```bash
python training/train_unet.py \
    --data data/deepglobe/train \
    --epochs 60 --batch-size 8 --crop 256 \
    --output backend/models/unet_model.keras
```

Key flags (all optional except `--data`):

| Flag | Default | Meaning |
|---|---|---|
| `--data` | — | directory of `*_sat.jpg` / `*_mask.png` pairs |
| `--epochs` | 60 | training epochs |
| `--batch-size` | 8 | mini-batch size (lower on smaller GPUs) |
| `--crop` | 256 | random side length cropped from the 1024² tiles |
| `--val-split` | 0.15 | reproducible validation fraction |
| `--seed` | 42 | split / augmentation seed (reproducibility) |
| `--output` | `backend/models/unet_model.keras` | output `.keras` file |

The script:

- loads image + mask pairs (`dataset_loader.py`),
- converts each RGB mask to a class-index mask with the **verified DeepGlobe
  palette** (channels binarised at threshold 128, as recommended by the
  challenge),
- uses a reproducible shuffled train/validation split,
- trains a U-Net built with a **7-channel softmax head** (channel order =
  `backend/config/model_config.py CLASS_NAMES`),
- minimises **categorical focal loss + CategoricalCrossentropy** blend
  (segmentation loss; robust against seven-way class imbalance), and tracks
  Mean IoU (Unknown excluded, per the challenge) and Dice,
- saves **only** the trained architecture + weights — never metrics.

## 4. Evaluate

```bash
python training/evaluate.py \
    --model backend/models/unet_model.keras \
    --data data/deepglobe/train
```

Computes, on the held-out validation tiles with the **same seed/split** as
training: pixel accuracy, Mean IoU (excluding Unknown) and Dice coefficient,
then writes genuine numbers to:

```
backend/models/evaluation.json
{
  "unet": {"training_accuracy": ..., "validation_loss": ..., "mean_iou": ..., "dice": ...},
  "source": "offline evaluation of unet_model.keras over ... tiles",
  "generated": "ISO date"
}
```

`GET /api/metrics` picks this file up; the Model Performance section then shows
real numbers. Until that file exists the site correctly shows
**"Evaluation pending"**.

## 5. What is deliberately NOT in this repo

- the dataset itself,
- any trained `.keras`/`.h5` binaries (gitignored),
- any accuracy/IoU/Dice numbers not produced by *your own* evaluation run.
