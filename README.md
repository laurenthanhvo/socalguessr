# SoCalGuessr

Starter repo for the DSC 140B final project: train a neural network to predict which Southern California city a Street View image came from.

This repo is organized so you can:

1. train and validate locally,
2. keep everything clean in GitHub,
3. export a **Gradescope-safe submission** with a top-level `predict.py` and one checkpoint file.

The project spec requires your submitted `predict.py` to expose a function named `predict(image_path)` that returns a dictionary mapping each test filename to a predicted city. It also notes that you may upload `predict.py` together with a weights file in a zip, and that those files must be at the **top level** of the zip. It also limits the combined file size to **under 50MB** and runtime/memory on Gradescope. fileciteturn0file0

## Recommended repo layout

```text
socalguessr_repo/
├── .gitignore
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── predict.py                # standalone Gradescope submission file
├── checkpoints/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── reports/
│   └── report_template.md
├── scripts/
│   └── make_submission_zip.sh
└── src/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── engine.py
    ├── models.py
    └── utils.py
```

## What goes where

- `data/`
  - put the extracted training images here
  - expected filenames look like `<city>-<random-id>.jpg`
- `train.py`
  - trains your model and saves the best checkpoint to `checkpoints/`
- `evaluate.py`
  - evaluates a saved checkpoint on a validation split and saves a confusion matrix + metrics
- `predict.py`
  - self-contained inference file for Gradescope
- `outputs/`
  - training curve, confusion matrix, logs
- `reports/report_template.md`
  - fill this in for the written report
- `scripts/make_submission_zip.sh`
  - packages only the files needed for Gradescope

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

The project PDF links the training set zip. Download it, unzip it, and place the images in `data/` (or a subfolder like `data/train_images/`). The labels come from the filename prefix. fileciteturn0file0

## Train

```bash
python train.py \
  --data-dir data \
  --arch efficientnet_b0 \
  --epochs 12 \
  --batch-size 64 \
  --lr 1e-4
```

This will:
- parse labels from image filenames,
- create a train/validation split,
- fine-tune a pretrained vision model,
- save the best checkpoint to `checkpoints/best_model.pt`,
- save plots/metrics to `outputs/`.

## Evaluate

```bash
python evaluate.py \
  --data-dir data \
  --checkpoint checkpoints/best_model.pt
```

## Build submission zip

```bash
bash scripts/make_submission_zip.sh checkpoints/best_model.pt
```

This creates `submission.zip` containing:
- `predict.py`
- `best_model.pt`

both at the top level, matching the project instructions. fileciteturn0file0

## Notes

- Start with `efficientnet_b0` for a strong baseline.
- If your checkpoint ends up too large, switch to `mobilenet_v3_small`.
- Keep your final checkpoint and `predict.py` under the 50MB combined submission limit. fileciteturn0file0
