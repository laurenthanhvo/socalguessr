from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    SoCalGuessrDataset,
    build_class_names,
    build_eval_transform,
    make_labeled_samples,
    split_samples,
)
from src.engine import evaluate_epoch
from src.models import build_model
from src.utils import ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SoCalGuessr model")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    arch = checkpoint["arch"]
    image_size = checkpoint.get("image_size", 224)

    samples = make_labeled_samples(args.data_dir)
    _, val_samples = split_samples(samples, val_size=0.2, random_seed=args.seed)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    val_ds = SoCalGuessrDataset(
        val_samples,
        transform=build_eval_transform(image_size),
        class_to_idx=class_to_idx,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, y_true, y_pred = evaluate_epoch(model, val_loader, criterion, device)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    out_path = Path(args.out_dir) / "confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
