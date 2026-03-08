from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.data import (
    SoCalGuessrDataset,
    build_class_names,
    build_eval_transform,
    build_train_transform,
    make_labeled_samples,
    split_samples,
)
from src.engine import checkpoint_state, evaluate_epoch, train_epoch
from src.models import build_model, freeze_backbone
from src.utils import ensure_dir, get_device, save_training_curve, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SoCalGuessr model")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--arch", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        image_size=args.image_size,
        val_size=args.val_size,
        random_seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        arch=args.arch,
        freeze_backbone=args.freeze_backbone,
        early_stopping_patience=args.patience,
    )

    set_seed(cfg.random_seed)
    device = get_device()
    ensure_dir(args.out_dir)
    ensure_dir(args.checkpoint_dir)

    samples = make_labeled_samples(args.data_dir)
    class_names = build_class_names(samples)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_samples, val_samples = split_samples(
        samples,
        val_size=cfg.val_size,
        random_seed=cfg.random_seed,
    )

    train_ds = SoCalGuessrDataset(
        train_samples,
        transform=build_train_transform(cfg.image_size),
        class_to_idx=class_to_idx,
    )
    val_ds = SoCalGuessrDataset(
        val_samples,
        transform=build_eval_transform(cfg.image_size),
        class_to_idx=class_to_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg.arch, num_classes=len(class_names), pretrained=True)
    if cfg.freeze_backbone:
        freeze_backbone(model, cfg.arch)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    patience_counter = 0
    ckpt_path = Path(args.checkpoint_dir) / "best_model.pt"

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                checkpoint_state(model, class_names, cfg.arch, cfg.image_size),
                ckpt_path,
            )
            print(f"Saved best checkpoint to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print("Early stopping triggered")
                break

    curve_path = Path(args.out_dir) / "training_curve.png"
    save_training_curve(history, curve_path)
    print(f"Saved training curve to {curve_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
