from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    return mean_loss, acc, y_true, y_pred


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    return mean_loss, acc


def checkpoint_state(
    model: nn.Module,
    class_names: List[str],
    arch: str,
    image_size: int,
) -> Dict:
    return {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "arch": arch,
        "image_size": image_size,
    }
