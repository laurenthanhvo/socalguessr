"""
Standalone inference script for Gradescope.

This file is intentionally self-contained so you can submit it together with a
single checkpoint file in a top-level zip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvm
import torchvision.transforms as T


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CHECKPOINT_CANDIDATES = [
    "best_model.pt",
    "model.pt",
    "weights.pt",
    "checkpoint.pt",
]


class InferenceDataset(Dataset):
    def __init__(self, image_dir: str | Path, image_size: int = 224) -> None:
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            [
                p
                for p in self.image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, path.name


def _build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "mobilenet_v3_small":
        model = tvm.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet18":
        model = tvm.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture in checkpoint: {arch}")


def _find_checkpoint(base_dir: Path) -> Path:
    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        candidate_path = base_dir / candidate
        if candidate_path.exists():
            return candidate_path

    pt_files = sorted(base_dir.glob("*.pt"))
    if pt_files:
        return pt_files[0]

    raise FileNotFoundError(
        "No checkpoint file found next to predict.py. "
        "Expected something like best_model.pt"
    )


def predict(image_path: str) -> Dict[str, str]:
    """
    Parameters
    ----------
    image_path : str
        Path to directory containing test images.

    Returns
    -------
    Dict[str, str]
        Mapping from image filename to predicted city label.
    """
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = _find_checkpoint(script_dir)
    device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names: List[str] = checkpoint["class_names"]
    arch = checkpoint["arch"]
    image_size = checkpoint.get("image_size", 224)

    model = _build_model(arch, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = InferenceDataset(image_path, image_size=image_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions: Dict[str, str] = {}
    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)
            logits = model(images)
            pred_idx = logits.argmax(dim=1).cpu().tolist()
            for filename, idx in zip(filenames, pred_idx):
                predictions[filename] = class_names[idx]

    return predictions
