from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    path: Path
    label: Optional[str] = None


class SoCalGuessrDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if sample.label is None:
            return image, sample.path.name

        if self.class_to_idx is None:
            raise ValueError("class_to_idx is required for labeled samples")

        target = self.class_to_idx[sample.label]
        return image, target


def list_images(data_dir: str | Path) -> List[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    image_paths: List[Path] = []
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(path)

    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    return sorted(image_paths)


def extract_label_from_filename(filename: str) -> str:
    if "-" not in filename:
        raise ValueError(
            f"Filename '{filename}' does not match expected '<city>-<id>.jpg' format"
        )
    return filename.split("-", 1)[0]


def make_labeled_samples(data_dir: str | Path) -> List[Sample]:
    samples: List[Sample] = []
    for path in list_images(data_dir):
        label = extract_label_from_filename(path.name)
        samples.append(Sample(path=path, label=label))
    return samples


def make_unlabeled_samples(data_dir: str | Path) -> List[Sample]:
    return [Sample(path=path, label=None) for path in list_images(data_dir)]


def build_class_names(samples: Sequence[Sample]) -> List[str]:
    labels = sorted({sample.label for sample in samples if sample.label is not None})
    return labels


def split_samples(
    samples: Sequence[Sample],
    val_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    labels = [sample.label for sample in samples]
    train_idx, val_idx = train_test_split(
        list(range(len(samples))),
        test_size=val_size,
        random_state=random_seed,
        stratify=labels,
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def build_train_transform(image_size: int = 224):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size: int = 224):
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
