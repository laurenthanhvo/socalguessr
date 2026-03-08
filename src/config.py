from dataclasses import dataclass


@dataclass
class TrainConfig:
    image_size: int = 224
    val_size: float = 0.2
    random_seed: int = 42
    num_workers: int = 4
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    arch: str = "efficientnet_b0"
    freeze_backbone: bool = False
    early_stopping_patience: int = 4
