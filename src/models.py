from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


SUPPORTED_ARCHES = {
    "efficientnet_b0",
    "mobilenet_v3_small",
    "resnet18",
}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch not in SUPPORTED_ARCHES:
        raise ValueError(f"Unsupported arch '{arch}'. Supported: {sorted(SUPPORTED_ARCHES)}")

    if arch == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "mobilenet_v3_small":
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise AssertionError("Unreachable")


def freeze_backbone(model: nn.Module, arch: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if arch in {"efficientnet_b0", "mobilenet_v3_small"}:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True
