"""Model definitions for emotion classification."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision import models


EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


class EmotionClassifier(nn.Module):
    """Emotion classifier backed by torchvision models.

    Supports grayscale FER-like inputs by adapting the first convolution
    to one channel and replacing the classification head.
    """

    def __init__(
        self,
        num_classes: int = len(EMOTION_CLASSES),
        backbone: Literal["resnet18", "efficientnet_b0"] = "resnet18",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            self.model.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            self.model.features[0][0] = nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Choose from ['resnet18', 'efficientnet_b0']."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



def load_model(
    checkpoint_path: str | None,
    device: torch.device,
    backbone: Literal["resnet18", "efficientnet_b0"] = "resnet18",
    num_classes: int = len(EMOTION_CLASSES),
) -> EmotionClassifier:
    """Build and optionally load model weights from a checkpoint file."""
    model = EmotionClassifier(num_classes=num_classes, backbone=backbone, pretrained=False)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
