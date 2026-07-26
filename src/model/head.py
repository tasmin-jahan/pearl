"""
Classification head module (v3).

Architecture:
    GlobalAvgPool -> Linear(in_features, hidden_dim) -> SiLU
    -> Linear(hidden_dim, hidden_dim) -> SiLU
    -> Linear(hidden_dim, num_classes)

SiLU matches the native activation used inside EfficientNet/ConvNeXt/
MobileNetV3 backbones and is stdlib in PyTorch. The head is shallow
enough that SiLU's smooth, non-monotonic shape is safe here.
"""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """Custom classification head for transfer learning.

    Args:
        in_features: Number of input features from the backbone.
        head_config: Dict with 'hidden_dim' and 'dropout' keys.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        in_features: int,
        head_config: dict,
        num_classes: int = 2,
    ):
        super().__init__()
        hidden_dim = head_config.get("hidden_dim", 256)
        dropout = head_config.get("dropout", 0.5)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # 2-layer head with SiLU activations (matches backbone-native activation).
        # BN between FC layers stabilizes head training; dropout only after first
        # activation (it's the most regularization-sensitive spot).
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 512 else 128),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(hidden_dim // 2 if hidden_dim >= 512 else 128),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 512 else 128, num_classes),
        )

    def forward(self, x):
        # x may be (B, C, H, W) or (B, C) depending on backbone
        if x.dim() == 4:
            x = self.pool(x)
            x = self.flatten(x)
        return self.fc(x)