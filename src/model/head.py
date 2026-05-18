"""
Classification head module.

GlobalAveragePooling → Linear(in_features, 256) → ReLU → Dropout(0.5) → Linear(256, 2)
"""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """Custom classification head for transfer learning.

    Args:
        in_features: Number of input features from the backbone.
        head_config: Dict with 'hidden_dim' and 'dropout' keys.
    """

    def __init__(self, in_features: int, head_config: dict):
        super().__init__()
        hidden_dim = head_config.get("hidden_dim", 256)
        dropout = head_config.get("dropout", 0.5)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        # x may be (B, C, H, W) or (B, C) depending on backbone
        if x.dim() == 4:
            x = self.pool(x)
            x = self.flatten(x)
        return self.fc(x)
