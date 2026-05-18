"""
Model builder: timm backbone + freeze lower layers + custom classification head.
"""

import timm
import torch.nn as nn

from src.model.head import ClassificationHead


def build_model(model_config: dict) -> nn.Module:
    """Build a transfer-learning model from config.

    Steps:
      1. Load pretrained backbone from timm (no default head)
      2. Freeze lower freeze_fraction of parameters
      3. Attach custom ClassificationHead

    Args:
        model_config: Model config dict with keys: timm_name, pretrained,
                      freeze_fraction, head, num_classes.

    Returns:
        nn.Module ready for training.
    """
    # Create backbone without classifier head
    backbone = timm.create_model(
        model_config["timm_name"],
        pretrained=model_config.get("pretrained", True),
        num_classes=0,  # removes default head
    )

    # Freeze lower fraction of parameters
    freeze_fraction = model_config.get("freeze_fraction", 0.60)
    params = list(backbone.named_parameters())
    freeze_up_to = int(len(params) * freeze_fraction)
    for _, param in params[:freeze_up_to]:
        param.requires_grad = False

    frozen = sum(1 for _, p in backbone.named_parameters() if not p.requires_grad)
    total = len(params)
    print(f"[Builder] Frozen {frozen}/{total} parameters ({frozen/total*100:.1f}%)")

    # Get feature dimension from backbone
    in_features = backbone.num_features

    # Attach custom classification head
    head_config = model_config.get("head", {"hidden_dim": 256, "dropout": 0.5})
    head = ClassificationHead(in_features, head_config)

    model = _ModelWithHead(backbone, head)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Builder] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    return model


class _ModelWithHead(nn.Module):
    """Wrapper combining a timm backbone with a custom head."""

    def __init__(self, backbone: nn.Module, head: ClassificationHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)  # (B, num_features) or (B, C, H, W)
        logits = self.head(features)
        return logits
