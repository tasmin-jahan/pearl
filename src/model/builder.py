"""
Model builder: timm backbone + freeze lower layers + custom classification head.
"""

import timm
import torch
import torch.nn as nn

from src.model.head import ClassificationHead


def build_model(model_config: dict) -> nn.Module:
    """Build a transfer-learning model from config.

    Steps:
      1. Load pretrained backbone from timm (no default head)
      2. Freeze lower freeze_fraction of parameters
      3. Probe the backbone's actual forward-pass output width (more
         reliable than ``backbone.num_features`` for some timm models
         — e.g. ``mobilenetv3_large_100`` in timm 1.0.x reports 960 but
         actually outputs 1280 features)
      4. Attach custom ClassificationHead

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

    # ---- Probe the backbone's true output feature width ----
    # Some timm models (notably mobilenetv3_large_100 in newer timm
    # releases) report ``num_features`` that disagrees with the actual
    # forward output shape, which causes a shape-mismatch crash when
    # the head is built from the wrong dimension. Run a dummy forward
    # pass and trust the real output, not the metadata.
    in_features = _probe_in_features(
        backbone,
        input_size=model_config.get("input_size", 224),
    )
    reported = backbone.num_features
    if in_features != reported:
        print(
            f"[Builder] WARNING: backbone.num_features={reported} but forward "
            f"output is {in_features} — using actual forward output."
        )

    # Attach custom classification head
    head_config = model_config.get("head", {"hidden_dim": 256, "dropout": 0.5})
    num_classes = model_config.get("num_classes", 2)
    head = ClassificationHead(in_features, head_config, num_classes=num_classes)

    model = _ModelWithHead(backbone, head)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Builder] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    return model


def _probe_in_features(backbone: nn.Module, input_size: int = 224) -> int:
    """Run a dummy forward pass and return the actual output width.

    Handles backbones that emit either a flat ``(B, C)`` tensor or a
    spatial ``(B, C, H, W)`` tensor (pooled down inside the head).
    """
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, input_size, input_size)
        try:
            out = backbone(dummy)
        except Exception:
            # Some timm models need explicit forward_intermediates; try a
            # smaller dummy if the larger one fails (channel mismatches).
            out = backbone(dummy)
    if out.dim() == 2:
        return out.shape[1]
    if out.dim() == 4:
        # (B, C, H, W) — the head will AdaptiveAvgPool to (B, C, 1, 1).
        return out.shape[1]
    # 3D or other unusual shape — fall back to the metadata.
    return getattr(backbone, "num_features", 0) or 0


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
