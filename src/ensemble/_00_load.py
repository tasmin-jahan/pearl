"""
00_load — Step 0 of the ensemble pipeline.

Build N models from configs and load each from its own checkpoint.
Pure probability-averaging inference later (01_predict) assumes the
returned models are already ``.eval()`` on the correct device.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

from src.model.builder import build_model


def load_ensemble(
    model_configs: Sequence[dict],
    checkpoint_paths: Sequence[str],
    device: str = "cpu",
) -> List[torch.nn.Module]:
    """Build N models and load each from its own checkpoint.

    Args:
        model_configs: One model-build config per ensemble member.
        checkpoint_paths: Parallel sequence of checkpoint paths.
        device: Device to materialize each model on.

    Returns:
        List of ``nn.Module`` in eval mode, same order as inputs.
    """
    if len(model_configs) != len(checkpoint_paths):
        raise ValueError("model_configs and checkpoint_paths must be same length")
    models = []
    for cfg, path in zip(model_configs, checkpoint_paths):
        m = build_model(cfg).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
    return models
