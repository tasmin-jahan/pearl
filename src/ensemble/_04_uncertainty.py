"""
04_uncertainty — Step 4 of the ensemble pipeline.

MC-Dropout ensemble: run ``mc_dropout_inference`` per model, average
the per-model mean probabilities across the ensemble, then take
predictive entropy of the averaged mean.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.uncertainty.mc_dropout import mc_dropout_inference


def mc_dropout_ensemble(
    models: Sequence[torch.nn.Module],
    loader: DataLoader,
    n_passes: int,
    device: str,
) -> tuple:
    """Run MC dropout per model and average the resulting mean probs.

    Args:
        models: Sequence of ``nn.Module`` (each in eval-but-with-dropout
            mode).
        loader: Test DataLoader.
        n_passes: Number of stochastic forward passes per model.
        device: Device string.

    Returns:
        ``(mean_probs, entropy, labels)`` — all numpy arrays.
    """
    sum_probs = None
    for model in models:
        mean_probs, _ = mc_dropout_inference(
            model, loader, n_passes=n_passes, device=device
        )
        if sum_probs is None:
            sum_probs = mean_probs
        else:
            sum_probs = sum_probs + mean_probs
    avg_probs = sum_probs / len(models)
    entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum(dim=1)

    labels = []
    for _, lab in loader:
        if isinstance(lab, torch.Tensor):
            labels.extend(lab.cpu().numpy())
        else:
            labels.extend(lab)
    return avg_probs.numpy(), entropy.numpy(), np.array(labels)
