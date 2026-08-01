"""
02_per_model_calibration — Step 2 of the ensemble pipeline.

Pass 1 of two-pass ensemble calibration: each member gets its own
optimal temperature, then per-model calibrated probs are averaged.
Averaging individually-calibrated models does not guarantee the
average itself is calibrated (mixture does not preserve per-model T),
so the ensemble-level Pass 3 is needed afterwards.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import DataLoader

from src.calibration.temperature_scaling import (
    apply_temperature,
    fit_temperature,
)


def collect_logits(model, dataloader, device: str = "cuda") -> tuple:
    """Run a model over a dataloader and return ``(logits, labels)`` tensors."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels, dtype=torch.long)
            logits = model(images)
            all_logits.append(logits.float().cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_per_model_temperature(
    models: Sequence[torch.nn.Module],
    val_loaders: Sequence[DataLoader],
    device: str = "cuda",
) -> list:
    """Fit a scalar ``T_i`` for each model on its own validation loader.

    Args:
        models: Sequence of ``nn.Module`` (one per ensemble member).
        val_loaders: Parallel sequence of val DataLoaders.
        device: Device string.

    Returns:
        List of optimal temperatures, one per model.
    """
    temps = []
    for m, loader in zip(models, val_loaders):
        logits, labels = collect_logits(m, loader, device=device)
        T = fit_temperature(logits, labels)
        temps.append(T)
    return temps


def apply_per_model_temperature(
    models: Sequence[torch.nn.Module],
    test_loader: DataLoader,
    temperatures: Sequence[float],
    device: str = "cuda",
) -> tuple:
    """Run inference with per-model temperature scaling applied.

    Returns:
        ``(averaged_probs, labels)`` as numpy arrays.
    """
    all_labels = None
    sum_probs = None
    for m, T in zip(models, temperatures):
        logits, labels = collect_logits(m, test_loader, device=device)
        probs = apply_temperature(logits, T).numpy()
        if sum_probs is None:
            sum_probs = probs
            all_labels = labels.numpy()
        else:
            sum_probs = sum_probs + probs
    sum_probs /= len(models)
    return sum_probs, all_labels
