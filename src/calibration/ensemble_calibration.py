"""
v3 Phase 7 — two-pass calibration:

  Pass 1: per-model temperature scaling (each model gets its own T)
  Pass 2: ensemble-level temperature scaling (after probability averaging)

Averaging individually-calibrated models does not guarantee the average
itself is calibrated, because the resulting mixture's effective
temperature is the harmonic-mean-like mix of individual Ts. We fit a
fresh T on the ensemble's held-out calibration set.
"""

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.calibration.temperature_scaling import (
    fit_temperature, apply_temperature,
)


def collect_logits(model, dataloader, device: str = "cuda") -> tuple:
    """Run a model over a dataloader and return (logits, labels) as numpy arrays."""
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
    """Fit a scalar T_i for each model on its own validation loader.

    Args:
        models: Sequence of nn.Modules (one per architecture in the ensemble).
        val_loaders: Parallel sequence of val DataLoaders, one per model.
        device: Device to run inference on.

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
        (averaged_probs, labels) as numpy arrays.
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


def fit_ensemble_temperature(
    averaged_probs: np.ndarray,
    labels: np.ndarray,
    bounds: tuple = (0.1, 10.0),
) -> float:
    """Fit a scalar T on the ensemble's averaged probabilities.

    Note: with probs already in [0,1], we work in log-prob space
    (equivalent to logit space with a fixed reference).

    Args:
        averaged_probs: Per-sample averaged probabilities, shape (N, C).
        labels: Ground-truth labels, shape (N,).
        bounds: Search bounds for T.

    Returns:
        Optimal ensemble-level temperature.
    """
    eps = 1e-12
    log_p = np.log(np.clip(averaged_probs, eps, 1.0)).astype(np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)

    def nll(T):
        scaled = log_p / T
        max_vals = scaled.max(axis=1, keepdims=True)
        shifted = scaled - max_vals
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_q = shifted - log_sum_exp
        return -log_q[np.arange(len(labels_np)), labels_np].mean()

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    T_ens = float(result.x)
    print(f"[EnsembleTempScaling] Optimal ensemble T: {T_ens:.4f}")
    print(f"[EnsembleTempScaling] NLL before: {nll(1.0):.4f}, after: {nll(T_ens):.4f}")
    return T_ens


def apply_ensemble_temperature(
    averaged_probs: np.ndarray, T: float,
) -> np.ndarray:
    """Apply a learned T to averaged ensemble probabilities."""
    eps = 1e-12
    log_p = np.log(np.clip(averaged_probs, eps, 1.0))
    scaled = log_p / T
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_q = shifted - log_sum_exp
    return np.exp(log_q)