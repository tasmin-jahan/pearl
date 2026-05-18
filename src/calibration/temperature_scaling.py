"""
Temperature scaling for post-hoc calibration.

Learns a scalar temperature T on the validation set by minimizing NLL.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


class TemperatureScaling(nn.Module):
    """Temperature scaling wrapper.

    Divides logits by a learned scalar temperature before softmax.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bounds: tuple = (0.1, 10.0),
) -> float:
    """Find optimal temperature by minimizing NLL on validation logits.

    Args:
        logits: Raw model logits, shape (N, num_classes).
        labels: Ground-truth labels, shape (N,).
        bounds: Search bounds for temperature.

    Returns:
        Optimal temperature scalar.
    """
    logits_np = logits.numpy().astype(np.float64)
    labels_np = labels.numpy().astype(np.int64)

    def nll(T):
        scaled = logits_np / T
        # Log-softmax for numerical stability
        max_vals = scaled.max(axis=1, keepdims=True)
        shifted = scaled - max_vals
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp
        # Negative log-likelihood
        return -log_probs[np.arange(len(labels_np)), labels_np].mean()

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    optimal_T = result.x

    print(f"[TempScaling] Optimal temperature: {optimal_T:.4f}")
    print(f"[TempScaling] NLL before: {nll(1.0):.4f}, after: {nll(optimal_T):.4f}")

    return float(optimal_T)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw logits, shape (N, num_classes).
        temperature: Temperature scalar.

    Returns:
        Calibrated probabilities, shape (N, num_classes).
    """
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=1)
