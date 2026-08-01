"""
03_ensemble_calibration — Step 3 of the ensemble pipeline.

Pass 2 of two-pass ensemble calibration: a single scalar ``T_ens`` is
fit on the val-set average of per-model-calibrated probs and applied
to test-set averages. Necessary because averaging softened softmaxes
does not commute with calibration — the mixture has an effective
temperature of its own.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def fit_ensemble_temperature(
    averaged_probs: np.ndarray,
    labels: np.ndarray,
    bounds: tuple = (0.1, 10.0),
) -> float:
    """Fit a scalar ``T`` on the ensemble's averaged probabilities.

    With probs already in ``[0, 1]``, we work in log-prob space
    (equivalent to logit space with a fixed reference).

    Args:
        averaged_probs: Per-sample averaged probabilities, shape ``(N, C)``.
        labels: Ground-truth labels, shape ``(N,)``.
        bounds: Search bounds for ``T``.

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

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    T_ens = float(result.x)
    print(f"[EnsembleTempScaling] Optimal ensemble T: {T_ens:.4f}")
    print(f"[EnsembleTempScaling] NLL before: {nll(1.0):.4f}, after: {nll(T_ens):.4f}")
    return T_ens


def apply_ensemble_temperature(
    averaged_probs: np.ndarray, T: float,
) -> np.ndarray:
    """Apply a learned ``T`` to averaged ensemble probabilities."""
    eps = 1e-12
    log_p = np.log(np.clip(averaged_probs, eps, 1.0))
    scaled = log_p / T
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_q = shifted - log_sum_exp
    return np.exp(log_q)
