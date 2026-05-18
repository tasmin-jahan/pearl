"""
Expected Calibration Error (ECE) computation with 15 equal-width bins.
"""

import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error.

    Args:
        probs: Predicted probabilities for the positive class, shape (N,).
        labels: Ground-truth binary labels, shape (N,).
        n_bins: Number of equal-width bins (default 15).

    Returns:
        Scalar ECE value.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        mean_confidence = probs[mask].mean()
        fraction_correct = labels[mask].mean()
        ece += (n_bin / n_total) * abs(mean_confidence - fraction_correct)

    return float(ece)


def compute_bin_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list:
    """Compute per-bin calibration data for reliability diagrams.

    Args:
        probs: Predicted probabilities, shape (N,).
        labels: Ground-truth labels, shape (N,).
        n_bins: Number of bins.

    Returns:
        List of dicts with bin_lower, bin_upper, bin_midpoint,
        mean_confidence, fraction_correct, n_samples.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_bin = mask.sum()
        bin_data.append({
            "bin_lower": round(lo, 4),
            "bin_upper": round(hi, 4),
            "bin_midpoint": round((lo + hi) / 2, 4),
            "mean_confidence": round(float(probs[mask].mean()), 4) if n_bin > 0 else 0.0,
            "fraction_correct": round(float(labels[mask].mean()), 4) if n_bin > 0 else 0.0,
            "n_samples": int(n_bin),
        })

    return bin_data
