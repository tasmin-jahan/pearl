"""
Referral system: sweeps coverage thresholds and computes accuracy-vs-coverage.

Images with entropy above a cutoff are "referred" to a human expert.
"""

import numpy as np
from typing import List


def compute_referral_curve(
    entropy: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    coverage_thresholds: List[float] = None,
) -> list:
    """Compute coverage-accuracy curve for the referral system.

    At each coverage level, only the most confident samples (lowest entropy)
    are auto-decided; the rest are referred.

    Args:
        entropy: Predictive entropy per sample, shape (N,).
        labels: Ground-truth labels, shape (N,).
        predictions: Predicted labels, shape (N,).
        coverage_thresholds: List of coverage fractions (e.g. [1.0, 0.9, ...]).

    Returns:
        List of dicts with coverage_threshold, n_auto_decided, n_referred,
        accuracy_on_auto, entropy_cutoff.
    """
    if coverage_thresholds is None:
        coverage_thresholds = [1.0, 0.9, 0.8, 0.7, 0.6]

    n_total = len(entropy)
    sorted_indices = np.argsort(entropy)  # lowest entropy first

    results = []
    for cov in sorted(coverage_thresholds, reverse=True):
        n_auto = int(np.ceil(n_total * cov))
        n_auto = min(n_auto, n_total)

        auto_indices = sorted_indices[:n_auto]
        auto_preds = predictions[auto_indices]
        auto_labels = labels[auto_indices]

        accuracy = (auto_preds == auto_labels).mean() if n_auto > 0 else 0.0
        entropy_cutoff = float(entropy[sorted_indices[n_auto - 1]]) if n_auto > 0 else float("inf")

        results.append({
            "coverage_threshold": round(cov, 2),
            "n_auto_decided": n_auto,
            "n_referred": n_total - n_auto,
            "accuracy_on_auto": round(float(accuracy), 4),
            "entropy_cutoff": round(entropy_cutoff, 4),
        })

    return results
