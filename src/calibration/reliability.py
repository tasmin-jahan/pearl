"""
Reliability diagram plotting for model calibration analysis.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    bin_data: list,
    title: str = "Reliability Diagram",
    save_path: str = None,
) -> None:
    """Plot a reliability diagram from per-bin calibration data.

    Args:
        bin_data: List of dicts from compute_bin_data().
        title: Plot title.
        save_path: If provided, save the figure to this path.
    """
    midpoints = [b["bin_midpoint"] for b in bin_data]
    fractions = [b["fraction_correct"] for b in bin_data]
    confidences = [b["mean_confidence"] for b in bin_data]
    counts = [b["n_samples"] for b in bin_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Main reliability diagram
    bar_width = 1.0 / len(midpoints) * 0.8
    ax1.bar(midpoints, fractions, width=bar_width, alpha=0.7,
            color="tab:blue", edgecolor="black", linewidth=0.5, label="Outputs")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfectly calibrated")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions per bin
    ax2.bar(midpoints, counts, width=bar_width, alpha=0.7,
            color="tab:orange", edgecolor="black", linewidth=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Reliability] Saved diagram to {save_path}")

    plt.close(fig)
