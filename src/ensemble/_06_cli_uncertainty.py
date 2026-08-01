"""
06_cli_uncertainty — Step 6 of the ensemble pipeline.

Orchestrates MC-Dropout predictive entropy over an ensemble. Used by
``src.uncertainty.run_uncertainty`` when ``--ensemble`` is set (or
when more than one ``--model``/``--checkpoint`` pair is supplied).

Pipeline:
    00_load → 04_uncertainty (per-model MC dropout + averaging)
            → entropy histogram, coverage-vs-accuracy curve,
              per-sample CSV, results JSON.

Single-model uncertainty continues to live in
``src.uncertainty.run_uncertainty`` because it has no ensemble content.
"""
from __future__ import annotations

import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ensemble import load_ensemble, mc_dropout_ensemble
from src.uncertainty.referral import compute_referral_curve
from src.utils.config import load_config


def _model_config_from_name(
    model_name: str, model_configs_dir: str = "configs/model",
) -> dict:
    """Resolve a model name or YAML path to a model-build config dict."""
    cfg_path = (
        model_name
        if model_name.endswith((".yaml", ".yml"))
        else os.path.join(model_configs_dir, f"{model_name}.yaml")
    )
    return load_config(cfg_path)


def _summarize(
    mean_probs: np.ndarray,
    entropy: np.ndarray,
    labels: np.ndarray,
    arch_names: List[str],
    mc_passes: int,
    coverage_thresholds: List[float],
    out_dir: str,
) -> dict:
    """Write entropy histogram, coverage curve, per-sample CSV, JSON."""
    os.makedirs(out_dir, exist_ok=True)
    predictions = mean_probs.argmax(axis=1)
    correct = predictions == labels

    per_sample = pd.DataFrame({
        "sample_id": range(len(labels)),
        "true_label": labels,
        "mean_prob_pcos": np.round(mean_probs[:, 1], 4),
        "mean_prob_non_pcos": np.round(mean_probs[:, 0], 4),
        "predicted_label": predictions,
        "correct": correct,
        "entropy_nats": np.round(entropy, 4),
    })
    per_sample.to_csv(
        os.path.join(out_dir, "entropy_per_sample.csv"), index=False
    )

    referral = compute_referral_curve(
        entropy, labels, predictions, coverage_thresholds
    )
    pd.DataFrame(referral).to_csv(
        os.path.join(out_dir, "referral_curve.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropy[correct], bins=50, alpha=0.7, label="Correct", color="tab:green")
    ax.hist(entropy[~correct], bins=50, alpha=0.7, label="Incorrect", color="tab:red")
    ax.set_xlabel("Predictive Entropy (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(out_dir, "entropy_histogram.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    covs = [r["coverage_threshold"] for r in referral]
    accs = [r["accuracy_on_auto"] for r in referral]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(covs, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy on Auto-decided")
    ax.set_title("Coverage vs Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 1.05)
    fig.savefig(
        os.path.join(out_dir, "coverage_accuracy_curve.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    summary = {
        "archs": arch_names,
        "ensemble": len(arch_names) > 1,
        "mc_passes": mc_passes,
        "n_test": int(len(labels)),
        "mean_entropy": round(float(entropy.mean()), 4),
        "median_entropy": round(float(np.median(entropy)), 4),
        "accuracy_full": round(float(correct.mean()), 4),
    }
    with open(os.path.join(out_dir, "uncertainty_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Ensemble Uncertainty] Saved to {out_dir}")
    return summary


def run_uncertainty_ensemble(
    dataset_dir: str,
    preproc_config: dict,
    model_specs: list,
    test_loader,
    mc_passes: int,
    coverage_thresholds: List[float],
    out_dir: str,
    device: str,
    model_configs_dir: str = "configs/model",
) -> dict:
    """Run MC dropout ensemble + summarize artifacts.

    Args:
        dataset_dir: Preprocessed dataset root (used here for metadata
            only — the test loader is passed in by the caller to keep
            single-model and ensemble code paths in lockstep).
        preproc_config: Preprocessing YAML loaded as dict.
        model_specs: Sequence of ``(model_name_or_yaml, checkpoint_path)``.
        test_loader: Test DataLoader.
        mc_passes: Number of stochastic forward passes per model.
        coverage_thresholds: List of coverage levels for the referral
            curve.
        out_dir: Where to write artifacts.
        device: Device string.
        model_configs_dir: Folder holding model-name YAMLs.

    Returns:
        Summary dict (also written to ``uncertainty_results.json``).
    """
    # ---- 00_load ----
    model_configs = [_model_config_from_name(name, model_configs_dir) for name, _ in model_specs]
    ckpt_paths = [path for _, path in model_specs]
    models = load_ensemble(model_configs, ckpt_paths, device=device)
    arch_names = [name for name, _ in model_specs]

    # ---- 04_uncertainty ----
    mean_probs, entropy, labels = mc_dropout_ensemble(
        models, test_loader, mc_passes, device,
    )
    return _summarize(
        mean_probs=mean_probs,
        entropy=entropy,
        labels=labels,
        arch_names=arch_names,
        mc_passes=mc_passes,
        coverage_thresholds=coverage_thresholds,
        out_dir=out_dir,
    )