#!/usr/bin/env python3
"""
CLI entrypoint for MC Dropout uncertainty quantification + referral system.

Usage:
    python scripts/run_uncertainty.py \
        --model configs/model/swin_tiny.yaml \
        --preprocessing configs/preprocessing/srad.yaml \
        --checkpoint results/checkpoints/srad/swin_tiny.pt \
        --mc_passes 50 \
        --entropy_threshold 0.35 \
        --coverage_thresholds 1.0 0.9 0.8 0.7 0.6
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.uncertainty.referral import compute_referral_curve


def main():
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--preprocessing", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mc_passes", type=int, default=50)
    parser.add_argument("--entropy_threshold", type=float, default=0.35)
    parser.add_argument("--coverage_thresholds", type=float, nargs="+",
                        default=[1.0, 0.9, 0.8, 0.7, 0.6])
    parser.add_argument("--run_dir", type=str, default=None,
                        help="If set, write artifacts under <run_dir>/uncertainty/ "
                             "instead of legacy results/uncertainty/<arch>__<prep>/.")
    args = parser.parse_args()

    model_config = load_config(args.model)
    preproc_config = load_config(args.preprocessing)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = model_config["name"]
    preproc_name = preproc_config["name"]

    if args.run_dir:
        out_dir = os.path.join(args.run_dir, "uncertainty")
    else:
        out_dir = os.path.join("results", "uncertainty", f"{arch}__{preproc_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Data
    input_size = model_config.get("input_size", 224)
    _, _, test_loader = build_dataloaders(
        preproc_config, batch_size=32, input_size=input_size,
    )

    # Model
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)

    # MC Dropout inference
    print(f"[Uncertainty] Running {args.mc_passes} MC Dropout passes...")
    mean_probs, entropy = mc_dropout_inference(
        model, test_loader, n_passes=args.mc_passes, device=device,
    )

    # Collect labels
    all_labels = []
    for _, labels in test_loader:
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.numpy())
        else:
            all_labels.extend(labels)
    all_labels = np.array(all_labels)

    entropy_np = entropy.numpy()
    mean_probs_np = mean_probs.numpy()
    predictions = mean_probs_np.argmax(axis=1)
    correct = (predictions == all_labels)

    # Per-sample CSV
    per_sample = pd.DataFrame({
        "sample_id": range(len(all_labels)),
        "true_label": all_labels,
        "mean_prob_pcos": np.round(mean_probs_np[:, 1], 4),
        "mean_prob_non_pcos": np.round(mean_probs_np[:, 0], 4),
        "predicted_label": predictions,
        "correct": correct,
        "entropy_nats": np.round(entropy_np, 4),
    })
    per_sample.to_csv(os.path.join(out_dir, "entropy_per_sample.csv"), index=False)

    # Referral curve
    referral_data = compute_referral_curve(
        entropy_np, all_labels, predictions, args.coverage_thresholds,
    )
    pd.DataFrame(referral_data).to_csv(
        os.path.join(out_dir, "referral_curve.csv"), index=False,
    )

    # Entropy histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(entropy_np[correct], bins=50, alpha=0.7, label="Correct", color="tab:green")
    ax.hist(entropy_np[~correct], bins=50, alpha=0.7, label="Incorrect", color="tab:red")
    ax.axvline(args.entropy_threshold, color="black", linestyle="--",
               label=f"Threshold ({args.entropy_threshold})")
    ax.set_xlabel("Predictive Entropy (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Entropy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "entropy_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Coverage-accuracy curve
    covs = [r["coverage_threshold"] for r in referral_data]
    accs = [r["accuracy_on_auto"] for r in referral_data]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(covs, accs, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy on Auto-decided")
    ax.set_title("Coverage vs Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.55, 1.05)
    fig.savefig(os.path.join(out_dir, "coverage_accuracy_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary JSON
    low_entropy_mask = entropy_np < args.entropy_threshold
    summary = {
        "arch": arch,
        "preprocessing": preproc_name,
        "mc_passes": args.mc_passes,
        "entropy_threshold_nats": args.entropy_threshold,
        "n_test": len(all_labels),
        "n_low_entropy": int(low_entropy_mask.sum()),
        "pct_low_entropy": round(float(low_entropy_mask.mean() * 100), 1),
        "mean_entropy": round(float(entropy_np.mean()), 4),
        "median_entropy": round(float(np.median(entropy_np)), 4),
        "accuracy_full": round(float(correct.mean()), 4),
        "accuracy_low_entropy_only": round(
            float(correct[low_entropy_mask].mean()) if low_entropy_mask.any() else 0.0, 4
        ),
    }
    with open(os.path.join(out_dir, "uncertainty_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Uncertainty] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
