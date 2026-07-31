#!/usr/bin/env python3
"""
CLI entrypoint for MC Dropout uncertainty quantification + referral system.

Two modes:

1. Figshare mode (legacy):
       python scripts/uncertainty/run_uncertainty.py \
           --model configs/model/swin_tiny.yaml \
           --preprocessing configs/preprocessing.yaml \
           --checkpoint results/checkpoints/srad/swin_tiny.pt \
           --mc_passes 50

2. Zenodo mode (--test_dir set, --run_dir set):
       python scripts/uncertainty/run_uncertainty.py \
           --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
           --test_dir data_external/test \
           --n_passes 50

Preprocessing is now the unified config (configs/preprocessing.yaml).

This file replaces both `run_uncertainty.py` (original Figshare-only) and
`run_uncertainty_zenodo.py` (Zenodo-only).
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.preprocessing.preprocess import Preprocessor
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.uncertainty.referral import compute_referral_curve
from torch.utils.data import DataLoader


def _run_figshare(args, model_config, preproc_config, device, out_dir):
    """Figshare mode: standard Figshare train/val/test loader."""
    input_size = model_config.get("input_size", 224)
    _, _, test_loader = build_dataloaders(
        preproc_config, batch_size=32, input_size=input_size,
    )

    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)

    print(f"[Uncertainty] Running {args.mc_passes} MC Dropout passes...")
    mean_probs, entropy = mc_dropout_inference(
        model, test_loader, n_passes=args.mc_passes, device=device,
    )

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

    referral_data = compute_referral_curve(
        entropy_np, all_labels, predictions, args.coverage_thresholds,
    )
    pd.DataFrame(referral_data).to_csv(
        os.path.join(out_dir, "referral_curve.csv"), index=False,
    )

    # Plots
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

    low_entropy_mask = entropy_np < args.entropy_threshold
    summary = {
        "arch": model_config["name"],
        "preprocessing": preproc_config["name"],
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


def _run_zenodo(args, device, out_dir):
    """Zenodo mode: load from run_dir, run MC Dropout on Zenodo test set."""
    cfg_path = os.path.join(args.run_dir, "config.yaml")
    cfg = load_config(cfg_path)
    model_config = cfg["model"]
    preproc_config = cfg["preprocessing"]

    model_config_freeze = dict(model_config)
    model_config_freeze["freeze_fraction"] = 0.0
    model = build_model(model_config_freeze)
    load_checkpoint(
        model, os.path.join(args.run_dir, "best.pt"),
        optimizer=None, scheduler=None, ema=None, device=device,
    )
    model.to(device)
    model.eval()

    pairs = discover_zenodo_pairs(args.test_dir)
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False,
    )

    print(f"[Uncertainty] Running {args.n_passes} MC Dropout passes on Zenodo test...")
    mean_probs, entropy = mc_dropout_inference(
        model, loader, n_passes=args.n_passes, device=device,
    )
    labels = np.array([l for _, l in pairs])
    prob_infected = mean_probs[:, 1].numpy()
    entropy_np = entropy.numpy()

    df = pd.DataFrame({
        "path": [p for p, _ in pairs],
        "label": labels,
        "prob_infected": prob_infected,
        "predictive_entropy": entropy_np,
    })
    df.to_csv(os.path.join(out_dir, "mc_dropout_predictions.csv"), index=False)

    try:
        ref = compute_referral_curve(
            labels, prob_infected, entropy_np,
            coverage_thresholds=args.coverage_thresholds,
        )
    except Exception as e:
        print(f"[Uncertainty] WARN: referral curve failed: {e}")
        ref = {}

    results = {
        "arch": model_config.get("name"),
        "preprocessing": preproc_config.get("name"),
        "n_test_samples": len(labels),
        "n_passes": args.n_passes,
        "mean_entropy": float(entropy_np.mean()),
        "median_entropy": float(np.median(entropy_np)),
        "referral": ref,
    }
    with open(os.path.join(out_dir, "mc_dropout_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Uncertainty] Mean predictive entropy: {entropy_np.mean():.4f}")
    print(f"[Uncertainty] Saved to {out_dir}/mc_dropout_results.json")


def main():
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty analysis")
    parser.add_argument("--model", type=str, default=None,
                        help="Figshare mode: model config YAML")
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="Figshare mode: preprocessing config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Figshare mode: path to checkpoint .pt")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run dir containing config.yaml + best.pt "
                             "(Zenodo mode uses this; Figshare mode uses it for output only)")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Zenodo mode: external test directory (e.g. data_external/test). "
                             "If set, switches to Zenodo mode.")
    parser.add_argument("--mc_passes", "--n_passes", type=int, default=50, dest="mc_passes")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--entropy_threshold", type=float, default=0.35)
    parser.add_argument("--coverage_thresholds", type=float, nargs="+",
                        default=[1.0, 0.9, 0.8, 0.7, 0.6])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_dir is not None:
        # Zenodo mode
        if args.run_dir is None:
            parser.error("--run_dir is required when --test_dir is set")
        out_dir = os.path.join(args.run_dir, "uncertainty")
        os.makedirs(out_dir, exist_ok=True)
        _run_zenodo(args, device, out_dir)
    else:
        # Figshare mode
        if not (args.model and args.preprocessing and args.checkpoint):
            parser.error(
                "Figshare mode requires --model, --preprocessing, --checkpoint "
                "(or use --test_dir for Zenodo mode)"
            )
        model_config = load_config(args.model)
        preproc_config = load_config(args.preprocessing)
        if args.run_dir:
            out_dir = os.path.join(args.run_dir, "uncertainty")
        else:
            out_dir = os.path.join(
                "results", "uncertainty",
                f"{model_config['name']}__{preproc_config['name']}",
            )
        os.makedirs(out_dir, exist_ok=True)
        _run_figshare(args, model_config, preproc_config, device, out_dir)


if __name__ == "__main__":
    main()
