#!/usr/bin/env python3
"""
CLI entrypoint for calibration analysis: ECE + temperature scaling + reliability diagrams.

Usage:
    python scripts/run_calibration.py \
        --model configs/model/swin_tiny.yaml \
        --preprocessing configs/preprocessing/srad.yaml \
        --checkpoint results/checkpoints/srad/swin_tiny.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.evaluation.evaluator import collect_logits_and_labels
from src.calibration.ece import compute_ece, compute_bin_data
from src.calibration.reliability import plot_reliability_diagram
from src.calibration.temperature_scaling import fit_temperature, apply_temperature


def main():
    parser = argparse.ArgumentParser(description="Run calibration analysis")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--preprocessing", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    args = parser.parse_args()

    model_config = load_config(args.model)
    preproc_config = load_config(args.preprocessing)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = model_config["name"]
    preproc_name = preproc_config["name"]

    # Output directory
    out_dir = os.path.join("results", "calibration", f"{arch}__{preproc_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Data
    input_size = model_config.get("input_size", 224)
    _, val_loader, test_loader = build_dataloaders(
        preproc_config, batch_size=32, input_size=input_size,
    )

    # Model
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)

    # Collect logits on val set (for fitting T) and test set (for ECE)
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)

    # Before calibration
    import torch.nn.functional as F
    test_probs_before = F.softmax(test_logits, dim=1)[:, 1].numpy()
    test_labels_np = test_labels.numpy()

    ece_before = compute_ece(test_probs_before, test_labels_np, n_bins=args.n_bins)
    bin_data_before = compute_bin_data(test_probs_before, test_labels_np, n_bins=args.n_bins)

    print(f"[Calibration] ECE before: {ece_before:.4f}")

    # Reliability diagram before
    plot_reliability_diagram(
        bin_data_before,
        title=f"Reliability Diagram (Before) — {arch}",
        save_path=os.path.join(out_dir, "reliability_diagram_before.png"),
    )

    # Fit temperature on validation set
    optimal_T = fit_temperature(val_logits, val_labels)

    # After calibration
    test_probs_after = apply_temperature(test_logits, optimal_T)[:, 1].numpy()
    ece_after = compute_ece(test_probs_after, test_labels_np, n_bins=args.n_bins)
    bin_data_after = compute_bin_data(test_probs_after, test_labels_np, n_bins=args.n_bins)

    print(f"[Calibration] ECE after: {ece_after:.4f}")

    # Reliability diagram after
    plot_reliability_diagram(
        bin_data_after,
        title=f"Reliability Diagram (After T={optimal_T:.2f}) — {arch}",
        save_path=os.path.join(out_dir, "reliability_diagram_after.png"),
    )

    # Save bin data CSV
    df = pd.DataFrame(bin_data_after)
    df.to_csv(os.path.join(out_dir, "bin_data.csv"), index=False)

    # Compute NLL before/after
    def nll(logits, labels, T=1.0):
        scaled = logits / T
        log_probs = F.log_softmax(scaled, dim=1)
        return -log_probs[range(len(labels)), labels].mean().item()

    val_nll_before = nll(val_logits, val_labels, 1.0)
    val_nll_after = nll(val_logits, val_labels, optimal_T)

    # Save results JSON
    results = {
        "arch": arch,
        "preprocessing": preproc_name,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "optimal_temperature": round(optimal_T, 4),
        "val_nll_before": round(val_nll_before, 4),
        "val_nll_after": round(val_nll_after, 4),
        "n_bins": args.n_bins,
        "n_test_samples": len(test_labels_np),
    }
    with open(os.path.join(out_dir, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Calibration] Results saved to {out_dir}")


if __name__ == "__main__":
    main()
