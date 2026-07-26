#!/usr/bin/env python3
"""
v3 Phase 7 — Ensemble-level calibration (second pass).

After per-model temperature scaling has been applied and the ensemble
probability has been averaged, fit a single scalar T on the ensemble's
held-out calibration set and report the second-pass ECE.

Usage:
    python scripts/run_calibration_ensemble.py \
        --model_configs configs/ensemble/finalists.yaml \
        --checkpoints results/finalists.txt \
        --preprocessing configs/preprocessing/srad_clahe.yaml
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
from src.evaluation.ensemble import load_ensemble
from src.calibration.ensemble_calibration import (
    fit_per_model_temperature, apply_per_model_temperature,
    fit_ensemble_temperature, apply_ensemble_temperature,
)
from src.calibration.ece import compute_ece, compute_bin_data
from src.calibration.reliability import plot_reliability_diagram
from src.data.dataloader import build_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Two-pass ensemble calibration")
    parser.add_argument("--model_configs", type=str, nargs="+", required=True,
                        help="One model YAML per ensemble member")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="One checkpoint path per ensemble member")
    parser.add_argument("--preprocessing", type=str, required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="results/calibration_ensemble")
    args = parser.parse_args()

    if len(args.model_configs) != len(args.checkpoints):
        raise ValueError("Need same number of configs and checkpoints")

    preproc_config = load_config(args.preprocessing)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = [load_config(p) for p in args.model_configs]
    arch_names = [c["name"] for c in model_configs]

    os.makedirs(args.out_dir, exist_ok=True)

    # Build loaders once (shared across ensemble members)
    input_size = model_configs[0].get("input_size", 224)
    _, val_loader, test_loader = build_dataloaders(
        preproc_config, batch_size=32, input_size=input_size,
    )

    # Load ensemble
    models = load_ensemble(model_configs, args.checkpoints, device=device)
    print(f"[Ensemble] {len(models)} models: {arch_names}")

    # ---- Pass 1: per-model temperature ----
    print("\n[Pass 1] Fitting per-model temperatures on val set...")
    # Each model shares the same val_loader (model-agnostic val set)
    val_loaders = [val_loader] * len(models)
    temperatures = fit_per_model_temperature(models, val_loaders, device=device)

    # Apply per-model temps on test, then average
    avg_probs, test_labels = apply_per_model_temperature(
        models, test_loader, temperatures, device=device,
    )

    # ECE after per-model calibration (before ensemble T)
    test_labels_np = test_labels if isinstance(test_labels, np.ndarray) else test_labels.numpy()
    pos_probs = avg_probs[:, 1]
    ece_after_pass1 = compute_ece(pos_probs, test_labels_np, n_bins=args.n_bins)
    print(f"[Pass 1] ECE after per-model calibration: {ece_after_pass1:.4f}")

    # ---- Pass 2: ensemble-level temperature ----
    print("\n[Pass 2] Fitting ensemble-level temperature...")
    # We re-collect averaged probs on val (using same per-model Ts)
    val_avg_probs, val_labels = apply_per_model_temperature(
        models, val_loader, temperatures, device=device,
    )
    val_labels_np = val_labels if isinstance(val_labels, np.ndarray) else val_labels.numpy()
    T_ens = fit_ensemble_temperature(val_avg_probs, val_labels_np)

    # Apply T_ens to test averaged probs
    cal_probs = apply_ensemble_temperature(avg_probs, T_ens)
    cal_pos_probs = cal_probs[:, 1]
    ece_after_pass2 = compute_ece(cal_pos_probs, test_labels_np, n_bins=args.n_bins)
    print(f"[Pass 2] ECE after ensemble calibration: {ece_after_pass2:.4f}")

    # Reliability diagrams
    bin_data_before = compute_bin_data(
        # Before any calibration: use uncalibrated averaged probs
        # Re-collect uncalibrated for the report
        np.zeros(0), test_labels_np, n_bins=args.n_bins,
    )
    # (Forwarded via pass1 raw probs for clarity:)
    bin_data_p1 = compute_bin_data(pos_probs, test_labels_np, n_bins=args.n_bins)
    bin_data_p2 = compute_bin_data(cal_pos_probs, test_labels_np, n_bins=args.n_bins)

    plot_reliability_diagram(
        bin_data_p1,
        title=f"Ensemble — After per-model T",
        save_path=os.path.join(args.out_dir, "reliability_p1.png"),
    )
    plot_reliability_diagram(
        bin_data_p2,
        title=f"Ensemble — After ensemble T (T={T_ens:.2f})",
        save_path=os.path.join(args.out_dir, "reliability_p2.png"),
    )

    # CSV outputs
    pd.DataFrame(bin_data_p1).to_csv(
        os.path.join(args.out_dir, "bin_data_pass1.csv"), index=False,
    )
    pd.DataFrame(bin_data_p2).to_csv(
        os.path.join(args.out_dir, "bin_data_pass2.csv"), index=False,
    )

    # JSON summary
    summary = {
        "archs": arch_names,
        "per_model_temperatures": [round(t, 4) for t in temperatures],
        "ensemble_temperature": round(T_ens, 4),
        "ece_after_pass1": round(ece_after_pass1, 4),
        "ece_after_pass2": round(ece_after_pass2, 4),
        "n_bins": args.n_bins,
        "n_test_samples": len(test_labels_np),
    }
    with open(os.path.join(args.out_dir, "calibration_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Ensemble Calibration] Results in {args.out_dir}")


if __name__ == "__main__":
    main()
