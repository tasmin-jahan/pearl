#!/usr/bin/env python3
"""Calibration analysis on the Zenodo held-out test set.

Usage:
    python scripts/run_calibration_zenodo.py \
        --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121
"""
import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.preprocessing.preprocess import Preprocessor
from src.calibration.ece import compute_ece, compute_bin_data
from src.calibration.reliability import plot_reliability_diagram
from src.calibration.temperature_scaling import fit_temperature, apply_temperature
from torch.utils.data import DataLoader


def _load_checkpoint_for_run(run_dir: str, device: str):
    """Load model and preprocessing config from a run_dir (which contains
    best.pt and config.yaml). Returns (model, model_config, preproc_config)."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.yaml under {run_dir}")
    cfg = load_config(cfg_path)
    model_config = cfg["model"]
    preproc_config = cfg["preprocessing"]
    model = build_model(model_config)
    model_config_freeze = dict(model_config)
    model_config_freeze["freeze_fraction"] = 0.0
    model = build_model(model_config_freeze)
    load_checkpoint(
        model, os.path.join(run_dir, "best.pt"),
        optimizer=None, scheduler=None, ema=None, device=device,
    )
    model.to(device)
    model.eval()
    return model, model_config, preproc_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--external_dir", default="data_external/test")
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model + preprocessing
    model, model_config, preproc_config = _load_checkpoint_for_run(args.run_dir, device)
    arch = model_config.get("name", "?")
    preproc_name = preproc_config.get("name", "?")

    # Test loader (Zenodo)
    pairs = discover_zenodo_pairs(args.external_dir)
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False,
    )

    # Output dir
    out_dir = os.path.join(args.run_dir, "calibration")
    os.makedirs(out_dir, exist_ok=True)

    # Collect logits
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.extend([int(v) for v in y])
    test_logits = torch.cat(all_logits, dim=0)
    test_labels = torch.tensor(all_labels, dtype=torch.long)
    test_probs_before = F.softmax(test_logits, dim=1)[:, 1].numpy()
    test_labels_np = test_labels.numpy()

    # ECE before
    ece_before = compute_ece(test_probs_before, test_labels_np, n_bins=args.n_bins)
    bin_data_before = compute_bin_data(test_probs_before, test_labels_np, n_bins=args.n_bins)

    # Reliability diagram before
    plot_reliability_diagram(
        bin_data_before,
        title=f"Reliability Diagram (Before) — {arch} + {preproc_name}",
        save_path=os.path.join(out_dir, "reliability_diagram_before.png"),
    )

    # Fit temperature on the test set itself (no separate val set in our setup)
    # This is a limitation — for honest calibration fit on val; we use a
    # held-out fraction of the test set as a calibration val.
    # We do a 50/50 split inside the test set: first half for T-fitting,
    # second half for ECE computation. This gives an honest "post-T" ECE.
    half = len(test_labels_np) // 2
    val_logits = test_logits[:half]
    val_labels = test_labels[:half]
    test_logits_2 = test_logits[half:]
    test_labels_np2 = test_labels_np[half:]
    test_probs_before_2 = test_probs_before[half:]

    optimal_T = fit_temperature(val_logits, val_labels)
    test_probs_after_2 = apply_temperature(test_logits_2, optimal_T)[:, 1].numpy()

    ece_before_2 = compute_ece(test_probs_before_2, test_labels_np2, n_bins=args.n_bins)
    bin_data_before_2 = compute_bin_data(test_probs_before_2, test_labels_np2, n_bins=args.n_bins)
    ece_after_2 = compute_ece(test_probs_after_2, test_labels_np2, n_bins=args.n_bins)
    bin_data_after_2 = compute_bin_data(test_probs_after_2, test_labels_np2, n_bins=args.n_bins)

    print(f"[Calibration] {arch} + {preproc_name}")
    print(f"  ECE (full test, before T):  {ece_before:.4f}")
    print(f"  ECE (test-half, before T):  {ece_before_2:.4f}")
    print(f"  ECE (test-half, after T={optimal_T:.2f}): {ece_after_2:.4f}")

    # Reliability diagrams
    plot_reliability_diagram(
        bin_data_before_2,
        title=f"Reliability Diagram (Before T) — {arch} + {preproc_name}",
        save_path=os.path.join(out_dir, "reliability_diagram_before.png"),
    )
    plot_reliability_diagram(
        bin_data_after_2,
        title=f"Reliability Diagram (After T={optimal_T:.2f}) — {arch} + {preproc_name}",
        save_path=os.path.join(out_dir, "reliability_diagram_after.png"),
    )

    # Bin CSV
    pd.DataFrame(bin_data_after_2).to_csv(
        os.path.join(out_dir, "bin_data.csv"), index=False,
    )

    # Save JSON
    results = {
        "arch": arch,
        "preprocessing": preproc_name,
        "n_test_samples": int(len(test_labels_np)),
        "ece_full_before": float(ece_before),
        "ece_held_before": float(ece_before_2),
        "ece_held_after_temp": float(ece_after_2),
        "optimal_temperature": float(optimal_T),
        "n_bins": args.n_bins,
    }
    with open(os.path.join(out_dir, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Calibration] Saved to {out_dir}/calibration_results.json")


if __name__ == "__main__":
    main()