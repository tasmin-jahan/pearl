#!/usr/bin/env python3
"""MC Dropout uncertainty on the Zenodo held-out test set.

Usage:
    python scripts/run_uncertainty_zenodo.py \
        --run_dir results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
        --n_passes 50
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.preprocessing.preprocess import Preprocessor
from src.uncertainty.mc_dropout import mc_dropout_inference
from src.uncertainty.referral import compute_referral_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--external_dir", default="data_external/test")
    parser.add_argument("--n_passes", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--entropy_threshold", type=float, default=0.35)
    parser.add_argument("--coverage_thresholds", type=float, nargs="+",
                        default=[1.0, 0.9, 0.8, 0.7, 0.6])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load configs from run_dir
    cfg_path = os.path.join(args.run_dir, "config.yaml")
    cfg = load_config(cfg_path)
    model_config = cfg["model"]
    preproc_config = cfg["preprocessing"]

    # Build model
    model_config_freeze = dict(model_config)
    model_config_freeze["freeze_fraction"] = 0.0
    model = build_model(model_config_freeze)
    load_checkpoint(
        model, os.path.join(args.run_dir, "best.pt"),
        optimizer=None, scheduler=None, ema=None, device=device,
    )
    model.to(device)
    model.eval()

    # Test loader (Zenodo)
    pairs = discover_zenodo_pairs(args.external_dir)
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False,
    )

    # MC Dropout
    print(f"[Uncertainty] Running {args.n_passes} MC Dropout passes on Zenodo test...")
    mean_probs, entropy = mc_dropout_inference(
        model, loader, n_passes=args.n_passes, device=device,
    )
    labels = np.array([l for _, l in pairs])
    prob_infected = mean_probs[:, 1].numpy()
    entropy_np = entropy.numpy()

    # Output dir
    out_dir = os.path.join(args.run_dir, "uncertainty")
    os.makedirs(out_dir, exist_ok=True)

    # Per-image CSV
    df = pd.DataFrame({
        "path": [p for p, _ in pairs],
        "label": labels,
        "prob_infected": prob_infected,
        "predictive_entropy": entropy_np,
    })
    df.to_csv(os.path.join(out_dir, "mc_dropout_predictions.csv"), index=False)

    # Referral curves
    try:
        ref = compute_referral_curve(
            labels, prob_infected, entropy_np,
            coverage_thresholds=args.coverage_thresholds,
        )
    except Exception as e:
        print(f"[Uncertainty] WARN: referral curve failed: {e}")
        ref = {}

    # Save JSON
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
    print(f"[Uncertainty] Median predictive entropy: {np.median(entropy_np):.4f}")
    print(f"[Uncertainty] Saved to {out_dir}/mc_dropout_results.json")


if __name__ == "__main__":
    main()