#!/usr/bin/env python3
"""Two-pass ensemble calibration for Zenodo fine-tuned models.

Loads the Zenodo test set on-the-fly (no .npy cache), runs each ensemble
member, averages probabilities, and applies (1) per-model temperature
fitting on val and (2) ensemble-level temperature fitting.

Usage:
    python scripts/run_calibration_zenodo_ensemble.py \
        --run_dirs results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
                   results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny \
                   results/finetune_zenodo/checkpoints/gauss_nopad/vit_base \
        --out_dir results/finetune_zenodo/ensemble/top3_pre_hpo/calibration
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset, build_zenodo_loader
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.preprocessing.preprocess import Preprocessor
from src.calibration.ece import compute_ece, compute_bin_data
from src.calibration.reliability import plot_reliability_diagram
from src.calibration.temperature_scaling import fit_temperature, apply_temperature
from torch.utils.data import DataLoader


def _load_model(run_dir, device):
    cfg_path = os.path.join(run_dir, "config.yaml")
    cfg = load_config(cfg_path)
    model_config = cfg["model"]
    model_config = dict(model_config)
    model_config["freeze_fraction"] = 0.0
    model = build_model(model_config)
    load_checkpoint(
        model, os.path.join(run_dir, "best.pt"),
        optimizer=None, scheduler=None, ema=None, device=device,
    )
    model.to(device)
    model.eval()
    return model, model_config, cfg["preprocessing"]


def _predict_probs(model, pairs, model_config, preproc_config, device, batch_size=64):
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.extend([int(v) for v in y])
    logits = torch.cat(all_logits, dim=0)
    labels = np.array(all_labels)
    probs = F.softmax(logits, dim=1).numpy()
    return logits.numpy(), labels, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--external_dir", default="data_external/test")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = discover_zenodo_pairs(args.external_dir)
    n_inf = sum(1 for _, l in pairs if l == 1)
    n_h = len(pairs) - n_inf
    print(f"[EnsembleCalib] Test set: {len(pairs)} pairs ({n_inf} infected, {n_h} healthy)")

    # ---- Pass 1: per-model temperature ----
    per_model_logits = []
    per_model_probs = []
    labels = None
    per_model_T = []
    arch_names = []

    # 50/50 split for honest held-out calibration
    half = len(pairs) // 2

    for run_dir in args.run_dirs:
        model, model_config, preproc_config = _load_model(run_dir, device)
        arch = model_config.get("name", "?")
        arch_names.append(arch)
        print(f"\n[EnsembleCalib] === {arch} ===")

        logits, labels, probs = _predict_probs(
            model, pairs, model_config, preproc_config, device, args.batch_size,
        )
        # Fit T on first half (held-out calibration set)
        logits_t = torch.from_numpy(logits)
        labels_t = torch.from_numpy(labels).long()
        T = fit_temperature(logits_t[:half], labels_t[:half])
        per_model_T.append(float(T))
        print(f"  Per-model T={T:.4f}")

        per_model_logits.append(logits)
        per_model_probs.append(probs)

        del model
        torch.cuda.empty_cache()

    # Apply per-model Ts on the SECOND HALF of test, then average
    test_labels = labels[half:]
    pos_probs_per_model = []
    for logits, T in zip(per_model_logits, per_model_T):
        T_t = torch.tensor(T, dtype=torch.float32)
        cal = apply_temperature(torch.from_numpy(logits[half:]), T_t).numpy()
        pos_probs_per_model.append(cal[:, 1])
    pos_probs_pass1 = np.mean(np.stack(pos_probs_per_model), axis=0)

    ece_p1 = compute_ece(pos_probs_pass1, test_labels, n_bins=args.n_bins)
    print(f"\n[Pass 1] ECE after per-model T: {ece_p1:.4f}")

    # ---- Pass 2: ensemble-level temperature ----
    # Fit T_ens on FIRST HALF (calibration) using per-model-T-applied averaged probs
    cal_first_half_pos = []
    for logits, T in zip(per_model_logits, per_model_T):
        T_t = torch.tensor(T, dtype=torch.float32)
        cal = apply_temperature(torch.from_numpy(logits[:half]), T_t).numpy()
        cal_first_half_pos.append(cal[:, 1])
    avg_probs_first_half = np.mean(np.stack(cal_first_half_pos), axis=0)
    labels_first_half = labels[:half]

    # Convert avg to 2D for fit_temperature
    # fit_temperature expects logits; we treat avg probabilities as pseudo-logits
    # via log — fit a single T on the averaged probabilities
    eps = 1e-12
    pseudo_logits = np.log(np.stack([1 - avg_probs_first_half, avg_probs_first_half], axis=1) + eps)
    pseudo_logits_t = torch.from_numpy(pseudo_logits).float()
    labels_first_half_t = torch.from_numpy(labels_first_half).long()
    T_ens = fit_temperature(pseudo_logits_t, labels_first_half_t)

    # Apply T_ens to test
    eps = 1e-12
    pseudo_logits_test = np.log(
        np.stack([1 - pos_probs_pass1, pos_probs_pass1], axis=1) + eps,
    )
    pos_probs_pass2 = apply_temperature(
        torch.from_numpy(pseudo_logits_test).float(), torch.tensor(T_ens),
    ).numpy()[:, 1]

    ece_p2 = compute_ece(pos_probs_pass2, test_labels, n_bins=args.n_bins)
    print(f"[Pass 2] ECE after ensemble T (T={T_ens:.2f}): {ece_p2:.4f}")

    # Reliability diagrams
    bin_data_p1 = compute_bin_data(pos_probs_pass1, test_labels, n_bins=args.n_bins)
    bin_data_p2 = compute_bin_data(pos_probs_pass2, test_labels, n_bins=args.n_bins)
    plot_reliability_diagram(
        bin_data_p1,
        title=f"Ensemble — After per-model T (top-3)",
        save_path=os.path.join(args.out_dir, "reliability_p1.png"),
    )
    plot_reliability_diagram(
        bin_data_p2,
        title=f"Ensemble — After ensemble T (T={T_ens:.2f})",
        save_path=os.path.join(args.out_dir, "reliability_p2.png"),
    )

    pd.DataFrame(bin_data_p1).to_csv(
        os.path.join(args.out_dir, "bin_data_pass1.csv"), index=False,
    )
    pd.DataFrame(bin_data_p2).to_csv(
        os.path.join(args.out_dir, "bin_data_pass2.csv"), index=False,
    )

    summary = {
        "archs": arch_names,
        "per_model_temperatures": [round(t, 4) for t in per_model_T],
        "ensemble_temperature": round(float(T_ens), 4),
        "ece_after_pass1": round(float(ece_p1), 4),
        "ece_after_pass2": round(float(ece_p2), 4),
        "n_bins": args.n_bins,
        "n_test_samples": int(len(test_labels)),
    }
    with open(os.path.join(args.out_dir, "calibration_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[EnsembleCalib] Results in {args.out_dir}")


if __name__ == "__main__":
    main()