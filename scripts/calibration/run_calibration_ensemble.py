#!/usr/bin/env python3
"""
Ensemble-level calibration (second pass).

After per-model temperature scaling has been applied and the ensemble
probability has been averaged, fit a single scalar T on the ensemble's
held-out calibration set and report the second-pass ECE.

Two modes:

1. Figshare mode (default):
       python scripts/calibration/run_calibration_ensemble.py \
           --model_configs configs/model/resnet50.yaml ... \
           --checkpoints results/checkpoints/srad/resnet50/best.pt ... \
           --preprocessing configs/preprocessing/srad.yaml

2. Zenodo mode (--test_dir set, --run_dirs set):
       python scripts/calibration/run_calibration_ensemble.py \
           --run_dirs results/finetune_zenodo/checkpoints/<prep>/<arch>/... \
           --test_dir data_external/test

This file replaces both `run_calibration_ensemble.py` (Figshare-only) and
`run_calibration_zenodo_ensemble.py` (Zenodo-only).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

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
    # Figshare mode
    parser.add_argument("--model_configs", type=str, nargs="+", default=None,
                        help="Figshare mode: one model YAML per ensemble member")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Figshare mode: one checkpoint path per ensemble member")
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="Figshare mode: preprocessing config YAML")
    # Zenodo mode
    parser.add_argument("--run_dirs", type=str, nargs="+", default=None,
                        help="Zenodo mode: ensemble run dirs (each contains config.yaml + best.pt)")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Zenodo mode: external test directory (e.g. data_external/test)")
    parser.add_argument("--external_dir", type=str, default=None,
                        help="Alias kept for backward compatibility with run_calibration_zenodo_ensemble")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Defaults to results/calibration_ensemble "
                             "(Figshare) or <run_dirs[0]>/calibration (Zenodo).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Zenodo mode ----
    if args.test_dir or args.run_dirs or args.external_dir:
        if not args.run_dirs:
            parser.error("--run_dirs is required in Zenodo mode (one ensemble member dir each)")
        test_dir = args.test_dir or args.external_dir or "data_external/test"
        run_dirs = args.run_dirs
        out_dir = args.out_dir or os.path.join(run_dirs[0], "calibration")
        os.makedirs(out_dir, exist_ok=True)

        # Load each member's config + build model
        from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
        from src.preprocessing.preprocess import Preprocessor
        from torch.utils.data import DataLoader

        member_configs = []
        models = []
        for rd in run_dirs:
            cfg = load_config(os.path.join(rd, "config.yaml"))
            mc = dict(cfg["model"])
            mc["freeze_fraction"] = 0.0
            member_configs.append(mc)
            m = build_model(mc)
            load_checkpoint(
                m, os.path.join(rd, "best.pt"),
                optimizer=None, scheduler=None, ema=None, device=device,
            )
            m.to(device); m.eval()
            models.append(m)
        arch_names = [c.get("name", "?") for c in member_configs]
        preproc = cfg["preprocessing"]

        # Single shared Zenodo test loader
        pairs = discover_zenodo_pairs(test_dir)
        pre = Preprocessor(preproc, input_size=member_configs[0].get("input_size", 224))
        ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False,
        )

        # Pass 1: per-model T on 50/50 split (held-out half fits T)
        import torch.nn.functional as F

        # First, collect per-model logits on full test set
        all_logits = []  # list of [N, 2] tensors
        all_labels = []
        with torch.no_grad():
            for x, y in loader:
                lbls = [int(v) for v in y]
                all_labels.extend(lbls)
                for m in models:
                    all_logits.append(F.softmax(m(x.to(device)), dim=1).cpu())
        all_labels = np.array(all_labels)
        # Stack per model: shape (n_models, N, 2)
        per_model_probs = np.stack(
            [np.concatenate([np.atleast_2d(l) for l in [all_logits[i]]]) for i in range(len(models))],
            axis=0,
        )
        # Simpler: average across batches to get (n_models, N, 2)
        # The above stack is wrong -- recompute correctly
        per_member_probs = []
        idx = 0
        with torch.no_grad():
            for x, _ in loader:
                batch_probs = []
                for m in models:
                    p = F.softmax(m(x.to(device)), dim=1).cpu().numpy()
                    batch_probs.append(p)
                per_member_probs.append(np.stack(batch_probs, axis=0))
        per_member_probs = np.concatenate(per_member_probs, axis=1)  # (n_models, N, 2)
        N = per_member_probs.shape[1]

        # 50/50 split: fit T on first half, evaluate on second half
        half = N // 2
        test_labels_np = all_labels
        temperatures = []
        for k in range(len(models)):
            # We have probabilities (post-softmax), so reconstructing logits
            # loses precision; instead sweep T and pick the one with lowest NLL.
            eps = 1e-12
            logits_first = np.log(per_member_probs[k, :half] + eps)
            labels_first = test_labels_np[:half]
            best_T, best_nll = 1.0, float("inf")
            for T in np.linspace(0.5, 3.0, 51):
                scaled = logits_first / T
                scaled -= scaled.max(axis=1, keepdims=True)
                exp = np.exp(scaled)
                soft = exp / exp.sum(axis=1, keepdims=True)
                nll = -np.log(soft[np.arange(half), labels_first] + eps).mean()
                if nll < best_nll:
                    best_nll, best_T = nll, T
            temperatures.append(best_T)
        print(f"[Pass 1] Per-model temperatures: {temperatures}")

        # Apply per-model T to second half, average
        cal_second = np.zeros_like(per_member_probs[:, half:, :])
        for k in range(len(models)):
            eps = 1e-12
            logits = np.log(per_member_probs[k, half:] + eps)
            scaled = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(scaled / temperatures[k])
            cal_second[k] = exp / exp.sum(axis=1, keepdims=True)
        avg_probs = cal_second.mean(axis=0)
        ece_after_pass1 = compute_ece(avg_probs[:, 1], test_labels_np[half:], n_bins=args.n_bins)
        print(f"[Pass 1] ECE after per-model calibration: {ece_after_pass1:.4f}")

        # Pass 2: ensemble T
        T_ens = fit_ensemble_temperature(avg_probs, test_labels_np[half:])
        cal_probs = apply_ensemble_temperature(avg_probs, T_ens)
        ece_after_pass2 = compute_ece(cal_probs[:, 1], test_labels_np[half:], n_bins=args.n_bins)
        print(f"[Pass 2] ECE after ensemble calibration: {ece_after_pass2:.4f}")

        # Reliability diagrams
        bin_data_p1 = compute_bin_data(avg_probs[:, 1], test_labels_np[half:], n_bins=args.n_bins)
        bin_data_p2 = compute_bin_data(cal_probs[:, 1], test_labels_np[half:], n_bins=args.n_bins)
        plot_reliability_diagram(
            bin_data_p1, title="Ensemble — After per-model T",
            save_path=os.path.join(out_dir, "reliability_p1.png"),
        )
        plot_reliability_diagram(
            bin_data_p2, title=f"Ensemble — After ensemble T (T={T_ens:.2f})",
            save_path=os.path.join(out_dir, "reliability_p2.png"),
        )
        pd.DataFrame(bin_data_p1).to_csv(os.path.join(out_dir, "bin_data_pass1.csv"), index=False)
        pd.DataFrame(bin_data_p2).to_csv(os.path.join(out_dir, "bin_data_pass2.csv"), index=False)

        summary = {
            "archs": arch_names,
            "per_model_temperatures": [round(t, 4) for t in temperatures],
            "ensemble_temperature": round(float(T_ens), 4),
            "ece_after_pass1": round(float(ece_after_pass1), 4),
            "ece_after_pass2": round(float(ece_after_pass2), 4),
            "n_bins": args.n_bins,
            "n_test_samples": int(N),
        }
        with open(os.path.join(out_dir, "calibration_results.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Ensemble Calibration] Results in {out_dir}")
        return

    # ---- Figshare mode (legacy CLI) ----
    if not (args.model_configs and args.checkpoints and args.preprocessing):
        parser.error(
            "Figshare mode requires --model_configs --checkpoints --preprocessing "
            "(or use --run_dirs/--test_dir for Zenodo mode)"
        )
    if len(args.model_configs) != len(args.checkpoints):
        raise ValueError("Need same number of configs and checkpoints")

    preproc_config = load_config(args.preprocessing)
    out_dir = args.out_dir or "results/calibration_ensemble"
    os.makedirs(out_dir, exist_ok=True)

    model_configs = [load_config(p) for p in args.model_configs]
    arch_names = [c["name"] for c in model_configs]

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
    val_loaders = [val_loader] * len(models)
    temperatures = fit_per_model_temperature(models, val_loaders, device=device)

    avg_probs, test_labels = apply_per_model_temperature(
        models, test_loader, temperatures, device=device,
    )

    test_labels_np = test_labels if isinstance(test_labels, np.ndarray) else test_labels.numpy()
    pos_probs = avg_probs[:, 1]
    ece_after_pass1 = compute_ece(pos_probs, test_labels_np, n_bins=args.n_bins)
    print(f"[Pass 1] ECE after per-model calibration: {ece_after_pass1:.4f}")

    # ---- Pass 2: ensemble-level temperature ----
    print("\n[Pass 2] Fitting ensemble-level temperature...")
    val_avg_probs, val_labels = apply_per_model_temperature(
        models, val_loader, temperatures, device=device,
    )
    val_labels_np = val_labels if isinstance(val_labels, np.ndarray) else val_labels.numpy()
    T_ens = fit_ensemble_temperature(val_avg_probs, val_labels_np)

    cal_probs = apply_ensemble_temperature(avg_probs, T_ens)
    cal_pos_probs = cal_probs[:, 1]
    ece_after_pass2 = compute_ece(cal_pos_probs, test_labels_np, n_bins=args.n_bins)
    print(f"[Pass 2] ECE after ensemble calibration: {ece_after_pass2:.4f}")

    bin_data_p1 = compute_bin_data(pos_probs, test_labels_np, n_bins=args.n_bins)
    bin_data_p2 = compute_bin_data(cal_pos_probs, test_labels_np, n_bins=args.n_bins)

    plot_reliability_diagram(
        bin_data_p1,
        title=f"Ensemble — After per-model T",
        save_path=os.path.join(out_dir, "reliability_p1.png"),
    )
    plot_reliability_diagram(
        bin_data_p2,
        title=f"Ensemble — After ensemble T (T={T_ens:.2f})",
        save_path=os.path.join(out_dir, "reliability_p2.png"),
    )

    pd.DataFrame(bin_data_p1).to_csv(
        os.path.join(out_dir, "bin_data_pass1.csv"), index=False,
    )
    pd.DataFrame(bin_data_p2).to_csv(
        os.path.join(out_dir, "bin_data_pass2.csv"), index=False,
    )

    summary = {
        "archs": arch_names,
        "per_model_temperatures": [round(t, 4) for t in temperatures],
        "ensemble_temperature": round(T_ens, 4),
        "ece_after_pass1": round(ece_after_pass1, 4),
        "ece_after_pass2": round(ece_after_pass2, 4),
        "n_bins": args.n_bins,
        "n_test_samples": len(test_labels_np),
    }
    with open(os.path.join(out_dir, "calibration_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Ensemble Calibration] Results in {out_dir}")


if __name__ == "__main__":
    main()
