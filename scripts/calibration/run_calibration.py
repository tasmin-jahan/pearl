#!/usr/bin/env python3
"""
CLI entrypoint for calibration analysis: ECE + temperature scaling + reliability diagrams.

Supports two evaluation contexts via --test_dir:

1. Figshare mode (default): runs calibration on the standard Figshare train/val/test
   loaders. Use:
       python scripts/calibration/run_calibration.py \
           --model configs/model/swin_tiny.yaml \
           --preprocessing configs/preprocessing.yaml \
           --checkpoint results/checkpoints/srad/swin_tiny.pt
       python scripts/calibration/run_calibration.py --run_dir <run_dir> [--test_dir <path>]

2. Zenodo mode (--test_dir set, --run_dir set): runs calibration on an external
   held-out test set (e.g. Zenodo PCOSgen) with a 50/50 split inside the test set
   to fit T honestly without using val data.

Preprocessing is now the unified config (configs/preprocessing.yaml).

This file replaces both `run_calibration.py` (original Figshare-only version) and
`run_calibration_zenodo.py` (Zenodo-only version).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.evaluation.evaluator import collect_logits_and_labels
from src.calibration.ece import compute_ece, compute_bin_data
from src.calibration.reliability import plot_reliability_diagram
from src.calibration.temperature_scaling import fit_temperature, apply_temperature
from src.preprocessing.preprocess import Preprocessor
from torch.utils.data import DataLoader


def _load_model_from_run_dir(run_dir, device):
    """Load model and preprocessing config from a run_dir (which contains
    best.pt and config.yaml). Returns (model, model_config, preproc_config)."""
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.yaml under {run_dir}")
    cfg = load_config(cfg_path)
    model_config = cfg["model"]
    preproc_config = cfg["preprocessing"]
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


def _calibrate_figshare(args, model_config, preproc_config, device, out_dir):
    """Standard Figshare train/val/test calibration."""
    input_size = model_config.get("input_size", 224)
    _, val_loader, test_loader = build_dataloaders(
        preproc_config, batch_size=32, input_size=input_size,
    )
    # Build the same model the legacy CLI did (build_model + load_checkpoint).
    model = build_model(model_config)
    load_checkpoint(model, args.checkpoint, device=device)

    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)

    test_probs_before = F.softmax(test_logits, dim=1)[:, 1].numpy()
    test_labels_np = test_labels.numpy()

    ece_before = compute_ece(test_probs_before, test_labels_np, n_bins=args.n_bins)
    bin_data_before = compute_bin_data(test_probs_before, test_labels_np, n_bins=args.n_bins)
    print(f"[Calibration] ECE before: {ece_before:.4f}")

    plot_reliability_diagram(
        bin_data_before,
        title=f"Reliability Diagram (Before) — {model_config['name']}",
        save_path=os.path.join(out_dir, "reliability_diagram_before.png"),
    )

    optimal_T = fit_temperature(val_logits, val_labels)

    test_probs_after = apply_temperature(test_logits, optimal_T)[:, 1].numpy()
    ece_after = compute_ece(test_probs_after, test_labels_np, n_bins=args.n_bins)
    bin_data_after = compute_bin_data(test_probs_after, test_labels_np, n_bins=args.n_bins)

    print(f"[Calibration] ECE after: {ece_after:.4f}")

    plot_reliability_diagram(
        bin_data_after,
        title=f"Reliability Diagram (After T={optimal_T:.2f}) — {model_config['name']}",
        save_path=os.path.join(out_dir, "reliability_diagram_after.png"),
    )

    pd.DataFrame(bin_data_after).to_csv(os.path.join(out_dir, "bin_data.csv"), index=False)

    def nll(logits, labels, T=1.0):
        scaled = logits / T
        log_probs = F.log_softmax(scaled, dim=1)
        return -log_probs[range(len(labels)), labels].mean().item()

    results = {
        "arch": model_config["name"],
        "preprocessing": preproc_config["name"],
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "optimal_temperature": round(optimal_T, 4),
        "val_nll_before": round(nll(val_logits, val_labels, 1.0), 4),
        "val_nll_after": round(nll(val_logits, val_labels, optimal_T), 4),
        "n_bins": args.n_bins,
        "n_test_samples": len(test_labels_np),
    }
    with open(os.path.join(out_dir, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Calibration] Results saved to {out_dir}")


def _calibrate_zenodo(args, device, out_dir):
    """Zenodo-style calibration: 50/50 split inside the test set, fit T on the
    first half, compute honest post-T ECE on the second half."""
    model, model_config, preproc_config = _load_model_from_run_dir(args.run_dir, device)
    arch = model_config.get("name", "?")
    preproc_name = preproc_config.get("name", "?")

    pairs = discover_zenodo_pairs(args.test_dir)
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False,
    )

    all_logits, all_labels = [], []
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

    ece_before = compute_ece(test_probs_before, test_labels_np, n_bins=args.n_bins)
    bin_data_before = compute_bin_data(test_probs_before, test_labels_np, n_bins=args.n_bins)

    plot_reliability_diagram(
        bin_data_before,
        title=f"Reliability Diagram (Before) — {arch} + {preproc_name}",
        save_path=os.path.join(out_dir, "reliability_diagram_before.png"),
    )

    # 50/50 split: half for T-fit, half for honest ECE.
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

    pd.DataFrame(bin_data_after_2).to_csv(
        os.path.join(out_dir, "bin_data.csv"), index=False,
    )

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


def main():
    parser = argparse.ArgumentParser(description="Run calibration analysis")
    # --- Figshare mode (legacy) ---
    parser.add_argument("--model", type=str, default=None,
                        help="Model config YAML (Figshare mode)")
    parser.add_argument("--preprocessing", type=str, default=None,
                        help="Preprocessing config YAML (Figshare mode)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt (Figshare mode)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="If set, write artifacts under <run_dir>/calibration/. "
                             "In Figshare mode with --checkpoint, expects standard "
                             "YAML discovery. In Zenodo mode, reads config.yaml+best.pt.")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Zenodo mode: external test directory (e.g. data_external/test). "
                             "If set, switches to Zenodo mode (50/50 split + temp fit).")
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.test_dir is not None:
        # Zenodo mode
        if args.run_dir is None:
            parser.error("--run_dir is required when --test_dir is set")
        out_dir = os.path.join(args.run_dir, "calibration")
        os.makedirs(out_dir, exist_ok=True)
        _calibrate_zenodo(args, device, out_dir)
    else:
        # Figshare mode (legacy CLI: --model/--preprocessing/--checkpoint)
        if not (args.model and args.preprocessing and args.checkpoint):
            parser.error(
                "Figshare mode requires --model, --preprocessing, --checkpoint "
                "(or use --test_dir for Zenodo mode)"
            )
        if args.run_dir:
            out_dir = os.path.join(args.run_dir, "calibration")
        else:
            arch_name = load_config(args.model)["name"]
            prep_name = load_config(args.preprocessing)["name"]
            out_dir = os.path.join(
                "results", "calibration", f"{arch_name}__{prep_name}",
            )
        os.makedirs(out_dir, exist_ok=True)
        model_config = load_config(args.model)
        preproc_config = load_config(args.preprocessing)
        _calibrate_figshare(args, model_config, preproc_config, device, out_dir)


if __name__ == "__main__":
    main()
