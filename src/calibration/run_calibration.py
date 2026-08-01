"""
Calibration analysis CLI: temperature scaling, reliability diagrams,
ECE + ensemble-level calibration.

This is the only entry point for calibration. It expects a fully
preprocessed PNG dataset and trained checkpoints; it never touches
raw data or raw datasets.

Single-model mode::

    python -m src.calibration.run_calibration \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny \
        --checkpoint results/figshare/swin_tiny/best.pt \
        --out-dir results/calibration/figshare/swin_tiny

Ensemble mode::

    python -m src.calibration.run_calibration \
        --dataset-dir data/preprocessed/figshare \
        --model swin_tiny --checkpoint results/.../swin_tiny/best.pt \
        --model densenet169 --checkpoint results/.../densenet169/best.pt \
        --ensemble

Single-model calibration lives here; the two-pass ensemble calibration
is delegated to ``src.ensemble.run_calibration_ensemble`` so all
ensemble code is in one place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.calibration.ece import compute_bin_data, compute_ece
from src.calibration.reliability import plot_reliability_diagram
from src.calibration.temperature_scaling import (
    apply_temperature,
    fit_temperature,
)
from src.data.dataloader import build_dataloaders
from src.evaluation.evaluator import collect_logits_and_labels
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.utils.config import DEFAULT_PREPROCESSING_CONFIG, load_config
from src.utils.seed import set_seed


def _load_model(model_name: str, checkpoint: str, device: str) -> torch.nn.Module:
    cfg_path = (
        model_name
        if model_name.endswith((".yaml", ".yml"))
        else os.path.join("configs", "model", f"{model_name}.yaml")
    )
    model_config = load_config(cfg_path)
    model = build_model(model_config)
    load_checkpoint(model, checkpoint, device=device)
    model.to(device)
    model.eval()
    return model


def _run_single(
    dataset_dir: str,
    preproc_config: dict,
    model_name: str,
    checkpoint: str,
    n_bins: int,
    batch_size: int,
    out_dir: str,
    device: str,
) -> Dict[str, float]:
    _, val_loader, test_loader = build_dataloaders(
        dataset_dir,
        preproc_config,
        batch_size=batch_size,
        num_workers=2,
        sampler="none",
        drop_last=False,
    )
    model = _load_model(model_name, checkpoint, device)
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)

    test_probs_before = F.softmax(test_logits, dim=1)[:, 1].numpy()
    test_labels_np = test_labels.numpy()

    ece_before = compute_ece(test_probs_before, test_labels_np, n_bins=n_bins)
    bin_data_before = compute_bin_data(test_probs_before, test_labels_np, n_bins=n_bins)
    print(f"[Calibration] ECE before: {ece_before:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    plot_reliability_diagram(
        bin_data_before,
        title=f"Reliability — Before ({model_name})",
        save_path=os.path.join(out_dir, "reliability_diagram_before.png"),
    )

    optimal_T = fit_temperature(val_logits, val_labels)
    test_probs_after = apply_temperature(test_logits, optimal_T)[:, 1].numpy()
    ece_after = compute_ece(test_probs_after, test_labels_np, n_bins=n_bins)
    bin_data_after = compute_bin_data(test_probs_after, test_labels_np, n_bins=n_bins)
    print(f"[Calibration] ECE after T={optimal_T:.3f}: {ece_after:.4f}")

    plot_reliability_diagram(
        bin_data_after,
        title=f"Reliability — After T={optimal_T:.2f} ({model_name})",
        save_path=os.path.join(out_dir, "reliability_diagram_after.png"),
    )

    pd.DataFrame(bin_data_after).to_csv(
        os.path.join(out_dir, "bin_data.csv"), index=False
    )

    def _nll(logits, labels, T):
        scaled = logits / T
        log_probs = F.log_softmax(scaled, dim=1)
        return -log_probs[range(len(labels)), labels].mean().item()

    results = {
        "model": model_name,
        "dataset": dataset_dir,
        "checkpoint": checkpoint,
        "ece_before": round(float(ece_before), 4),
        "ece_after": round(float(ece_after), 4),
        "optimal_temperature": round(float(optimal_T), 4),
        "val_nll_before": round(float(_nll(val_logits, val_labels, 1.0)), 4),
        "val_nll_after": round(float(_nll(val_logits, val_labels, optimal_T)), 4),
        "n_bins": n_bins,
        "n_test_samples": int(len(test_labels_np)),
    }
    with open(os.path.join(out_dir, "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Calibration] Saved to {out_dir}")
    return results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibration analysis on a preprocessed PNG dataset."
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Preprocessed dataset root with train/val/test.",
    )
    parser.add_argument(
        "--preprocessing", default=DEFAULT_PREPROCESSING_CONFIG,
        help=f"Preprocessing config (default: {DEFAULT_PREPROCESSING_CONFIG}).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Run two-pass ensemble calibration (delegates to src.ensemble).",
    )
    parser.add_argument(
        "--model", action="append", default=[],
        help="Model name or YAML path. Repeat for ensemble members.",
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[],
        help="Checkpoint path; pair with each --model.",
    )
    args = parser.parse_args(argv)

    if len(args.model) != len(args.checkpoint) or not args.model:
        parser.error("Provide equal numbers of --model and --checkpoint flags")

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    preproc_config = load_config(args.preprocessing)

    if args.ensemble:
        from src.ensemble import run_calibration_ensemble
        run_calibration_ensemble(
            dataset_dir=args.dataset_dir,
            preproc_config=preproc_config,
            model_specs=list(zip(args.model, args.checkpoint)),
            n_bins=args.n_bins,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            device=device,
        )
    else:
        _run_single(
            dataset_dir=args.dataset_dir,
            preproc_config=preproc_config,
            model_name=args.model[0],
            checkpoint=args.checkpoint[0],
            n_bins=args.n_bins,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            device=device,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())