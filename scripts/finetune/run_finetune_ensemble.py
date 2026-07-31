#!/usr/bin/env python3
"""Ensemble of top-K fine-tuned checkpoints, evaluated on Zenodo test.

Loads the best.pt from each --run_dir, computes probability-averaged predictions
on the Zenodo test set (1468 images), writes metrics + per-image CSV to a
new ensemble directory, and optionally runs calibration on the ensemble.

Usage:
    python scripts/run_finetune_ensemble.py \
        --run_dirs results/finetune_zenodo/checkpoints/srad_nopad/resnet50 \
                   results/finetune_zenodo/checkpoints/srad_nopad/efficientnet_b0 \
                   results/finetune_zenodo/checkpoints/gauss_nopad/resnet50 \
        --model configs/model/resnet50.yaml \
        --preprocessing configs/preprocessing.yaml \
        --out_dir results/finetune_zenodo/ensemble/top3_srad_gauss \
        --external_dir data_external/test

Each --run_dir must share the same preprocessing config (we apply the same
preprocessor to the test set as was used to train each member). The ensemble
output is written to:
    <out_dir>/external_validation/pcosgen.json
    <out_dir>/external_validation/pcosgen.csv

Preprocessing is now the unified config (configs/preprocessing.yaml).
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.model.builder import build_model
from src.training.checkpoint import load_checkpoint
from src.preprocessing.preprocess import Preprocessor
from src.data.zenodo_dataset import ZenodoDataset, discover_zenodo_pairs
from src.evaluation.metrics import compute_all_metrics


def _load_one_model(run_dir: str, model_config: dict, device: str) -> Tuple[torch.nn.Module, str]:
    """Build a model + load weights from a per-arch run dir. Returns (model, arch_name)."""
    model = build_model(model_config)
    # The model config name tells us the architecture; we use the user-supplied
    # model_config directly here so all members share the same arch config.
    # But ensembles typically mix arches — so we look up the per-arch config.
    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    load_checkpoint(model, ckpt_path, optimizer=None, scheduler=None, ema=None, device=device)
    model.eval()
    model.to(device)
    return model, os.path.basename(os.path.normpath(run_dir))


def _build_loader(model_config: dict, preproc_config: dict, pairs, batch_size: int) -> DataLoader:
    pre = Preprocessor(preproc_config, input_size=model_config.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)


def _predict_one(model: torch.nn.Module, loader: DataLoader, device: str):
    paths, labels, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = F.softmax(logits, dim=1)[:, 1]
            paths.extend([None] * len(y))  # we don't have paths through ZenodoDataset
            labels.extend([int(v) for v in y])
            probs.extend(p.cpu().numpy().tolist())
    return np.array(labels), np.array(probs)


def _predict_with_paths(model: torch.nn.Module, pairs: List[Tuple[str, int]], preproc_config: dict,
                        model_config: dict, device: str, batch_size: int = 32):
    """Same as _predict_one but yields path info too (used for CSV output)."""
    loader = _build_loader(model_config, preproc_config, pairs, batch_size)
    paths, labels, probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            p = F.softmax(logits, dim=1)[:, 1]
            labels.extend([int(v) for v in batch_y])
            probs.extend(p.cpu().numpy().tolist())
    # Recover paths from the underlying dataset indices (simple linear walk).
    # We use the raw pairs list order to keep paths aligned with labels.
    base_paths = [p for p, _ in pairs]
    assert len(base_paths) == len(labels), (
        f"Path/label length mismatch: {len(base_paths)} vs {len(labels)}"
    )
    return base_paths, np.array(labels), np.array(probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True,
                        help="Per-arch run dirs containing best.pt (each must "
                             "share the same preprocessing config).")
    parser.add_argument("--model", required=True,
                        help="Model config YAML (used for the ensemble's "
                             "model-template). The script regenerates per-arch "
                             "configs from the run_dir names.")
    parser.add_argument("--preprocessing", required=True,
                        help="Preprocessing config YAML applied to the test set.")
    parser.add_argument("--external_dir", default="data_external/test",
                        help="Zenodo test root.")
    parser.add_argument("--out_dir", required=True,
                        help="Where to write the ensemble metrics + CSV.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Discover test pairs (path + label)
    pairs = discover_zenodo_pairs(args.external_dir)
    if not pairs:
        raise RuntimeError(f"No test images found at {args.external_dir}")
    n_inf = sum(1 for _, l in pairs if l == 1)
    n_h = len(pairs) - n_inf
    print(f"[Ensemble] External test set: {len(pairs)} pairs ({n_inf} infected, {n_h} healthy)")

    # Build per-arch model config + load each member
    # Convention: <run_dir>/best.pt + the model name is the last path component.
    member_results = []
    preproc_config = load_config(args.preprocessing)
    base_model_config = load_config(args.model)

    # For ensembles, the *base* model config is just a template; each member
    # needs its own arch config. We resolve that by looking at the per-arch
    # config.yaml that the trainer already saved inside each run_dir.
    probs_acc = None
    labels_ref = None
    paths_ref = None
    for run_dir in args.run_dirs:
        cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Missing config.yaml under {run_dir}. The trainer normally writes this."
            )
        per_run_config = load_config(cfg_path)
        # The trainer's config has a "model" key with the arch config dict.
        arch_config = per_run_config.get("model", base_model_config)
        # Strip freeze_fraction so we don't print confusing "frozen" lines during eval.
        arch_config = dict(arch_config)
        arch_config["freeze_fraction"] = 0.0

        ckpt_path = os.path.join(run_dir, "best.pt")
        print(f"[Ensemble] Loading {run_dir} ({arch_config.get('name', '?')})")
        model = build_model(arch_config)
        load_checkpoint(model, ckpt_path, optimizer=None, scheduler=None, ema=None, device=device)
        model.eval()
        model.to(device)

        paths, labels, probs = _predict_with_paths(
            model, pairs, preproc_config, arch_config, device, args.batch_size,
        )
        member_results.append({
            "run_dir": run_dir,
            "arch": arch_config.get("name", "?"),
            "preprocessing": preproc_config.get("name", "?"),
            "individual_test_auc": float(compute_all_metrics(labels, (probs >= 0.5).astype(int), probs)["test_auc_roc"]),
        })
        if probs_acc is None:
            probs_acc = probs.copy()
            labels_ref = labels
            paths_ref = paths
        else:
            # Sum (then divide at the end for the average)
            probs_acc = probs_acc + probs
        del model
        torch.cuda.empty_cache()

    # Average probabilities across members
    probs_avg = probs_acc / len(args.run_dirs)
    preds = (probs_avg >= 0.5).astype(int)

    # Compute ensemble metrics
    metrics = compute_all_metrics(labels_ref, preds, probs_avg)
    metrics["n_members"] = len(args.run_dirs)
    metrics["n_samples"] = len(labels_ref)
    metrics["n_infected"] = n_inf
    metrics["n_healthy"] = n_h
    metrics["members"] = member_results
    metrics["architecture"] = f"ensemble[{','.join(m['arch'] for m in member_results)}]"
    metrics["preprocessing"] = preproc_config.get("name", "?")

    # Write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    ext_dir = os.path.join(args.out_dir, "external_validation")
    os.makedirs(ext_dir, exist_ok=True)
    json_path = os.path.join(ext_dir, "pcosgen.json")
    csv_path = os.path.join(ext_dir, "pcosgen.csv")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred", "prob_infected"])
        for p, l, pr, pb in zip(paths_ref, labels_ref, preds, probs_avg):
            w.writerow([p, int(l), int(pr), f"{pb:.6f}"])
    print(f"\n[Ensemble] Metrics:")
    for k in ("test_auc_roc", "test_accuracy", "test_f1", "test_mcc",
              "test_sensitivity", "test_specificity", "test_precision"):
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")
    print(f"[Ensemble] Wrote {json_path}")
    print(f"[Ensemble] Wrote {csv_path}")


if __name__ == "__main__":
    main()