#!/usr/bin/env python3
"""
v3 Phase 6.3 — k-fold cross-validation on top-k finalists only.

Reads ``finalists.csv`` (output of ``scripts/sweep_hpo.py``) and runs
k-fold CV on each finalist using its tuned hyperparameters from
``best_params.yaml``. Reports mean ± std AUC per finalist and writes
a combined ``kfold_summary.csv``.

Usage:
    python scripts/kfold_finalists.py \
        --experiment configs/experiment/kfold_finalists.yaml \
        --finalists results/sweep_hpo/finalists.csv \
        --params_dir results/sweep_hpo/ \
        --preprocessing srad_clahe
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.splitter import (
    get_image_paths_and_labels, stratified_split, _infer_patient_id,
)
from src.data.dataloader import _make_dataset_and_loaders
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer
from src.training.swa import run_swa_on_recent


def build_kfold_indices(paths, labels, groups, n_folds: int, seed: int):
    """Stratified group k-fold indices.

    Each group gets a single label (majority), then we use
    StratifiedKFold-style logic on groups.
    """
    from sklearn.model_selection import StratifiedKFold

    unique_groups = sorted(set(groups))
    group_labels = []
    for g in unique_groups:
        mask = np.array(groups) == g
        l_in_g = np.array(labels)[mask]
        uniq, counts = np.unique(l_in_g, return_counts=True)
        group_labels.append(int(uniq[counts.argmax()]))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_g_idx, val_g_idx in skf.split(unique_groups, group_labels):
        train_groups = set(np.array(unique_groups)[train_g_idx].tolist())
        val_groups = set(np.array(unique_groups)[val_g_idx].tolist())
        train_idx = [i for i, g in enumerate(groups) if g in train_groups]
        val_idx = [i for i, g in enumerate(groups) if g in val_groups]
        folds.append((train_idx, val_idx))
    return folds, unique_groups, group_labels


def run_one_fold(args, arch, params, paths, labels, groups, fold_idx, train_idx, val_idx):
    """Train + evaluate one fold. Returns metrics dict."""
    seed = args.seed + fold_idx
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_paths = [paths[i] for i in train_idx]
    fold_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Build model with overridden config
    model_config = load_config(f"configs/model/{arch}.yaml")
    model_config["freeze_fraction"] = params["freeze_fraction"]
    model_config["head"]["dropout"] = params["dropout"]
    model_config["lr"] = params["lr"]
    model_config["weight_decay"] = params["weight_decay"]
    model = build_model(model_config)

    # Build train/val loaders (in-memory, no preprocessing to disk)
    train_loader, val_loader = _make_dataset_and_loaders(
        fold_paths, fold_labels, val_paths, val_labels,
        args.preprocessing_config, batch_size=int(params["batch_size"]),
        input_size=model_config.get("input_size", 224),
    )

    weights = train_loader.dataset.get_class_weights()
    criterion = build_weighted_loss(weights, device=device)

    fold_dir = os.path.join(args.out_dir, arch, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    ckpt_path = os.path.join(fold_dir, "best.pt")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=val_loader, criterion=criterion,
        config={
            "lr": params["lr"], "weight_decay": params["weight_decay"],
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "warmup_epochs": 2, "freeze_epochs": 1,
            "bf16": True, "channels_last": True,
            "grad_clip_norm": 1.0, "ema": True, "ema_decay": 0.999,
        },
        logger=_make_logger(fold_dir, arch),
        device=device, checkpoint_path=ckpt_path, silent=False,
    )
    metrics = trainer.train()
    return metrics


def _make_logger(fold_dir, arch):
    from src.utils.logging import ExperimentLogger, EPOCH_LOG_HEADER
    # Lazy import to avoid pulling at module top.

    class _L(ExperimentLogger):
        def __init__(self):
            super().__init__(fold_dir, {"arch": arch})
    return _L()


def main():
    parser = argparse.ArgumentParser(description="k-fold CV on top-k finalists")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--finalists", type=str, required=True)
    parser.add_argument("--params_dir", type=str, required=True)
    parser.add_argument("--preprocessing", type=str, default="srad_clahe")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/kfold/")
    args = parser.parse_args()

    exp_config = load_config(args.experiment)
    args.experiment_config = exp_config
    args.seed = exp_config.get("seed", 42)
    args.max_epochs = exp_config.get("training", {}).get("max_epochs", 100)
    args.patience = exp_config.get("training", {}).get("early_stopping_patience", 20)
    args.n_folds = exp_config.get("n_folds", 5)

    args.preprocessing_config = load_config(
        f"configs/preprocessing/{args.preprocessing}.yaml"
    )

    finalists = pd.read_csv(args.finalists)
    args.finalists_list = finalists["arch"].tolist()
    os.makedirs(args.out_dir, exist_ok=True)

    # Get all paths once (in-memory CV doesn't write preprocessed data)
    all_paths, all_labels, all_groups = get_image_paths_and_labels(args.data_dir)

    summary_rows = []
    for arch in args.finalists_list:
        print(f"\n{'='*60}\n[K-Fold] Architecture: {arch}\n{'='*60}")
        params_path = os.path.join(args.params_dir, arch, "best_params.yaml")
        with open(params_path) as f:
            params = yaml.safe_load(f)

        folds, _, _ = build_kfold_indices(
            all_paths, all_labels, all_groups,
            n_folds=args.n_folds, seed=args.seed,
        )

        per_fold = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            try:
                metrics = run_one_fold(
                    args, arch, params,
                    all_paths, all_labels, all_groups,
                    fold_idx, train_idx, val_idx,
                )
                per_fold.append(metrics)
                summary_rows.append({
                    "arch": arch, "fold": fold_idx,
                    "val_auc": metrics["val_auc_best"],
                    "test_auc": metrics.get("test_auc_roc", float("nan")),
                })
            except Exception as e:
                print(f"[K-Fold] {arch} fold {fold_idx} failed: {e}")

        if per_fold:
            aucs = [m["val_auc_best"] for m in per_fold]
            print(f"[K-Fold] {arch}: mean AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # ---- Summary CSV ----
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out_dir, "kfold_summary.csv"), index=False,
    )
    print(f"[K-Fold] Summary written to {args.out_dir}/kfold_summary.csv")


if __name__ == "__main__":
    main()
