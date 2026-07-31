#!/usr/bin/env python3
"""Retrain top-K fine-tuned checkpoints with their best HPO params.

For each arch we:
  1. Read best_params.yaml from results/finetune_hpo/<prep>_<arch>/<arch>/.
  2. Backup the existing fine-tuned best.pt → best_before_hpo.pt.
  3. Build model with the HPO dropout (head) + freeze_fraction.
  4. Load the Figshare weights (the original foundation).
  5. Train for the full 15 epochs with the HPO LR / wd / batch / augmentation.
  6. Save the new best.pt and final_metrics.json in-place.

Usage:
    python scripts/retrain_with_hpo.py \
        --preprocessing srad_nopad \
        --arch densenet121 \
        --hpo_dir results/finetune_hpo/srad_nopad_densenet121/densenet121 \
        --resume_root results/finetune_zenodo/checkpoints/ \
        --figshare_root results/ablation_nopad/checkpoints/ \
        --train_split data_external/zenodo_splits/train.json \
        --val_split data_external/zenodo_splits/val.json \
        --max_epochs 15
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import ExperimentLogger
from src.data.zenodo_dataset import build_zenodo_loader
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer
from src.training.checkpoint import load_checkpoint


def _load_split(path):
    with open(path) as f:
        d = json.load(f)
    return [(item["path"], int(item["label"])) for item in d["items"]]


def retrain_one(prep, arch, hpo_dir, resume_root, figshare_root,
                train_split, val_split, max_epochs, seed):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load HPO best params
    params_file = os.path.join(hpo_dir, "best_params.yaml")
    if not os.path.isfile(params_file):
        raise FileNotFoundError(f"Missing {params_file}")
    with open(params_file) as f:
        params = yaml.safe_load(f)
    print(f"[Retrain] {arch} + {prep} — HPO params: {params}")

    # Splits
    train_pairs = _load_split(train_split)
    val_pairs = _load_split(val_split)
    print(f"[Retrain] Splits: {len(train_pairs)} train, {len(val_pairs)} val")
    print(f"  train class dist: {dict(Counter(l for _, l in train_pairs))}")

    # Per-arch configs
    model_config = load_config(f"configs/model/{arch}.yaml")
    preproc_config = load_config(f"configs/preprocessing/{prep}.yaml")
    model_config = dict(model_config)
    model_config["freeze_fraction"] = float(params["freeze_fraction"])
    head = dict(model_config.get("head", {}))
    head["dropout"] = float(params["dropout"])
    model_config["head"] = head

    # Training config — take HPO params + extend training schedule
    cfg = {
        "optimizer": "adamw",
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "batch_size": int(params["batch_size"]),
        "scheduler": "cosine_annealing",
        "max_epochs": int(max_epochs),
        "early_stopping_patience": 5,
        "warmup_epochs": 1,
        "freeze_epochs": 0,
        "monitor": "val_auc",
        "bf16": True,
        "channels_last": True,
        "grad_clip_norm": 1.0,
        "ema": True,
        "sampler": "weighted",
        "keep_last_n": 1,
        "augmentation": {
            "rotation": int(params["rotation"]),
            "horizontal_flip": True,
            "scale": 0.1,
        },
    }

    # Output dir
    arch_dir = os.path.join(resume_root, prep, arch)
    os.makedirs(arch_dir, exist_ok=True)
    ckpt_path = os.path.join(arch_dir, "best.pt")
    backup_path = os.path.join(arch_dir, "best_before_hpo.pt")
    if os.path.isfile(ckpt_path) and not os.path.isfile(backup_path):
        shutil.copy2(ckpt_path, backup_path)
        print(f"[Retrain] Backed up best.pt → {backup_path}")

    # Build model
    model = build_model(model_config)
    # Load Figshare weights as the foundation (Honest transfer)
    figshare_ckpt = os.path.join(figshare_root, prep, arch, "best.pt")
    if not os.path.isfile(figshare_ckpt):
        raise FileNotFoundError(f"Missing Figshare weights: {figshare_ckpt}")
    print(f"[Retrain] Loading Figshare weights from {figshare_ckpt}")
    load_checkpoint(model, figshare_ckpt, optimizer=None, scheduler=None,
                    ema=None, device=device)

    # Loaders
    input_size = model_config.get("input_size", 224)
    train_loader, val_loader = build_zenodo_loader(
        train_pairs=train_pairs, val_pairs=val_pairs,
        preproc_config=preproc_config,
        batch_size=int(params["batch_size"]),
        input_size=input_size,
        sampler="weighted",
    )

    # Loss
    class_weights = train_loader.dataset.get_class_weights()
    criterion = build_weighted_loss(class_weights, device=device)

    # Logger
    full_config = {
        "model": model_config,
        "preprocessing": preproc_config,
        "hpo_params": params,
        "experiment": {"max_epochs": max_epochs, "seed": seed},
        "seed": seed,
    }
    logger = ExperimentLogger(arch_dir, full_config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=val_loader,
        criterion=criterion,
        config=cfg,
        logger=logger,
        device=device,
        checkpoint_path=ckpt_path,
    )
    metrics = trainer.train()
    metrics["arch"] = arch
    metrics["preprocessing"] = prep
    metrics["hpo_params"] = params
    metrics["figshare_resume_from"] = figshare_ckpt
    print(f"[Retrain] Done. best val_auc = {metrics.get('val_auc_best', '?'):.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--hpo_dir", required=True)
    parser.add_argument("--resume_root", default="results/finetune_zenodo/checkpoints/")
    parser.add_argument("--figshare_root", default="results/ablation_nopad/checkpoints/")
    parser.add_argument("--train_split", default="data_external/zenodo_splits/train.json")
    parser.add_argument("--val_split", default="data_external/zenodo_splits/val.json")
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    retrain_one(
        prep=args.preprocessing,
        arch=args.arch,
        hpo_dir=args.hpo_dir,
        resume_root=args.resume_root,
        figshare_root=args.figshare_root,
        train_split=args.train_split,
        val_split=args.val_split,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
