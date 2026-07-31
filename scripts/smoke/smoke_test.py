#!/usr/bin/env python3
"""
Quick smoke test: 15-second, 2-epoch, 10-synthetic-image sanity check.

Run before committing to a long sweep to catch integration breakage
fast. Always uses CPU unless CUDA is unavoidable.

Usage:
    python scripts/smoke_test.py [--device cuda]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer


class _SyntheticDataset(Dataset):
    """Tiny synthetic dataset for smoke tests."""

    def __init__(self, n: int = 10, h: int = 224, w: int = 224, channels: int = 3):
        self.n = n
        self.images = torch.rand(n, channels, h, w, dtype=torch.float32)
        # Half class 0, half class 1
        self.labels = torch.tensor([0] * (n // 2) + [1] * (n - n // 2), dtype=torch.long)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_class_weights(self) -> torch.Tensor:
        # Inverse frequency
        counts = torch.bincount(self.labels, minlength=2).float()
        weights = self.n / (2.0 * counts.clamp_min(1))
        return weights


class _NullLogger:
    def log_epoch(self, *a, **kw): pass
    def log_final_metrics(self, *a, **kw): pass
    def plot_training_curves(self, *a, **kw): pass
    def close(self, *a, **kw): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(42)
    print("[Smoke] Starting 2-epoch synthetic run...")

    train_ds = _SyntheticDataset(n=10)
    val_ds = _SyntheticDataset(n=4)
    test_ds = _SyntheticDataset(n=4)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    # Build a tiny model — use resnet50 but force CPU + no compile
    model_config = {
        "name": "smoke",
        "timm_name": "resnet50",
        "input_size": 224,
        "pretrained": False,  # no network required
        "num_classes": 2,
        "freeze_fraction": 0.5,
        "head": {"hidden_dim": 64, "dropout": 0.3},
    }
    model = build_model(model_config)

    weights = train_ds.get_class_weights()
    criterion = build_weighted_loss(weights, device=args.device)

    training_config = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "max_epochs": 2,
        "early_stopping_patience": 10,
        "warmup_epochs": 1,
        "freeze_epochs": 0,
        "bf16": False,  # CPU
        "channels_last": False,
        "compile": False,
        "ema": False,
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        config=training_config,
        logger=_NullLogger(),
        device=args.device,
        checkpoint_path="results/smoke_best.pt",
        silent=True,
    )

    t0 = time.time()
    metrics = trainer.train()
    elapsed = time.time() - t0

    print(f"\n[Smoke] Done in {elapsed:.1f}s")
    print(f"[Smoke] val_auc={metrics['val_auc_best']:.4f}, "
          f"test_acc={metrics.get('test_accuracy', 0):.4f}")
    if elapsed > 60:
        print(f"[Smoke] WARNING: took {elapsed:.1f}s (target: <15s)")
    print("[Smoke] PASSED")


if __name__ == "__main__":
    main()
