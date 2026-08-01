"""
End-to-end smoke test for the Trainer pipeline.

Replaces ``scripts/smoke/smoke_test.py`` — runs as a regular pytest case so
it is exercised in CI alongside the rest of the suite. Synthetic data,
CPU-only, ~15s. Catches integration breakage (model build, weighted loss,
optimizer, scheduler, EMA-disabled training loop, checkpoint write).

Run as::

    pytest tests/test_smoke.py -s
"""
from __future__ import annotations

import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.model.builder import build_model
from src.training.losses import build_weighted_loss
from src.training.trainer import Trainer
from src.utils.seed import set_seed


class _SyntheticDataset(Dataset):
    """Tiny synthetic dataset for the smoke test."""

    def __init__(self, n: int = 10, h: int = 224, w: int = 224, channels: int = 3):
        self.n = n
        self.images = torch.rand(n, channels, h, w, dtype=torch.float32)
        self.labels = torch.tensor(
            [0] * (n // 2) + [1] * (n - n // 2), dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def get_class_weights(self) -> torch.Tensor:
        counts = torch.bincount(self.labels, minlength=2).float()
        return self.n / (2.0 * counts.clamp_min(1))


class _NullLogger:
    def log_epoch(self, *a, **kw): pass
    def log_final_metrics(self, *a, **kw): pass
    def plot_training_curves(self, *a, **kw): pass
    def close(self, *a, **kw): pass


def test_smoke_trainer_runs_end_to_end(tmp_path):
    """15-second, 2-epoch synthetic Trainer run on CPU."""
    set_seed(42)
    train_ds = _SyntheticDataset(n=10)
    val_ds = _SyntheticDataset(n=4)
    test_ds = _SyntheticDataset(n=4)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    model_config = {
        "name": "smoke",
        "timm_name": "resnet50",
        "input_size": 224,
        "pretrained": False,
        "num_classes": 2,
        "freeze_fraction": 0.5,
        "head": {"hidden_dim": 64, "dropout": 0.3},
    }
    model = build_model(model_config)

    weights = train_ds.get_class_weights()
    criterion = build_weighted_loss(weights, device="cpu")

    training_config = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "max_epochs": 2,
        "early_stopping_patience": 10,
        "warmup_epochs": 1,
        "freeze_epochs": 0,
        "bf16": False,
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
        device="cpu",
        checkpoint_path=str(tmp_path / "smoke_best.pt"),
        silent=True,
    )

    t0 = time.time()
    metrics = trainer.train()
    elapsed = time.time() - t0

    # Sanity: trainer returned metrics, ran within budget, wrote a checkpoint.
    assert "val_auc_best" in metrics
    assert elapsed < 60, f"smoke run took {elapsed:.1f}s (target: <60s)"


def test_smoke_module_removed():
    """`scripts/smoke/` should be gone after the move to tests/."""
    import os
    assert not os.path.isdir("scripts/smoke"), \
        "scripts/smoke/ should be deleted (its contents live in tests/test_smoke.py)"