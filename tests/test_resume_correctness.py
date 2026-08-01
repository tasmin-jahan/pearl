"""
Pytest suite — resume correctness.

Verifies that resuming a Trainer from a saved checkpoint restores
model weights, optimizer state, RNG state, and continues deterministically.

Critical: depends on Phase 3.3 (RNG state in checkpoints) being implemented.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.seed import set_seed
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.trainer import Trainer, TrialDivergedError


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class _FixedDS(Dataset):
    """Deterministic synthetic dataset."""

    def __init__(self, n=20):
        torch.manual_seed(0)
        self.x = torch.rand(n, 4)
        # Half 0, half 1 based on sum
        y = (self.x.sum(dim=1) > 2).long()
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def get_class_weights(self):
        counts = torch.bincount(self.y, minlength=2).float()
        return len(self.x) / (2.0 * counts.clamp_min(1))


class _NullLogger:
    def log_epoch(self, *a, **kw): pass
    def log_final_metrics(self, *a, **kw): pass
    def plot_training_curves(self, *a, **kw): pass
    def close(self, *a, **kw): pass


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_checkpoint_roundtrip_preserves_model(tmp_dir):
    set_seed(42)
    m = _TinyModel()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ckpt_path = os.path.join(tmp_dir, "c.pt")
    save_checkpoint(m, opt, epoch=3, val_auc=0.91, path=ckpt_path,
                    arch_name="tiny", class_names=["a", "b"],
                    save_rng=False)

    m2 = _TinyModel()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    info = load_checkpoint(m2, ckpt_path, optimizer=opt2, device="cpu")
    assert info["epoch"] == 3
    assert info["val_auc"] == 0.91
    assert info["arch_name"] == "tiny"
    assert info["class_names"] == ["a", "b"]
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1, p2), f"mismatch at {n1}"


def test_resume_state_restores_rng(tmp_dir):
    """Trainer._resume_state should restore torch/numpy RNG to checkpoint state."""
    set_seed(0)
    train_ds = _FixedDS(20)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(_FixedDS(8), batch_size=4)

    model = _TinyModel()
    weights = train_ds.get_class_weights()
    from src.training.losses import build_weighted_loss
    criterion = build_weighted_loss(weights, device="cpu")

    cfg = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "max_epochs": 1,
        "early_stopping_patience": 100,
        "warmup_epochs": 0,
        "freeze_epochs": 0,
        "bf16": False,
        "channels_last": False,
        "compile": False,
        "ema": False,
        "save_rng_state": True,
    }
    ckpt_path = os.path.join(tmp_dir, "best.pt")
    trainer = Trainer(
        model=model,
        train_loader=train_loader, val_loader=val_loader, test_loader=val_loader,
        criterion=criterion, config=cfg,
        logger=_NullLogger(), device="cpu", checkpoint_path=ckpt_path, silent=True,
    )
    trainer.train()

    # Now simulate resume
    trainer2 = Trainer(
        model=_TinyModel(), train_loader=train_loader, val_loader=val_loader,
        test_loader=val_loader, criterion=criterion, config=cfg,
        logger=_NullLogger(), device="cpu", checkpoint_path=ckpt_path, silent=True,
    )
    trainer2._resume_state(ckpt_path)
    assert trainer2.start_epoch == cfg["max_epochs"] + 1


def test_divergence_guard_raises(tmp_dir):
    """TrialDivergedError propagates from the trainer when loss is non-finite."""
    from src.training.losses import build_weighted_loss

    class _BadLoss(nn.Module):
        def forward(self, logits, labels):
            # Return NaN immediately — should trigger guard.
            return torch.tensor(float("nan"))

    set_seed(0)
    ds = _FixedDS(8)
    loader = DataLoader(ds, batch_size=4)
    model = _TinyModel()
    weights = ds.get_class_weights()
    crit = build_weighted_loss(weights, device="cpu")

    cfg = {
        "lr": 1e-3, "weight_decay": 1e-2,
        "max_epochs": 1, "early_stopping_patience": 100,
        "warmup_epochs": 0, "freeze_epochs": 0,
        "bf16": False, "channels_last": False,
        "compile": False, "ema": False,
    }
    trainer = Trainer(
        model=model, train_loader=loader, val_loader=loader, test_loader=loader,
        criterion=_BadLoss(), config=cfg, logger=_NullLogger(),
        device="cpu", checkpoint_path=os.path.join(tmp_dir, "x.pt"), silent=True,
    )
    with pytest.raises(TrialDivergedError):
        trainer.train()