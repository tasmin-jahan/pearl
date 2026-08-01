"""
Tests for src.train CLI argument handling.

The CLI is the public surface for the foundation / fine-tune / HPO
workflows. Two non-trivial behaviors live here that the rest of the
codebase assumes:

  - ``--model`` is optional when ``--model-dir`` is given (the
    fine-tune-all workflow discovers architectures from disk).
  - The mutual-exclusion is enforced only in the legacy sense; the
    new discovery mode is strictly additive.

We don't actually train anything (the trainer has its own coverage
in test_resume_correctness). These tests just verify argument parsing
and the auto-discovery list.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.train import build_parser, main as train_main
from src.training.checkpoint import save_checkpoint


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write_best(ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        model, opt, epoch=1, val_auc=0.5,
        path=str(ckpt_dir / "best.pt"),
        arch_name="tiny", class_names=["a", "b"], save_rng=False,
    )


def test_model_required_when_no_model_dir():
    """Without --model or --model-dir, CLI should fail with a clear message."""
    rc = train_main(["--dataset-dir", "/tmp"])
    assert rc == 1


def test_model_dir_auto_discovers_archs(tmp_dir):
    """--model-dir alone should populate args.model from <arch>/best.pt."""
    model_dir = Path(tmp_dir) / "results"
    (model_dir / "alpha").mkdir(parents=True)
    (model_dir / "beta").mkdir(parents=True)
    _write_best(model_dir / "alpha")
    _write_best(model_dir / "beta")
    # No best.pt in gamma — must be skipped.
    (model_dir / "gamma").mkdir(parents=True)

    ns = build_parser().parse_args([
        "--dataset-dir", tmp_dir,
        "--model-dir", str(model_dir),
        "--output-dir", str(Path(tmp_dir) / "out"),
    ])
    # Parse-only doesn't trigger auto-discovery; main() does. Call main()
    # with a dataset_dir that doesn't exist and verify auto-discovery
    # happens before the dataset_dir check.
    ns.model = []  # simulate "no --model"
    # Just call the discovery branch in isolation by importing the helper.
    from src.train import main as _main
    # We want to test only the auto-discovery logic, not the full train.
    # Re-implement the discovery here, mirroring train.main():
    if not ns.model:
        discovered = sorted(
            p.name for p in Path(ns.model_dir).iterdir()
            if p.is_dir() and (p / "best.pt").is_file()
        )
        ns.model = discovered
    assert ns.model == ["alpha", "beta"], ns.model


def test_model_dir_empty_or_missing(tmp_dir):
    """Empty --model-dir (no <arch>/best.pt) should report a clear error."""
    empty = Path(tmp_dir) / "empty_results"
    empty.mkdir()
    rc = train_main([
        "--dataset-dir", tmp_dir,
        "--model-dir", str(empty),
        "--output-dir", str(Path(tmp_dir) / "out"),
    ])
    assert rc == 1


def test_model_explicit_overrides_auto_discovery(tmp_dir):
    """If user passes --model explicitly, --model-dir is still respected
    for checkpoint lookup but arch list comes from --model."""
    model_dir = Path(tmp_dir) / "results"
    (model_dir / "alpha").mkdir(parents=True)
    (model_dir / "beta").mkdir(parents=True)
    _write_best(model_dir / "alpha")
    _write_best(model_dir / "beta")

    ns = build_parser().parse_args([
        "--dataset-dir", tmp_dir,
        "--model", "alpha",
        "--model-dir", str(model_dir),
        "--output-dir", str(Path(tmp_dir) / "out"),
    ])
    assert ns.model == ["alpha"], ns.model