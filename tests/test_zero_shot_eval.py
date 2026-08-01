"""
Tests for the dataset-agnostic multi-model evaluator.

Two things to verify:

1. ``build_test_loader`` accepts both the dataset root and a single
   split directory (``<root>/test``) — important for the zero-shot
   ``--test-dataset-dir data/preprocessed/pcosgen/test`` use case.
2. ``run_evaluate`` discovers every ``<arch>/best.pt`` under
   ``--model-dir`` and writes one metrics JSON per arch plus an
   aggregated ``all_metrics.json``.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.data.dataloader import build_test_loader
from src.evaluation import run_evaluate
from src.training.checkpoint import save_checkpoint
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write_png(path: Path, value: int) -> None:
    """Tiny 8x8 RGB PNG with a constant channel value (for label = 0/1)."""
    img = Image.new("RGB", (8, 8), color=(value * 255, value * 255, value * 255))
    img.save(path)


def _make_preprocessed_split(split_dir: Path, n_pos: int = 4, n_neg: int = 4) -> None:
    """Materialize a synthetic ``<split>/{images,label.csv}`` pair."""
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_pos):
        name = f"pos_{i:03d}.png"
        _write_png(split_dir / "images" / name, 1)
        rows.append({"imagePath": name, "label": "visible"})
    for i in range(n_neg):
        name = f"neg_{i:03d}.png"
        _write_png(split_dir / "images" / name, 0)
        rows.append({"imagePath": name, "label": "not-visible"})
    import csv
    with open(split_dir / "label.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["imagePath", "label"])
        writer.writeheader()
        writer.writerows(rows)


class _TinyConv(nn.Module):
    """Tiny model so we can save a real checkpoint without timm download."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, 2)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = h.flatten(1)
        return self.fc(h)


def _make_timm_ckpt(ckpt_path: Path, timm_name: str) -> None:
    """Save a checkpoint whose model matches the given timm config.

    Uses pretrained=False so the test is fully offline. The state dict
    shape is whatever timm gives us — ``build_model`` + ``load_checkpoint``
    must round-trip them without complaint.
    """
    import timm
    from src.model.builder import build_model
    from src.model.head import ClassificationHead

    backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
    head = ClassificationHead(backbone.num_features, {"hidden_dim": 4, "dropout": 0.0},
                                num_classes=2)
    model = backbone
    # Reuse _ModelWithHead for head attachment so the state-dict layout
    # matches what build_model produces at load time.
    from src.model.builder import _ModelWithHead
    model = _ModelWithHead(backbone, head)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        model, opt, epoch=1, val_auc=0.5, path=str(ckpt_path),
        arch_name=timm_name, class_names=["a", "b"], save_rng=False,
    )


# ---------------------------------------------------------------------------
# build_test_loader
# ---------------------------------------------------------------------------

def test_build_test_loader_accepts_dataset_root(tmp_dir):
    root = Path(tmp_dir) / "preproc"
    _make_preprocessed_split(root / "test", n_pos=3, n_neg=3)
    preproc = load_config("configs/preprocessing.yaml")
    loader = build_test_loader(str(root), preproc, batch_size=2, num_workers=0)
    assert len(loader.dataset) == 6
    x, y = next(iter(loader))
    # PCOSDataset does not resize — the upstream preprocessing pass
    # already materializes 224x224 PNGs. For this test the source PNGs
    # are 8x8, so we just verify the dataset emits the right number of
    # channels and preserves spatial shape.
    assert x.shape[0] == 2
    assert x.shape[1] == 3
    assert y.shape == (2,)


def test_build_test_loader_accepts_split_dir(tmp_dir):
    root = Path(tmp_dir) / "preproc"
    test_split = root / "test"
    _make_preprocessed_split(test_split, n_pos=2, n_neg=2)
    preproc = load_config("configs/preprocessing.yaml")
    # Pointing directly at the test/ split dir should resolve up
    # one level and find the same samples.
    loader = build_test_loader(str(test_split), preproc, batch_size=2, num_workers=0)
    assert len(loader.dataset) == 4


# ---------------------------------------------------------------------------
# run_evaluate multi-model discovery
# ---------------------------------------------------------------------------

def test_run_evaluate_multi_model_discovers_archs(tmp_dir, capsys, monkeypatch):
    # Two timm models that are small enough to instantiate from scratch
    # in CI. ``pretrained: false`` keeps the test offline.
    archs = [("alpha", "resnet18"), ("beta", "efficientnet_b0")]

    model_dir = Path(tmp_dir) / "results"
    cfg_dir = Path(tmp_dir) / "cfgs"
    cfg_dir.mkdir()
    for arch_name, timm_name in archs:
        (model_dir / arch_name).mkdir(parents=True)
        _make_timm_ckpt(model_dir / arch_name / "best.pt", timm_name)
        (cfg_dir / f"{arch_name}.yaml").write_text(
            f"timm_name: {timm_name}\npretrained: false\nnum_classes: 2\n"
            f"freeze_fraction: 0.0\ninput_size: 8\n"
            f"head: {{hidden_dim: 4, dropout: 0.0}}\n"
        )

    test_root = Path(tmp_dir) / "preproc"
    _make_preprocessed_split(test_root / "test", n_pos=3, n_neg=3)
    output_dir = Path(tmp_dir) / "out"

    rc = run_evaluate.main([
        "--model-dir", str(model_dir),
        "--test-dataset-dir", str(test_root),
        "--model-configs-dir", str(cfg_dir),
        "--preprocessing-config", "configs/preprocessing.yaml",
        "--output-dir", str(output_dir),
        "--batch-size", "2",
        "--num-workers", "0",
        "--seed", "0",
        "--device", "cpu",
    ])
    assert rc == 0

    summary = json.loads((output_dir / "all_metrics.json").read_text())
    assert set(summary.keys()) == {"alpha", "beta"}, summary.keys()
    for arch in ("alpha", "beta"):
        m = summary[arch]
        assert "test_auc_roc" in m, m
        assert "test_accuracy" in m
        assert m["checkpoint"].endswith(f"{arch}/best.pt")
        assert (output_dir / arch / "metrics.json").is_file()


def test_run_evaluate_requires_exactly_one_mode():
    """Passing neither --checkpoint-dir nor --model-dir should fail."""
    with pytest.raises(SystemExit):
        run_evaluate.main([
            "--test-dataset-dir", "/tmp/nope",
        ])
