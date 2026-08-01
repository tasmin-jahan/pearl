"""
Smoke test for src.evaluation.run_evaluate_ensemble.

Verifies that the ensemble CLI:
  - Loads each model from its own checkpoint
  - Probability-averages across all members (no truncation)
  - Writes a metrics JSON keyed the same way as the single-model CLI
  - Refuses to run with fewer than two members
"""

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.evaluation import run_evaluate_ensemble
from src.training.checkpoint import save_checkpoint


def _write_png(path: Path, value: int) -> None:
    # Swin/ViT backbones enforce strict 224x224 in eval mode. Test PNGs
    # are tiny grayscale fills, so the on-disk cost is negligible (~5KB
    # at 224x224). This keeps the test fully realistic — same forward
    # path as production inference.
    img = Image.new("RGB", (224, 224), color=(value * 255, value * 255, value * 255))
    img.save(path)


def _make_split(split_dir: Path, n_pos: int = 4, n_neg: int = 4) -> None:
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
    with open(split_dir / "label.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["imagePath", "label"])
        writer.writeheader()
        writer.writerows(rows)


def _make_timm_ckpt(ckpt_path: Path, timm_name: str) -> None:
    """Build a real timm model + custom head and save a checkpoint."""
    import timm
    from src.model.builder import _ModelWithHead
    from src.model.head import ClassificationHead

    backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
    head = ClassificationHead(backbone.num_features,
                              {"hidden_dim": 4, "dropout": 0.0},
                              num_classes=2)
    model = _ModelWithHead(backbone, head)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    save_checkpoint(
        model, opt, epoch=1, val_auc=0.5, path=str(ckpt_path),
        arch_name=timm_name, class_names=["a", "b"], save_rng=False,
    )


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _build_ensemble_setup(tmp_dir: str, archs: list[tuple[str, str]]):
    """Create ckpt dir, configs dir, preprocessed test set, and return paths."""
    model_dir = Path(tmp_dir) / "results"
    cfg_dir = Path(tmp_dir) / "cfgs"
    cfg_dir.mkdir()
    test_root = Path(tmp_dir) / "preproc"
    _make_split(test_root / "test", n_pos=3, n_neg=3)
    for arch_name, timm_name in archs:
        (model_dir / arch_name).mkdir(parents=True)
        _make_timm_ckpt(model_dir / arch_name / "best.pt", timm_name)
        (cfg_dir / f"{arch_name}.yaml").write_text(
            f"timm_name: {timm_name}\npretrained: false\nnum_classes: 2\n"
            f"freeze_fraction: 0.0\ninput_size: 224\n"
            f"head: {{hidden_dim: 4, dropout: 0.0}}\n"
        )
    return model_dir, cfg_dir, test_root


def test_ensemble_eval_aggregates_all_members(tmp_dir):
    """All passed --model/--checkpoint pairs should appear in the output."""
    model_dir, cfg_dir, test_root = _build_ensemble_setup(
        tmp_dir,
        # Use the exact 5 architectures the production pipeline supports,
        # to verify the CLI's <arch>.yaml resolution works for the real
        # model lineup end-to-end.
        [
            ("swin_tiny",        "swin_tiny_patch4_window7_224"),
            ("vit_base",         "vit_base_patch16_224"),
            ("convnext_tiny",    "convnext_tiny.fb_in22k_ft_in1k"),
            ("densenet169",      "densenet169"),
            ("efficientnet_b0",  "efficientnet_b0"),
        ],
    )
    output = Path(tmp_dir) / "ensemble.json"
    archs = ["swin_tiny", "vit_base", "convnext_tiny", "densenet169", "efficientnet_b0"]
    rc = run_evaluate_ensemble.main([
        "--test-dataset-dir", str(test_root),
        "--model-configs-dir", str(cfg_dir),
        "--output", str(output),
        "--batch-size", "2",
        "--num-workers", "0",
        "--seed", "0",
        "--device", "cpu",
        *[v for arch in archs for v in ("--model", arch,
                                          "--checkpoint", str(model_dir / arch / "best.pt"))],
    ])
    assert rc == 0
    metrics = json.loads(output.read_text())
    assert metrics["aggregation"] == "probability_mean"
    assert len(metrics["ensemble_members"]) == 5
    assert {m["arch"] for m in metrics["ensemble_members"]} == {
        "swin_tiny", "vit_base", "convnext_tiny", "densenet169", "efficientnet_b0"
    }
    # Headline metric keys must be present (same names as single-model CLI)
    assert "test_auc_roc" in metrics
    assert "test_accuracy" in metrics


def test_ensemble_eval_refuses_single_member(tmp_dir):
    with pytest.raises(SystemExit):
        run_evaluate_ensemble.main([
            "--test-dataset-dir", "/tmp",
            "--output", "/tmp/x.json",
            "--model", "alpha",
            "--checkpoint", "/tmp/alpha.pt",
        ])


def test_ensemble_eval_model_dir_auto_discovers(tmp_dir):
    """--model-dir should populate --model/--checkpoint from disk."""
    model_dir, cfg_dir, test_root = _build_ensemble_setup(
        tmp_dir,
        [
            ("swin_tiny",        "swin_tiny_patch4_window7_224"),
            ("vit_base",         "vit_base_patch16_224"),
            ("convnext_tiny",    "convnext_tiny.fb_in22k_ft_in1k"),
            ("densenet169",      "densenet169"),
            ("efficientnet_b0",  "efficientnet_b0"),
        ],
    )
    output = Path(tmp_dir) / "ensemble.json"
    rc = run_evaluate_ensemble.main([
        "--test-dataset-dir", str(test_root),
        "--model-configs-dir", str(cfg_dir),
        "--model-dir", str(model_dir),
        "--output", str(output),
        "--batch-size", "2",
        "--num-workers", "0",
        "--seed", "0",
        "--device", "cpu",
    ])
    assert rc == 0
    metrics = json.loads(output.read_text())
    assert len(metrics["ensemble_members"]) == 5
    # Each member's checkpoint path should match the auto-discovered
    # <model_dir>/<arch>/best.pt layout.
    for m in metrics["ensemble_members"]:
        expected = str(model_dir / m["arch"] / "best.pt")
        assert m["checkpoint"] == expected, m


def test_ensemble_eval_requires_matched_pairs(tmp_dir):
    """Passing 2 models but 1 checkpoint must fail."""
    with pytest.raises(SystemExit):
        run_evaluate_ensemble.main([
            "--test-dataset-dir", "/tmp",
            "--output", "/tmp/x.json",
            "--model", "alpha",
            "--model", "beta",
            "--checkpoint", "/tmp/alpha.pt",
        ])