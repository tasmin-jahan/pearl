"""
Tests for the canonical PNG/CSV dataloader.

Covers:
  - PNG discovery + label parsing from canonical label.csv
  - Rejection of non-PNG references (forces pipeline compliance)
  - Missing/dataset dir raises FileNotFoundError pointing at run_preprocessing
  - Tensor shape and dtype
  - Weighted sampler balance
  - Z-score toggle
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.dataloader import (
    PCOSDataset,
    _detect_label_column,
    _label_from_value,
    build_dataloaders,
    make_weighted_sampler,
)


def _write_png(path: str, value: int = 50) -> None:
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    cv2.imwrite(path, img)


def _make_dataset(
    root: str,
    *,
    split_counts=("train", 4, "val", 2, "test", 2),
    label_col: str = "label",
) -> None:
    for split, n in zip(split_counts[0::2], split_counts[1::2]):
        split_dir = Path(root) / split
        (split_dir / "images").mkdir(parents=True)
        rows = []
        for i in range(n):
            class_label = i % 2
            name = f"img_{i:04d}.png"
            _write_png(str(split_dir / "images" / name), class_label * 100)
            rows.append({"imagePath": name, label_col: class_label})
        with open(split_dir / "label.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["imagePath", label_col])
            writer.writeheader()
            writer.writerows(rows)


def test_detect_label_column_priority():
    assert _detect_label_column(["label", "imagePath"]) == "label"
    assert _detect_label_column(["imagePath", "PCOS-visible"]) == "PCOS-visible"
    assert _detect_label_column(["imagePath", "Class label (visible or not)"]) == \
        "Class label (visible or not)"
    with pytest.raises(ValueError):
        _detect_label_column(["imagePath", "size", "color"])


def test_label_from_value_handles_datasets():
    assert _label_from_value("Visible") == 1
    assert _label_from_value("Not-visible") == 0
    assert _label_from_value("noninfected") == 0
    assert _label_from_value("infected") == 1
    assert _label_from_value("") == 0
    assert _label_from_value("nan") == 0
    assert _label_from_value(None) == 0
    assert _label_from_value("1") == 1
    assert _label_from_value("0") == 0


def test_build_dataloaders_returns_three_loaders():
    with tempfile.TemporaryDirectory() as root:
        _make_dataset(root)
        cfg = {
            "steps": {"zscore_normalize": {"enabled": True}},
            "augmentation": {"horizontal_flip": True},
        }
        tl, vl, xl = build_dataloaders(
            root, cfg, batch_size=2, num_workers=0, sampler="none",
        )
        x, y = next(iter(tl))
        assert x.shape[1:] == (3, 64, 64)
        assert x.dtype == torch.float32
        assert set(y.tolist()).issubset({0, 1})
        # All three splits are present and non-empty.
        assert len(tl.dataset) == 4
        assert len(vl.dataset) == 2
        assert len(xl.dataset) == 2


def test_rejects_non_png_references():
    with tempfile.TemporaryDirectory() as root:
        _make_dataset(root)
        # Replace one PNG with a .jpg reference in the CSV.
        split_dir = Path(root) / "train"
        with open(split_dir / "label.csv", "w") as f:
            f.write("imagePath,label\n")
            f.write("img_0000.png,0\n")
            f.write("img_0001.jpg,1\n")
        with pytest.raises(ValueError, match="non-PNG"):
            build_dataloaders(
                root,
                batch_size=2,
                num_workers=0,
                sampler="none",
            )


def test_missing_split_raises_with_clear_error():
    with tempfile.TemporaryDirectory() as root:
        # Only create train/
        split_dir = Path(root) / "train"
        (split_dir / "images").mkdir(parents=True)
        _write_png(str(split_dir / "images" / "a.png"))
        with open(split_dir / "label.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["imagePath", "label"])
            writer.writeheader()
            writer.writerow({"imagePath": "a.png", "label": 0})
        with pytest.raises(FileNotFoundError, match="run_preprocessing"):
            build_dataloaders(root, batch_size=2, num_workers=0, sampler="none")


def test_dataset_returns_tensor_and_label():
    samples = [("/tmp/never.png", 1)]
    ds = PCOSDataset(samples, zscore=False)
    # Manually inject a HxWxC uint8 png in /tmp for the test.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        _write_png(f.name, 200)
        samples = [(f.name, 1)]
        ds = PCOSDataset(samples, zscore=False)
        x, label = ds[0]
        assert x.shape == (3, 64, 64)
        assert x.dtype == torch.float32
        assert float(label) == 1.0
        # Without zscore, values should be in [0, 1].
        assert x.max() <= 1.0 + 1e-6
        assert x.min() >= 0.0


def test_weighted_sampler_with_imbalanced_labels():
    labels = [0] * 80 + [1] * 20
    sampler = make_weighted_sampler(labels)
    drawn = [labels[int(i)] for i in list(sampler)[:2000]]
    frac = sum(drawn) / len(drawn)
    assert 0.40 < frac < 0.60
