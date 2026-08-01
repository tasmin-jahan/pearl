"""
Tests for dataset-wide PNG materialization.

The materializer must preserve the dataset's train/val/test boundaries,
copy labels into canonical CSV form, write PNGs only, and reject missing
pipeline stages.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.preprocessing.run_preprocessing import run_preprocessing


def _write_raw_dataset(root: Path) -> dict:
    expected = {}
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        split_dir = root / split
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True)
        rows = []
        for i in range(count):
            label = "Visible" if i % 2 else "Not-visible"
            name = f"raw_{i:04d}.jpg"
            image = np.full((80, 120, 3), 40 + i * 20, dtype=np.uint8)
            cv2.imwrite(str(img_dir / name), image)
            rows.append({"imagePath": name, "PCOS-visible": label})
        with open(split_dir / "label.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["imagePath", "PCOS-visible"]
            )
            writer.writeheader()
            writer.writerows(rows)
        expected[split] = count
    return expected


def _write_config(path: Path) -> None:
    config = {
        "name": "test",
        "steps": {
            "resize": True,
            "padding": "none",
            "clahe": {"enabled": False},
            "gaussian": {"enabled": False},
            "srad": {"enabled": False},
            "to_grayscale": {"enabled": False},
            # Keep z-score off for a deterministic uint8 round trip.
            "zscore_normalize": {"enabled": False},
        },
        "augmentation": {},
    }
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


def test_materializer_preserves_splits_and_writes_png(tmp_path):
    raw = tmp_path / "raw" / "figshare"
    out = tmp_path / "preprocessed" / "figshare"
    expected = _write_raw_dataset(raw)
    cfg = tmp_path / "preprocessing.yaml"
    _write_config(cfg)

    summary = run_preprocessing(
        dataset_dir=str(raw),
        output_dir=str(out),
        config_path=str(cfg),
        input_size=64,
    )

    for split, count in expected.items():
        image_files = sorted((out / split / "images").glob("*"))
        assert len(image_files) == count
        assert all(p.suffix.lower() == ".png" for p in image_files)
        assert not list((out / split / "images").glob("*.npy"))

        with open(out / split / "label.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == count
        assert {row["label"] for row in rows} == {"0", "1"}
        assert all(row["imagePath"].endswith(".png") for row in rows)
        assert summary[split]["infected"] == count // 2
        assert summary[split]["noninfected"] == count // 2

    assert (out / "preprocessing_metadata.json").is_file()


def test_materializer_never_modifies_raw_dataset(tmp_path):
    raw = tmp_path / "raw" / "pcosgen"
    out = tmp_path / "preprocessed" / "pcosgen"
    _write_raw_dataset(raw)
    cfg = tmp_path / "preprocessing.yaml"
    _write_config(cfg)

    original = {
        p.relative_to(raw): p.read_bytes()
        for p in raw.rglob("*")
        if p.is_file()
    }
    run_preprocessing(
        dataset_dir=str(raw),
        output_dir=str(out),
        config_path=str(cfg),
        input_size=64,
    )
    after = {
        p.relative_to(raw): p.read_bytes()
        for p in raw.rglob("*")
        if p.is_file()
    }
    assert after == original


def test_missing_val_split_fails_clear(tmp_path):
    raw = tmp_path / "raw" / "figshare"
    _write_raw_dataset(raw)
    # Simulate a skipped split stage.
    import shutil
    shutil.rmtree(raw / "val")
    out = tmp_path / "preprocessed" / "figshare"
    cfg = tmp_path / "preprocessing.yaml"
    _write_config(cfg)

    with pytest.raises(FileNotFoundError, match="src.preprocessing.split"):
        run_preprocessing(
            dataset_dir=str(raw),
            output_dir=str(out),
            config_path=str(cfg),
            input_size=64,
        )


def test_preprocessed_png_has_expected_shape(tmp_path):
    raw = tmp_path / "raw" / "figshare"
    out = tmp_path / "preprocessed" / "figshare"
    _write_raw_dataset(raw)
    cfg = tmp_path / "preprocessing.yaml"
    _write_config(cfg)

    run_preprocessing(
        dataset_dir=str(raw),
        output_dir=str(out),
        config_path=str(cfg),
        input_size=96,
    )
    png = next((out / "train" / "images").glob("*.png"))
    image = cv2.imread(str(png))
    assert image is not None
    assert image.shape == (96, 96, 3)
    assert image.dtype == np.uint8
