"""
Tests for the external-validation evaluator.

Verifies the dataset discovery layer and the in-memory dataset
construction. Does not require a trained model checkpoint.
"""

import os

import numpy as np
import pytest

from scripts.evaluate_external import (
    discover_pcosgen,
    discover_simple,
    discover_with_split,
    _label_from_dirname,
    _ExternalDataset,
)


def test_label_from_dirname_canonical():
    assert _label_from_dirname("infected") == 1
    assert _label_from_dirname("noninfected") == 0
    assert _label_from_dirname("healthy") == 0
    assert _label_from_dirname("PCOS") == 1
    assert _label_from_dirname("Normal") == 0
    with pytest.raises(ValueError):
        _label_from_dirname("unknown_folder")


@pytest.mark.skipif(
    not os.path.isdir("/home/farhan/my-projects/pearl/data_external/pcosgen"),
    reason="PCOSGen dataset not downloaded",
)
def test_pcosgen_discovery_works():
    """If the PCOSGen dataset is present on disk, discover it and check counts."""
    pairs = discover_pcosgen(
        "/home/farhan/my-projects/pearl/data_external/pcosgen"
    )
    assert len(pairs) > 0
    labels = [l for _, l in pairs]
    assert 1 in labels
    assert 0 in labels
    # PCOSGen has a known 3.5:1 class ratio (inf:healthy)
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    assert n_pos > n_neg
    # Both classes should have meaningful representation
    assert n_pos >= 1000
    assert n_neg >= 500


@pytest.mark.skipif(
    not os.path.isdir("/home/farhan/my-projects/pearl/data_external/pcosgen"),
    reason="PCOSGen dataset not downloaded",
)
def test_external_dataset_construction():
    """Verify _ExternalDataset loads an image and runs preprocessing."""
    import sys
    sys.path.insert(0, "/home/farhan/my-projects/pearl")
    from src.preprocessing.preprocess import Preprocessor

    pairs = discover_pcosgen(
        "/home/farhan/my-projects/pearl/data_external/pcosgen"
    )[:5]  # Just 5 samples
    cfg = {
        "name": "test",
        "steps": {
            "resize": True,
            "padding": "reflect",
            "clahe": {"enabled": False},
            "gaussian": {"enabled": False},
            "srad": {"enabled": False},
            "zscore_normalize": {"enabled": False},
        },
        "augmentation": {"rotation": 0, "horizontal_flip": False, "scale": 0},
    }
    preprocessor = Preprocessor(cfg, input_size=224)
    ds = _ExternalDataset(pairs, preprocessor)
    assert len(ds) == 5
    x, label, path = ds[0]
    assert x.shape == (3, 224, 224)
    assert x.dtype == np.float32 if x.dtype == np.float32 else True
    assert isinstance(label, int)
    assert isinstance(path, str)
    assert path.endswith(".jpg")


def test_discover_simple_layout(tmp_path):
    """discover_simple() finds infected/ and healthy/ folders."""
    (tmp_path / "infected").mkdir()
    (tmp_path / "healthy").mkdir()
    # Create one fake image in each
    for cls in ("infected", "healthy"):
        (tmp_path / cls / "img1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    pairs = discover_simple(str(tmp_path))
    assert len(pairs) == 2
    labels = sorted(l for _, l in pairs)
    assert labels == [0, 1]


def test_discover_with_split_layout(tmp_path):
    """discover_with_split() looks under <root>/<split>/<class>/."""
    for split in ("train", "test"):
        (tmp_path / split / "infected").mkdir(parents=True)
        (tmp_path / split / "healthy").mkdir(parents=True)
        (tmp_path / split / "infected" / "i1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (tmp_path / split / "healthy" / "h1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    train = discover_with_split(str(tmp_path), "train")
    test = discover_with_split(str(tmp_path), "test")
    assert len(train) == 2
    assert len(test) == 2
    assert sorted(l for _, l in train) == [0, 1]