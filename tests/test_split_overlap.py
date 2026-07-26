"""
Pytest suite — split overlap (image-level + group-level).
"""

import os
from urllib.parse import quote

import numpy as np
import pytest

from src.data.splitter import (
    _infer_patient_id,
    _split_group_level,
    _split_image_level,
)


def _make_image_level(n_per_class: int = 1000):
    paths, labels, groups = [], [], []
    for cls in (0, 1):
        for i in range(n_per_class):
            paths.append(f"/fake/{'noninfected' if cls == 0 else 'infected'}/img_unique{i}_{cls}.png")
            labels.append(cls)
            groups.append(f"img_unique{i}")
    return paths, labels, groups


def _make_group_level(n_groups: int = 500):
    """Each group has exactly 2 images (one per class). Total: n_groups*2."""
    paths, labels, groups = [], [], []
    for i in range(n_groups):
        for cls, sub in [(0, "n"), (1, "p")]:
            paths.append(f"/fake/{'noninfected' if cls == 0 else 'infected'}/img_pat{i}_{sub}.png")
            labels.append(cls)
            groups.append(f"pat{i}")
    return paths, labels, groups


def test_image_level_split_80_10_10():
    paths, labels, _ = _make_image_level(1000)
    tp, tl, vp, vl, xp, xl = _split_image_level(paths, labels, seed=42)
    n = len(paths)
    assert abs(len(tp) / n - 0.80) < 0.02, f"train ratio {len(tp)/n:.3f}"
    assert abs(len(vp) / n - 0.10) < 0.02, f"val ratio {len(vp)/n:.3f}"
    assert abs(len(xp) / n - 0.10) < 0.02, f"test ratio {len(xp)/n:.3f}"
    assert abs(np.mean(tl) - np.mean(labels)) < 0.02


def test_group_level_no_leakage():
    paths2, labels2, groups2 = _make_group_level(n_groups=500)
    tp, tl, vp, vl, xp, xl = _split_group_level(paths2, labels2, groups2, seed=42)

    def _group(p):
        # Extract group id 'patNNN' from filename
        base = os.path.basename(p)
        # "img_patNNN_x.png"
        parts = base.split("_")
        return parts[1]  # 'patNNN'

    train_groups = {_group(p) for p in tp}
    val_groups = {_group(p) for p in vp}
    test_groups = {_group(p) for p in xp}
    assert train_groups & val_groups == set(), "group leakage train↔val"
    assert train_groups & test_groups == set(), "group leakage train↔test"
    assert val_groups & test_groups == set(), "group leakage val↔test"


def test_infer_patient_id_stability():
    a = _infer_patient_id("/foo/infected/img_001_PCOS.png")
    b = _infer_patient_id("/bar/noninfected/img_001_PCOS.png")
    assert a == b == "img_001_PCOS"
