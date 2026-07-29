"""
Zenodo PCOSgen dataset adapters.

Source layout (Zenodo PCOSgen):
    data_external/train/
        images/<file>.jpg
        class_label.xlsx          # columns: imagePath, ..., 'Class label (whether polycsytic ovary is visible or not visible)'
    data_external/test/
        images/<file>.jpg
        class label.csv           # same column structure

This module provides:
  - `load_zenodo_labels(root_dir)` -> {filename: 0|1}
  - `ZenodoDataset(paths, labels, preprocessor, augment)` -> torch Dataset
  - `build_zenodo_loader(...)` -> (train_loader, val_loader)
  - `discover_zenodo_pairs(root_dir, split)` -> [(path, label), ...]  (used by
    evaluate_external.py's new `zenodo_labeled` layout)

The Zenodo label is read from the LAST column of the spreadsheet / CSV,
which is the "polycystic ovary visible" column. "Visible" -> 1, anything
else -> 0 (matching the convention used in `prepare_pcosgen_layout.py`).

The dataset class follows the same pattern as `_InMemoryDataset` in
`src/data/dataloader.py`: load raw jpg via cv2, apply preprocessing on-the-fly,
return (tensor, label). Augmentation is enabled only for the training split.
"""

import csv
import os
import random
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_xlsx(path: str) -> List[List[str]]:
    """Minimal xlsx reader — first sheet, rows as lists of strings.

    Sufficient for the small Zenodo label files (no formulas, no styles).
    """
    with zipfile.ZipFile(path) as z:
        try:
            shared = [
                (t.text or "")
                for t in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{NS}t")
            ]
        except KeyError:
            shared = []
        sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = []
    for row in sh.iter(f"{NS}row"):
        cells = []
        for c in row:
            t = c.attrib.get("t", "n")
            v_node = c.find(f"{NS}v")
            v = v_node.text if v_node is not None else ""
            if t == "s":
                v = shared[int(v)]
            elif t == "inlineStr":
                tnode = c.find(f"{NS}is/{NS}t")
                v = tnode.text if tnode.text is not None else ""
            cells.append(v)
        out.append(cells)
    return out


def _read_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _label_from_value(value: str) -> int:
    """Per the PCOSgen convention, "Visible" -> 1, anything else -> 0."""
    return 1 if (value or "").strip().lower().startswith("visible") else 0


def load_zenodo_labels(root_dir: str) -> Dict[str, int]:
    """Return {filename: 0|1} for the Zenodo PCOSgen split at `root_dir`.

    Reads `class_label.xlsx` for the train split and `class label.csv`
    for the test split. The imagePATH column is the file name only
    (no path); the actual image is at `root_dir/images/`.
    """
    xlsx_path = os.path.join(root_dir, "class_label.xlsx")
    csv_path = os.path.join(root_dir, "class label.csv")

    if os.path.isfile(xlsx_path):
        rows = _read_xlsx(xlsx_path)
        if not rows:
            return {}
        header = rows[0]
        last_col = len(header) - 1
        out = {}
        for r in rows[1:]:
            if len(r) < 2:
                continue
            img = r[0]
            lbl = _label_from_value(r[last_col])
            if img and str(img).lower() != "nan":
                out[img] = lbl
        return out

    if os.path.isfile(csv_path):
        rows = _read_csv(csv_path)
        if not rows:
            return {}
        header = list(rows[0].keys())
        last_col = header[-1]
        first_col = header[0]
        out = {}
        for row in rows:
            img = row[first_col]
            lbl = _label_from_value(row[last_col])
            if img and str(img).lower() != "nan":
                out[img] = lbl
        return out

    raise FileNotFoundError(
        f"No label file found at {root_dir}: expected class_label.xlsx or 'class label.csv'"
    )


# ---------------------------------------------------------------------
# Paths & split discovery
# ---------------------------------------------------------------------


def discover_zenodo_pairs(root_dir: str, split: str = "train") -> List[Tuple[str, int]]:
    """Discover (path, label) pairs from the Zenodo source layout.

    Args:
        root_dir: Root containing `images/` and a label file.
        split: Unused today (kept for symmetry with `discover_simple`).

    Returns:
        Sorted list of (path, label) pairs.
    """
    labels = load_zenodo_labels(root_dir)
    img_dir = os.path.join(root_dir, "images")
    out = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if fname not in labels:
            # Skip files without a label row (e.g. NaN rows in the xlsx).
            continue
        out.append((os.path.join(img_dir, fname), labels[fname]))
    return out


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


class ZenodoDataset(Dataset):
    """Reads raw Zenodo jpg images and applies the preprocessing pipeline
    on-the-fly. Mirrors the pattern of `_InMemoryDataset` in
    `src/data/dataloader.py`.
    """

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        preprocessor,
        augment: bool = False,
    ):
        self.paths = list(paths)
        self.labels = list(labels)
        self.preprocessor = preprocessor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.paths[idx])
        if img is None:
            img = np.zeros(
                (self.preprocessor.input_size, self.preprocessor.input_size, 3),
                dtype=np.uint8,
            )
        x = self.preprocessor.apply(img, augment=self.augment)
        if x.ndim == 2:
            x = x[None, :, :]
        else:
            x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x.copy()).float(), self.labels[idx]

    def get_class_counts(self) -> Dict[int, int]:
        return dict(Counter(self.labels))

    def get_class_weights(self) -> torch.Tensor:
        counts = self.get_class_counts()
        n = len(self.paths)
        w0 = n / (2.0 * max(counts.get(0, 1), 1))
        w1 = n / (2.0 * max(counts.get(1, 1), 1))
        return torch.tensor([w0, w1], dtype=torch.float32)


# ---------------------------------------------------------------------
# Loader builder
# ---------------------------------------------------------------------


def build_zenodo_loader(
    train_pairs: Sequence[Tuple[str, int]],
    val_pairs: Sequence[Tuple[str, int]],
    preproc_config: dict,
    batch_size: int = 32,
    input_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler: str = "weighted",
):
    """Build (train_loader, val_loader) using the Zenodo source layout.

    The training loader uses the configured sampler (default `weighted` because
    PCOSgen is ~8× imbalanced). The val loader is deterministic.
    """
    from src.preprocessing.preprocess import Preprocessor

    train_pre = Preprocessor(preproc_config, input_size=input_size)
    val_pre = Preprocessor(preproc_config, input_size=input_size)

    train_paths, train_labels = zip(*train_pairs) if train_pairs else ([], [])
    val_paths, val_labels = zip(*val_pairs) if val_pairs else ([], [])

    train_ds = ZenodoDataset(train_paths, train_labels, train_pre, augment=True)
    val_ds = ZenodoDataset(val_paths, val_labels, val_pre, augment=False)

    if sampler == "weighted":
        from src.data.dataloader import make_weighted_sampler

        train_sampler = make_weighted_sampler(list(train_labels))
        shuffle_flag = False
    elif sampler == "none":
        train_sampler = None
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------
# Class-balanced split (used by build_zenodo_splits.py)
# ---------------------------------------------------------------------


def stratified_split(
    pairs: List[Tuple[str, int]],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Stratified train/val split preserving the class ratio.

    Returns (train_pairs, val_pairs). Random is seeded for reproducibility.
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[str, int]]] = {}
    for p, l in pairs:
        by_label.setdefault(l, []).append((p, l))
    train, val = [], []
    for lbl, items in by_label.items():
        rng.shuffle(items)
        cut = max(1, int(round(len(items) * val_frac)))
        val.extend(items[:cut])
        train.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


__all__ = [
    "discover_zenodo_pairs",
    "load_zenodo_labels",
    "ZenodoDataset",
    "build_zenodo_loader",
    "stratified_split",
]
