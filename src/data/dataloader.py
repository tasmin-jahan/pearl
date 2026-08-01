"""
Canonical PNG dataset and DataLoader construction for PEARL.

This module is intentionally read-only with respect to dataset structure.
It does not split, deduplicate, preprocess, or write data. It expects the
upstream pipeline to have completed successfully::

    data/preprocessed/<dataset>/
        train/{images/*.png,label.csv}
        val/{images/*.png,label.csv}
        test/{images/*.png,label.csv}

If any required split is missing, ``build_dataloaders`` raises a clear
``FileNotFoundError`` pointing at the upstream preprocessing command.

The separation of concerns is deliberate:

- ``src.preprocessing.dedup`` and ``src.preprocessing.split`` decide the
  data boundaries on the raw dataset.
- ``src.preprocessing.run_preprocessing`` materializes PNGs while
  preserving those boundaries.
- This module only reads the completed PNG dataset and builds PyTorch
  DataLoaders.
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


FIGSHARE_LABEL = "PCOS-visible"
POSITIVE_TOKENS = ("visible", "infected")
NEGATIVE_TOKENS = ("not-visible", "noninfected", "notinfected")
REQUIRED_SPLITS = ("train", "val", "test")


def _detect_label_column(header: Sequence[str]) -> str:
    """Pick the label column from a canonical split CSV header.

    Priority:
      1. Exact ``label`` (preprocessed canonical layout)
      2. Exact ``PCOS-visible`` (Figshare)
      3. Any header containing ``visible`` (PCOSGen's long header)
      4. Any header containing ``polycyst``
    """
    for col in header:
        if col.strip().lower() == "label":
            return col
    for col in header:
        if col.strip() == FIGSHARE_LABEL:
            return col
    for col in header:
        if "visible" in col.lower():
            return col
    for col in header:
        if "polycyst" in col.lower():
            return col
    raise ValueError(
        f"Could not detect label column in CSV header: {list(header)}. "
        "Expected 'label', 'PCOS-visible', or a column containing 'visible'."
    )


def _label_from_value(value: str) -> int:
    """Map a raw label cell to integer class 0 or 1."""
    if value is None:
        return 0
    v = str(value).strip().lower()
    if not v or v == "nan":
        return 0
    # Check the negative tokens first because "not-visible" contains
    # "visible" as a substring.
    if any(tok in v for tok in NEGATIVE_TOKENS):
        return 0
    if any(tok in v for tok in POSITIVE_TOKENS):
        return 1
    try:
        return 1 if int(float(v)) == 1 else 0
    except (ValueError, TypeError):
        return 0


def _read_split(dataset_dir: str, split: str) -> List[Tuple[str, int]]:
    """Read one preprocessed split and return ``[(png_path, label), ...]``.

    The CSV is the source of truth. Every nonblank CSV row must reference
    an existing PNG file; non-PNG references are rejected because the
    preprocessing stage must have completed before training.
    """
    split_dir = Path(dataset_dir) / split
    img_dir = split_dir / "images"
    csv_path = split_dir / "label.csv"

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing required split {split_dir}/. Run "
            "`python -m src.preprocessing.run_preprocessing "
            "--dataset-dir data/raw/<dataset> "
            "--output-dir data/preprocessed/<dataset>` first."
        )
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing required image directory {img_dir}/")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing required label CSV {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not header:
        raise RuntimeError(f"{csv_path} has no header")
    if not rows:
        raise RuntimeError(f"{csv_path} has no rows")

    label_col = _detect_label_column(header)
    image_col = "imagePath" if "imagePath" in header else header[0]
    samples: List[Tuple[str, int]] = []
    for row_number, row in enumerate(rows, start=2):
        image_name = (row.get(image_col) or "").strip()
        if not image_name or image_name.lower() == "nan":
            continue
        if Path(image_name).suffix.lower() != ".png":
            raise ValueError(
                f"{csv_path}:{row_number} references non-PNG image "
                f"{image_name!r}. The dataloader only accepts the output of "
                "src.preprocessing.run_preprocessing."
            )
        image_path = img_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{csv_path}:{row_number} references missing PNG {image_path}"
            )
        samples.append((str(image_path), _label_from_value(row.get(label_col))))

    if not samples:
        raise RuntimeError(f"{csv_path} contains no usable PNG rows")
    return samples


class PCOSDataset(Dataset):
    """Dataset for the canonical preprocessed PNG layout.

    Args:
        samples: ``[(png_path, label), ...]``.
        augment: Apply stochastic training augmentation in memory.
        augmentation_config: ``configs/preprocessing.yaml`` augmentation
            block. Only simple geometric augmentation is applied here;
            deterministic SRAD/CLAHE was already materialized to PNG.
        zscore: Apply per-channel z-score normalization after loading.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        *,
        augment: bool = False,
        augmentation_config: Optional[dict] = None,
        zscore: bool = True,
    ):
        self.samples = samples
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.zscore = zscore
        self.labels = [label for _, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode PNG: {path}")

        # cv2 reads BGR; convert once to RGB for ImageNet-pretrained models.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.augment:
            image = self._apply_augmentation(image)

        if self.zscore:
            image = self._zscore(image)
        else:
            image = image / 255.0

        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image.copy()).float()
        # Trainer expects `(image, label)`. Callers that need the
        # source path can read `dataset.samples[idx][0]` directly.
        return tensor, label

    @staticmethod
    def _zscore(image: np.ndarray) -> np.ndarray:
        """Per-channel z-score normalization."""
        out = image.astype(np.float32, copy=True)
        for channel in range(out.shape[2]):
            values = out[:, :, channel]
            out[:, :, channel] = (values - values.mean()) / (values.std() + 1e-8)
        return out

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply stochastic geometric augmentation to a loaded PNG."""
        cfg = self.augmentation_config
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        rotation = float(cfg.get("rotation", 0) or 0)
        if rotation > 0:
            angle = np.random.uniform(-rotation, rotation)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )

        if cfg.get("horizontal_flip", False) and np.random.random() > 0.5:
            image = cv2.flip(image, 1)

        scale_range = float(cfg.get("scale", 0) or 0)
        if scale_range > 0:
            scale = 1.0 + np.random.uniform(-scale_range, scale_range)
            matrix = cv2.getRotationMatrix2D(center, 0.0, scale)
            image = cv2.warpAffine(
                image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )

        return image

    def get_class_counts(self) -> Dict[int, int]:
        return dict(Counter(self.labels))

    def get_class_weights(self) -> torch.Tensor:
        counts = self.get_class_counts()
        n = len(self.labels)
        w0 = n / (2.0 * max(counts.get(0, 1), 1))
        w1 = n / (2.0 * max(counts.get(1, 1), 1))
        return torch.tensor([w0, w1], dtype=torch.float32)


def make_weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Build an inverse-frequency per-sample weighted sampler."""
    if not labels:
        raise ValueError("Cannot build a weighted sampler for an empty dataset")
    counts = Counter(int(label) for label in labels)
    weights = [1.0 / max(counts[int(label)], 1) for label in labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(
    dataset_dir: str,
    preproc_config: Optional[dict] = None,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler: str = "weighted",
    drop_last: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders from an already-preprocessed dataset.

    All three splits are required. This function never carves validation
    or otherwise mutates data. ``dataset_dir`` should be a path such as
    ``data/preprocessed/figshare``.
    """
    preproc_config = preproc_config or {}
    augmentation = preproc_config.get("augmentation", {}) or {}
    zscore_cfg = (preproc_config.get("steps", {}) or {}).get(
        "zscore_normalize", {}
    ) or {}
    zscore = bool(zscore_cfg.get("enabled", True))

    train_samples = _read_split(dataset_dir, "train")
    val_samples = _read_split(dataset_dir, "val")
    test_samples = _read_split(dataset_dir, "test")

    train_ds = PCOSDataset(
        train_samples,
        augment=True,
        augmentation_config=augmentation,
        zscore=zscore,
    )
    val_ds = PCOSDataset(val_samples, augment=False, zscore=zscore)
    test_ds = PCOSDataset(test_samples, augment=False, zscore=zscore)

    if sampler == "weighted":
        train_sampler = make_weighted_sampler(train_ds.labels)
        shuffle = False
    elif sampler == "none":
        train_sampler = None
        shuffle = False
    elif sampler == "shuffle":
        train_sampler = None
        shuffle = True
    else:
        raise ValueError(
            f"Unknown sampler {sampler!r}; expected 'weighted', 'shuffle', or 'none'"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"[Data] dataset={dataset_dir} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )
    return train_loader, val_loader, test_loader


# Backward-compatible alias for callers being migrated. The canonical
# function name is build_dataloaders.
build_loaders = build_dataloaders


__all__ = [
    "PCOSDataset",
    "build_dataloaders",
    "build_loaders",
    "make_weighted_sampler",
    "_read_split",
    "_detect_label_column",
    "_label_from_value",
]
