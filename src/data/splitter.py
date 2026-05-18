"""
Stratified data splitter for PCOS dataset.

Implements a 70/15/15 train/val/test split using StratifiedShuffleSplit
to preserve class balance across all splits.
"""

import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def get_image_paths_and_labels(data_dir: str) -> Tuple[List[str], List[int]]:
    """Scan dataset directory and return image paths with labels.

    Expected directory structure:
        data_dir/
            infected/      ← PCOS (label=1)
            notinfected/   ← non-PCOS (label=0)

    Args:
        data_dir: Root directory of the raw dataset.

    Returns:
        Tuple of (image_paths, labels) where labels are 0 or 1.
    """
    image_paths = []
    labels = []

    class_map = {"notinfected": 0, "infected": 1}

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class directory not found: {class_dir}"
            )
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    print(f"[Splitter] Found {len(image_paths)} images total")
    print(f"  Class 0 (non-PCOS): {labels.count(0)}")
    print(f"  Class 1 (PCOS):     {labels.count(1)}")

    return image_paths, labels


def stratified_split(
    image_paths: List[str],
    labels: List[int],
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Perform stratified 70/15/15 train/val/test split.

    Strategy:
      1. First split: 85% train+val / 15% test
      2. Second split: from train+val → 82.35% train / 17.65% val
         (which gives 70% / 15% of total)

    Args:
        image_paths: List of image file paths.
        labels: List of integer labels (0 or 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels,
                  test_paths, test_labels).
    """
    paths = np.array(image_paths)
    labs = np.array(labels)

    # Step 1: Split off test set (15% of total)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(paths, labs))

    trainval_paths = paths[trainval_idx]
    trainval_labels = labs[trainval_idx]
    test_paths = paths[test_idx]
    test_labels = labs[test_idx]

    # Step 2: Split train+val into train (70% of total) and val (15% of total)
    # 15% of total / 85% of total ≈ 17.65% of trainval
    val_fraction = 0.15 / 0.85  # ≈ 0.17647
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(sss_val.split(trainval_paths, trainval_labels))

    train_paths = trainval_paths[train_idx]
    train_labels = trainval_labels[train_idx]
    val_paths = trainval_paths[val_idx]
    val_labels = trainval_labels[val_idx]

    print(f"[Splitter] Split sizes:")
    print(f"  Train: {len(train_paths)} (class 0: {(train_labels==0).sum()}, class 1: {(train_labels==1).sum()})")
    print(f"  Val:   {len(val_paths)} (class 0: {(val_labels==0).sum()}, class 1: {(val_labels==1).sum()})")
    print(f"  Test:  {len(test_paths)} (class 0: {(test_labels==0).sum()}, class 1: {(test_labels==1).sum()})")

    return (
        train_paths.tolist(),
        train_labels.tolist(),
        val_paths.tolist(),
        val_labels.tolist(),
        test_paths.tolist(),
        test_labels.tolist(),
    )
