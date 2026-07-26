"""
Stratified data splitter for PCOS dataset.

Implements an 80/10/10 train/val/test split using StratifiedShuffleSplit
to preserve class balance across all splits.

If patient IDs can be inferred from filenames / metadata, the splitter
falls back to a group-aware split (StratifiedGroupKFold-style logic)
to prevent patient-level leakage across train/val/test.
"""

import os
import re
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def _infer_patient_id(path: str) -> str:
    """Best-effort extraction of a patient identifier from a file path.

    For the Figshare PCOS dataset we don't have an explicit patient_id
    field in filenames. We use the filename stem as a stable per-image
    group proxy, which is the safest default: identical filenames land
    in the same split. If filename patterns change upstream (e.g. a
    metadata CSV with explicit patient IDs), replace this function with
    a metadata lookup.

    Args:
        path: An image file path like .../infected/img001.png

    Returns:
        Patient identifier string.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    # Figshare filenames are like "img_001_...", "image_12_PCOS", etc.
    # Strip common suffixes; keep the basename as the group key.
    return stem


def get_image_paths_and_labels(
    data_dir: str, patient_id_fn=None,
) -> Tuple[List[str], List[int], List[str]]:
    """Scan dataset directory and return image paths with labels.

    Expected directory structure:
        data_dir/
            infected/      ← PCOS (label=1)
            notinfected/   ← non-PCOS (label=0)

    Args:
        data_dir: Root directory of the raw dataset.
        patient_id_fn: Optional callable (path → patient_id_str). If None,
            uses :func:`_infer_patient_id`.

    Returns:
        Tuple of (image_paths, labels, patient_ids) where labels are 0/1
        and patient_ids is a parallel list of group keys.
    """
    image_paths = []
    labels = []
    patient_ids = []

    class_map = {"noninfected": 0, "infected": 1}
    pid_fn = patient_id_fn or _infer_patient_id

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class directory not found: {class_dir}"
            )
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                full = os.path.join(class_dir, fname)
                image_paths.append(full)
                labels.append(label)
                patient_ids.append(pid_fn(full))

    print(f"[Splitter] Found {len(image_paths)} images total")
    print(f"  Class 0 (non-PCOS): {labels.count(0)}")
    print(f"  Class 1 (PCOS):     {labels.count(1)}")
    print(f"  Unique patient groups: {len(set(patient_ids))}")

    return image_paths, labels, patient_ids


def _split_image_level(paths, labels, seed):
    """Image-level stratified 80/10/10 split (legacy behavior)."""
    paths = np.array(paths)
    labs = np.array(labels)

    # Step 1: 80% train+val+test reserved → 10% test
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=seed
    )
    trainval_idx, test_idx = next(sss_test.split(paths, labs))

    trainval_paths = paths[trainval_idx]
    trainval_labels = labs[trainval_idx]
    test_paths = paths[test_idx]
    test_labels = labs[test_idx]

    # Step 2: From train+val (90% remaining), val is 10/90 = 11.11%
    val_fraction = 0.10 / 0.90
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(sss_val.split(trainval_paths, trainval_labels))

    train_paths = trainval_paths[train_idx]
    train_labels = trainval_labels[train_idx]
    val_paths = trainval_paths[val_idx]
    val_labels = trainval_labels[val_idx]

    return (
        train_paths.tolist(), train_labels.tolist(),
        val_paths.tolist(), val_labels.tolist(),
        test_paths.tolist(), test_labels.tolist(),
    )


def _split_group_level(paths, labels, groups, seed):
    """Group-aware stratified 80/10/10 split.

    Stratification is done at the group level: each group is assigned a
    single label (majority/only), then we choose 10% of groups for test,
    10% of remaining groups for val, etc.
    """
    paths = np.array(paths)
    labs = np.array(labels)
    groups = np.array(groups)

    unique_groups = np.array(sorted(set(groups.tolist())))
    # Assign a per-group label: majority label within the group, falling back
    # to the first observed label if no majority.
    group_labels = []
    for g in unique_groups:
        mask = groups == g
        l_in_g = labs[mask]
        # If a group is single-class, that's unambiguous.
        uniq, counts = np.unique(l_in_g, return_counts=True)
        idx = counts.argmax()
        group_labels.append(int(uniq[idx]))
    group_labels = np.array(group_labels)

    # 80/10/10 split on groups: split off test first, then val from train+val
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=0.10, random_state=seed
    )
    trainval_g_idx, test_g_idx = next(
        sss_test.split(unique_groups, group_labels)
    )

    trainval_groups = unique_groups[trainval_g_idx]
    trainval_group_labels = group_labels[trainval_g_idx]

    val_fraction = 0.10 / 0.90
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_g_idx, val_g_idx = next(
        sss_val.split(trainval_groups, trainval_group_labels)
    )
    train_groups = set(trainval_groups[train_g_idx].tolist())
    val_groups = set(trainval_groups[val_g_idx].tolist())
    test_groups = set(unique_groups[test_g_idx].tolist())

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    for p, l, g in zip(paths.tolist(), labs.tolist(), groups.tolist()):
        if g in train_groups:
            train_paths.append(p); train_labels.append(l)
        elif g in val_groups:
            val_paths.append(p); val_labels.append(l)
        elif g in test_groups:
            test_paths.append(p); test_labels.append(l)

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
    )


def stratified_split(
    image_paths: List[str],
    labels: List[int],
    patient_ids: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Perform stratified 80/10/10 train/val/test split.

    If ``patient_ids`` is provided and distinct from per-image keys, a
    group-aware split is used to prevent patient-level leakage between
    train/val/test. Otherwise the plain image-level stratified split is
    used.

    Args:
        image_paths: List of image file paths.
        labels: List of integer labels (0 or 1).
        patient_ids: Optional parallel list of group identifiers. If the
            number of unique ids equals len(paths), the image-level split
            is used. If fewer unique ids, group-level split is used.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels,
                  test_paths, test_labels).
    """
    use_groups = False
    if patient_ids is not None and len(patient_ids) == len(image_paths):
        n_unique = len(set(patient_ids))
        if n_unique < len(image_paths):
            use_groups = True
            print(f"[Splitter] Using GROUP-AWARE split "
                  f"({n_unique} groups vs {len(image_paths)} images) to prevent "
                  f"patient-level leakage.")
        else:
            print(f"[Splitter] Each image has a unique patient_id; using image-level split.")

    if use_groups:
        result = _split_group_level(image_paths, labels, patient_ids, seed)
    else:
        result = _split_image_level(image_paths, labels, seed)

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = result

    print(f"[Splitter] Split sizes:")
    print(f"  Train: {len(train_paths)} (class 0: {(np.array(train_labels)==0).sum()}, class 1: {(np.array(train_labels)==1).sum()})")
    print(f"  Val:   {len(val_paths)} (class 0: {(np.array(val_labels)==0).sum()}, class 1: {(np.array(val_labels)==1).sum()})")
    print(f"  Test:  {len(test_paths)} (class 0: {(np.array(test_labels)==0).sum()}, class 1: {(np.array(test_labels)==1).sum()})")

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
    )
