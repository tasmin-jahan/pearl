#!/usr/bin/env python3
"""
CLI entrypoint for preprocessing.

Reads a preprocessing config, applies the pipeline to all images in the
raw dataset, and saves preprocessed train/val/test splits to disk.

Usage:
    python scripts/preprocess.py \\
        --config configs/preprocessing/srad.yaml \\
        --data_dir /path/to/figshare_raw \\
        --split_seed 42
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.splitter import (
    get_image_paths_and_labels, stratified_split, _infer_patient_id,
)
from src.preprocessing.preprocess import Preprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess PCOS dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to preprocessing config YAML",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to raw dataset directory (with infected/ and notinfected/ subdirs)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for the stratified split (default: 42)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Resize target size (default: 224)",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Skip clearing the output_dir before writing (preserve old runs).",
    )
    args = parser.parse_args()

    set_seed(args.split_seed)

    # Load preprocessing config
    config = load_config(args.config)
    output_dir = config.get("output_dir", "results/preprocessed/default")
    print(f"[Preprocess] Config: {config['name']}")
    print(f"[Preprocess] Output: {output_dir}")

    # ---- Clean output_dir to avoid stale files from a previous split_seed ----
    if os.path.isdir(output_dir):
        if args.no_clean:
            print(f"[Preprocess] --no_clean set: keeping existing {output_dir}")
        else:
            import shutil
            shutil.rmtree(output_dir)
            print(f"[Preprocess] Cleared previous {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Get image paths, labels, and patient_ids (group keys)
    image_paths, labels, patient_ids = get_image_paths_and_labels(args.data_dir)

    # Stratified split (group-aware if patient_ids have repeats)
    (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
    ) = stratified_split(
        image_paths, labels,
        patient_ids=patient_ids, seed=args.split_seed,
    )

    # Initialize preprocessor (NO augmentation — that happens at train time)
    preprocessor = Preprocessor(config, input_size=args.input_size)

    # Process and save each split
    class_names = {0: "noninfected", 1: "infected"}
    for split_name, paths, split_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        print(f"\n[Preprocess] Processing {split_name} split ({len(paths)} images)...")
        for img_path, label in zip(paths, split_labels):
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"  WARNING: Could not read {img_path}, skipping.")
                continue

            # Apply preprocessing (no augmentation at save time)
            processed = preprocessor.apply(image, augment=False)

            # Save to output directory preserving class structure
            class_name = class_names[label]
            save_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(save_dir, exist_ok=True)

            fname = os.path.basename(img_path)
            # Save as numpy array (.npy) to preserve float values
            save_path = os.path.join(save_dir, fname.rsplit(".", 1)[0] + ".npy")
            np.save(save_path, processed)

        # Log class counts
        unique, counts = np.unique(split_labels, return_counts=True)
        count_str = ", ".join(
            f"{class_names[u]}: {c}" for u, c in zip(unique, counts)
        )
        print(f"  {split_name} class counts: {count_str}")

    # ---- Post-run verification (Bug 3 fix) ----
    # Confirms that the output directory contains exactly the expected
    # number of .npy files (one per input image, across all three splits)
    # and that class ratios roughly mirror the input. If split_seed was
    # changed between runs, this catches leftover stale files in adjacent
    # split folders.
    print(f"\n[Preprocess] Verifying output...")
    expected_total = len(image_paths)
    actual_total = 0
    class_counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}
    for split_name in ("train", "val", "test"):
        for cls_name, cls_label in [("infected", 1), ("noninfected", 0)]:
            cls_dir = os.path.join(output_dir, split_name, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            n = sum(1 for f in os.listdir(cls_dir) if f.endswith(".npy"))
            actual_total += n
            class_counts[split_name][cls_label] = n
    if actual_total != expected_total:
        print(
            f"  WARNING: expected {expected_total} files, found {actual_total}."
        )
        print(
            f"  This usually means --no_clean was set or the output_dir had"
        )
        print(
            f"  leftover files from a previous run with a different --split_seed."
        )
    else:
        print(f"  OK: {actual_total} .npy files match expected (one per input image).")
    print(f"\n  Final split table:")
    print(f"  {'split':<8} {'noninfected':>14} {'infected':>12} {'ratio':>10}")
    for split_name in ("train", "val", "test"):
        n_neg = class_counts[split_name][0]
        n_pos = class_counts[split_name][1]
        ratio = f"{n_pos / max(n_neg, 1):.2f}:1" if n_neg > 0 else "—"
        print(f"  {split_name:<8} {n_neg:>14} {n_pos:>12} {ratio:>10}")
    print(
        f"\n[Preprocess] Done! Preprocessed data saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
