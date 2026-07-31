#!/usr/bin/env python3
"""
CLI entrypoint for preprocessing.

Reads a preprocessing config (the unified configs/preprocessing.yaml) and
applies the pipeline to every image in a pre-split dataset directory
(see scripts/preprocessing/{dedup,split}.py). The data dir must already
contain the three splits::

    data_dir/
        train/{infected,notinfected}/*.jpg
        val/{infected,notinfected}/*.jpg
        test/{infected,notinfected}/*.jpg

Outputs: ``<output_dir>/{train,val,test}/{infected,noninfected}/*.npy``
(legacy "noninfected" folder name is preserved).

Usage:
    python scripts/preprocessing/preprocessing.py \\
        --config configs/preprocessing.yaml \\
        --data_dir data/figshare_5x \\
        --output_dir results/preprocessed/figshare_5x
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
from src.preprocessing.preprocess import Preprocessor


SPLITS = ("train", "val", "test")
CLASS_FOLDERS = {
    0: "notinfected",   # input dir name (kept for back-compat)
    1: "infected",
}
OUTPUT_CLASS_FOLDERS = {
    0: "noninfected",   # legacy output dir name (matches PCOSDataset)
    1: "infected",
}


def _find_class_dir(split_dir: str, label: int) -> str:
    """Locate the class folder under a split dir, accepting both
    ``infected`` and ``noninfected``/``notinfected`` naming."""
    candidates = ["infected", "notinfected", "noninfected"]
    for c in candidates:
        d = os.path.join(split_dir, c)
        if os.path.isdir(d):
            if c == "infected":
                return d
            # 'notinfected' and 'noninfected' both map to label 0
            if label == 0:
                return d
    return None


def _walk_split(split_dir: str):
    """Return [(path, label), ...] for a split dir."""
    items = []
    for label in (1, 0):
        cdir = _find_class_dir(split_dir, label)
        if cdir is None:
            continue
        for fname in sorted(os.listdir(cdir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                items.append((os.path.join(cdir, fname), label))
    return items


def main():
    parser = argparse.ArgumentParser(description="Preprocess PCOS dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to preprocessing config YAML (the unified configs/preprocessing.yaml).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to pre-split dataset directory containing train/, val/, test/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to write preprocessed .npy files (a per-config folder).",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Resize target size (default: 224).",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Skip clearing the output_dir before writing (preserve old runs).",
    )
    args = parser.parse_args()

    set_seed(42)

    config = load_config(args.config)
    print(f"[Preprocess] Config: {config.get('name', args.config)}")
    print(f"[Preprocess] Output: {args.output_dir}")

    if os.path.isdir(args.output_dir):
        if args.no_clean:
            print(f"[Preprocess] --no_clean set: keeping existing {args.output_dir}")
        else:
            import shutil
            shutil.rmtree(args.output_dir)
            print(f"[Preprocess] Cleared previous {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    preprocessor = Preprocessor(config, input_size=args.input_size)

    expected_total = 0
    actual_total = 0
    class_counts = {s: {0: 0, 1: 0} for s in SPLITS}

    for split_name in SPLITS:
        split_dir = os.path.join(args.data_dir, split_name)
        if not os.path.isdir(split_dir):
            print(f"[Preprocess] WARNING: missing {split_dir} — skipping")
            continue
        items = _walk_split(split_dir)
        print(f"\n[Preprocess] Processing {split_name} split ({len(items)} images)...")
        for img_path, label in items:
            image = cv2.imread(img_path)
            if image is None:
                print(f"  WARNING: Could not read {img_path}, skipping.")
                continue
            processed = preprocessor.apply(image, augment=False)
            out_dir = os.path.join(
                args.output_dir, split_name, OUTPUT_CLASS_FOLDERS[label],
            )
            os.makedirs(out_dir, exist_ok=True)
            fname = os.path.basename(img_path)
            save_path = os.path.join(out_dir, fname.rsplit(".", 1)[0] + ".npy")
            np.save(save_path, processed)
            class_counts[split_name][label] += 1
            actual_total += 1
        # Log class counts
        n0 = class_counts[split_name][0]
        n1 = class_counts[split_name][1]
        print(f"  {split_name} class counts: noninfected={n0}, infected={n1}")

    # Compute expected total under the assumption: every input image gets
    # exactly one .npy output. We use the discovered items list as
    # ground truth.
    expected_total = sum(
        len(_walk_split(os.path.join(args.data_dir, s))) for s in SPLITS
        if os.path.isdir(os.path.join(args.data_dir, s))
    )

    if actual_total != expected_total:
        print(
            f"\n[Preprocess] WARNING: expected {expected_total} files, "
            f"wrote {actual_total}."
        )
    else:
        print(f"\n[Preprocess] OK: {actual_total} .npy files written.")
    print(f"\n  Final split table:")
    print(f"  {'split':<8} {'noninfected':>14} {'infected':>12} {'ratio':>10}")
    for split_name in SPLITS:
        n_neg = class_counts[split_name][0]
        n_pos = class_counts[split_name][1]
        ratio = f"{n_pos / max(n_neg, 1):.2f}:1" if n_neg > 0 else "—"
        print(f"  {split_name:<8} {n_neg:>14} {n_pos:>12} {ratio:>10}")
    print(f"\n[Preprocess] Done! Preprocessed data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
