#!/usr/bin/env python3
"""
Carve a validation split out of an existing train/ folder.

Reads data/raw/<dataset>/train/{images,label.csv}, takes a stratified
fraction (default 10%) as validation, and writes the remaining images
back to train/ plus a new val/{images,label.csv}.

Both Figshare and PCOSGen use the same canonical layout, so this
script works on either dataset:

    data/raw/figshare/
        train/
            images/
            label.csv       <- input
        val/                <- output (10% of train)
            images/
            label.csv

    data/raw/pcosgen/
        train/
            images/
            label.csv       <- input
        val/                <- output
            images/
            label.csv

Naming rewrites: validation images are renumbered to start at 0 within
val/images/, and the surviving train images are renumbered to start at 0
within train/images/ (closing the gap left by the carved val set).
The master_label.csv at the dataset root is updated to reflect the new
train membership.

Usage:
    python scripts/preprocessing/split.py --dataset figshare
    python scripts/preprocessing/split.py --dataset pcosgen --val-frac 0.10
    python scripts/preprocessing/split.py --dataset figshare --val-frac 0.15 --seed 7
"""

import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


LABEL_COLUMN = "PCOS-visible"
NAME_PREFIX = "image"
NAME_EXT = ".jpg"

DEFAULT_VAL_FRAC = 0.10
DEFAULT_SEED = 42


def pad_width(n: int) -> int:
    return max(4, len(str(n)))


def stratified_split(items: list, val_frac: float, seed: int) -> tuple:
    """Stratified 2-way split preserving class proportions.

    items: list of (path, label). Returns (train, val).
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)
    train, val = [], []
    for label, group in by_label.items():
        group = sorted(group)
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_frac)))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["figshare", "pcosgen"],
                        help="Which dataset to split")
    parser.add_argument("--data-root", type=str, default="data/raw",
                        help="Root of raw dataset tree (default: data/raw)")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                        help=f"Fraction of train to carve as val "
                             f"(default {DEFAULT_VAL_FRAC})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default {DEFAULT_SEED})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite val/ if it already exists")
    args = parser.parse_args()

    data_root = Path(args.data_root) / args.dataset
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    train_csv = train_dir / "label.csv"
    train_img_dir = train_dir / "images"

    if not train_csv.is_file():
        sys.exit(f"ERROR: {train_csv} not found. Run dedup.py first.")
    if not train_img_dir.is_dir():
        sys.exit(f"ERROR: {train_img_dir} not found.")

    if val_dir.exists() and not args.force:
        sys.exit(f"ERROR: {val_dir} already exists. Pass --force to overwrite.")

    # Read train label.csv. Detect the label column: Figshare uses
    # "PCOS-visible"; PCOSGen uses "Class label (whether polycsytic ovary
    # is visible or not visible)". We pick the column whose values contain
    # "Visible" or "Not-visible" (or our short alias).
    train_items = []
    label_col = None
    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for cand in reader.fieldnames:
            if cand == LABEL_COLUMN or "PCOS-visible" in cand or "polycsytic" in cand.lower() or "visible" in cand.lower():
                label_col = cand
                break
        if label_col is None:
            sys.exit(f"ERROR: could not find label column in {train_csv}. "
                     f"Got columns: {reader.fieldnames}")
        for row in reader:
            # Skip blank rows (PCOSGen label.csv has stray label-only rows).
            img_name = (row.get("imagePath") or "").strip()
            if not img_name or img_name.lower() == "nan":
                continue
            img_path = train_img_dir / img_name
            if not img_path.is_file():
                sys.exit(f"ERROR: train references missing file {img_path}")
            train_items.append((img_path, row[label_col]))
    print(f"[Split] Reading label column: '{label_col}'")

    train, val = stratified_split(train_items, args.val_frac, args.seed)

    n = len(train_items)
    width = pad_width(n)
    print(f"[Split] Carving {args.val_frac:.0%} val from {args.dataset}/train "
          f"(seed={args.seed}):")
    print(f"  total train = {n}  ->  train = {len(train)}, val = {len(val)}")
    for split_name, split in [("train", train), ("val", val)]:
        counts = defaultdict(int)
        for _, lbl in split:
            counts[lbl] += 1
        print(f"  {split_name}: " + ", ".join(
            f"{lbl}={c}" for lbl, c in sorted(counts.items())))

    # ---- Write val/ ----
    val_img_dir = val_dir / "images"
    if val_img_dir.exists():
        for old in val_img_dir.iterdir():
            if old.is_file():
                old.unlink()
    val_img_dir.mkdir(parents=True, exist_ok=True)

    val_rows = []
    for i, (src, label) in enumerate(val):
        new_name = f"{NAME_PREFIX}{i:0{width}d}{NAME_EXT}"
        dest = val_img_dir / new_name
        shutil.copy2(src, dest)
        val_rows.append({"imagePath": new_name, LABEL_COLUMN: label})
    with open(val_dir / "label.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["imagePath", LABEL_COLUMN])
        w.writeheader()
        w.writerows(val_rows)
    print(f"\n[Split] Wrote {len(val)} val images to {val_img_dir}/ + {val_dir / 'label.csv'}")

    # ---- Rewrite train/ (renumber to close the gap) ----
    # Write the new train items to a temporary directory first so a crash
    # mid-rewrite can't leave train/ in an inconsistent state with a
    # label.csv pointing at missing files. Then atomically swap.
    tmp_dir = train_img_dir.parent / f".{train_img_dir.name}_swap"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    train_rows = []
    for i, (src, label) in enumerate(train):
        new_name = f"{NAME_PREFIX}{i:0{width}d}{NAME_EXT}"
        dest = tmp_dir / new_name
        shutil.copy2(src, dest)
        train_rows.append({"imagePath": new_name, LABEL_COLUMN: label})

    # All new files written successfully — now swap.
    if train_img_dir.exists():
        for old in train_img_dir.iterdir():
            if old.is_file():
                old.unlink()
        train_img_dir.rmdir()
    tmp_dir.rename(train_img_dir)

    with open(train_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["imagePath", LABEL_COLUMN])
        w.writeheader()
        w.writerows(train_rows)
    print(f"[Split] Rewrote {len(train)} train images to {train_img_dir}/ + {train_csv}")

    # ---- Update master_label.csv if it exists (Figshare only) ----
    master_csv = data_root / "master_label.csv"
    if master_csv.is_file():
        # master_label.csv was the pre-split list; rewrite it to mirror
        # the new train/ membership so dedup.py --split can be re-run on
        # the same dataset without re-renaming.
        with open(master_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["imagePath", LABEL_COLUMN])
            w.writeheader()
            # train rows carry the same image names as in train/label.csv
            for row in train_rows:
                w.writerow(row)
            # val rows also belong to the master "train-like" pool
            for row in val_rows:
                w.writerow(row)
        print(f"[Split] Updated {master_csv} to {len(train_rows) + len(val_rows)} rows")


if __name__ == "__main__":
    main()