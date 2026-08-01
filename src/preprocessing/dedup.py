#!/usr/bin/env python3
"""
Figshare dedup + canonical rename + train/test stratified split.

Three stages, run sequentially by default:

    1. dedup       — exact-byte (md5) duplicates + near-duplicates
                     (perceptual hash with Hamming distance <= threshold).
                     Quarantines the dupes into _duplicates/ (per-class AND
                     cross-class if --cross-dedup is set).
    2. rename      — copy survivors to _renamed/ with canonical padded
                     filenames (image0000.jpg..image{N-1}.jpg).
                     Noninfected first (label = Not-visible), then
                     infected (label = Visible).
                     Writes master_label.csv (imagePath, PCOS-visible).
    3. split       — stratified 15% test split (default), materialised as
                     data/raw/figshare/{train,test}/{images,label.csv}.
                     This matches the data/raw/pcosgen/{train,test}/ layout.

The output is the canonical Figshare layout:
    data/raw/figshare/
        _duplicates/      # quarantined dupes (per-class or cross-class)
        _renamed/         # canonical copies used to build master_label.csv
        master_label.csv  # imagePath,PCOS-visible
        infected/         # original raw files (untouched)
        noninfected/      # original raw files (untouched)
        train/{images,label.csv}
        test/{images,label.csv}

After this script finishes, run split.py to carve val/ out of train/.

Usage:
    python scripts/preprocessing/dedup.py --all
    python scripts/preprocessing/dedup.py dedup --cross-dedup
    python scripts/preprocessing/dedup.py rename
    python scripts/preprocessing/dedup.py split --test-frac 0.15
    python scripts/preprocessing/dedup.py --all --cross-dedup --test-frac 0.15
"""

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# --- Layout constants ------------------------------------------------------

DATA_ROOT = Path("data/raw/figshare")
QUARANTINE_DIR = DATA_ROOT / "_duplicates"
RENAMED_DIR = DATA_ROOT / "_renamed"
MASTER_LABEL_CSV = DATA_ROOT / "master_label.csv"
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

NAME_PREFIX = "image"
NAME_EXT = ".jpg"

# Class labels — same vocabulary as the PCOSGen label.csv last column.
NEGATIVE_LABEL = "Not-visible"   # noninfected
POSITIVE_LABEL = "Visible"       # infected
LABEL_COLUMN = "PCOS-visible"

# Perceptual-hash config: 8x8 dHash = 64 bits; Hamming distance <= N means
# "visually similar enough to be a near-duplicate". The default threshold
# is 0/64: we treat as duplicates only images that are identical at the
# 9x8 grayscale level. This catches re-encoded versions of the same
# image (different JPEG bytes, same visual content) without
# over-quarantining distinct images — for ultrasound, mean inter-image
# Hamming distance is ~26-30 bits, so any non-zero threshold catches a
# large fraction of legitimate distinct images. Set --near-threshold 1
# or 2 to be more aggressive if your dataset has heavier near-dup
# contamination.
DHASH_SIZE = 8
NEAR_DUP_HAMMING_THRESHOLD = 0

DEFAULT_TEST_FRAC = 0.15
DEFAULT_SEED = 42


# --- Filesystem helpers ----------------------------------------------------

def md5_of(path: Path, chunk: int = 1 << 16) -> str:
    """Stream-hash a file's bytes (constant memory)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def collect_hashes(class_dir: Path, extensions: tuple) -> dict:
    """{md5: [path, ...]} for every image under class_dir.

    First lexicographic occurrence is the canonical keeper.
    """
    hashes: dict = defaultdict(list)
    for fname in sorted(os.listdir(class_dir)):
        if not fname.lower().endswith(extensions):
            continue
        p = class_dir / fname
        if not p.is_file():
            continue
        hashes[md5_of(p)].append(p)
    return hashes


def list_images(class_dir: Path, extensions: tuple) -> list:
    """Sorted list of image paths under class_dir."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def pad_width(n: int) -> int:
    """Zero-pad width: enough digits to hold `n` items (>= 4)."""
    return max(4, len(str(n)))


# --- Near-dedup (perceptual hash) ------------------------------------------

def dhash(image_path: Path, hash_size: int = DHASH_SIZE) -> int:
    """Compute a perceptual dHash for an image file.

    Algorithm:
      1. Read image (grayscale).
      2. Resize to (hash_size+1, hash_size) — i.e. 9x8 for the default.
      3. Compare adjacent pixels horizontally: bit i is 1 if left > right.
      4. Pack the bits into an integer.

    Two images with dHashes whose Hamming distance is small (<= threshold)
    are visually near-duplicates (cropped, resized, slightly re-encoded).

    Args:
        image_path: Path to an image file.
        hash_size: Hash side length (default 8 -> 64-bit hash).

    Returns:
        Integer hash (64 bits for hash_size=8).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Unreadable file: treat as a unique sentinel hash that won't
        # collide with anything legitimate. We let the caller decide
        # whether to keep or quarantine.
        return -1
    resized = cv2.resize(
        img, (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA,
    )
    diff = resized[:, 1:] > resized[:, :-1]
    # Pack 8 rows of (hash_size) bits into one int.
    out = 0
    for row in diff:
        for bit in row:
            out = (out << 1) | int(bit)
    return out


def hamming_distance(a: int, b: int) -> int:
    """Bit-popcount of XOR of a and b (number of differing bits)."""
    return bin(a ^ b).count("1")


def find_near_duplicates(
    paths: list, threshold: int = NEAR_DUP_HAMMING_THRESHOLD,
) -> dict:
    """Group paths into near-duplicate clusters via dHash + Hamming distance.

    Returns {canonical_path: [duplicate_paths, ...]} where canonical_path is
    the lexicographically-first path in each cluster. Paths that have no
    near-duplicate are NOT included in the returned dict.

    O(N^2) in the worst case but N is small (~4000 images) and we early-
    exit clusters as soon as a canonical is assigned.
    """
    print(f"[NearDup] Computing dHash for {len(paths)} images...")
    hashes = []
    for i, p in enumerate(paths):
        h = dhash(p)
        hashes.append(h)
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(paths)}")

    # Union-find for clustering
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Bucket by hamming buckets for speed: only compare hashes in same/buck
    # would be ideal, but the naive O(N^2) is fine for ~4000 images.
    n = len(paths)
    for i in range(n):
        if hashes[i] == -1:
            continue
        for j in range(i + 1, n):
            if hashes[j] == -1:
                continue
            if find(i) == find(j):
                continue  # already clustered
            d = hamming_distance(hashes[i], hashes[j])
            if d <= threshold:
                union(i, j)

    # Group by cluster root
    clusters = defaultdict(list)
    for i in range(n):
        if hashes[i] == -1:
            continue
        clusters[find(i)].append(i)

    # Build output: only return clusters with >1 member
    dup_clusters = {}
    for root, members in clusters.items():
        if len(members) > 1:
            sorted_members = sorted(paths[m] for m in members)
            kept = sorted_members[0]
            dup_clusters[kept] = sorted_members[1:]
    return dup_clusters


# --- Stage 1: dedup --------------------------------------------------------

def stage_dedup(args) -> None:
    """Quarantine exact-byte duplicates, then near-duplicates.

    Modes:
      - per-class (default): find dupes within infected/ and within
        noninfected/ independently.
      - cross-class (--cross-dedup): also find dupes that appear in BOTH
        classes (same image labeled twice with conflicting labels — a
        potential label-noise source).

    Quarantined files go to _duplicates/<class>/. Near-duplicates found
    cross-class go to _duplicates/cross/<class_of_kept>/.
    """
    extensions = tuple(e.lower() if e.startswith(".") else "." + e.lower()
                       for e in args.extensions)
    print(f"[Dedup] Mode: {'exact+near, cross-class' if args.cross_dedup else 'exact+near, per-class'}")

    # ---- Pass 1: exact-byte dedup (per-class) ----
    classes = ["noninfected", "infected"]
    for cls in classes:
        cls_dir = DATA_ROOT / cls
        if not cls_dir.is_dir():
            sys.exit(f"ERROR: missing class folder {cls_dir}")

        qdir = QUARANTINE_DIR / cls
        qdir.mkdir(parents=True, exist_ok=True)

        hashes = collect_hashes(cls_dir, extensions)
        n_total = sum(len(v) for v in hashes.values())
        n_unique = len(hashes)
        n_quarantined = 0
        for h, paths in hashes.items():
            kept = paths[0]
            for p in paths[1:]:
                dest = qdir / p.name
                if dest.exists():
                    stem, suf = dest.stem, dest.suffix
                    i = 1
                    while True:
                        cand = qdir / f"{stem}__{i}{suf}"
                        if not cand.exists():
                            dest = cand
                            break
                        i += 1
                if not args.dry_run:
                    shutil.move(str(p), str(dest))
                n_quarantined += 1

        print(f"  [exact] {cls:14s} files={n_total:>5,}  unique={n_unique:>5,}  "
              f"quarantined={n_quarantined:>5,}")

    # ---- Pass 2: near-duplicate dedup (per-class) ----
    for cls in classes:
        cls_dir = DATA_ROOT / cls
        if not cls_dir.is_dir():
            continue
        survivors = list_images(cls_dir, extensions)
        if not survivors:
            print(f"  [near ] {cls:14s} files=0 (skip)")
            continue

        clusters = find_near_duplicates(
            survivors, threshold=args.near_threshold,
        )
        qdir = QUARANTINE_DIR / cls
        n_quarantined = 0
        for kept, dupes in clusters.items():
            for p in dupes:
                dest = qdir / p.name
                if dest.exists():
                    stem, suf = dest.stem, dest.suffix
                    i = 1
                    while True:
                        cand = qdir / f"{stem}_near{i}{suf}"
                        if not cand.exists():
                            dest = cand
                            break
                        i += 1
                if not args.dry_run:
                    shutil.move(str(p), str(dest))
                n_quarantined += 1

        n_clusters = len(clusters)
        print(f"  [near ] {cls:14s} files={len(survivors):>5,}  clusters={n_clusters:>4,}  "
              f"quarantined={n_quarantined:>5,}")

    # ---- Pass 3 (optional): cross-class near-dedup ----
    if args.cross_dedup:
        all_survivors = []
        for cls in classes:
            cls_dir = DATA_ROOT / cls
            if cls_dir.is_dir():
                for p in list_images(cls_dir, extensions):
                    all_survivors.append((p, cls))

        print(f"[Dedup] Cross-class near-dedup on {len(all_survivors)} images...")
        paths_only = [p for p, _ in all_survivors]
        clusters = find_near_duplicates(
            paths_only, threshold=args.near_threshold,
        )

        qdir = QUARANTINE_DIR / "cross"
        n_quarantined = 0
        cross_label_conflicts = 0
        for kept, dupes in clusters.items():
            kept_cls = next(c for p, c in all_survivors if p == kept)
            # Detect label conflicts (a near-dup with different class)
            dup_classes = {all_survivors[i][1] for i in (
                paths_only.index(p) for p in dupes
            )}
            if dup_classes != {kept_cls}:
                cross_label_conflicts += 1
            for p in dupes:
                dest = qdir / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    stem, suf = dest.stem, dest.suffix
                    i = 1
                    while True:
                        cand = qdir / f"{stem}_x{i}{suf}"
                        if not cand.exists():
                            dest = cand
                            break
                        i += 1
                if not args.dry_run:
                    shutil.move(str(p), str(dest))
                n_quarantined += 1

        n_clusters = len(clusters)
        print(f"  [cross] clusters={n_clusters:>4,}  "
              f"with-label-conflict={cross_label_conflicts:>4,}  "
              f"quarantined={n_quarantined:>5,}")

    print(f"\n[Dedup] Done. Quarantined files in {QUARANTINE_DIR}/")


# --- Stage 2: rename -------------------------------------------------------

def stage_rename(args) -> None:
    """Copy survivors to _renamed/ with canonical padded filenames.

    Order: noninfected first (Not-visible), then infected (Visible).
    Writes master_label.csv (imagePath,PCOS-visible).
    """
    extensions = tuple(e.lower() if e.startswith(".") else "." + e.lower()
                       for e in args.extensions)

    if not args.force and MASTER_LABEL_CSV.exists() and RENAMED_DIR.exists():
        n_lines = sum(1 for _ in MASTER_LABEL_CSV.open()) - 1
        n_imgs = sum(1 for _ in RENAMED_DIR.iterdir() if _.is_file())
        if n_lines == n_imgs and n_lines > 0:
            print(f"[Rename] Skipping — master_label.csv and _renamed/ already "
                  f"exist ({n_imgs} images). Pass --force to redo.")
            return

    noninfected_files = list_images(DATA_ROOT / "noninfected", extensions)
    infected_files = list_images(DATA_ROOT / "infected", extensions)
    all_files = [(p, NEGATIVE_LABEL) for p in noninfected_files] + \
                [(p, POSITIVE_LABEL) for p in infected_files]

    n = len(all_files)
    width = pad_width(n)
    print(f"[Rename] {len(noninfected_files)} noninfected + {len(infected_files)} "
          f"infected = {n} total images")
    print(f"[Rename] Writing to {RENAMED_DIR}/ with width {width} digits")

    if args.dry_run:
        for i, (p, label) in enumerate(all_files[:3]):
            print(f"  would rename {p.name} -> {NAME_PREFIX}{i:0{width}d}"
                  f"{NAME_EXT}  [{label}]")
        print(f"  ... ({n - 3} more)")
        return

    if RENAMED_DIR.exists():
        shutil.rmtree(RENAMED_DIR)
    RENAMED_DIR.mkdir(parents=True)

    rows = []
    for i, (src, label) in enumerate(all_files):
        new_name = f"{NAME_PREFIX}{i:0{width}d}{NAME_EXT}"
        dest = RENAMED_DIR / new_name
        shutil.copy2(src, dest)
        rows.append({"imagePath": new_name, LABEL_COLUMN: label})

    with open(MASTER_LABEL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["imagePath", LABEL_COLUMN])
        w.writeheader()
        w.writerows(rows)

    print(f"[Rename] Wrote {n} rows to {MASTER_LABEL_CSV}")
    print(f"[Rename] Wrote {n} files to {RENAMED_DIR}/")


# --- Stage 3: train/test split --------------------------------------------

def stratified_split_2way(items: list, test_frac: float, seed: int) -> tuple:
    """Stratified 2-way split preserving class proportions.

    items: list of (path, label). Returns (train, test).
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)
    train, test = [], []
    for label, group in by_label.items():
        group = sorted(group)
        rng.shuffle(group)
        n_test = max(1, int(round(len(group) * test_frac)))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_split(split_dir: Path, items: list, name_prefix: str,
                width: int, name_ext: str) -> None:
    """Materialise one split: copy images to split_dir/images/ and write
    split_dir/label.csv.

    Uses a temp-dir-then-swap pattern so a crash mid-write can't leave
    split_dir with an empty images/ folder and a stale label.csv.
    """
    img_dir = split_dir / "images"
    tmp_dir = split_dir / ".images_swap"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    rows = []
    for i, (src, label) in enumerate(items):
        new_name = f"{name_prefix}{i:0{width}d}{name_ext}"
        dest = tmp_dir / new_name
        shutil.copy2(src, dest)
        rows.append({"imagePath": new_name, LABEL_COLUMN: label})

    # All new files written — swap.
    if img_dir.exists():
        for old in img_dir.iterdir():
            if old.is_file():
                old.unlink()
        img_dir.rmdir()
    tmp_dir.rename(img_dir)

    csv_path = split_dir / "label.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["imagePath", LABEL_COLUMN])
        w.writeheader()
        w.writerows(rows)


def stage_split(args) -> None:
    """Stratified train/test split (default test_frac=0.15).

    Reads master_label.csv, splits stratified by label, writes
    train/{images,label.csv} and test/{images,label.csv}.
    """
    if not MASTER_LABEL_CSV.exists():
        sys.exit(f"ERROR: {MASTER_LABEL_CSV} not found. Run `rename` first.")

    if not args.force and (TRAIN_DIR / "label.csv").exists() and \
                            (TEST_DIR / "label.csv").exists():
        n_train = sum(1 for _ in (TRAIN_DIR / "images").iterdir())
        n_test = sum(1 for _ in (TEST_DIR / "images").iterdir())
        if n_train > 0 and n_test > 0:
            print(f"[Split] Skipping — train/ ({n_train}) and test/ ({n_test}) "
                  f"already exist. Pass --force to redo.")
            return

    items = []
    with open(MASTER_LABEL_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = RENAMED_DIR / row["imagePath"]
            if not img_path.is_file():
                sys.exit(f"ERROR: master references missing file {img_path}")
            items.append((img_path, row[LABEL_COLUMN]))

    train, test = stratified_split_2way(items, args.test_frac, args.seed)

    n = len(items)
    width = pad_width(n)
    print(f"[Split] Stratified {args.test_frac:.0%} test split (seed={args.seed}):")
    print(f"  total = {n}  ->  train = {len(train)}, test = {len(test)}")
    for split_name, split in [("train", train), ("test", test)]:
        counts = defaultdict(int)
        for _, lbl in split:
            counts[lbl] += 1
        print(f"  {split_name}: " + ", ".join(
            f"{lbl}={c}" for lbl, c in sorted(counts.items())))

    if args.dry_run:
        print("[Split] Dry run — no files written.")
        return

    write_split(TRAIN_DIR, train, NAME_PREFIX, width, NAME_EXT)
    write_split(TEST_DIR, test, NAME_PREFIX, width, NAME_EXT)
    print(f"\n[Split] Done. PCOSGen-compatible layout at:")
    print(f"  {TRAIN_DIR}/{{images,label.csv}}  ({len(train)} imgs)")
    print(f"  {TEST_DIR}/{{images,label.csv}}   ({len(test)} imgs)")


# --- Stage: all -----------------------------------------------------------

def stage_all(args) -> None:
    stage_dedup(args)
    stage_rename(args)
    stage_split(args)


# --- CLI ------------------------------------------------------------------

def _add_common_stage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen, change nothing")
    p.add_argument("--extensions", nargs="+",
                   default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                   help="File extensions to consider")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all stages in order (dedup -> rename -> split)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extensions", nargs="+",
                        default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"])
    parser.add_argument("--cross-dedup", action="store_true",
                        help="Also run cross-class near-dedup (catches the "
                             "same image labeled as both infected and "
                             "noninfected)")
    parser.add_argument("--near-threshold", type=int,
                        default=NEAR_DUP_HAMMING_THRESHOLD,
                        help=f"Hamming distance threshold for near-dup "
                             f"(default {NEAR_DUP_HAMMING_THRESHOLD}/64 bits)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default {DEFAULT_SEED})")
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC,
                        help=f"Test fraction (default {DEFAULT_TEST_FRAC})")

    sub = parser.add_subparsers(dest="cmd", required=False)

    p_dedup = sub.add_parser("dedup",
                            help="Stage 1: exact-byte + near-duplicate dedup")
    _add_common_stage_args(p_dedup)
    p_dedup.add_argument("--cross-dedup", action="store_true",
                        help="Also run cross-class near-dedup")
    p_dedup.add_argument("--near-threshold", type=int,
                        default=NEAR_DUP_HAMMING_THRESHOLD)

    p_rename = sub.add_parser("rename",
                              help="Stage 2: canonical rename + master_label.csv")
    _add_common_stage_args(p_rename)

    p_split = sub.add_parser("split",
                            help="Stage 3: stratified train/test split")
    _add_common_stage_args(p_split)
    p_split.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_split.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)

    args = parser.parse_args()

    if args.all:
        stage_all(args)
        return

    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    handler = {
        "dedup": stage_dedup,
        "rename": stage_rename,
        "split": stage_split,
    }[args.cmd]
    handler(args)


if __name__ == "__main__":
    main()