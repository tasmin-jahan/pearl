#!/usr/bin/env python3
"""Stage 02: Prepare & Preprocess Data for PEARL.

End-to-end data pipeline:
  1. Unpacks raw archives in data/:
     - data/PCOS.zip -> data/raw/figshare/{infected,noninfected}
     - data/updated train dataset.zip -> data/raw/pcosgen/train/{images,label.csv}
     - data/updated test dataset.zip -> data/raw/pcosgen/test/{images,label.csv}
  2. Deduplication and splitting for Figshare (src.preprocessing.dedup and split):
     - MD5 and perceptual hash deduplication
     - Stratified test split (15%)
     - Stratified validation split (10%)
  3. Splitting for PCOSgen:
     - Stratified validation split (20% carved from train)
  4. Deterministic preprocessing materialization (src.preprocessing.run_preprocessing):
     - Applies SRAD / CLAHE / adaptive filters
     - Generates canonical PNGs under data/preprocessed/figshare and data/preprocessed/pcosgen

Usage::

    python scripts/02_prepare_data.py
    python scripts/02_prepare_data.py --skip-unpack  # if already extracted
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def unpack_figshare(force: bool = False) -> None:
    zip_path = REPO_ROOT / "data" / "PCOS.zip"
    target_dir = REPO_ROOT / "data" / "raw" / "figshare"
    infected_dir = target_dir / "infected"
    noninfected_dir = target_dir / "noninfected"

    if infected_dir.exists() and noninfected_dir.exists() and not force:
        print("[1/4 - Figshare] Raw directories already exist. Skipping extraction.")
        return

    if not zip_path.exists():
        print(f"[WARN] {zip_path} not found. Skipping Figshare unpack.")
        return

    print(f"[1/4 - Figshare] Unpacking {zip_path.name} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.endswith("/"):
                continue
            # Members typically formatted: PCOS/infected/image.jpg
            parts = member.split("/")
            if len(parts) >= 3 and parts[0] == "PCOS" and parts[1] in ("infected", "noninfected"):
                category = parts[1]
                filename = parts[-1]
                dest_path = target_dir / category / filename
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print("  [OK] Figshare raw dataset extracted.")


def unpack_pcosgen(force: bool = False) -> None:
    train_zip = REPO_ROOT / "data" / "updated train dataset.zip"
    test_zip = REPO_ROOT / "data" / "updated test dataset.zip"
    target_dir = REPO_ROOT / "data" / "raw" / "pcosgen"

    train_out = target_dir / "train"
    test_out = target_dir / "test"

    if (train_out / "images").exists() and (test_out / "images").exists() and not force:
        print("[1/4 - PCOSgen] Raw directories already exist. Skipping extraction.")
        return

    print("[1/4 - PCOSgen] Unpacking PCOSgen train & test sets...")
    target_dir.mkdir(parents=True, exist_ok=True)

    if train_zip.exists():
        train_out.mkdir(parents=True, exist_ok=True)
        (train_out / "images").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(train_zip, "r") as z:
            for member in z.namelist():
                if member.endswith("/"):
                    continue
                if member == "class_label.xlsx":
                    content = z.read(member)
                    df = pd.read_excel(io.BytesIO(content))
                    # Standardize column name
                    df.to_csv(train_out / "label.csv", index=False)
                elif "images/" in member:
                    fname = os.path.basename(member)
                    if fname:
                        dest = train_out / "images" / fname
                        with z.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        print("  [OK] PCOSgen train pool extracted and label.csv created.")

    if test_zip.exists():
        test_out.mkdir(parents=True, exist_ok=True)
        (test_out / "images").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(test_zip, "r") as z:
            for member in z.namelist():
                if member.endswith("/"):
                    continue
                if member == "class label.csv":
                    content = z.read(member)
                    df = pd.read_csv(io.BytesIO(content))
                    df.to_csv(test_out / "label.csv", index=False)
                elif "images/" in member:
                    fname = os.path.basename(member)
                    if fname:
                        dest = test_out / "images" / fname
                        with z.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        print("  [OK] PCOSgen test set extracted and label.csv created.")


def run_figshare_dedup_and_split() -> None:
    raw_figshare = REPO_ROOT / "data" / "raw" / "figshare"
    if not (raw_figshare / "infected").exists():
        print("[2/4] Figshare raw files not found. Skipping dedup & split.")
        return

    train_dir = raw_figshare / "train"
    val_dir = raw_figshare / "val"
    test_dir = raw_figshare / "test"

    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        print("[2/4] Figshare splits (train/val/test) already prepared.")
        return

    print("\n[2/4] Running Figshare deduplication, renaming, and test split...")
    cmd_dedup = [
        sys.executable, "-m", "src.preprocessing.dedup",
        "--all", "--cross-dedup", "--test-frac", "0.15",
    ]
    subprocess.run(cmd_dedup, cwd=str(REPO_ROOT), check=True)

    print("  Carving Figshare validation split (10%)...")
    cmd_split = [
        sys.executable, "-m", "src.preprocessing.split",
        "--dataset", "figshare", "--val-frac", "0.10",
    ]
    subprocess.run(cmd_split, cwd=str(REPO_ROOT), check=True)
    print("  [OK] Figshare raw splits ready.")


def run_pcosgen_split() -> None:
    raw_pcosgen = REPO_ROOT / "data" / "raw" / "pcosgen"
    if not (raw_pcosgen / "train").exists():
        print("[3/4] PCOSgen raw train folder not found. Skipping split.")
        return

    if (raw_pcosgen / "val").exists():
        print("[3/4] PCOSgen val split already carved.")
        return

    print("\n[3/4] Carving PCOSgen validation split (20% of train pool)...")
    cmd_split = [
        sys.executable, "-m", "src.preprocessing.split",
        "--dataset", "pcosgen", "--val-frac", "0.20",
    ]
    subprocess.run(cmd_split, cwd=str(REPO_ROOT), check=True)
    print("  [OK] PCOSgen raw splits ready.")


def materialize_preprocessed_datasets(dataset: str = "both") -> None:
    print("\n[4/4] Materializing preprocessed PNG datasets...")
    targets = []
    if dataset in ("both", "figshare"):
        targets.append(("data/raw/figshare", "data/preprocessed/figshare"))
    if dataset in ("both", "pcosgen"):
        targets.append(("data/raw/pcosgen", "data/preprocessed/pcosgen"))

    for src_rel, dst_rel in targets:
        src = REPO_ROOT / src_rel
        dst = REPO_ROOT / dst_rel
        if not src.exists() or not (src / "train").exists():
            print(f"  [SKIP] {src_rel} does not have required train/ split.")
            continue
        print(f"  Materializing {src_rel} -> {dst_rel}...")
        cmd = [
            sys.executable, "-m", "src.preprocessing.run_preprocessing",
            "--dataset-dir", str(src),
            "--output-dir", str(dst),
        ]
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        print(f"  [OK] Preprocessed dataset ready at {dst_rel}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PEARL Data Preparation.")
    parser.add_argument("--skip-unpack", action="store_true", help="Skip unzipping raw files.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip PNG materialization pass.")
    parser.add_argument("--dataset", choices=("both", "figshare", "pcosgen"), default="both")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing splits.")
    args = parser.parse_args(argv)

    print("========================================")
    print("   PEARL Pipeline - Data Preparation    ")
    print("========================================")

    if not args.skip_unpack:
        if args.dataset in ("both", "figshare"):
            unpack_figshare(force=args.force)
        if args.dataset in ("both", "pcosgen"):
            unpack_pcosgen(force=args.force)

    if args.dataset in ("both", "figshare"):
        run_figshare_dedup_and_split()
    if args.dataset in ("both", "pcosgen"):
        run_pcosgen_split()

    if not args.skip_preprocess:
        materialize_preprocessed_datasets(dataset=args.dataset)

    print("\n========================================")
    print("  STATUS: Data preparation completed!")
    print("  Next Step: python scripts/03_train_models.py")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
