#!/usr/bin/env python3
"""Build reproducible train/val splits for fine-tuning on Zenodo PCOSgen.

Uses the new `src/data/zenodo_dataset.py` adapter to read the Zenodo source
layout (`<root>/images/*.jpg` + `class_label.xlsx` / `class label.csv`).

Output: `data_external/zenodo_splits/{train,val}.json` — list of objects with
`path` and `label` fields. The Zenodo test split is reserved as the HELD-OUT
external test set and is NOT touched by this script.

Usage:
    python scripts/build_zenodo_splits.py
    python scripts/build_zenodo_splits.py --val_frac 0.2 --seed 42

Default: 80/20 stratified split, seed 42.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.zenodo_dataset import discover_zenodo_pairs, stratified_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data_external/train",
                        help="Zenodo source root containing images/ and the label file")
    parser.add_argument("--out_dir", default="data_external/zenodo_splits",
                        help="Output dir for train.json + val.json")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = discover_zenodo_pairs(args.source)
    if not pairs:
        raise RuntimeError(f"No image+label pairs found under {args.source}")

    train, val = stratified_split(pairs, val_frac=args.val_frac, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {
        "source": os.path.abspath(args.source),
        "n_total": len(pairs),
        "n_train": len(train),
        "n_val": len(val),
        "val_frac": args.val_frac,
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    for name, items in (("train", train), ("val", val)):
        out_path = os.path.join(args.out_dir, f"{name}.json")
        with open(out_path, "w") as f:
            json.dump({
                "summary": summary if name == "train" else None,
                "items": [{"path": p, "label": int(l)} for p, l in items],
            }, f, indent=2)
        print(f"  wrote {len(items):5d} items to {out_path}")

    from collections import Counter
    print(f"\nTrain label counts: {dict(Counter(l for _, l in train))}")
    print(f"Val   label counts: {dict(Counter(l for _, l in val))}")
    print(f"Total: {len(pairs)} (val_frac={args.val_frac}, seed={args.seed})")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
