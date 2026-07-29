#!/usr/bin/env python3
"""External evaluation of all fine-tuned checkpoints on the Zenodo held-out test set.

For each (preprocessing, arch) pair under results/finetune_zenodo/checkpoints/,
runs `scripts/evaluate_external.py --layout zenodo_labeled --external_dir data_external/test`
and writes the per-checkpoint JSON + CSV into
    <run_dir>/external_validation/pcosgen.{json,csv}

Then aggregates all per-arch JSONs into:
    results/finetune_zenodo/external_validation/summary.csv

Usage:
    python scripts/eval_finetune_zenodo.py
    python scripts/eval_finetune_zenodo.py --dry_run     # print what would run
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _find_finetune_dirs(root: str):
    pattern = os.path.join(root, "checkpoints", "*", "*", "best.pt")
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/finetune_zenodo/")
    parser.add_argument("--external_dir", default="data_external/test")
    parser.add_argument("--layout", default="zenodo_labeled")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")

    ckpts = _find_finetune_dirs(args.results_dir)
    if not ckpts:
        raise RuntimeError(
            f"No best.pt files found under {args.results_dir}/checkpoints/*/*/"
        )

    print(f"[EvalFinetune] Found {len(ckpts)} fine-tuned checkpoints under {args.results_dir}")

    results_root = []
    for ckpt in ckpts:
        parts = ckpt.split("/")
        # .../checkpoints/<prep>/<arch>/best.pt
        preproc = parts[-3]
        arch = parts[-2]
        run_dir = os.path.dirname(ckpt)
        out_json = os.path.join(run_dir, "external_validation", "pcosgen.json")
        out_csv = os.path.join(run_dir, "external_validation", "pcosgen.csv")

        model_cfg = os.path.join("configs", "model", f"{arch}.yaml")
        preproc_cfg = os.path.join("configs", "preprocessing", f"{preproc}.yaml")
        if not os.path.isfile(model_cfg):
            print(f"[EvalFinetune] WARN: missing {model_cfg}; skipping")
            continue
        if not os.path.isfile(preproc_cfg):
            print(f"[EvalFinetune] WARN: missing {preproc_cfg}; skipping")
            continue

        cmd = [
            ".venv/bin/python", "scripts/evaluate_external.py",
            "--model", model_cfg,
            "--preprocessing", preproc_cfg,
            "--checkpoint", ckpt,
            "--external_dir", args.external_dir,
            "--layout", args.layout,
            "--run_dir", run_dir,
            "--dataset_slug", "pcosgen",
        ]
        print(f"\n[EvalFinetune] {arch} + {preproc}")
        if args.dry_run:
            print("  ", " ".join(cmd))
            continue

        if os.path.isfile(out_json):
            print(f"  already evaluated → {out_json} (skipping)")
        else:
            subprocess.run(cmd, check=True, cwd=ROOT)

        if os.path.isfile(out_json):
            with open(out_json) as f:
                d = json.load(f)
            results_root.append({
                "arch": arch,
                "preprocessing": preproc,
                "auc": d.get("test_auc_roc"),
                "accuracy": d.get("test_accuracy"),
                "f1": d.get("test_f1"),
                "mcc": d.get("test_mcc"),
                "sensitivity": d.get("test_recall"),
                "specificity": d.get("test_specificity"),
                "precision": d.get("test_precision"),
                "n_samples": d.get("n_samples"),
            })

    # Aggregate summary
    if not args.dry_run and results_root:
        df = pd.DataFrame(results_root).sort_values("auc", ascending=False)
        out_csv = os.path.join(args.results_dir, "external_validation", "summary.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n[EvalFinetune] Wrote {len(df)} rows to {out_csv}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()