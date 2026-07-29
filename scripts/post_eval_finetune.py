#!/usr/bin/env python3
"""Post-training analysis runner for the top-K fine-tuned checkpoints.

For each <run_dir> given, runs (in order):
  - scripts/run_calibration.py
  - scripts/run_uncertainty.py
  - scripts/run_xai.py --method gradcam

Then aggregates the JSON outputs into a single per-(arch,prep) table at
<out_dir>/post_eval_summary.csv.

Usage:
    python scripts/post_eval_finetune.py \
        --run_dirs results/finetune_zenodo/checkpoints/srad_nopad/resnet50 \
                   results/finetune_zenodo/checkpoints/srad_nopad/efficientnet_b0
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd: List[str], cwd: str = ROOT, check: bool = True):
    print(f"[PostEval] {' '.join(cmd)}")
    subprocess.run(cmd, check=check, cwd=cwd)


def _read_metrics(run_dir: str) -> dict:
    out = {"run_dir": run_dir}
    # External eval (Zenodo test)
    f = os.path.join(run_dir, "external_validation", "pcosgen.json")
    if os.path.isfile(f):
        with open(f) as fh:
            d = json.load(fh)
        for k in ("test_auc_roc", "test_accuracy", "test_f1", "test_mcc",
                  "test_sensitivity", "test_specificity", "test_precision"):
            out[k] = d.get(k)
    # Calibration (Zenodo)
    f = os.path.join(run_dir, "calibration", "calibration_results.json")
    if os.path.isfile(f):
        with open(f) as fh:
            d = json.load(fh)
        out["ece_full_before"] = d.get("ece_full_before")
        out["ece_held_before"] = d.get("ece_held_before")
        out["ece_held_after_temp"] = d.get("ece_held_after_temp")
        out["optimal_temperature"] = d.get("optimal_temperature")
    # Uncertainty (Zenodo)
    f = os.path.join(run_dir, "uncertainty", "mc_dropout_results.json")
    if os.path.isfile(f):
        with open(f) as fh:
            d = json.load(fh)
        out["mean_entropy"] = d.get("mean_entropy")
        out["median_entropy"] = d.get("median_entropy")
        out["n_passes"] = d.get("n_passes")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--n_passes", type=int, default=50)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument("--skip_uncertainty", action="store_true")
    parser.add_argument("--skip_xai", action="store_true")
    args = parser.parse_args()

    summary = []
    for run_dir in args.run_dirs:
        print(f"\n[PostEval] === {run_dir} ===")

        if not args.skip_calibration:
            cmd = [
                ".venv/bin/python", "scripts/run_calibration_zenodo.py",
                "--run_dir", run_dir,
            ]
            try:
                _run(cmd)
            except subprocess.CalledProcessError as e:
                print(f"  calibration failed: {e}")

        if not args.skip_uncertainty:
            cmd = [
                ".venv/bin/python", "scripts/run_uncertainty_zenodo.py",
                "--run_dir", run_dir,
                "--n_passes", str(args.n_passes),
            ]
            try:
                _run(cmd)
            except subprocess.CalledProcessError as e:
                print(f"  uncertainty failed: {e}")

        if not args.skip_xai:
            cmd = [
                ".venv/bin/python", "scripts/run_xai_zenodo.py",
                "--run_dir", run_dir,
                "--method", "gradcam",
            ]
            try:
                _run(cmd)
            except subprocess.CalledProcessError as e:
                print(f"  xai failed: {e}")

        summary.append(_read_metrics(run_dir))

    # Write summary table
    out_csv = os.path.join(args.run_dirs[0], "..", "..", "post_eval_summary.csv")
    out_csv = os.path.normpath(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = sorted({k for s in summary for k in s.keys()})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(f"[PostEval] Wrote {out_csv}")


if __name__ == "__main__":
    main()