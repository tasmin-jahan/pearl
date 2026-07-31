#!/usr/bin/env python3
"""Top-level orchestrator for the top-3 fine-tune pipeline.

Runs (per finalist):
  1. HPO search (sweep_hpo_finetune.py) — already done; this script reads the
     resulting best_params.yaml.
  2. Retrain with best HPO params (retrain_with_hpo.py) — full 15 epochs.
  3. Evaluate on Zenodo test (scripts/evaluate_external.py --layout zenodo_labeled).
  4. Aggregate results into results/finetune_zenodo/top3_final_results.json.

Usage:
    python scripts/run_top3_finetune_pipeline.py \
        --finalists configs/top3_finalists.yaml
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _run(cmd, cwd=ROOT, check=True):
    print(f"[Top3] >>> {' '.join(cmd)}")
    res = subprocess.run(cmd, check=check, cwd=cwd)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalists", required=True,
                        help="YAML listing top-3 finalists with their HPO dirs.")
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--skip_retrain", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.finalists) as f:
        cfg = yaml.safe_load(f)
    finalists = cfg.get("finalists", [])
    if not finalists:
        raise SystemExit("No finalists found in YAML")

    results = []
    for f in finalists:
        prep = f["preprocessing"]
        arch = f["arch"]
        hpo_dir = f["hpo_dir"]
        run_dir = os.path.join("results/finetune_zenodo/checkpoints", prep, arch)
        print(f"\n{'='*70}\n[Top3] {arch} + {prep}\n{'='*70}")

        # 1. Retrain
        if not args.skip_retrain:
            cmd = [
                ".venv/bin/python", "scripts/retrain_with_hpo.py",
                "--preprocessing", prep,
                "--arch", arch,
                "--hpo_dir", hpo_dir,
                "--max_epochs", str(args.max_epochs),
            ]
            _run(cmd)

        # 2. Evaluate
        if not args.skip_eval:
            model_cfg = f"configs/model/{arch}.yaml"
            preproc_cfg = "configs/preprocessing.yaml"
            ckpt = os.path.join(run_dir, "best.pt")
            cmd = [
                ".venv/bin/python", "scripts/evaluate_external.py",
                "--run_dir", run_dir,
                "--model", model_cfg,
                "--preprocessing", preproc_cfg,
                "--checkpoint", ckpt,
                "--external_dir", "data_external/test",
                "--layout", "zenodo_labeled",
            ]
            _run(cmd)

        # 3. Read metrics
        fjson = os.path.join(run_dir, "external_validation", "pcosgen.json")
        if os.path.isfile(fjson):
            with open(fjson) as fh:
                m = json.load(fh)
            results.append({
                "arch": arch,
                "preprocessing": prep,
                "test_auc_roc": m.get("test_auc_roc"),
                "test_accuracy": m.get("test_accuracy"),
                "test_f1": m.get("test_f1"),
                "test_sensitivity": m.get("test_sensitivity"),
                "test_specificity": m.get("test_specificity"),
                "test_precision": m.get("test_precision"),
                "test_mcc": m.get("test_mcc"),
                "run_dir": run_dir,
            })

    # Aggregate
    out_path = "results/finetune_zenodo/top3_final_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Top3] Wrote {out_path}")
    print("\n[Top3] Summary:")
    for r in results:
        print(f"  {r['arch']}+{r['preprocessing']}: AUC={r['test_auc_roc']:.4f}, "
              f"F1={r['test_f1']:.4f}, Sens={r['test_sensitivity']:.4f}, "
              f"Spec={r['test_specificity']:.4f}")


if __name__ == "__main__":
    main()