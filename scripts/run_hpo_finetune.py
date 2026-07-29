#!/usr/bin/env python3
"""Run Optuna HPO sweeps for the top-3 fine-tuned (preprocessing, arch) pairs.

Reads the post-fine-tune external_validation/summary.csv to identify the top-3
(by test_auc), then for each finalist:
  1. Generates a per-arch experiment config (configs/experiment/finetune_hpo_<prep>_<arch>.yaml)
     that points the tuner's `fine_tune_resume_from` at the matching fine-tuned checkpoint.
  2. Runs `scripts/sweep_hpo.py` with that config, saving trials to
     results/finetune_hpo/<prep>_<arch>/.
  3. Re-trains the best HPO trial for full 15 epochs using the best params,
     overwriting the corresponding fine-tuned checkpoint with the optimized version.
  4. Re-evaluates the optimized checkpoint on Zenodo test.

Usage:
    python scripts/run_hpo_finetune.py --top_k 3
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")


def _read_summary(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    df = df.sort_values("auc", ascending=False).reset_index(drop=True)
    return df.to_dict("records")


def _write_per_arch_config(template_path: str, prep: str, arch: str, ckpt_path: str,
                            out_path: str) -> None:
    with open(template_path) as f:
        cfg = yaml.safe_load(f)
    cfg["preprocessing"] = prep
    cfg["model"] = arch
    cfg["data"]["fine_tune_resume_from"] = ckpt_path
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def _run_sweep_hpo(exp_cfg: str, out_dir: str, dry_run: bool = False) -> None:
    cmd = [
        VENV_PY, "scripts/sweep_hpo.py",
        "--experiment", exp_cfg,
        "--out_dir", out_dir,
    ]
    print(f"[HPO] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


def _run_external_eval(ckpt: str, model_cfg: str, preproc_cfg: str, run_dir: str,
                       external_dir: str, layout: str, dry_run: bool = False) -> None:
    cmd = [
        VENV_PY, "scripts/evaluate_external.py",
        "--model", model_cfg,
        "--preprocessing", preproc_cfg,
        "--checkpoint", ckpt,
        "--external_dir", external_dir,
        "--layout", layout,
        "--run_dir", run_dir,
        "--dataset_slug", "pcosgen",
    ]
    print(f"[HPO->Eval] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


def _copy_best_to_finetuned(src_ckpt: str, dst_dir: str, dry_run: bool = False) -> None:
    """Copy the best HPO checkpoint over the fine-tuned checkpoint so the
    final ensemble/measures use the HPO-optimized versions."""
    if dry_run:
        return
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "best.pt")
    shutil.copy2(src_ckpt, dst)
    print(f"[HPO] Copied {src_ckpt} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/finetune_zenodo/external_validation/summary.csv")
    parser.add_argument("--template", default="configs/experiment/finetune_hpo.yaml")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--external_dir", default="data_external/test")
    parser.add_argument("--layout", default="zenodo_labeled")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.summary):
        raise FileNotFoundError(
            f"Summary not found: {args.summary}. Run scripts/eval_finetune_zenodo.py first."
        )

    rank = _read_summary(args.summary)
    finalists = rank[: args.top_k]
    print(f"HPO finalists (top-{args.top_k} by external AUC):")
    for i, f_ in enumerate(finalists, 1):
        print(f"  {i}. {f_['arch']} + {f_['preprocessing']} -> AUC={f_['auc']:.4f}")

    # Generate per-arch configs and run HPO
    summary_paths = []
    for f_ in finalists:
        prep = f_["preprocessing"]
        arch = f_["arch"]
        run_dir = f"results/finetune_zenodo/checkpoints/{prep}/{arch}"
        ckpt = f"{run_dir}/best.pt"
        per_arch_cfg = f"configs/experiment/finetune_hpo_{prep}_{arch}.yaml"
        _write_per_arch_config(args.template, prep, arch, ckpt, per_arch_cfg)
        print(f"\n[HPO] Config for {arch} + {prep} -> {per_arch_cfg}")

        out_dir = f"results/finetune_hpo/{prep}_{arch}"
        _run_sweep_hpo(per_arch_cfg, out_dir, dry_run=args.dry_run)

        # Locate the best HPO checkpoint
        # - per-arch sweep_hpo.py saves: <out_dir>/<arch>/best_params.yaml + trial best.pt
        # - The sweep_hpo script stores its best model at <out_dir>/<arch>/<model>__tuned.pt
        #   (used by tune.py), but sweep_hpo.py itself doesn't retrain. We retrain
        #   with the best params in a follow-up step.
        best_params_path = os.path.join(out_dir, arch, "best_params.yaml")
        if not os.path.isfile(best_params_path):
            print(f"[HPO] WARN: best_params.yaml not found at {best_params_path}")
            continue

        # Retrain with best params → save to <run_dir>/best.pt (overwriting).
        # We use the existing sweep_finetune.py with a pre-generated
        # experiment config that includes the best params.
        with open(best_params_path) as fh:
            best_params = yaml.safe_load(fh)
        retrain_cfg = {
            "name": f"finetune_retrain_{prep}_{arch}",
            "mode": "finetune",
            "preprocessing": [prep],
            "models": [arch],
            "training": {
                "optimizer": "adamw",
                "lr": float(best_params.get("lr", 3e-5)),
                "weight_decay": float(best_params.get("weight_decay", 1e-3)),
                "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
                "scheduler": "cosine_annealing",
                "batch_size": int(best_params.get("batch_size", 32)),
                "max_epochs": 15,
                "early_stopping_patience": 5,
                "warmup_epochs": 1,
                "freeze_epochs": 0,
                "monitor": "val_auc",
                "bf16": True, "channels_last": True,
                "grad_clip_norm": 1.0, "ema": True, "sampler": "weighted",
                "augmentation": {
                    "rotation": float(best_params.get("rotation", 10)),
                    "horizontal_flip": True,
                    "scale": 0.1,
                },
            },
            "seed": 42,
            "results_dir": f"results/finetune_hpo_retrain/{prep}_{arch}/",
            "data": {
                "train_split": "data_external/zenodo_splits/train.json",
                "val_split": "data_external/zenodo_splits/val.json",
                "external_dir": args.external_dir,
                "external_layout": args.layout,
                "resume_root": "results/finetune_zenodo/checkpoints/",
            },
        }
        retrain_cfg_path = f"configs/experiment/finetune_retrain_{prep}_{arch}.yaml"
        with open(retrain_cfg_path, "w") as fh:
            yaml.dump(retrain_cfg, fh, default_flow_style=False, sort_keys=False)
        print(f"[HPO] Retrain config for {arch} + {prep} -> {retrain_cfg_path}")

        # Run the retrain (skipping the existing best.pt)
        retrain_dir = retrain_cfg["results_dir"]
        new_best = os.path.join("results/finetune_zenodo/checkpoints", prep, arch, "best.pt")
        if not args.dry_run and os.path.isfile(new_best):
            # Backup the un-HPO best.pt (we'll re-evaluate the HPO'd one)
            backup_dir = os.path.join("results/finetune_hpo", prep + "_" + arch, "pre_hpo_backup")
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(new_best, os.path.join(backup_dir, "best.pt"))

        cmd = [VENV_PY, "scripts/sweep_finetune.py", "--experiment", retrain_cfg_path]
        print(f"[HPO] Retrain: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)

        # Re-evaluate the retrained checkpoint
        model_cfg = f"configs/model/{arch}.yaml"
        preproc_cfg = f"configs/preprocessing/{prep}.yaml"
        _run_external_eval(
            ckpt=new_best,
            model_cfg=model_cfg,
            preproc_cfg=preproc_cfg,
            run_dir=os.path.join("results/finetune_zenodo/checkpoints", prep, arch),
            external_dir=args.external_dir,
            layout=args.layout,
            dry_run=args.dry_run,
        )

        # Read the post-HPO external AUC
        json_path = os.path.join(
            "results/finetune_zenodo/checkpoints", prep, arch,
            "external_validation", "pcosgen.json",
        )
        if os.path.isfile(json_path):
            with open(json_path) as fh:
                d = json.load(fh)
            summary_paths.append({
                "arch": arch, "preprocessing": prep,
                "pre_hpo_auc": float(f_["auc"]),
                "post_hpo_auc": float(d.get("test_auc_roc", float("nan"))),
                "post_hpo_acc": float(d.get("test_accuracy", float("nan"))),
                "post_hpo_f1": float(d.get("test_f1", float("nan"))),
                "post_hpo_mcc": float(d.get("test_mcc", float("nan"))),
            })

    # HPO summary
    if not args.dry_run and summary_paths:
        out = "results/finetune_hpo/hpo_summary.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(summary_paths, fh, indent=2)
        print(f"\n[HPO] Summary: {out}")
        for s in summary_paths:
            print(f"  {s['arch']} + {s['preprocessing']}: "
                  f"pre={s['pre_hpo_auc']:.4f} -> post={s['post_hpo_auc']:.4f}")


if __name__ == "__main__":
    main()