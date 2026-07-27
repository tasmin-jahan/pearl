#!/usr/bin/env python3
"""Compare Figshare internal-val and PCOSgen external-val metrics across
the three preprocessing ablations: reflect-padding (srad/gauss), no-padding
(srad_nopad/gauss_nopad), and no-padding+greyscale
(srad_nopad_gray/gauss_nopad_gray).

Emits one CSV per comparison (internal val, external val) plus a small
markdown summary printed to stdout.

Run after each ablation completes:
  python scripts/compare_ablation_runs.py
"""
import csv
import glob
import json
import os
from collections import defaultdict


ABLATIONS = {
    "reflect (orig)": "results/ablation",
    "no-pad":        "results/ablation_nopad",
    "no-pad+gray":   "results/ablation_nopad_gray",
}
ARCHES = ['resnet50', 'resnet101', 'densenet121', 'densenet169',
          'efficientnet_b0', 'convnext_tiny', 'mobilenetv3_large',
          'vit_base', 'swin_tiny']
PREPS_PER_ABLATION = {
    "reflect (orig)": ["srad", "gauss"],
    "no-pad":        ["srad_nopad", "gauss_nopad"],
    "no-pad+gray":   ["srad_nopad_gray", "gauss_nopad_gray"],
}


def load_per_run_internal(root):
    """{ablation: {arch: {prep: {val_auc, test_auc, test_acc, test_f1}}}}"""
    out = {a: {ar: {} for ar in ARCHES} for a in ABLATIONS}
    for ablation, ab_root in ABLATIONS.items():
        for prep in PREPS_PER_ABLATION[ablation]:
            for arch in ARCHES:
                p = os.path.join(ab_root, "checkpoints", prep, arch, "final_metrics.json")
                if not os.path.isfile(p):
                    continue
                with open(p) as f:
                    m = json.load(f)
                out[ablation][arch][prep] = {
                    "val_auc":  m.get("val_auc_best"),
                    "test_auc": m.get("test_auc_roc"),
                    "test_acc": m.get("test_accuracy"),
                    "test_f1":  m.get("test_f1"),
                    "test_mcc": m.get("test_mcc"),
                }
    return out


def load_external_per_arch():
    """{ablation: {arch: {prep: {auc, accuracy, f1}}}}"""
    out = {a: {ar: {} for ar in ARCHES} for a in ABLATIONS}
    for ablation, ab_root in ABLATIONS.items():
        ext_root = os.path.join(ab_root, "checkpoints")
        for prep in PREPS_PER_ABLATION[ablation]:
            for arch in ARCHES:
                p = os.path.join(ext_root, prep, arch,
                                 "external_validation", "pcosgen.json")
                if not os.path.isfile(p):
                    continue
                with open(p) as f:
                    m = json.load(f)
                out[ablation][arch][prep] = {
                    "auc":      m.get("test_auc_roc"),
                    "accuracy": m.get("test_accuracy"),
                    "f1":       m.get("test_f1"),
                    "mcc":      m.get("test_mcc"),
                    "n":        m.get("n_samples"),
                }
    return out


def to_csv(rows, out_path):
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out_path} ({len(rows)} rows)")


def build_rows(per_run, metric_path, metric_keys, per_arch_means=False):
    """per_arch_means=False -> one row per (ablation, arch, prep, metric)."""
    rows = []
    for ablation in ABLATIONS:
        for arch in ARCHES:
            for prep in PREPS_PER_ABLATION[ablation]:
                m = per_run.get(ablation, {}).get(arch, {}).get(prep)
                if not m:
                    continue
                row = {"ablation": ablation, "arch": arch, "prep": prep}
                for k in metric_keys:
                    row[k] = m.get(k)
                rows.append(row)
    if per_arch_means:
        # collapse across preps -> per-(ablation, arch)
        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            key = (r["ablation"], r["arch"])
            for k in metric_keys:
                if r[k] is not None:
                    agg[key][k].append(r[k])
        rows = []
        for (ablation, arch), ms in sorted(agg.items()):
            row = {"ablation": ablation, "arch": arch}
            for k in metric_keys:
                vals = ms[k]
                row[f"{k}_mean"] = sum(vals)/len(vals) if vals else None
            rows.append(row)
    return rows


def print_summary(internal, external):
    print("\n=== Per-arch mean AUC (averaged across preps) ===")
    print(f"{'ablation':<18} {'arch':<22} {'val_auc':>9} {'test_auc':>9} "
          f"{'ext_auc':>9} {'ext_acc':>9} {'ext_f1':>9}")
    for ablation in ABLATIONS:
        for arch in ARCHES:
            internal_vals = [m.get("val_auc") for m in
                             internal[ablation][arch].values()
                             if m.get("val_auc") is not None]
            test_vals = [m.get("test_auc") for m in
                         internal[ablation][arch].values()
                         if m.get("test_auc") is not None]
            ext_vals = [m.get("auc") for m in
                        external[ablation][arch].values()
                        if m.get("auc") is not None]
            ext_acc = [m.get("accuracy") for m in
                       external[ablation][arch].values()
                       if m.get("accuracy") is not None]
            ext_f1  = [m.get("f1") for m in
                       external[ablation][arch].values()
                       if m.get("f1") is not None]
            def mean(xs):
                xs = [x for x in xs if x is not None]
                return sum(xs)/len(xs) if xs else float("nan")
            print(f"{ablation:<18} {arch:<22} {mean(internal_vals):>9.4f} "
                  f"{mean(test_vals):>9.4f} {mean(ext_vals):>9.4f} "
                  f"{mean(ext_acc):>9.4f} {mean(ext_f1):>9.4f}")


def main():
    print("[compare] Loading per-run metrics...")
    internal = load_per_run_internal("results")
    external = load_external_per_arch()

    os.makedirs("results/comparison", exist_ok=True)

    internal_rows = build_rows(internal, "internal",
                               ["val_auc", "test_auc", "test_acc",
                                "test_f1", "test_mcc"])
    external_rows = build_rows(external, "external",
                               ["auc", "accuracy", "f1", "mcc", "n"])

    to_csv(internal_rows, "results/comparison/internal_per_run.csv")
    to_csv(external_rows, "results/comparison/external_per_run.csv")

    int_arch = build_rows(internal, "internal",
                          ["val_auc", "test_auc", "test_acc",
                           "test_f1", "test_mcc"],
                          per_arch_means=True)
    ext_arch = build_rows(external, "external",
                          ["auc", "accuracy", "f1", "mcc"],
                          per_arch_means=True)
    to_csv(int_arch, "results/comparison/internal_per_arch.csv")
    to_csv(ext_arch, "results/comparison/external_per_arch.csv")

    print_summary(internal, external)


if __name__ == "__main__":
    main()