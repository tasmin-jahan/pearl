#!/usr/bin/env python3
"""Generate thesis-specific figures and tables for the PEARL thesis.

Outputs to:
  docs/thesis/figures/  PNGs
  docs/thesis/tables/   .tex fragments

Reuses JSON/CSV result artefacts under results/ -- does not re-run training.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(".")
OUT_FIG = ROOT / "docs" / "thesis" / "figures"
OUT_TAB = ROOT / "docs" / "thesis" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

# ---------------------------------------------------------------------
# 1. Dataset composition table
# ---------------------------------------------------------------------

def dataset_composition_table(out_path: pathlib.Path) -> None:
    """Table 4.1: dataset composition across Figshare, PCOSgen, splits."""
    rows = [
        ("Figshare raw",      "11,000+", "raw corpus before dedup"),
        ("Figshare PCOS",     "3,184",   "md5-unique infected"),
        ("Figshare non-PCOS", "812",     "md5-unique non-infected"),
        ("Figshare unique",   "3,996",   "after md5 dedup; 3.92:1 imbalance"),
        ("PCOSgen raw",       "4,668",   "3,627 PCOS / 1,041 healthy"),
        ("PCOSgen train pool", "3,200",  "2560 train + 640 val; 362 PCOS / 2838 healthy"),
        ("Fine-tune train",   "2,560",   "after stratified split"),
        ("Fine-tune val",     "640",     "best-epoch selection"),
        ("Held-out test",     "1,468",   "549 PCOS / 919 healthy; never seen in training"),
    ]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Dataset composition across the executed pipeline.}")
    lines.append(r"  \label{tab:dataset-composition}")
    lines.append(r"  \begin{tabular}{lr@{\hspace{1em}}l}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Partition} & \textbf{Images} & \textbf{Notes} \\")
    lines.append(r"    \midrule")
    for name, n, note in rows:
        lines.append(rf"    {name} & {n} & {note} \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 2. Model roster table
# ---------------------------------------------------------------------

def model_roster_table(out_path: pathlib.Path) -> None:
    """Table 6.1: model roster with parameters, freeze_fraction executed."""
    rows = [
        ("ResNet-50",        "resnet50",                    "25.6", "25.6", "0"),
        ("ResNet-101",       "resnet101",                   "44.5", "44.5", "0"),
        ("DenseNet-121",     "densenet121",                 "8.1",  "8.1",  "0"),
        ("DenseNet-169",     "densenet169",                 "14.1", "14.1", "0"),
        ("EfficientNet-B0",  "tf\\_efficientnet\\_b0",     "5.3",  "5.3",  "0"),
        ("ConvNeXt-Tiny",    "convnext\\_tiny",            "28.6", "28.6", "0"),
        ("MobileNetV3-Large","mobilenetv3\\_large\\_100", "5.4",  "5.4",  "0"),
        ("ViT-Base/16",      "vit\\_base\\_patch16\\_224", "86.6", "86.6", "0"),
        ("Swin-Tiny",        "swin\\_tiny\\_patch4\\_window7\\_224", "28.3", "28.3", "0"),
    ]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Architecture roster. The \texttt{freeze\_fraction} column records the value used in the executed top-3 fine-tune; Optuna selected \texttt{freeze\_fraction=0} for all three members (see~\cref{tab:hpo-best}).}")
    lines.append(r"  \label{tab:model-roster}")
    lines.append(r"  \begin{tabular}{llrrl}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Name} & \textbf{timm key} & \textbf{Params (M)} & \textbf{Trainable (M)} & \textbf{Freeze frac.} \\")
    lines.append(r"    \midrule")
    for n, k, p, t, ff in rows:
        lines.append(rf"    {n} & \texttt{{{k}}} & {p} & {t} & {ff} \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 3. HPO best params table
# ---------------------------------------------------------------------

def hpo_best_table(out_path: pathlib.Path) -> None:
    """Table 6.2: Optuna best params for top-3 players."""
    files = [
        ("DenseNet-121 (srad)",   "results/finetune_hpo/srad_nopad_densenet121/densenet121/best_params.yaml"),
        ("ConvNeXt-Tiny (srad)",  "results/finetune_hpo/srad_nopad_convnext_tiny/convnext_tiny/best_params.yaml"),
        ("ViT-B/16 (gauss)",      "results/finetune_hpo/gauss_nopad_vit_base/vit_base/best_params.yaml"),
    ]
    rows = []
    for arch, path in files:
        if not pathlib.Path(path).exists():
            continue
        cfg = json.loads(pathlib.Path(path).read_text()) if path.endswith(".json") else None
        if cfg is None:
            # yaml
            import yaml
            cfg = yaml.safe_load(pathlib.Path(path).read_text())
        rows.append((arch, cfg))

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Optuna best hyperparameters for the three ensemble members. All three selected \texttt{freeze\_fraction=0} (full backbone fine-tuning).}")
    lines.append(r"  \label{tab:hpo-best}")
    lines.append(r"  \begin{tabular}{lcccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Architecture} & \textbf{LR} & \textbf{WD} & \textbf{Dropout} & \textbf{Rot.} & \textbf{LS} & \textbf{Batch} \\")
    lines.append(r"    \midrule")
    for arch, cfg in rows:
        lr = cfg["lr"]
        wd = cfg["weight_decay"]
        dr = cfg["dropout"]
        rot = cfg.get("rotation", 0)
        ls = cfg.get("label_smoothing", 0)
        bs = cfg["batch_size"]
        lines.append(rf"    {arch} & {lr:.2e} & {wd:.2e} & {dr:.2f} & {rot} & {ls:.3f} & {bs} \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 4. Per-class precision/recall bar chart
# ---------------------------------------------------------------------

def per_class_pr_figure(out_path: pathlib.Path) -> None:
    files = [
        ("DenseNet-121",   "results/finetune_zenodo/checkpoints/srad_nopad/densenet121/final_metrics.json"),
        ("ConvNeXt-Tiny",  "results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny/final_metrics.json"),
        ("ViT-B/16",       "results/finetune_zenodo/checkpoints/gauss_nopad/vit_base/final_metrics.json"),
        ("Ensemble",       "results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.json"),
    ]
    archs, prec, rec, f1, auc = [], [], [], [], []
    for name, path in files:
        if not pathlib.Path(path).exists():
            continue
        d = json.loads(pathlib.Path(path).read_text())
        if "members" in d and "test_precision" not in d:
            # ensemble JSON; can't compute per-class PR here without probabilities;
            # use the top-3 P/R from per-member final_metrics
            continue
        archs.append(name)
        prec.append(d.get("test_precision", float("nan")))
        rec.append(d.get("test_recall", float("nan")))
        f1.append(d.get("test_f1", float("nan")))
        auc.append(d.get("test_auc_roc", float("nan")))
    # Ensemble precision/recall fallback
    ens_csv = "results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.csv"
    if pathlib.Path(ens_csv).exists():
        tp = fp = fn = tn = 0
        with open(ens_csv) as f:
            for row in csv.DictReader(f):
                y = int(row["label"])
                p = int(float(row["prob_infected"]) >= 0.5)
                if y == 1 and p == 1: tp += 1
                elif y == 0 and p == 1: fp += 1
                elif y == 1 and p == 0: fn += 1
                else: tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        # Insert/replace ensemble entry
        if archs and archs[-1] == "Ensemble":
            prec[-1] = precision
            rec[-1] = recall
        else:
            archs.append("Ensemble")
            prec.append(precision)
            rec.append(recall)
            f1.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
            auc.append(0.9487)

    x = np.arange(len(archs))
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    width = 0.25
    ax.bar(x - width, prec, width, label="Precision", color="#5b8fb8")
    ax.bar(x,         rec,  width, label="Recall",    color="#d77a61")
    ax.bar(x + width, f1,   width, label="F1",        color="#7a9b6e")
    for i, (p, r, f) in enumerate(zip(prec, rec, f1)):
        # Anchor labels on top of each bar; for low-bar cases use a small
        # fixed offset that places the text just above the bar height.
        for xpos, val in [(-width, p), (0, r), (width, f)]:
            ax.text(i + xpos, max(val, 0.02) + 0.015, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=9)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall, and F1 on the held-out PCOSgen test set\n(threshold = 0.5)", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=8, facecolor="white", edgecolor="#ccc")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 5. Threshold sweep line plot
# ---------------------------------------------------------------------

def threshold_sweep_figure(out_path: pathlib.Path) -> None:
    csv_path = "results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.csv"
    if not pathlib.Path(csv_path).exists():
        print(f"missing {csv_path}; skipping")
        return
    probs = []
    labels = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            probs.append(float(row["prob_infected"]))
            labels.append(int(row["label"]))
    probs = np.array(probs); labels = np.array(labels)
    thresholds = np.linspace(0.05, 0.95, 91)
    sens, spec, f1s = [], [], []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        f1s.append(f1)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(thresholds, sens, label="Sensitivity", color="#d77a61")
    ax.plot(thresholds, spec, label="Specificity", color="#5b8fb8")
    ax.plot(thresholds, f1s,  label="F1",          color="#7a9b6e")
    for t0, lbl in [(0.5, "0.5"), (0.6198, "Youden"), (0.7597, "sens=0.95")]:
        if 0.05 <= t0 <= 0.95:
            ax.axvline(t0, linestyle="--", alpha=0.5)
            ax.text(t0 + 0.005, 0.05, lbl, rotation=90, fontsize=7, color="#555")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Pre-HPO ensemble: threshold sweep on the held-out test set", fontsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 6. Uncertainty histogram (correct vs incorrect)
# ---------------------------------------------------------------------

def uncertainty_figure(out_path: pathlib.Path) -> None:
    """Load MC-Dropout entropy per checkpoint and compare correct vs incorrect."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    titles = [
        ("DenseNet-121",  "results/finetune_zenodo/checkpoints/srad_nopad/densenet121/uncertainty/mc_dropout_results.json"),
        ("ConvNeXt-Tiny", "results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny/uncertainty/mc_dropout_results.json"),
        ("ViT-B/16",      "results/finetune_zenodo/checkpoints/gauss_nopad/vit_base/uncertainty/mc_dropout_results.json"),
    ]
    for ax, (name, p) in zip(axes, titles):
        if not pathlib.Path(p).exists():
            ax.set_title(f"{name} (not found)", fontsize=10)
            continue
        d = json.loads(pathlib.Path(p).read_text())
        # Per-sample entropy is stored separately; join it to the validation
        # CSV by image path so correctness is computed from ground truth.
        csv_path = p.replace("mc_dropout_results.json", "mc_dropout_predictions.csv")
        pred_path = p.replace("/uncertainty/mc_dropout_results.json", "/external_validation/pcosgen.csv")
        if not pathlib.Path(csv_path).exists() or not pathlib.Path(pred_path).exists():
            ax.set_title(f"{name} (per-sample data not found)", fontsize=9)
            continue
        labels_by_path = {}
        pred_by_path = {}
        with open(pred_path) as f:
            for row in csv.DictReader(f):
                labels_by_path[row["path"]] = int(row["label"])
                pred_by_path[row["path"]] = int(row["pred"])
        ents_correct, ents_wrong = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ent = float(row["predictive_entropy"])
                path_key = row["path"]
                if path_key not in labels_by_path:
                    continue
                correct = labels_by_path[path_key] == pred_by_path[path_key]
                (ents_correct if correct else ents_wrong).append(ent)
        ax.hist(ents_correct, bins=20, alpha=0.6, color="#5b8fb8", label="correct")
        ax.hist(ents_wrong,  bins=20, alpha=0.6, color="#d77a61", label="incorrect")
        ax.set_title(f"{name}\nmean={d['mean_entropy']:.3f} nats", fontsize=9)
        ax.set_xlabel("Predictive entropy (nats)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("MC-Dropout entropy by correctness (50 passes)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 7. Combined training curve figure
# ---------------------------------------------------------------------

def training_curves_figure(out_path: pathlib.Path) -> None:
    """Plot training curves from the existing final per-arch JSONs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    archs = [
        ("DenseNet-121",  "results/finetune_zenodo/checkpoints/srad_nopad/densenet121/epoch_log.csv"),
        ("ConvNeXt-Tiny", "results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny/epoch_log.csv"),
        ("ViT-B/16",      "results/finetune_zenodo/checkpoints/gauss_nopad/vit_base/epoch_log.csv"),
    ]
    colors = ["#d77a61", "#5b8fb8", "#7a9b6e"]
    for (name, path), col in zip(archs, colors):
        if not pathlib.Path(path).exists():
            continue
        epochs, train_loss, val_loss, val_auc = [], [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                epochs.append(int(row["epoch"]) + 1)
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                val_auc.append(float(row["val_auc"]))
        axes[0].plot(epochs, train_loss, color=col, label=f"{name} train", linewidth=1.5)
        axes[0].plot(epochs, val_loss,   color=col, linestyle="--", label=f"{name} val", linewidth=1.5)
        axes[1].plot(epochs, val_auc,    color=col, label=name, linewidth=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation AUC")
    axes[1].set_title("Validation AUC during fine-tuning")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 8. Calibration before/after table
# ---------------------------------------------------------------------

def calibration_table(out_path: pathlib.Path) -> None:
    """Table 7.1: per-model calibration before/after temperature scaling."""
    src = "results/finetune_zenodo/ensemble/top3_pre_hpo/calibration/calibration_results.json"
    if not pathlib.Path(src).exists():
        print(f"missing {src}; skipping")
        return
    d = json.loads(pathlib.Path(src).read_text())
    archs = d["archs"]
    T = d["per_model_temperatures"]
    n = d["n_test_samples"]
    ece1 = d["ece_after_pass1"]
    ece2 = d["ece_after_pass2"]
    T_ens = d["ensemble_temperature"]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Two-pass temperature scaling. Per-model temperatures scale the logits of each member before probability averaging; the ensemble temperature then scales the averaged logits.}")
    lines.append(r"  \label{tab:calibration-detail}")
    lines.append(r"  \begin{tabular}{lr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Component} & \textbf{Value} \\")
    lines.append(r"    \midrule")
    for a, t in zip(archs, T):
        lines.append(rf"    {a} \,\,$T_i$ & {t:.4f} \\")
    lines.append(rf"    Ensemble $T_\text{{ens}}$ & {T_ens:.4f} \\")
    lines.append(rf"    ECE after pass 1 & {ece1:.4f} \\")
    lines.append(rf"    ECE after pass 2 & {ece2:.4f} \\")
    lines.append(rf"    Test samples & {n} \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 9. Stage confusion-matrix numerics table
# ---------------------------------------------------------------------

def stage_confusion_table(out_path: pathlib.Path) -> None:
    """Table 8.1: numerics matching the three-stage confusion-matrix figure."""
    def cm_from(path, thr=0.5):
        cm = [[0, 0], [0, 0]]
        with open(path) as f:
            for row in csv.DictReader(f):
                y = int(row["label"]); p = int(float(row["prob_infected"]) >= thr)
                cm[y][p] += 1
        return cm
    cm1 = cm_from("results/ablation/checkpoints/srad/resnet50/external_validation/pcosgen.csv")
    cm2 = cm_from("results/finetune_zenodo/checkpoints/srad_nopad/densenet121/external_validation/pcosgen.csv")
    cm3 = cm_from("results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.csv")
    rows = [
        ("1. Foundation (SRAD + ResNet-50)",  cm1, 4668, "Figshare train, PCOSgen full"),
        ("2. Fine-tune (SRAD + DenseNet-121)", cm2, 1468, "PCOSgen train, held-out test"),
        ("3. Pre-HPO ensemble",                cm3, 1468, "PCOSgen train, held-out test"),
    ]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Confusion-matrix counts per pipeline stage. Stages 1 and 2--3 are not directly comparable: Stage 1 was scored on the full PCOSgen corpus before the train/test split existed; Stages 2 and 3 share the held-out test set.}")
    lines.append(r"  \label{tab:stage-cm}")
    lines.append(r"  \begin{tabular}{lrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Stage} & \textbf{TN} & \textbf{FP} & \textbf{FN} & \textbf{TP} & \textbf{$n$} \\")
    lines.append(r"    \midrule")
    for name, cm, n, _ in rows:
        tn, fp = cm[0]
        fn, tp = cm[1]
        lines.append(rf"    {name} & {tn} & {fp} & {fn} & {tp} & {n} \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# 10. Headline metrics table
# ---------------------------------------------------------------------

def headline_metrics_table(out_path: pathlib.Path) -> None:
    src = "results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.json"
    d = json.loads(pathlib.Path(src).read_text())
    cal = json.loads(pathlib.Path("results/finetune_zenodo/ensemble/top3_pre_hpo/calibration/calibration_results.json").read_text())
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Pre-HPO ensemble headline metrics on the held-out PCOSgen test set ($n=1468$).}")
    lines.append(r"  \label{tab:headline}")
    lines.append(r"  \begin{tabular}{lr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Metric} & \textbf{Value} \\")
    lines.append(r"    \midrule")
    lines.append(rf"    AUC-ROC (95\% CI) & {d['test_auc_roc']:.4f} ({d['test_auc_roc_ci_low']:.4f}--{d['test_auc_roc_ci_high']:.4f}) \\")
    lines.append(rf"    F1 & {d['test_f1']:.4f} \\")
    lines.append(rf"    MCC & {d['test_mcc']:.4f} \\")
    lines.append(rf"    Brier & {d['test_brier']:.4f} \\")
    lines.append(rf"    ECE (before pass 1) & {cal['ece_after_pass1']:.4f} \\")
    lines.append(rf"    ECE (after pass 2) & {cal['ece_after_pass2']:.4f} \\")
    if "test_ap" in d:
        lines.append(rf"    Average Precision (PR-AUC) & {d['test_ap']:.4f} \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    dataset_composition_table(OUT_TAB / "dataset_composition.tex")
    model_roster_table(OUT_TAB / "model_roster.tex")
    hpo_best_table(OUT_TAB / "hpo_best.tex")
    per_class_pr_figure(OUT_FIG / "thesis_per_class_pr.png")
    threshold_sweep_figure(OUT_FIG / "thesis_threshold_sweep.png")
    uncertainty_figure(OUT_FIG / "thesis_uncertainty_hist.png")
    training_curves_figure(OUT_FIG / "thesis_training_curves.png")
    calibration_table(OUT_TAB / "calibration_detail.tex")
    stage_confusion_table(OUT_TAB / "stage_confusion.tex")
    headline_metrics_table(OUT_TAB / "headline_metrics.tex")


if __name__ == "__main__":
    main()