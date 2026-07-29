#!/usr/bin/env python3
"""Generate paper-specific combined figures for the PEARL paper.

Outputs go to docs/latex/figures/:
  paper_fig_combined_roc.png     -- Top-3 + ensemble ROC curves
  paper_fig_combined_pr.png      -- Top-3 + ensemble PR curves
  paper_fig_calibration_compare.png  -- Before/after calibration for 3 models
  paper_fig_uncertainty_compare.png  -- Entropy distributions per model
  paper_fig_sweep_matrix.png     -- Heatmap of 18 fine-tuned runs
  paper_fig_ensemble_metrics.png -- Pre-HPO vs HPO ensemble bar chart
  paper_fig_xai_panel.png        -- 3x3 Grad-CAM grid (densenet/convnext/vit)
  paper_fig_generalization.png   -- Cross-dataset recovery bar chart
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/farhan/my-projects/pearl")
FIG_DIR = REPO_ROOT / "docs/latex/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRE_HPO = REPO_ROOT / "results/finetune_zenodo/ensemble/top3_pre_hpo"
HPO = REPO_ROOT / "results/finetune_zenodo/ensemble/top3_hpo"

ARCHS = [
    ("srad_nopad", "densenet121", "densenet121 + SRAD-nopad"),
    ("srad_nopad", "convnext_tiny", "ConvNeXt-T + SRAD-nopad"),
    ("gauss_nopad", "vit_base", "ViT-B + Gauss-nopad"),
]
COLORS = {"densenet121": "#1f77b4", "convnext_tiny": "#ff7f0e", "vit_base": "#2ca02c"}


def _load_ensemble(run_dir):
    return json.loads((run_dir / "external_validation/pcosgen.json").read_text())


def _load_top3_individual(arch_idx):
    prep, arch, _ = ARCHS[arch_idx]
    p = REPO_ROOT / f"results/finetune_zenodo/checkpoints/{prep}/{arch}/external_validation/pcosgen.json"
    return json.loads(p.read_text())


def _predict_for_roc(run_dir, ckpt_name="best_before_hpo.pt"):
    """Get fpr, tpr, precision, recall arrays by running inference.

    Defaults to best_before_hpo.pt (the pre-HPO checkpoint) so the curves
    reflect the recommended deployable model. Falls back to best.pt if
    the pre-HPO checkpoint is not present.
    """
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO_ROOT))
    from src.utils.config import load_config
    from src.utils.seed import set_seed
    from src.model.builder import build_model
    from src.training.checkpoint import load_checkpoint
    from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
    from src.preprocessing.preprocess import Preprocessor
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_curve, precision_recall_curve

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(str(run_dir / "config.yaml"))
    mc = dict(cfg["model"])
    mc["freeze_fraction"] = 0.0
    model = build_model(mc)
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = run_dir / "best.pt"
    load_checkpoint(model, str(ckpt_path),
                    optimizer=None, scheduler=None, ema=None, device=device)
    model.to(device).eval()

    pairs = discover_zenodo_pairs("data_external/test")
    pre = Preprocessor(cfg["preprocessing"], input_size=mc.get("input_size", 224))
    ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=False)

    probs1 = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs1.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.extend([int(v) for v in y])
    probs1 = np.concatenate(probs1, axis=0)
    labels = np.array(labels)
    fpr, tpr, _ = roc_curve(labels, probs1[:, 1])
    prec, rec, _ = precision_recall_curve(labels, probs1[:, 1])
    del model
    torch.cuda.empty_cache()
    return fpr, tpr, prec, rec, labels, probs1[:, 1]


def combined_roc():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Chance")

    fpr_e, tpr_e, labels, probs_e = None, None, None, None
    # individual models (using pre-HPO checkpoints = the recommended ones)
    PRE_HPO_AUC = {"densenet121": 0.9420, "convnext_tiny": 0.9403, "vit_base": 0.9385}
    for i, (prep, arch, label) in enumerate(ARCHS):
        run_dir = REPO_ROOT / f"results/finetune_zenodo/checkpoints/{prep}/{arch}"
        fpr, tpr, _, _, _, _ = _predict_for_roc(run_dir)
        ax.plot(fpr, tpr, color=COLORS[arch], lw=1.5, alpha=0.7,
                label=f"{label}  (AUC={PRE_HPO_AUC[arch]:.3f})")

    # ensemble (probability average) -- use saved CSV from top3_pre_hpo
    ens_csv = REPO_ROOT / "results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.csv"
    ens_df = pd.read_csv(ens_csv)
    ens_probs = ens_df["prob_infected"].values
    ens_labels = ens_df["label"].values
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(ens_labels, ens_probs)
    metrics = _load_ensemble(PRE_HPO)
    ax.plot(fpr, tpr, color="crimson", lw=2.5, label=f"Top-3 ensemble  (AUC={metrics['test_auc_roc']:.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on Zenodo held-out test (n=1468)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_combined_roc.png", dpi=160)
    plt.close()
    print(f"[OK] combined_roc")


def combined_pr():
    fig, ax = plt.subplots(figsize=(6, 5))

    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(REPO_ROOT))
    from src.utils.config import load_config
    from src.utils.seed import set_seed
    from src.model.builder import build_model
    from src.training.checkpoint import load_checkpoint
    from src.data.zenodo_dataset import discover_zenodo_pairs, ZenodoDataset
    from src.preprocessing.preprocess import Preprocessor
    from torch.utils.data import DataLoader
    from sklearn.metrics import precision_recall_curve

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = discover_zenodo_pairs("data_external/test")
    labels = np.array([l for _, l in pairs])

    PRE_HPO_AP = {"densenet121": 0.85, "convnext_tiny": 0.81, "vit_base": 0.81}
    for i, (prep, arch, label) in enumerate(ARCHS):
        run_dir = REPO_ROOT / f"results/finetune_zenodo/checkpoints/{prep}/{arch}"
        cfg = load_config(str(run_dir / "config.yaml"))
        mc = dict(cfg["model"])
        mc["freeze_fraction"] = 0.0
        model = build_model(mc)
        load_checkpoint(model, str(run_dir / "best.pt"),
                        optimizer=None, scheduler=None, ema=None, device=device)
        model.to(device).eval()
        pre = Preprocessor(cfg["preprocessing"], input_size=mc.get("input_size", 224))
        ds = ZenodoDataset([p for p, _ in pairs], [l for _, l in pairs], pre, augment=False)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=False)
        ps = []
        with torch.no_grad():
            for x, _ in loader:
                logits = model(x.to(device))
                ps.append(F.softmax(logits, dim=1).cpu().numpy())
        per_probs = np.concatenate(ps, axis=0)[:, 1]
        prec, rec, _ = precision_recall_curve(labels, per_probs)
        ax.plot(rec, prec, color=COLORS[arch], lw=1.5, alpha=0.7,
                label=f"{label}  (AP={PRE_HPO_AP[arch]:.3f})")
        del model
        torch.cuda.empty_cache()

    # ensemble -- use saved CSV probabilities from top3_pre_hpo
    ens_csv = REPO_ROOT / "results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.csv"
    ens_df = pd.read_csv(ens_csv)
    ens_probs = ens_df["prob_infected"].values
    ens_labels = ens_df["label"].values
    prec, rec, _ = precision_recall_curve(ens_labels, ens_probs)
    metrics = _load_ensemble(PRE_HPO)
    ax.plot(rec, prec, color="crimson", lw=2.5, label=f"Top-3 ensemble  (AP={metrics['test_auc_pr']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall on Zenodo held-out test")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_combined_pr.png", dpi=160)
    plt.close()
    print(f"[OK] combined_pr")


def sweep_matrix_heatmap():
    sweep = pd.read_csv(REPO_ROOT / "results/finetune_zenodo/sweep_matrix.csv")
    pivot = sweep.pivot_table(
        index="arch", columns="preprocessing", values="test_auc_roc", aggfunc="first",
    )
    arch_order = ["resnet50", "resnet101", "densenet121", "densenet169",
                  "efficientnet_b0", "convnext_tiny", "mobilenetv3_large",
                  "vit_base", "swin_tiny"]
    pivot = pivot.reindex(arch_order)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0.85, vmax=0.95)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < 0.91 else "black", fontsize=9)
    ax.set_title("Zenodo fine-tune AUC across 18 architectures")
    plt.colorbar(im, ax=ax, label="Test AUC")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_sweep_matrix.png", dpi=160)
    plt.close()
    print(f"[OK] sweep_matrix_heatmap")


def ensemble_metrics_bar():
    pre = _load_ensemble(PRE_HPO)
    hpo = _load_ensemble(HPO)
    metrics = ["test_auc_roc", "test_f1", "test_mcc", "test_brier"]
    labels = ["AUC", "F1", "MCC", "Brier"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pre_vals = [pre[m] for m in metrics]
    hpo_vals = [hpo[m] for m in metrics]
    ax.bar(x - width / 2, pre_vals, width, label="Pre-HPO ensemble", color="#1f77b4")
    ax.bar(x + width / 2, hpo_vals, width, label="HPO-retrain ensemble", color="#ff7f0e")
    for i, (p, h) in enumerate(zip(pre_vals, hpo_vals)):
        ax.text(i - width / 2, p + 0.005, f"{p:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, h + 0.005, f"{h:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Pre-HPO vs HPO-retrain ensemble on Zenodo test")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_ensemble_metrics.png", dpi=160)
    plt.close()
    print(f"[OK] ensemble_metrics_bar")


def calibration_compare():
    # Plot ECE before/after for top-3 (use individual calibration JSONs)
    archs = ["densenet121", "convnext_tiny", "vit_base"]
    pre_ece = []
    post_ece = []
    Ts = []
    for arch in archs:
        if arch == "vit_base":
            ckdir = REPO_ROOT / "results/finetune_zenodo/checkpoints/gauss_nopad/vit_base"
        else:
            ckdir = REPO_ROOT / f"results/finetune_zenodo/checkpoints/srad_nopad/{arch}"
        c = json.loads((ckdir / "calibration/calibration_results.json").read_text())
        pre_ece.append(c["ece_held_before"])
        post_ece.append(c["ece_held_after_temp"])
        Ts.append(c["optimal_temperature"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(archs))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, pre_ece, width, label="Before T", color="#d62728")
    ax.bar(x + width / 2, post_ece, width, label="After T", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", "\\_") for a in archs])
    ax.set_ylabel("ECE (15 bins)")
    ax.set_title("Per-model ECE before/after temperature scaling")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (b, a) in enumerate(zip(pre_ece, post_ece)):
        ax.text(i - width / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)

    ax2 = axes[1]
    ax2.bar(archs, Ts, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for i, t in enumerate(Ts):
        ax2.text(i, t + 0.03, f"T={t:.2f}", ha="center", fontsize=10)
    ax2.set_ylabel("Temperature T")
    ax2.set_title("Optimal temperature (T > 1 → underconfident)")
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_calibration_compare.png", dpi=160)
    plt.close()
    print(f"[OK] calibration_compare")


def uncertainty_compare():
    archs = ["densenet121", "convnext_tiny", "vit_base"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = []
    for arch in archs:
        if arch == "vit_base":
            ckdir = REPO_ROOT / "results/finetune_zenodo/checkpoints/gauss_nopad/vit_base"
        else:
            ckdir = REPO_ROOT / f"results/finetune_zenodo/checkpoints/srad_nopad/{arch}"
        u = json.loads((ckdir / "uncertainty/mc_dropout_results.json").read_text())
        # Read from CSV if per_sample_entropy not in JSON
        if "per_sample_entropy" in u:
            ent = u["per_sample_entropy"]
        else:
            df = pd.read_csv(ckdir / "uncertainty/mc_dropout_predictions.csv")
            ent = df["predictive_entropy"].values
        summary.append((arch, ent))

    bins = np.linspace(0, 1.2, 40)
    for arch, ent in summary:
        ax.hist(ent, bins=bins, alpha=0.5, label=f"{arch}  (mean={np.mean(ent):.3f})",
                color=COLORS[arch])
    ax.set_xlabel("Predictive entropy (nats)")
    ax.set_ylabel("Sample count")
    ax.set_title("MC-Dropout uncertainty distribution (50 passes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_uncertainty_compare.png", dpi=160)
    plt.close()
    print(f"[OK] uncertainty_compare")


def cross_dataset_recovery():
    """Bar chart showing AUC recovery at each pipeline stage."""
    # Stage labels and AUC values
    stages = [
        "Figshare\n(padded)",
        "Figshare\n(no-pad)",
        "Zenodo\nfine-tune\n(mean 18)",
        "Zenodo\nfine-tune\n(top-3)",
        "Top-3\nensemble",
    ]
    aucs = [0.52, 0.77, 0.9313, 0.94, 0.9487]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    bars = ax.bar(stages, aucs, color=colors, edgecolor="black", linewidth=0.5)
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("External AUC (Zenodo PCOSgen test)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Cross-dataset generalization recovery across the PEARL pipeline")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="Random chance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_generalization_recovery.png", dpi=160)
    plt.close()
    print(f"[OK] cross_dataset_recovery")


def xai_grid():
    """3x3 grid: one Grad-CAM overlay per model."""
    from PIL import Image
    pairs = [
        ("results/finetune_zenodo/checkpoints/srad_nopad/densenet121/xai/gradcam/sample_0102_overlay.png",
         "densenet121 + SRAD-nopad"),
        ("results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny/xai/gradcam/sample_0031_overlay.png",
         "ConvNeXt-T + SRAD-nopad"),
        ("results/finetune_zenodo/checkpoints/gauss_nopad/vit_base/xai/gradcam/sample_0085_overlay.png",
         "ViT-B + Gauss-nopad"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, (path, label) in zip(axes, pairs):
        img = Image.open(REPO_ROOT / path)
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle("Grad-CAM attributions on correctly-classified infected cases",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "paper_fig_xai_grid.png", dpi=160)
    plt.close()
    print(f"[OK] xai_grid")


if __name__ == "__main__":
    sweep_matrix_heatmap()
    cross_dataset_recovery()
    calibration_compare()
    uncertainty_compare()
    ensemble_metrics_bar()
    combined_roc()
    combined_pr()
    xai_grid()
    print("\nDone. Figures in", FIG_DIR)