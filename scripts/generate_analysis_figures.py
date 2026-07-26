#!/usr/bin/env python3
"""Generate all matplotlib figures for the analysis section.

All figures are written to docs/analyhsis/figures/ as PNGs at 200 dpi.
This script is idempotent — re-running it overwrites any existing files.
"""

import os
import json
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


FIG_DIR = "docs/analyhsis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Consistent style across figures
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

# Architecture + preprocessing display order
ARCHES = [
    "resnet50", "resnet101", "densenet121", "densenet169",
    "efficientnet_b0", "convnext_tiny", "mobilenetv3_large",
    "vit_base", "swin_tiny",
]
PREPS = ["srad", "gauss"]
ARCH_LABEL = {
    "resnet50": "ResNet-50", "resnet101": "ResNet-101",
    "densenet121": "DenseNet-121", "densenet169": "DenseNet-169",
    "efficientnet_b0": "EfficientNet-B0", "convnext_tiny": "ConvNeXt-Tiny",
    "mobilenetv3_large": "MobileNetV3-L", "vit_base": "ViT-B/16", "swin_tiny": "Swin-T",
}
PREP_LABEL = {"srad": "SRAD", "gauss": "Gauss"}


# =====================================================================
# Load data
# =====================================================================
def load_internal():
    """Load figshare-test metrics from per-run final_metrics.json."""
    out = {}
    for prep in PREPS:
        for arch in ARCHES:
            f = f"results/ablation/checkpoints/{prep}/{arch}/final_metrics.json"
            if os.path.isfile(f):
                out[(prep, arch)] = json.load(open(f))
    return out


def load_with_preproc():
    out = {}
    for prep in PREPS:
        for arch in ARCHES:
            f = f"results/ablation/checkpoints/{prep}/{arch}/external_validation/pcosgen.json"
            if os.path.isfile(f):
                out[(prep, arch)] = json.load(open(f))
    return out


def load_noproc():
    out = {}
    for prep in PREPS:
        for arch in ARCHES:
            f = f"results/ablation/checkpoints/{prep}/{arch}/external_validation_noproc/pcosgen_noproc.json"
            if os.path.isfile(f):
                out[(prep, arch)] = json.load(open(f))
    return out


internal = load_internal()
with_preproc = load_with_preproc()
noproc = load_noproc()


# =====================================================================
# Helpers
# =====================================================================
def get(rows, prep, arch, key, default=None):
    d = rows.get((prep, arch), {})
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def grouped_bars(rows, metric_key, title, ylabel, fname, *,
                 ylim=None, lower_better=False, fmt="{:.3f}",
                 external_only=False):
    """Two bars per architecture (srad, gauss) for a single metric."""
    x = np.arange(len(ARCHES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.2))

    for i, prep in enumerate(PREPS):
        vals = [get(rows, prep, a, metric_key, np.nan) for a in ARCHES]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=PREP_LABEL[prep],
                      color="#d77a61" if prep == "srad" else "#5b8fb8",
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.005 if not lower_better else -0.01),
                        fmt.format(v), ha="center", va="bottom" if not lower_better else "top",
                        fontsize=7, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHES], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3)
    if "external" in title.lower() or "PCOSgen" in title:
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5,
                   label="random AUC" if "AUC" in ylabel else None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Figure 1: Internal figshare-test AUC across all 18 runs
# =====================================================================
grouped_bars(
    internal, "test_auc_roc",
    "Internal test set (Figshare PCOS, n=406): AUC is saturated across all architectures",
    "Test AUC-ROC", "fig1_internal_auc.png",
    ylim=(0.998, 1.0005),
)
print("✓ fig1_internal_auc.png")

# =====================================================================
# Figure 2: External PCOSgen AUC across all 18 runs
# =====================================================================
grouped_bars(
    with_preproc, "test_auc_roc",
    "External (PCOSgen, n=4668): AUC collapses to near-random for every architecture",
    "Test AUC-ROC", "fig2_external_auc.png",
    ylim=(0.40, 0.62),
)
print("✓ fig2_external_auc.png")

# =====================================================================
# Figure 3: External specificity — the "predict-everything-as-PCOS" pattern
# =====================================================================
grouped_bars(
    with_preproc, "test_specificity",
    "Specificity on PCOSgen: every model predicts >93% of images as PCOS",
    "Specificity (true-negative rate)", "fig3_external_specificity.png",
    ylim=(0, 0.10),
)
print("✓ fig3_external_specificity.png")

# =====================================================================
# Figure 4: External MCC — the truthful single-number summary
# =====================================================================
grouped_bars(
    with_preproc, "test_mcc",
    "MCC on PCOSgen: effectively zero discrimination for every (prep × arch)",
    "Matthews Correlation Coefficient", "fig4_external_mcc.png",
    ylim=(-0.05, 0.10),
)
print("✓ fig4_external_mcc.png")

# =====================================================================
# Figure 5: Internal vs external AUC — paired drop
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(ARCHES))
width = 0.27
for i, prep in enumerate(PREPS):
    int_vals = [get(internal, prep, a, "test_auc_roc", np.nan) for a in ARCHES]
    ext_vals = [get(with_preproc, prep, a, "test_auc_roc", np.nan) for a in ARCHES]
    color = "#d77a61" if prep == "srad" else "#5b8fb8"
    ax.bar(x + (i - 0.5) * width, int_vals, width, label=f"{PREP_LABEL[prep]} — internal",
           color=color, edgecolor="black", linewidth=0.4, alpha=0.95)
    ax.bar(x + (i - 0.5) * width + 0.005, ext_vals, width, label=f"{PREP_LABEL[prep]} — external",
           color=color, edgecolor="black", linewidth=0.4, alpha=0.35, hatch="//")
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHES], rotation=25, ha="right")
ax.set_ylabel("AUC")
ax.set_ylim(0.3, 1.02)
ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="random AUC")
ax.set_title("Internal vs external AUC: 0.999 → 0.45–0.59 for every (prep × arch)")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_internal_vs_external_paired.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig5_internal_vs_external_paired.png")

# =====================================================================
# Figure 6: Drop distribution histogram
# =====================================================================
deltas = []
for prep in PREPS:
    for arch in ARCHES:
        i = get(internal, prep, arch, "test_auc_roc")
        e = get(with_preproc, prep, arch, "test_auc_roc")
        if i is not None and e is not None:
            deltas.append((prep, arch, i - e, i, e))
fig, ax = plt.subplots(figsize=(8, 4.2))
deltas_vals = [d for _, _, d, _, _ in deltas]
ax.hist(deltas_vals, bins=12, color="#9b7fb8", edgecolor="black", linewidth=0.5)
ax.axvline(np.mean(deltas_vals), color="red", linestyle="--",
           label=f"mean Δ = {np.mean(deltas_vals):.3f}")
ax.axvline(np.median(deltas_vals), color="black", linestyle=":",
           label=f"median Δ = {np.median(deltas_vals):.3f}")
ax.set_xlabel("Internal AUC − External AUC")
ax.set_ylabel("Number of (prep × arch) configurations")
ax.set_title(f"Distribution of AUC drop across {len(deltas)} runs (every run drops 0.41–0.55)")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_drop_distribution.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig6_drop_distribution.png")

# =====================================================================
# Figure 7: With-preproc vs no-preproc scatter
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 6.0))
colors = {"srad": "#d77a61", "gauss": "#5b8fb8"}
markers = {"srad": "o", "gauss": "s"}
for prep in PREPS:
    xs, ys = [], []
    for arch in ARCHES:
        w = get(with_preproc, prep, arch, "test_auc_roc")
        n = get(noproc, prep, arch, "test_auc_roc")
        if w is not None and n is not None:
            xs.append(n); ys.append(w)
            ax.scatter(n, w, c=colors[prep], marker=markers[prep],
                       s=80, edgecolor="black", linewidth=0.5, alpha=0.85)
            ax.annotate(ARCH_LABEL[arch].split("-")[0], (n, w),
                        fontsize=7, alpha=0.7, xytext=(3, 3),
                        textcoords="offset points")
# y=x reference line
lim = (0.30, 0.65)
ax.plot(lim, lim, color="grey", linestyle="--", alpha=0.5, label="y = x")
ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4)
ax.axvline(0.5, color="grey", linestyle=":", alpha=0.4)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("AUC with preprocessing disabled (resize + ImageNet normalize)")
ax.set_ylabel("AUC with original srad/gauss preprocessing")
ax.set_title("Preprocessing does not close the OOD gap\n(slight mean improvement, no architecture crosses 0.59)")
# legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["srad"],
           markeredgecolor="black", markersize=9, label="srad"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=colors["gauss"],
           markeredgecolor="black", markersize=9, label="gauss"),
    Line2D([0], [0], color="grey", linestyle="--", label="y = x"),
]
ax.legend(handles=legend_elements, frameon=False, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig7_with_vs_noproc_scatter.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig7_with_vs_noproc_scatter.png")

# =====================================================================
# Figure 8: Per-arch Δ preprocessing (helps / hurts bar)
# =====================================================================
diff_rows = []
for prep in PREPS:
    for arch in ARCHES:
        w = get(with_preproc, prep, arch, "test_auc_roc")
        n = get(noproc, prep, arch, "test_auc_roc")
        if w is not None and n is not None:
            diff_rows.append((prep, arch, w - n))
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(ARCHES))
width = 0.38
for i, prep in enumerate(PREPS):
    vals = [next((d for p, a, d in diff_rows if p == prep and a == arch), 0) for arch in ARCHES]
    bars = ax.bar(x + (i - 0.5) * width, vals, width,
                  label=PREP_LABEL[prep],
                  color="#d77a61" if prep == "srad" else "#5b8fb8",
                  edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.003 if v >= 0 else -0.003),
                f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHES], rotation=25, ha="right")
ax.set_ylabel("Δ AUC (with preprocessing − no preprocessing)")
ax.set_title("Effect of preprocessing on PCOSgen AUC: marginal, mixed sign")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig8_delta_preprocessing.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig8_delta_preprocessing.png")

# =====================================================================
# Figure 9: External scatter — Specificity vs Sensitivity
# =====================================================================
fig, ax = plt.subplots(figsize=(7.5, 6.0))
for prep in PREPS:
    xs, ys = [], []
    for arch in ARCHES:
        sens = get(with_preproc, prep, arch, "test_recall")
        spec = get(with_preproc, prep, arch, "test_specificity")
        if sens is not None and spec is not None:
            xs.append(1 - spec); ys.append(sens)
    ax.scatter(xs, ys, c=colors[prep], marker=markers[prep],
               s=90, edgecolor="black", linewidth=0.5, alpha=0.85,
               label=PREP_LABEL[prep])
ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4)
ax.axvline(0.5, color="grey", linestyle=":", alpha=0.4)
ax.set_xlim(0, 1.0); ax.set_ylim(0.5, 1.0)
ax.set_xlabel("False-Positive Rate (1 − Specificity)")
ax.set_ylabel("True-Positive Rate (Sensitivity)")
ax.set_title("ROC operating point on PCOSgen: every model is in the\ntop-right corner (high TPR, very high FPR)")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig9_sens_vs_fpr.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig9_sens_vs_fpr.png")

# =====================================================================
# Figure 10: Balanced accuracy — internal vs external
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(ARCHES))
width = 0.38
for i, prep in enumerate(PREPS):
    int_bal = []
    ext_bal = []
    for arch in ARCHES:
        i_sens = get(internal, prep, arch, "test_recall")
        i_spec = get(internal, prep, arch, "test_specificity")
        e_sens = get(with_preproc, prep, arch, "test_recall")
        e_spec = get(with_preproc, prep, arch, "test_specificity")
        int_bal.append((i_sens + i_spec) / 2 if i_sens and i_spec is not None else np.nan)
        ext_bal.append((e_sens + e_spec) / 2 if e_sens and e_spec is not None else np.nan)
    color = "#d77a61" if prep == "srad" else "#5b8fb8"
    ax.bar(x + (i - 0.5) * width, int_bal, width, label=f"{PREP_LABEL[prep]} — internal",
           color=color, edgecolor="black", linewidth=0.4)
    ax.bar(x + (i - 0.5) * width, ext_bal, width, label=f"{PREP_LABEL[prep]} — external",
           color=color, edgecolor="black", linewidth=0.4, alpha=0.35, hatch="//")
ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="random")
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHES], rotation=25, ha="right")
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0.4, 1.02)
ax.set_title("Balanced accuracy: internal ≈ 0.99 vs external ≈ 0.50")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig10_balanced_accuracy.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig10_balanced_accuracy.png")

# =====================================================================
# Figure 11: Internal-test sample size + class balance (data context)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 3.5))
splits = ["train", "val", "test"]
infected = [2549, 310, 325]
noninfected = [650, 81, 81]
x = np.arange(len(splits))
width = 0.38
ax.bar(x - width/2, infected, width, label="PCOS (infected)", color="#d77a61", edgecolor="black", linewidth=0.4)
ax.bar(x + width/2, noninfected, width, label="non-PCOS", color="#5b8fb8", edgecolor="black", linewidth=0.4)
for i, (inf, non) in enumerate(zip(infected, noninfected)):
    ax.text(i - width/2, inf + 30, str(inf), ha="center", fontsize=9)
    ax.text(i + width/2, non + 30, str(non), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(n={inf+non})" for s, inf, non in zip(splits, infected, noninfected)])
ax.set_ylabel("Number of images")
ax.set_title("Figshare PCOS splits: imbalanced 3.9:1 (PCOS:non-PCOS), test set has 81 negatives")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig11_internal_split_composition.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig11_internal_split_composition.png")

# =====================================================================
# Figure 12: PCOSgen test composition
# =====================================================================
fig, ax = plt.subplots(figsize=(6, 3.5))
n_infected = 3627
n_healthy = 1041
ax.bar([0], [n_infected], color="#d77a61", edgecolor="black", linewidth=0.4, label="PCOS")
ax.bar([1], [n_healthy], color="#5b8fb8", edgecolor="black", linewidth=0.4, label="non-PCOS")
ax.text(0, n_infected + 50, str(n_infected), ha="center", fontsize=11, fontweight="bold")
ax.text(1, n_healthy + 50, str(n_healthy), ha="center", fontsize=11, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["PCOS (infected)", "non-PCOS (healthy)"])
ax.set_ylabel("Image count")
ax.set_title(f"PCOSgen external cohort: n=4668, 3.5:1 imbalance ({(n_infected/(n_infected+n_healthy))*100:.1f}% PCOS)")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 4200)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig12_external_split_composition.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig12_external_split_composition.png")


# =====================================================================
# Figure 13: Image-size comparison (visual / acquisition proxy)
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
figshare_sizes = []
for f in os.listdir("data/infected"):
    p = os.path.join("data/infected", f)
    if os.path.isfile(p) and f.lower().endswith((".jpg", ".png")):
        try:
            import cv2
            img = cv2.imread(p)
            if img is not None:
                figshare_sizes.append((img.shape[1], img.shape[0]))  # (w, h)
        except Exception:
            pass
pcosgen_sizes = []
for fn in os.listdir("data_external/pcosgen/PCOSGen-train/train")[:2]:
    sub = os.path.join("data_external/pcosgen/PCOSGen-train/train", fn)
    if os.path.isdir(sub):
        for f in os.listdir(sub)[:200]:
            p = os.path.join(sub, f)
            if os.path.isfile(p) and f.lower().endswith((".jpg", ".png")):
                try:
                    img = cv2.imread(p)
                    if img is not None:
                        pcosgen_sizes.append((img.shape[1], img.shape[0]))
                except Exception:
                    pass

if figshare_sizes and pcosgen_sizes:
    ax.scatter([w for w, h in figshare_sizes], [h for w, h in figshare_sizes],
               s=8, alpha=0.4, c="#5b8fb8", label="Figshare (n=20 sampled)")
    ax.scatter([w for w, h in pcosgen_sizes], [h for w, h in pcosgen_sizes],
               s=8, alpha=0.4, c="#d77a61", label=f"PCOSgen (n={len(pcosgen_sizes)} sampled)")
    ax.set_xlabel("Image width (pixels)")
    ax.set_ylabel("Image height (pixels)")
    ax.set_title(f"Raw image resolution: Figshare {np.median([w for w,h in figshare_sizes]):.0f}×{np.median([h for w,h in figshare_sizes]):.0f} median vs "
                 f"PCOSgen {np.median([w for w,h in pcosgen_sizes]):.0f}×{np.median([h for w,h in pcosgen_sizes]):.0f} median")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig13_image_resolution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig13_image_resolution.png")

# =====================================================================
# Figure 14: Per-architecture external AUC rank (best-to-worst)
# =====================================================================
all_ext = []
for prep in PREPS:
    for arch in ARCHES:
        a = get(with_preproc, prep, arch, "test_auc_roc")
        if a is not None:
            all_ext.append((f"{PREP_LABEL[prep]}/{ARCH_LABEL[arch]}", a, prep))
all_ext.sort(key=lambda x: -x[1])
labels = [r[0] for r in all_ext]
vals = [r[1] for r in all_ext]
colors = ["#d77a61" if p == "srad" else "#5b8fb8" for _, _, p in all_ext]
fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(range(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5, label="random AUC")
ax.set_xlabel("External PCOSgen AUC")
ax.set_title("External AUC rank (best to worst): all 18 configurations span 0.137 around chance")
ax.legend(frameon=False)
ax.grid(axis="x", alpha=0.3)
for i, v in enumerate(vals):
    ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig14_external_auc_rank.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig14_external_auc_rank.png")

# =====================================================================
# Figure 15: Visual sanity check side-by-side (single composite figure)
# =====================================================================
fig, axes = plt.subplots(3, 4, figsize=(14, 10))

probe_files = [
    ("Image_001.jpg", "Figshare\ninfected"),
    ("Image_001.jpg", None),  # placeholder
    ("healthy_100image13.jpg", "PCOSgen\nhealthy"),
    ("infect_100image71.jpg", "PCOSgen\ninfected"),
]

# Special: figshare has ONE image, just put it in row 0 col 0
ax_layout = [
    # (row, col, src_file, label)
    (0, 0, "Image_001.jpg", "Raw (Figshare/infected)"),
    (0, 1, "Image_001.jpg", "SRAD preproc"),
    (0, 2, "Image_001.jpg", "Gauss preproc"),
    (0, 3, "Image_001.jpg", "No preproc"),
    (1, 0, "healthy_100image13.jpg", "Raw (PCOSgen/healthy)"),
    (1, 1, "healthy_100image13.jpg", "SRAD preproc"),
    (1, 2, "healthy_100image13.jpg", "Gauss preproc"),
    (1, 3, "healthy_100image13.jpg", "No preproc"),
    (2, 0, "infect_100image71.jpg", "Raw (PCOSgen/infected)"),
    (2, 1, "infect_100image71.jpg", "SRAD preproc"),
    (2, 2, "infect_100image71.jpg", "Gauss preproc"),
    (2, 3, "infect_100image71.jpg", "No preproc"),
]


def load_image_for_display(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_preproc_for_display(path):
    # load npy, percentile-stretch, convert to RGB uint8
    a = np.load(path)
    # The preprocessor saves HWC float32 in [-1, +3] range (already normalized).
    # We percentile-stretch for visual display — no need to denormalize.
    lo, hi = np.percentile(a, [1, 99])
    stretched = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if stretched.ndim == 3 and stretched.shape[-1] == 3:
        stretched = stretched[..., ::-1]  # BGR -> RGB (preprocessor saved BGR via cv2)
    return stretched


for row, col, fname, label in ax_layout:
    ax = axes[row, col]
    if fname == "Image_001.jpg":
        src = f"results/probe_preproc/raw_side_by_side/{fname}"
    else:
        src = f"results/probe_preproc/raw_side_by_side/{fname}"
    if "Raw" in label:
        img = load_image_for_display(src)
    elif "SRAD" in label:
        base = os.path.splitext(fname)[0]
        img = load_preproc_for_display(f"results/probe_preproc/proc_srad/{base}.npy")
    elif "Gauss" in label:
        base = os.path.splitext(fname)[0]
        img = load_preproc_for_display(f"results/probe_preproc/proc_gauss/{base}.npy")
    elif "No preproc" in label:
        # No preproc isn't a single file from that dir — reapply from raw
        import cv2, sys
        raw = cv2.imread(f"results/probe_preproc/raw_side_by_side/{fname}")
        if raw is not None:
            sys.path.insert(0, "scripts")
            from eval_external_noproc import noproc_apply
            np_img = noproc_apply(raw, 224)  # HWC float32 in [-1, +3]
            lo, hi = np.percentile(np_img, [1, 99])
            img = np.clip((np_img - lo) / max(hi - lo, 1e-6), 0, 1)
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img[..., ::-1]  # BGR -> RGB
        else:
            img = None
    if img is not None:
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=9)

plt.suptitle("Visual sanity check: preprocessing preserves ultrasound anatomy on both datasets",
             fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig15_visual_sanity_check.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig15_visual_sanity_check.png")

# =====================================================================
# Figure 16: Noproc AUC bar (companion to fig2)
# =====================================================================
grouped_bars(
    noproc, "test_auc_roc",
    "External (PCOSgen, n=4668) — preprocessing disabled: same collapse pattern",
    "Test AUC-ROC", "fig16_external_noproc_auc.png",
    ylim=(0.30, 0.62),
)
print("✓ fig16_external_noproc_auc.png")

# =====================================================================
# Figure 17: Confusion matrix from PCOSgen cross-check (existence proof)
# =====================================================================
fig, ax = plt.subplots(figsize=(5, 5))
cm = np.array([[41, 862], [69, 2228]])  # TN, FP / FN, TP
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred healthy", "Pred PCOS"])
ax.set_yticklabels(["True healthy", "True PCOS"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True (from PCOSgen CSV)")
ax.set_title("Confusion matrix (srad/resnet50, ground truth from PCOSgen CSV, n=3200)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.grid(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig17_confusion_matrix_csv_check.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig17_confusion_matrix_csv_check.png")

# =====================================================================
# Figure 18: All-metric summary heatmap (compact overview)
# =====================================================================
metrics_to_plot = [
    ("test_auc_roc", "AUC"),
    ("test_f1", "F1"),
    ("test_mcc", "MCC"),
    ("test_accuracy", "Accuracy"),
    ("test_recall", "Sensitivity"),
    ("test_specificity", "Specificity"),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for idx, (k, title) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    matrix = np.zeros((len(PREPS), len(ARCHES)))
    for i, prep in enumerate(PREPS):
        for j, arch in enumerate(ARCHES):
            v = get(with_preproc, prep, arch, k)
            matrix[i, j] = v if v is not None else np.nan
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ARCHES)))
    ax.set_xticklabels([ARCH_LABEL[a].split("-")[0] for a in ARCHES], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(PREPS)))
    ax.set_yticklabels([PREP_LABEL[p] for p in PREPS])
    ax.set_title(f"{title} on PCOSgen", fontsize=10)
    for i in range(len(PREPS)):
        for j in range(len(ARCHES)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle("All-metric heatmap on PCOSgen — every cell is failing", fontsize=12, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig18_metric_heatmap.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("✓ fig18_metric_heatmap.png")

print("\nAll figures written to docs/analyhsis/figures/")
print(f"Total: {len([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])} PNGs")
