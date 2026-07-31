#!/usr/bin/env python3
"""Generate publication figures for the shortcut-hypothesis section.

Reads every per-run shortcut_summary.json and per_image_gradcam.csv under
results/ablation/checkpoints/{srad,gauss}/<arch>/shortcut_audit/ and
emits 4 figures into docs/analysis/figures/:

  fig19_border_attention_per_class.png
      Per-architecture median border-attention fraction per class, with
      the 0.187 uniform-expected reference line.

  fig20_shortcut_signal_heatmap.png
      Per-(architecture, class) heatmap of border_attention_median
      normalized against the 0.187 uniform expectation.

  fig21_peak_location_distribution.png
      Peak-CAM-region distribution (top/bottom/left/right/center) per
      class, pooled across all 18 runs.

  fig22_example_panels.png
      Six representative Grad-CAM panels (one per "kind" of behaviour)
      chosen from the densenet121/srad audit: three PCOS-positive with
      heavy border attention, and three PCOS-positive with low border
      attention.

Run:  python scripts/generate_shortcut_figures.py
"""
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


# Output config
OUT_DIR = "docs/analysis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

ARCHES = ['swin_tiny', 'vit_base', 'convnext_tiny',
          'densenet169', 'efficientnet_b0']
PREPS = ['srad', 'gauss']
UNIFORM_BORDER = 0.187  # 5% strip on 224x224 -> ~18.7% area


def load_all():
    runs = []
    for prep in PREPS:
        for arch in ARCHES:
            p = f'results/ablation/checkpoints/{prep}/{arch}/shortcut_audit/shortcut_summary.json'
            if not os.path.isfile(p):
                continue
            s = json.load(open(p))
            s['prep'] = prep
            s['arch'] = arch
            s['csv_path'] = p.replace('shortcut_summary.json', 'per_image_gradcam.csv')
            runs.append(s)
    return runs


def fig19_border_per_class(runs):
    """Per-(prep, arch) median border attention, per class."""
    arches = list(dict.fromkeys(r['arch'] for r in runs))
    preps = list(dict.fromkeys(r['prep'] for r in runs))
    # 2 rows (preps) x 9 cols (arches)
    fig, axes = plt.subplots(len(preps), len(arches),
                             figsize=(2.0 * len(arches), 1.6 * len(preps)),
                             sharey=True)
    if len(preps) == 1:
        axes = [axes]
    fig.suptitle("Border attention fraction (5% outer strip)  "
                 "vs uniform expectation 18.7%", fontsize=12)
    for r_idx, prep in enumerate(preps):
        for c_idx, arch in enumerate(arches):
            ax = axes[r_idx][c_idx] if len(preps) > 1 else axes[c_idx]
            run = next((r for r in runs if r['prep'] == prep and r['arch'] == arch), None)
            if run is None:
                ax.text(0.5, 0.5, '—', ha='center', va='center', color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                if c_idx == 0:
                    ax.set_ylabel(prep, fontsize=10)
                continue
            bm = run['border_mass_frac_median']
            ax.bar(['class 0\n(healthy)', 'class 1\n(PCOS)'],
                   [bm.get('0', 0), bm.get('1', 0)],
                   color=['#4c956c', '#d63d3d'], alpha=0.85)
            ax.axhline(UNIFORM_BORDER, color='gray', linestyle=':', linewidth=1)
            ax.set_ylim(0, max(0.35, max(bm.get('0', 0), bm.get('1', 0)) + 0.05))
            if r_idx == 0:
                ax.set_title(arch, fontsize=8, rotation=20)
            if c_idx == 0:
                ax.set_ylabel(prep, fontsize=10)
            ax.tick_params(axis='both', labelsize=7)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "fig19_border_attention_per_class.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig19] -> {out}")


def fig20_shortcut_heatmap(runs):
    """Per-(arch, prep, class) heatmap of border attention with diff vs uniform."""
    arches = list(dict.fromkeys(r['arch'] for r in runs))
    preps = list(dict.fromkeys(r['prep'] for r in runs))
    M = np.zeros((len(arches), len(preps) * 2))
    labels = []
    for j, prep in enumerate(preps):
        labels.append(f"{prep}/c0")
        labels.append(f"{prep}/c1")
    for i, arch in enumerate(arches):
        for j, prep in enumerate(preps):
            r = next((r for r in runs if r['arch']==arch and r['prep']==prep), None)
            if r is None: continue
            bm = r['border_mass_frac_median']
            M[i, 2*j + 0] = bm.get('0', 0) - UNIFORM_BORDER
            M[i, 2*j + 1] = bm.get('1', 0) - UNIFORM_BORDER
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(arches) + 2))
    cmap = LinearSegmentedColormap.from_list(
        'rdgn', ['#1a9850', '#a6d96a', '#fdae61', '#d73027'])
    vmax = max(0.10, np.abs(M).max())
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_yticks(range(len(arches)))
    ax.set_yticklabels(arches, fontsize=9)
    for i in range(len(arches)):
        for j in range(len(labels)):
            v = M[i, j]
            txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if abs(v) > 0.05 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, label='Δ vs uniform (0.187)')
    ax.set_title("Border attention deviation from uniform (positive = uses border)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig20_shortcut_signal_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig20] -> {out}")


def fig21_peak_region_dist(runs):
    """Peak-region histogram per class, pooled across runs."""
    rows = []
    for r in runs:
        with open(r['csv_path']) as f:
            for row in csv.DictReader(f):
                row['arch'] = r['arch']; row['prep'] = r['prep']
                rows.append(row)
    def peak_region(r):
        y=int(r['peak_y']); x=int(r['peak_x'])
        b = 11  # 5% of 224
        if y < b: return 'TOP'
        if y > 224-b: return 'BOTTOM'
        if x < b: return 'LEFT'
        if x > 224-b: return 'RIGHT'
        return 'CENTER'
    cls_region = defaultdict(Counter)
    cls_border = defaultdict(lambda: [0, 0])
    for r in rows:
        c = int(r['true_label'])
        cls_region[c][peak_region(r)] += 1
        cls_border[c][int(r['is_border_peak']=='True')] += 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    regions = ['TOP', 'BOTTOM', 'LEFT', 'RIGHT', 'CENTER']
    colors = ['#d63d3d', '#f4a460', '#d63d3d', '#f4a460', '#4c956c']
    for ax_idx, c in enumerate((0, 1)):
        ax = axes[ax_idx]
        n = sum(cls_region[c].values())
        counts = [cls_region[c].get(k, 0) for k in regions]
        bars = ax.bar(regions, counts, color=colors)
        for k, b in zip(regions, bars):
            pct = 100 * counts[regions.index(k)] / n
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f"{pct:.1f}%", ha='center', fontsize=9)
        bp = cls_border[c][1] / n * 100
        ax.set_title(f"{'non-PCOS' if c==0 else 'PCOS-positive'}  "
                     f"(border-peak={bp:.1f}%, n={n})")
        ax.set_ylabel("# images (pooled across 18 runs)")
        ax.set_ylim(0, max(counts) * 1.15)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle("Peak Grad-CAM activation location per class  "
                 "(pooled over 1,440 image-level maps)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "fig21_peak_location_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig21] -> {out}")


def fig22_example_panels():
    """Six-panel figure: 3 obvious-shortcut + 3 honest-attention cases."""
    # We pick from densenet121/srad — it had the strongest shortcut signal
    src_dir = "results/ablation/checkpoints/srad/densenet121/shortcut_audit/panels"
    rows = list(csv.DictReader(open(
        'results/ablation/checkpoints/srad/densenet121/shortcut_audit/per_image_gradcam.csv')))
    # Keep only samples for which a panel was saved (the audit intentionally
    # saves the first 24 sampled panels, not all 80, to keep output compact).
    available = {
        int(os.path.splitext(os.path.basename(p))[0].split('_')[1])
        for p in glob.glob(os.path.join(src_dir, 'sample_*.png'))
    }
    inf_rows = [r for r in rows
                if r['true_label']=='1' and r['correct']=='1'
                and int(r['sample_id']) in available]
    inf_rows.sort(key=lambda r: -float(r['border_mass_frac']))
    # 3 panels showing strong border attention
    pick_shortcut = inf_rows[:3]
    # 3 panels showing weaker (but still correct) border attention
    honest_pool = [r for r in inf_rows if float(r['border_mass_frac']) < 0.12]
    pick_honest = honest_pool[-3:] if len(honest_pool) >= 3 else inf_rows[-3:]

    from PIL import Image
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, (r, tag) in enumerate(zip(pick_shortcut, ['strong #1', 'strong #2', 'strong #3'])):
        img = Image.open(os.path.join(src_dir, f"sample_{int(r['sample_id']):04d}.png"))
        axes[0, col].imshow(img); axes[0, col].axis('off')
        axes[0, col].set_title(f"PCOS — shortcut ({tag})\n"
                               f"border_frac={float(r['border_mass_frac']):.2f}, "
                               f"p(P)={float(r['prob_infected']):.2f}",
                               fontsize=9, color='red')
    for col, (r, tag) in enumerate(zip(pick_honest, ['honest #1', 'honest #2', 'honest #3'])):
        img = Image.open(os.path.join(src_dir, f"sample_{int(r['sample_id']):04d}.png"))
        axes[1, col].imshow(img); axes[1, col].axis('off')
        axes[1, col].set_title(f"PCOS — honest ({tag})\n"
                               f"border_frac={float(r['border_mass_frac']):.2f}, "
                               f"p(P)={float(r['prob_infected']):.2f}",
                               fontsize=9, color='green')
    fig.suptitle("densenet121 / SRAD — Grad-CAM on correctly-classified PCOS test images",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, "fig22_example_shortcut_panels.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig22] -> {out}")


def main():
    runs = load_all()
    print(f"Loaded {len(runs)} audit summaries")
    fig19_border_per_class(runs)
    fig20_shortcut_heatmap(runs)
    fig21_peak_region_dist(runs)
    fig22_example_panels()


if __name__ == "__main__":
    main()
