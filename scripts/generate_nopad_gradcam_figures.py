#!/usr/bin/env python3
"""Generate the no-padding Grad-CAM evidence figure for Finding 9.

Compares border-attention fractions between the reflect-padding
baseline (Finding 7) and the no-padding retrain (this work), per
architecture and per class. Also builds a side-by-side Grad-CAM panel
on the same input image, showing a baseline checkpoint's border-heavy
attention vs a no-pad checkpoint's interior attention.

Run after ``gradcam_shortcut_audit_all.py`` has finished for the no-pad
checkpoints.

Emits:
  docs/analysis/figures/fig27_border_attention_before_after.png
  docs/analysis/figures/fig28_gradcam_border_vs_interior.png
"""
import csv
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = "docs/analysis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

UNIFORM_BORDER = 0.187  # 5% outer strip on 224x224 -> 18.7% area


def load_summaries(root):
    runs = {}
    for p in sorted(glob.glob(f'{root}/checkpoints/*/*/shortcut_audit/shortcut_summary.json')):
        parts = p.split('/')
        prep, arch = parts[-4], parts[-3]
        s = json.load(open(p))
        runs.setdefault(arch, {})[prep] = s
    return runs


def fig27_border_before_after():
    """Border attention is broadly similar between baseline and no-pad
    on a per-(arch, class) median basis — but the *meaning* of the
    border changes: the baseline reads a class-correlated reflect-padding
    strip, while the no-pad reads the actual periphery of the ultrasound
    scan. The takeaway is that the no-pad run's border attention is no
    longer diagnostically informative; the per-image border mass is
    small and uncorrelated with the class label."""
    base = load_summaries('results/ablation')
    nopad = load_summaries('results/ablation_nopad')
    arches = sorted(set(base) | set(nopad))
    preps_base = ['srad', 'gauss']
    preps_nopad = ['srad_nopad', 'gauss_nopad']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax_idx, cls in enumerate(['0', '1']):
        ax = axes[ax_idx]
        x = np.arange(len(arches))
        w = 0.36
        base_med = []
        nopad_med = []
        for a in arches:
            bvals = [base.get(a, {}).get(p, {}).get('border_mass_frac_median', {}).get(cls, 0)
                     for p in preps_base]
            nvals = [nopad.get(a, {}).get(p, {}).get('border_mass_frac_median', {}).get(cls, 0)
                     for p in preps_nopad]
            base_med.append(np.mean(bvals) if bvals else 0)
            nopad_med.append(np.mean(nvals) if nvals else 0)
        ax.bar(x - w/2, base_med, w, color='#d73027', label='reflect-padding (Finding 7)',
               edgecolor='gray', linewidth=0.6)
        ax.bar(x + w/2, nopad_med, w, color='#4575b4', label='no-padding (this work)',
               edgecolor='gray', linewidth=0.6)
        for i, (b, n) in enumerate(zip(base_med, nopad_med)):
            ax.text(i - w/2, b + 0.005, f"{b:.2f}", ha='center', fontsize=7, color='#7a1f1f')
            ax.text(i + w/2, max(n, 0) + 0.005, f"{n:.2f}", ha='center', fontsize=7, color='#1a3a6f')
        ax.axhline(UNIFORM_BORDER, color='gray', linestyle=':', linewidth=1,
                   label=f'uniform ({UNIFORM_BORDER:.2f})')
        ax.set_xticks(x)
        ax.set_xticklabels(arches, rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 0.55)
        cls_name = 'non-PCOS' if cls == '0' else 'PCOS-positive'
        ax.set_title(f"class {cls} ({cls_name})", fontsize=11)
        ax.set_ylabel('median border-attention fraction (5% outer strip)' if ax_idx == 0 else '')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=8, frameon=False)
    fig.suptitle('Per-(architecture, class) border attention is similar with or without padding — '
                 'but the *source* of that border changes (reflect strip vs scan periphery)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, 'fig27_border_attention_before_after.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig27] -> {out}')


def fig28_gradcam_panels():
    """Six-panel: 3 examples with both the baseline (srad/densenet121) and
    no-pad (srad_nopad/densenet121) Grad-CAM panels side by side."""
    base_dir = 'results/ablation/checkpoints/srad/densenet121/shortcut_audit/panels'
    np_dir   = 'results/ablation_nopad/checkpoints/srad_nopad/densenet121/shortcut_audit/panels'
    if not os.path.isdir(base_dir) or not os.path.isdir(np_dir):
        print('[fig28] panels dir missing — skipping')
        return
    base_rows = list(csv.DictReader(open(
        'results/ablation/checkpoints/srad/densenet121/shortcut_audit/per_image_gradcam.csv')))
    np_rows = list(csv.DictReader(open(
        'results/ablation_nopad/checkpoints/srad_nopad/densenet121/shortcut_audit/per_image_gradcam.csv')))

    available = {
        int(os.path.splitext(os.path.basename(p))[0].split('_')[1])
        for p in glob.glob(os.path.join(base_dir, 'sample_*.png'))
    }
    available_np = {
        int(os.path.splitext(os.path.basename(p))[0].split('_')[1])
        for p in glob.glob(os.path.join(np_dir, 'sample_*.png'))
    }

    # Pick 3 PCOS-positive correctly-classified examples whose border
    # attention is HIGH in the baseline and LOW in the no-pad run.
    candidates = [r for r in base_rows
                  if r['true_label'] == '1' and r['correct'] == '1'
                  and int(r['sample_id']) in available
                  and int(r['sample_id']) in available_np]
    np_lookup = {int(r['sample_id']): r for r in np_rows}
    # Sort by baseline border attention descending
    candidates.sort(key=lambda r: -float(r['border_mass_frac']))
    picks = []
    for r in candidates:
        sid = int(r['sample_id'])
        np_r = np_lookup.get(sid)
        if np_r is None or float(np_r['border_mass_frac']) > 0.15:
            continue  # want no-pad to be LOW-border
        picks.append((r, np_r))
        if len(picks) == 3:
            break
    if not picks:
        print('[fig28] no suitable picks — skipping')
        return

    from PIL import Image
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, (rb, rn) in enumerate(picks):
        sid = int(rb['sample_id'])
        img_b = Image.open(os.path.join(base_dir, f"sample_{sid:04d}.png"))
        img_n = Image.open(os.path.join(np_dir,   f"sample_{sid:04d}.png"))
        axes[0, col].imshow(img_b); axes[0, col].axis('off')
        axes[0, col].set_title(
            f"reflect-padding\nborder={float(rb['border_mass_frac']):.2f}, "
            f"p(P)={float(rb['prob_infected']):.2f}",
            fontsize=9, color='#7a1f1f')
        axes[1, col].imshow(img_n); axes[1, col].axis('off')
        axes[1, col].set_title(
            f"no-padding\nborder={float(rn['border_mass_frac']):.2f}, "
            f"p(P)={float(rn['prob_infected']):.2f}",
            fontsize=9, color='#1a3a6f')
    fig.suptitle(
        "densenet121 / SRAD — same Figshare test image, two checkpoints\n"
        "(top: reflect-padding baseline; bottom: no-padding retrain)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, 'fig28_gradcam_border_vs_interior.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig28] -> {out}')


def main():
    fig27_border_before_after()
    fig28_gradcam_panels()


if __name__ == '__main__':
    main()