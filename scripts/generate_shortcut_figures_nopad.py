#!/usr/bin/env python3
"""Comparison figure for the no-padding ablation.

Generates one figure that puts the baseline (reflect-padding) external
PCOSgen AUC side-by-side with the no-padding AUC for each architecture.

Emits:
  docs/analysis/figures/fig25_padding_vs_nopad_external_auc.png
"""
import json
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = "docs/analysis/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load(root):
    out = {}
    for p in sorted(glob.glob(f'{root}/checkpoints/*/*/external_validation/pcosgen.json')):
        parts = p.split('/')
        prep, arch = parts[-4], parts[-3]
        m = json.load(open(p))
        out.setdefault(arch, []).append((prep, m['test_auc_roc']))
    return out


def main():
    base = load('results/ablation')
    new  = load('results/ablation_nopad')
    gray = load('results/ablation_nopad_gray')

    arches = sorted(set(base) | set(new) | set(gray))
    base_per = [np.mean([v for _, v in base.get(a, [])]) for a in arches]
    new_per  = [np.mean([v for _, v in new.get(a,  [])]) for a in arches]
    gray_per = [np.mean([v for _, v in gray.get(a, [])]) for a in arches]

    fig, ax = plt.subplots(figsize=(13, 4.8))
    x = np.arange(len(arches))
    w = 0.28
    ax.bar(x - w, base_per, w, color='#a6d96a', label='reflect-padding (baseline)',
           edgecolor='gray', linewidth=0.6)
    ax.bar(x,     new_per,  w, color='#d73027', label='no-padding',
           edgecolor='gray', linewidth=0.6)
    ax.bar(x + w, gray_per, w, color='#4575b4', label='no-padding + greyscale',
           edgecolor='gray', linewidth=0.6)
    for i, (b, n, g) in enumerate(zip(base_per, new_per, gray_per)):
        ax.text(i - w, b + 0.005, f"{b:.2f}", ha='center', fontsize=7, color='#3a6f3a')
        ax.text(i,     n + 0.005, f"{n:.2f}", ha='center', fontsize=7, color='#7a1f1f')
        ax.text(i + w, g + 0.005, f"{g:.2f}", ha='center', fontsize=7, color='#1a3a6f')
    ax.axhline(0.5, linestyle=':', color='gray', linewidth=1, label='chance (0.5)')
    ax.set_xticks(x); ax.set_xticklabels(arches, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('PCOSgen external ROC-AUC')
    ax.set_ylim(0.35, 0.95)
    ax.set_title('Removing the letterbox border recovers external AUC by +0.25 mean; '
                 'greyscale adds no further benefit')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(loc='upper left', fontsize=9, frameon=False, ncol=2)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig25_padding_vs_nopad_external_auc.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig25] -> {out_path}')

    # Per-(arch, prep) heatmap (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax_idx, (root, label) in enumerate([
        ('results/ablation',          'baseline (reflect-padding)'),
        ('results/ablation_nopad',    'no-padding'),
        ('results/ablation_nopad_gray','no-padding + greyscale'),
    ]):
        ax = axes[ax_idx]
        d = load(root)
        M = [sorted(d.get(a, []), key=lambda x: x[0]) for a in arches]
        max_len = max((len(r) for r in M), default=0)
        Mc = np.full((len(arches), max_len), np.nan)
        for i, r in enumerate(M):
            for j, (prep, v) in enumerate(r):
                Mc[i, j] = v
        im = ax.imshow(Mc, cmap='RdYlGn', vmin=0.40, vmax=0.92, aspect='auto')
        ax.set_xticks(range(max_len))
        ax.set_xticklabels(['srad', 'gauss'][:max_len], fontsize=9)
        ax.set_yticks(range(len(arches)))
        ax.set_yticklabels(arches, fontsize=9)
        for i in range(len(arches)):
            for j in range(max_len):
                v = Mc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                            color='white' if v < 0.65 else 'black', fontsize=8)
        ax.set_title(label, fontsize=11)
    fig.colorbar(im, ax=axes.tolist(), label='PCOSgen external AUC', shrink=0.7)
    fig.suptitle('Per-(architecture, preprocessing) external AUC',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, 'fig26_per_run_external_auc.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig26] -> {out_path}')


if __name__ == "__main__":
    main()