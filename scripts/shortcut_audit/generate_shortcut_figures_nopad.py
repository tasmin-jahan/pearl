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

    def stats(d):
        means, stds, ns = [], [], []
        for a in arches:
            vals = [v for _, v in d.get(a, [])]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals, ddof=0) if vals else 0.0)
            ns.append(len(vals))
        return np.array(means), np.array(stds), ns

    base_m, base_s, base_n = stats(base)
    new_m,  new_s,  new_n  = stats(new)
    gray_m, gray_s, gray_n = stats(gray)

    fig, ax = plt.subplots(figsize=(13, 5.0))
    x = np.arange(len(arches))
    w = 0.28
    def draw_bar(off, m, s, color, label):
        b = ax.bar(x + off, m, w, color=color, edgecolor='gray', linewidth=0.6,
                   label=label, yerr=s, ecolor='#333', capsize=3, error_kw={'lw': 0.8})
        for i, (v, e) in enumerate(zip(m, s)):
            ax.text(x[i] + off, max(v, 0) + (e if not np.isnan(e) else 0) + 0.005,
                    f"{v:.2f}", ha='center', fontsize=7)
    draw_bar(-w, base_m, base_s, '#a6d96a', 'reflect-padding (baseline)')
    draw_bar(  0, new_m,  new_s,  '#d73027', 'no-padding')
    draw_bar( w, gray_m, gray_s, '#4575b4', 'no-padding + greyscale')

    ax.axhline(0.5, linestyle=':', color='gray', linewidth=1, label='chance (0.5)')
    ax.set_xticks(x); ax.set_xticklabels(arches, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('PCOSgen external ROC-AUC')
    ax.set_ylim(0.35, 0.95)
    ax.set_title('Removing the letterbox border recovers external AUC by +0.25 mean; '
                 'greyscale adds no further benefit (mean ± std over srad / gauss)')
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