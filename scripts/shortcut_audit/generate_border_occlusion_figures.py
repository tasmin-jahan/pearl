#!/usr/bin/env python3
"""Generate figures for Finding 8: causal border-occlusion ablation."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "docs/analysis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

ARCHES = ['swin_tiny', 'vit_base', 'convnext_tiny',
          'densenet169', 'efficientnet_b0']
PREPS = ['srad', 'gauss']


def load_swap():
    out=[]
    for prep in PREPS:
        for arch in ARCHES:
            p=f'results/ablation/checkpoints/{prep}/{arch}/border_occlusion_swap/border_swap.json'
            if os.path.isfile(p):
                d=json.load(open(p)); d['prep']=prep; d['arch']=arch; out.append(d)
    return out


def load_median():
    out=[]
    for prep in PREPS:
        for arch in ARCHES:
            p=f'results/ablation/checkpoints/{prep}/{arch}/border_occlusion_median/border_occluded.json'
            if os.path.isfile(p):
                d=json.load(open(p)); d['prep']=prep; d['arch']=arch; out.append(d)
    return out


def fig23_swap_probability(runs):
    """Grouped bars: probability shift when opposite-class border is swapped."""
    labels=[]; d_inf=[]; d_nor=[]
    for prep in PREPS:
        for arch in ARCHES:
            r=next(x for x in runs if x['prep']==prep and x['arch']==arch)
            labels.append(f"{prep}/{arch}")
            d_inf.append(r['delta_inf'])
            d_nor.append(r['delta_nor'])
    x=np.arange(len(labels)); w=0.38
    fig,ax=plt.subplots(figsize=(15,6))
    ax.bar(x-w/2,d_inf,w,label='PCOS image + non-PCOS border',color='#4472c4')
    ax.bar(x+w/2,d_nor,w,label='non-PCOS image + PCOS border',color='#d63d3d')
    ax.axhline(0,color='black',linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=55,ha='right',fontsize=8)
    ax.set_ylabel('Δ p(PCOS) after opposite-class border swap')
    ax.set_title('Causal border-swap ablation: PCOS border acts as a positive-class trigger')
    ax.legend(frameon=False)
    for i,v in enumerate(d_nor):
        ax.text(i+w/2,v+0.012,f"{v:+.2f}",ha='center',va='bottom',fontsize=7,rotation=90)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.tight_layout()
    out=os.path.join(OUT_DIR,'fig23_border_swap_probability_shift.png')
    fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    print('[fig23] ->',out)


def fig24_median_occlusion(runs):
    """Two panels: AUC and specificity clean vs median-border-occluded."""
    labels=[]; auc_occ=[]; spec_occ=[]; auc_clean=[]; spec_clean=[]
    # Load original final_metrics
    orig={}
    for p in glob.glob('results/ablation/checkpoints/*/*/final_metrics.json'):
        parts=p.split('/'); prep=parts[-3]; arch=parts[-2]
        orig[(prep,arch)]=json.load(open(p))
    for prep in PREPS:
        for arch in ARCHES:
            r=next(x for x in runs if x['prep']==prep and x['arch']==arch)
            o=orig[(prep,arch)]
            labels.append(f"{prep}/{arch}")
            auc_clean.append(o['test_auc_roc']); auc_occ.append(r['test_auc_roc'])
            spec_clean.append(o['test_specificity']); spec_occ.append(r['test_specificity'])
    x=np.arange(len(labels)); w=0.36
    fig,axes=plt.subplots(2,1,figsize=(15,9),sharex=True)
    axes[0].bar(x-w/2,auc_clean,w,label='clean',color='#4c956c')
    axes[0].bar(x+w/2,auc_occ,w,label='median-border occluded',color='#d63d3d')
    axes[0].set_ylim(0.97,1.002); axes[0].set_ylabel('AUC')
    axes[0].set_title('Internal AUC remains saturated after neutralising the border')
    axes[0].legend(frameon=False)
    axes[1].bar(x-w/2,spec_clean,w,label='clean',color='#4c956c')
    axes[1].bar(x+w/2,spec_occ,w,label='median-border occluded',color='#d63d3d')
    axes[1].set_ylim(0.70,1.02); axes[1].set_ylabel('Specificity')
    axes[1].set_title('Specificity exposes the class-trigger effect')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels,rotation=55,ha='right',fontsize=8)
    for ax in axes:
        for s in ('top','right'): ax.spines[s].set_visible(False)
    fig.tight_layout()
    out=os.path.join(OUT_DIR,'fig24_median_border_occlusion_metrics.png')
    fig.savefig(out,dpi=150,bbox_inches='tight'); plt.close(fig)
    print('[fig24] ->',out)


def main():
    swap=load_swap(); median=load_median()
    print('loaded',len(swap),'swap and',len(median),'median runs')
    fig23_swap_probability(swap)
    fig24_median_occlusion(median)

if __name__=='__main__': main()
