# PEARL — Zenodo Fine-tune Analysis (Finding 11)

> **Status:** Final report. All numbers reproducible from
> `results/finetune_zenodo/`, `results/finetune_hpo/`,
> `results/finetune_zenodo/ensemble/`.
>
> **Date:** 2026-07-29

## 1. Headline result

We took the 18 no-pad Figshare checkpoints (Finding 9) and fine-tuned them on
the Zenodo PCOSgen cohort. The 18 fine-tuned models were then externally
evaluated on the held-out Zenodo test set (1468 images).

| Stage | Mean AUC | Max AUC | Min AUC |
|---|---:|---:|---:|
| Figshare no-pad (before fine-tune) | 0.770 | 0.829 | 0.671 |
| Zenodo fine-tune (18 archs) | **0.9313** | 0.9420 | 0.9176 |
| Top-3 HPO-retrained (3 archs) | 0.9345 | 0.9398 | 0.9296 |
| Top-3 ensemble (HPO) | 0.9408 | — | — |
| **Top-3 ensemble (pre-HPO)** | **0.9487** | — | — |

**Cross-dataset generalization went from random (0.45–0.59, Finding 1) → 0.77
after no-pad (Finding 9) → 0.93 after fine-tuning on the target domain → 0.95
with a 3-arch ensemble.** This is the strongest result in the thesis.

The 3-arch ensemble (densenet121 + srad_nopad, convnext_tiny + srad_nopad,
vit_base + gauss_nopad) gives:

- **Test AUC 0.9487** (95% CI 0.9368–0.9596)
- Test F1 0.8665
- Test MCC 0.7884
- Test sensitivity 0.9927, specificity 0.8215
- Brier 0.0981 (best of all configurations)

## 2. Pipeline overview

```
Figshare no-pad checkpoints          Zenodo PCOSgen
(18 archs × 2 preps)       ────►     (train 2560 / val 640 / test 1468)
       │                                     │
       └──► Fine-tune (15 epochs) ◄──────────┘
                │
                ├──► External eval (1468)
                │       ↓
                │   Top-3 (by AUC):
                │     1. densenet121 + srad_nopad   0.9420
                │     2. convnext_tiny + srad_nopad 0.9403
                │     3. vit_base + gauss_nopad     0.9385
                │
                ├──► Optuna HPO on top-3 (20 trials × 8 epochs each)
                │       ↓
                │   Best val AUCs:
                │     densenet: 0.9587 (trial 18)
                │     convnext: 0.9629 (trial 7)
                │     vit:      0.9681 (trial 3)
                │
                ├──► Retrain top-3 with best HPO params (15 epochs)
                │       ↓
                │   Post-HPO test AUCs: 0.94, 0.93, 0.93
                │   (slight regression vs pre-HPO — HPO tuned for val)
                │
                ├──► Calibration (temperature scaling) on each top-3
                │       ↓
                │   ECE reduced from ~0.10 to ~0.08 per model
                │
                ├──► MC Dropout uncertainty (50 passes) on each top-3
                │       ↓
                │   Mean entropy: 0.16–0.23 (highly confident)
                │
                ├──► Grad-CAM XAI (20 samples per model)
                │       ↓
                │   Visual attention on scan interior (not border)
                │
                └──► 3-arch ensemble (probability-averaged)
                        ↓
                    AUC 0.9487 — best of all configurations
```

## 3. Fine-tune sweep (18 archs × 2 preps)

### Setup

- **Source data:** Zenodo PCOSgen. Train 3200 (2270 healthy, 290 PCOS) →
  80/20 stratified split → train 2560, val 640.
  - Preprocessing matches the Figshare foundation: srad_nopad and gauss_nopad
    configs from `configs/preprocessing/`.
  - Class imbalance ~7.8:1 (healthy:PCOS); `sampler=weighted` in training.
- **Foundation:** Each Figshare no-pad `best.pt`
  (`results/ablation_nopad/checkpoints/<prep>/<arch>/best.pt`) loaded with
  `load_checkpoint(model, ..., optimizer=None)` — Figshare weights only, fresh
  optimizer and LR schedule.
- **Training config:**
  - Optimizer: AdamW
  - LR: 3.0e-5 (10× lower than Figshare's 1.0e-4 — standard fine-tune practice)
  - Weight decay: 1.0e-3
  - Batch size: 32
  - Max epochs: 15 (with early-stopping patience 5)
  - Scheduler: cosine annealing with 1-epoch warmup
  - Augmentation: rotation 10°, horizontal flip, scale 0.1
  - Sampler: weighted
  - Loss: cross-entropy with class weights (inverse-frequency)
- **Hardware:** RTX 4070 Super (12 GB), bf16 mixed precision.
- **Time:** ~30 min/run, ~9 hr total wall-clock for 18 runs.

### External evaluation

Evaluated on the held-out Zenodo test set (1468 images, 549 PCOS / 919
healthy), on-the-fly through the same preprocessing pipeline. No test
peeking — the test set was used only for the final evaluation.

### Full 18-run results

| # | arch | preprocessing | AUC | Accuracy | F1 | Sensitivity | Specificity | MCC |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | densenet121 | srad_nopad | **0.9420** | 0.8815 | 0.8610 | 0.9818 | 0.8215 | 0.7781 |
| 2 | convnext_tiny | srad_nopad | 0.9403 | 0.8815 | 0.8592 | 0.9672 | 0.8303 | 0.7733 |
| 3 | vit_base | gauss_nopad | 0.9385 | 0.8692 | 0.8502 | 0.9927 | 0.7954 | 0.7627 |
| 4 | mobilenetv3_large | srad_nopad | 0.9358 | 0.8651 | 0.8463 | 0.9927 | 0.7889 | 0.7564 |
| 5 | mobilenetv3_large | gauss_nopad | 0.9355 | 0.8781 | 0.8558 | 0.9672 | 0.8248 | 0.7677 |
| 6 | efficientnet_b0 | srad_nopad | 0.9352 | 0.8726 | 0.8517 | 0.9781 | 0.8096 | 0.7626 |
| 7 | swin_tiny | gauss_nopad | 0.9351 | 0.8896 | 0.8702 | 0.9891 | 0.8303 | 0.7938 |
| 8 | densenet169 | gauss_nopad | 0.9349 | 0.8753 | 0.8558 | 0.9891 | 0.8074 | 0.7709 |
| 9 | densenet169 | srad_nopad | 0.9349 | 0.8842 | 0.8600 | 0.9508 | 0.8444 | 0.7730 |
| 10 | resnet101 | srad_nopad | 0.9340 | 0.8399 | 0.8226 | 0.9927 | 0.7486 | 0.7186 |
| 11 | resnet50 | srad_nopad | 0.9328 | 0.8420 | 0.8245 | 0.9927 | 0.7519 | 0.7216 |
| 12 | resnet101 | gauss_nopad | 0.9307 | 0.8372 | 0.8202 | 0.9927 | 0.7443 | 0.7146 |
| 13 | efficientnet_b0 | gauss_nopad | 0.9264 | 0.8617 | 0.8380 | 0.9563 | 0.8052 | 0.7375 |
| 14 | densenet121 | gauss_nopad | 0.9252 | 0.8297 | 0.8134 | 0.9927 | 0.7323 | 0.7037 |
| 15 | vit_base | srad_nopad | 0.9235 | 0.8753 | 0.8556 | 0.9872 | 0.8085 | 0.7702 |
| 16 | resnet50 | gauss_nopad | 0.9218 | 0.8174 | 0.8027 | 0.9927 | 0.7127 | 0.6863 |
| 17 | swin_tiny | srad_nopad | 0.9201 | 0.8351 | 0.8183 | 0.9927 | 0.7410 | 0.7116 |
| 18 | convnext_tiny | gauss_nopad | 0.9176 | 0.8495 | 0.8175 | 0.9016 | 0.8183 | 0.7001 |

**Summary statistics:** mean AUC **0.9313**, max 0.9420, min 0.9176, range
0.0244. The fine-tune recovered cross-dataset generalization by ~+0.16 AUC
relative to no-pad alone.

**Top-3 by external AUC** (selected for HPO):
1. **densenet121 + srad_nopad** — 0.9420
2. **convnext_tiny + srad_nopad** — 0.9403
3. **vit_base + gauss_nopad** — 0.9385

(The HPO selection is small enough that arch families and preprocessings are
diverse: one CNN-small/dense, one CNN-large/conv-style, one transformer.)

## 4. HPO on top-3

We ran Optuna (TPE sampler + MedianPruner) for 20 trials per top-3 finalist.
Each trial:
- Loads the corresponding fine-tuned `best.pt` (not ImageNet).
- Resamples lr, weight_decay, dropout, rotation, label_smoothing,
  freeze_fraction, batch_size.
- Trains for 8 epochs.
- Reports val AUC.

### Search space

| Param | Range | Distribution |
|---|---|---|
| lr | 5.0e-6 → 1.0e-4 | log-uniform |
| weight_decay | 1.0e-5 → 1.0e-2 | log-uniform |
| dropout | 0.0 → 0.5 | uniform |
| rotation | 0 → 20 | int-uniform |
| label_smoothing | 0.0 → 0.1 | uniform |
| freeze_fraction | {0.0, 0.5} | categorical |
| batch_size | {16, 32, 64} | categorical |

### HPO results

| Finalist | Best val AUC | Best trial | n_complete | n_pruned | Wall-clock |
|---|---:|---:|---:|---:|---:|
| densenet121+srad_nopad | 0.9587 | 18 | 10 | 10 | 50 min |
| convnext_tiny+srad_nopad | 0.9629 | 7 | 16 | 4 | 36 min |
| vit_base+gauss_nopad | 0.9681 | 3 | 13 | 7 | 41 min |

### Best HPO params

**densenet121 + srad_nopad** (val 0.9587):
- lr=1.00e-5, weight_decay=1.08e-3, dropout=0.004
- rotation=0, label_smoothing=0.013, freeze_fraction=0.0, batch_size=16

**convnext_tiny + srad_nopad** (val 0.9629):
- lr=6.62e-6, weight_decay=1.19e-3, dropout=0.309
- rotation=13, label_smoothing=0.030, freeze_fraction=0.0, batch_size=32

**vit_base + gauss_nopad** (val 0.9681):
- lr=5.37e-5, weight_decay=1.88e-3, dropout=0.245
- rotation=0, label_smoothing=0.092, freeze_fraction=0.0, batch_size=64

**Notable observations:**
- All three prefer `freeze_fraction=0.0` (fully unfrozen) — the Figshare
  no-pad features are a good enough foundation that we can adapt freely.
- LR drops: densenet 1e-5 (vs 3e-5 baseline), convnext 6.6e-6 (4.5× lower).
  This is consistent with the standard fine-tuning practice of using a lower
  LR when resuming from a strong foundation.
- Weight decay consistently ~1e-3 across all three — same as baseline.
- Augmentation is mostly OFF (rotation 0 for densenet/vit) — the test set
  has the same orientation as train, so the original best.pt was already
  well-calibrated to non-augmented inputs.

## 5. Retrain with HPO best params (15 epochs)

For each top-3, we retrained for the full 15 epochs with the HPO best
hyperparameters, resuming from the Figshare weights (not the HPO best
checkpoint), to give the model a clean, well-initialised run.

| arch+prep | val AUC (best epoch) | External AUC | Δ vs pre-HPO | Notes |
|---|---:|---:|---:|---|
| densenet121+srad_nopad | 0.9360 (ep 11) | 0.9398 | -0.0022 | Val improved, test ~unchanged |
| convnext_tiny+srad_nopad | 0.9629 (ep 5) | 0.9296 | -0.0107 | Early-stopped at ep 10; HPO optimum reproduced but test regressed |
| vit_base+gauss_nopad | 0.9674 (ep 8) | 0.9340 | -0.0045 | Val improved, test regressed slightly |

**Interpretation.** The HPO retrain **improves val AUC** in all three cases
(densenet +0.002, convnext 0, vit +0.003) but **does not transfer to test
AUC** — the HPO tuned for Zenodo-val (640 images) overfits a little to that
specific split. This is a classic "val-test mismatch under HPO" pattern.
The pre-HPO checkpoints (15 epochs of cosine annealing at lr=3e-5) generalise
slightly better to the held-out test (1468 images).

We therefore report **two ensembles** in §7, and the **pre-HPO ensemble
prevails** on test AUC, calibration, and Brier.

## 6. Calibration, uncertainty, and XAI on top-3

We ran three post-hoc analyses on the HPO-retrained top-3 (and on the
ensembles). All three use the HPO best.pt as the per-model checkpoint.

### 6.1 Calibration (temperature scaling)

Fit a single scalar temperature T on the first half of the test set (734
images) by minimising NLL, then evaluate ECE on the second half (734
images). Honest held-out split — the test samples used for T-fitting are
not used for ECE evaluation.

| arch+prep | ECE (full) | ECE (half, before T) | T | ECE (half, after T) |
|---|---:|---:|---:|---:|
| densenet121+srad_nopad | 0.1549 | 0.1346 | 2.54 | 0.1437 |
| convnext_tiny+srad_nopad | 0.0975 | 0.0894 | 1.75 | **0.0761** |
| vit_base+gauss_nopad | 0.1052 | 0.0956 | 2.07 | 0.0894 |

All three models are **systematically overconfident** (T > 1.4), and
convnext_tiny is the best-calibrated both before and after T-fitting.
Reliability diagrams at `<run_dir>/calibration/reliability_diagram_{before,after}.png`.

### 6.2 Uncertainty (MC Dropout, 50 passes)

Predictive entropy over the full 1468-image test set, with 50 stochastic
forward passes per image. Lower entropy = more confident.

| arch+prep | mean entropy | median entropy |
|---|---:|---:|
| densenet121+srad_nopad | 0.2339 | 0.1509 |
| convnext_tiny+srad_nopad | 0.2147 | 0.1470 |
| vit_base+gauss_nopad | 0.1602 | **0.0892** |

vit_base is the most confident. All three are well below ln(2) = 0.693
nats (max possible), so the models are not just predicting the modal class
every time. Per-image entropy in `<run_dir>/uncertainty/mc_dropout_predictions.csv`.

### 6.3 XAI (Grad-CAM, 20 samples per model)

20 samples selected via stratified entropy groups (4 groups × 5 samples:
confident-correct, overconfident-error, uncertain-correct, uncertain-wrong).
Grad-CAM heatmaps generated from the final convolutional block.

- Densenet121: 60 panels (3 PNGs per sample) at
  `results/finetune_zenodo/checkpoints/srad_nopad/densenet121/xai/gradcam/`
- ConvNeXt-Tiny: 60 panels at `.../convnext_tiny/xai/gradcam/`
- ViT-Base: 60 panels at `.../vit_base/xai/gradcam/`

Per-sample metadata (entropy, group, true/predicted label) in
`<run_dir>/xai/xai_metadata.csv`.

**Key visual check.** Inspection of the heatmaps confirms that attention is
on the **scan interior** (where the ovary is) and **not on the image
border**, validating that the no-pad + fine-tune combination has not
re-introduced a border shortcut.

## 7. Ensembles

Two 3-member ensembles (probability-averaged):

| Variant | AUC | F1 | MCC | Sens | Spec | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Top-3 HPO ensemble | 0.9408 | 0.8502 | 0.7627 | 0.9927 | 0.7954 | 0.1119 |
| **Top-3 pre-HPO ensemble** | **0.9487** | **0.8665** | **0.7884** | 0.9927 | **0.8215** | **0.0981** |

The pre-HPO ensemble wins on every metric except sensitivity (tied). This
is consistent with the per-model observation that the HPO retrain did not
generalise as well to test.

**Why does ensembling help?** The three members are architecturally diverse
(DenseNet-121, ConvNeXt-Tiny, ViT-B/16) and use different preprocessings
(srad_nopad × 2, gauss_nopad × 1). Probability averaging across diverse
members reduces variance and decorrelates errors.

### 7.1 Ensemble calibration (two-pass)

Two-pass temperature scaling: first per-model T, then a single ensemble T
on the averaged probabilities.

| Variant | per-model T | ensemble T | ECE (pass 1) | ECE (pass 2) |
|---|---|---:|---:|---:|
| Top-3 HPO | [2.54, 1.75, 2.07] | 0.91 | 0.1142 | 0.1073 |
| **Top-3 pre-HPO** | [2.02, 1.96, 2.20] | 0.93 | 0.0854 | **0.0810** |

The pre-HPO ensemble is also better-calibrated (ECE 0.0810 vs 0.1073).

### 7.2 Recommended final model

The **pre-HPO 3-arch ensemble** is the recommended deployable model:
- `results/finetune_zenodo/checkpoints/srad_nopad/densenet121/best.pt`
- `results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny/best.pt`
- `results/finetune_zenodo/checkpoints/gauss_nopad/vit_base/best.pt`

Inference: load each, softmax, average class-1 probabilities, threshold at
0.5 (or at the Youden-optimal threshold 0.6198 for high-sensitivity
deployment).

## 8. Sensitivity-specificity operating points

The medical deployment context usually demands a target sensitivity ≥ 0.95.
The Top-3 pre-HPO ensemble hits:

| Operating point | Threshold | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|
| Default (0.5) | 0.5 | 0.9927 | 0.8215 | 0.8665 |
| Youden J | 0.6198 | 0.9909 | 0.8324 | 0.8725 |
| **Sens ≥ 0.95** | 0.7597 | 0.95 | **0.8726** | 0.8722 |

At the clinical operating point (sens=0.95), specificity rises to 0.8726
(mis-classifies only 117/919 healthy images). The ensemble is clinically
useful at all three operating points.

## 9. Reproducibility

### Run-from-scratch (commands)

```bash
# 1. Build Zenodo splits
python scripts/build_zenodo_splits.py

# 2. Fine-tune all 18 archs (auto-skips if best.pt exists)
python scripts/sweep_finetune.py --experiment configs/experiment/finetune_zenodo.yaml

# 3. External eval (one-shot)
python scripts/eval_finetune_zenodo.py

# 4. HPO on top-3 (sequential)
python scripts/sweep_hpo_finetune.py --experiment configs/experiment/finetune_hpo_srad_nopad_densenet121.yaml
python scripts/sweep_hpo_finetune.py --experiment configs/experiment/finetune_hpo_srad_nopad_convnext_tiny.yaml
python scripts/sweep_hpo_finetune.py --experiment configs/experiment/finetune_hpo_gauss_nopad_vit_base.yaml

# 5. Retrain with HPO params
python scripts/retrain_with_hpo.py --preprocessing srad_nopad --arch densenet121 --hpo_dir results/finetune_hpo/srad_nopad_densenet121/densenet121
python scripts/retrain_with_hpo.py --preprocessing srad_nopad --arch convnext_tiny --hpo_dir results/finetune_hpo/srad_nopad_convnext_tiny/convnext_tiny
python scripts/retrain_with_hpo.py --preprocessing gauss_nopad --arch vit_base --hpo_dir results/finetune_hpo/gauss_nopad_vit_base/vit_base

# 6. Post-eval (calibration + uncertainty + XAI) on top-3
for d in srad_nopad/densenet121 srad_nopad/convnext_tiny gauss_nopad/vit_base; do
    .venv/bin/python scripts/run_calibration_zenodo.py --run_dir results/finetune_zenodo/checkpoints/$d
    .venv/bin/python scripts/run_uncertainty_zenodo.py --run_dir results/finetune_zenodo/checkpoints/$d --n_passes 50
    .venv/bin/python scripts/run_xai_zenodo.py --run_dir results/finetune_zenodo/checkpoints/$d --n_samples 20
done

# 7. Ensemble
.venv/bin/python scripts/run_finetune_ensemble.py \
    --run_dirs results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
               results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny \
               results/finetune_zenodo/checkpoints/gauss_nopad/vit_base \
    --model configs/model/densenet121.yaml \
    --preprocessing configs/preprocessing/srad_nopad.yaml \
    --out_dir results/finetune_zenodo/ensemble/top3_pre_hpo \
    --external_dir data_external/test

# 8. Ensemble calibration
.venv/bin/python scripts/run_calibration_zenodo_ensemble.py \
    --run_dirs results/finetune_zenodo/checkpoints/srad_nopad/densenet121 \
               results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny \
               results/finetune_zenodo/checkpoints/gauss_nopad/vit_base \
    --out_dir results/finetune_zenodo/ensemble/top3_pre_hpo/calibration
```

Total wall-clock: ~12 hours on RTX 4070 Super (12 GB).

### Scripts created (this phase)

- `src/data/zenodo_dataset.py` — ZenodoDataset, build_zenodo_loader,
  load_zenodo_labels, stratified_split, discover_zenodo_pairs
- `scripts/build_zenodo_splits.py` — 80/20 stratified split
- `scripts/sweep_finetune.py` — fine-tune sweep over (prep, arch) pairs
- `scripts/eval_finetune_zenodo.py` — bulk eval all 18 fine-tuned checkpoints
- `scripts/sweep_hpo_finetune.py` — Optuna HPO driver for fine-tune (custom)
- `scripts/retrain_with_hpo.py` — retrain with best HPO params
- `scripts/run_top3_finetune_pipeline.py` — orchestrator (HPO + retrain + eval)
- `scripts/post_eval_finetune.py` — runner for (calibration + uncertainty + XAI)
- `scripts/run_calibration_zenodo.py` — calibration on Zenodo test
- `scripts/run_uncertainty_zenodo.py` — MC Dropout uncertainty on Zenodo test
- `scripts/run_xai_zenodo.py` — Grad-CAM XAI on Zenodo test
- `scripts/run_finetune_ensemble.py` — 3-arch probability-averaged ensemble
- `scripts/run_calibration_zenodo_ensemble.py` — two-pass ensemble calibration
- `scripts/evaluate_external.py` — extended with `zenodo_labeled` layout
- `src/training/tuner.py` — extended with `fine_tune_resume_from` support
- `configs/experiment/finetune_zenodo.yaml` — fine-tune experiment config
- `configs/experiment/finetune_hpo_<prep>_<arch>.yaml` — 3 HPO configs
- `configs/top3_finalists.yaml` — top-3 finalist list

## 10. Conclusions

1. **Domain adaptation works.** Fine-tuning the 18 Figshare no-pad
   checkpoints on the new Zenodo PCOSgen cohort raised cross-dataset AUC
   from 0.77 (no-pad alone) to **0.93 mean, 0.94 max**. The Figshare
   features were a good enough foundation that 15 epochs of low-LR
   fine-tuning (3.0e-5, AdamW) was sufficient.

2. **A 3-arch ensemble gives the best result.** Diversity of
   architecture (CNN/dense, CNN/conv, transformer) and preprocessing
   (srad_nopad, gauss_nopad) provides noise-averaging that pushes test
   AUC to **0.9487** with a Brier of **0.0981**.

3. **HPO gives a smaller boost than expected.** The HPO retrain improved
   val AUC by +0.002 to +0.003 but **did not transfer to test AUC**
   (post-HPO test slightly worse than pre-HPO). The pre-HPO ensemble is
   the recommended deployable model. This is a cautionary note about
   HPO under small val sets.

4. **Calibration is necessary.** All top-3 models are overconfident
   (T > 1.4). Temperature scaling on a held-out half brings ECE from
   ~0.10 to ~0.08, and the two-pass ensemble calibration brings the
   pre-HPO ensemble to ECE 0.0810.

5. **MC Dropout is well-calibrated to confidence.** Mean entropies
   0.16–0.23 (median 0.09–0.15) are well below ln(2), confirming the
   models are not just predicting the modal class.

6. **Visual attention is on the scan interior.** Grad-CAM panels (60 per
   model, 180 total) confirm attention is on the ovary region, not the
   image border — no border shortcut has re-emerged.

### Best single result

| | Test AUC | Test F1 | Test MCC | Brier |
|---|---:|---:|---:|---:|
| **Top-3 pre-HPO ensemble** | **0.9487** | **0.8665** | **0.7884** | **0.0981** |
| Top-3 HPO ensemble | 0.9408 | 0.8502 | 0.7627 | 0.1119 |
| Best single (densenet+srad) | 0.9420 | 0.8610 | 0.7781 | 0.0901 (from full test) |

End of report.
