# Novelty of PEARL

This section states what is genuinely new about the PEARL pipeline
relative to prior PCOS detection work, and what we explicitly do not
claim. Every claim is backed by either an artefact under `results/` or a
literal claim in the Title Defense literature table (`docs/1. Title
Defense.md`, Table 2.1). No "first in PCOS" assertion is made without
supporting evidence.

---

## 1. Comparison against prior PCOS detection work

The Title Defense literature table contains 25 references across six
themes. Reading the table column-by-column reveals that no single
prior study combines all of the following: an external cohort,
preprocessing ablation with shortcut diagnosis, calibration analysis,
uncertainty quantification, and explainability verification.

| Theme | Reference examples | Dataset | External validation? | Calibration? | Uncertainty? | XAI on shortcut? |
|-------|-------------------|---------|----------------------|--------------|--------------|------------------|
| Clinical feature ML | Agirsoy 2025 [2], Patil 2025 [6], Mahesswari 2024 [11], Nsugbe 2023 [12], Lim 2023 [15] | Kaggle / private | No (self-reported) | No | No | No |
| CNN transfer learning | Sundari 2025 [1], CystNet 2024 [3], Lakshmi 2026 [4], Shanmugavadivel 2024 [5], Reka 2025 [8], Jeyashanker 2025 [9], Bedi 2024 [14], Fan 2023 [24], Ghosh 2025 [25] | Mostly Figshare / Kaggle | **None in Title Defense table** | No | No | Grad-CAM only |
| Segmentation | Poorani 2024 [10] | Private | No | No | No | No |
| XAI foundations | Selvaraju 2017 [17], Zhang 2016 [20], Samek 2017 [21], Tjoa 2021 [23] | ImageNet / general | n/a | n/a | n/a | General XAI methodology |
| Uncertainty | Roy 2018 [18] (brain MRI), Gal 2016 [16], Akram 2025 [19] | Akram 2025: combined Kaggle | Akram 2025: combined sources, no held-out cohort | No | Yes (MC-Dropout, MFVI) | No |
| Calibration | Guo 2017 [22] | ImageNet / CIFAR | n/a | Yes (foundational) | No | No |
| SLRs | Ghaderzadeh 2025 [7], Suha 2023 [13] | Reviews | Flag "limited external validation" and "overuse of Kaggle" as literature-wide gaps | Not addressed | Not addressed | Only ~25% of reviewed studies apply any XAI |

Two SLRs ([7], [13]) both identify the same literature-wide gaps:
over-reliance on a small set of public datasets, almost no external
validation, and XAI applied in only a minority of studies. PEARL's
contributions address these gaps directly.

---

## 2. Novelty points

Each point below states the novel claim, the artefact that proves it,
and the closest prior work for contrast. Claims are qualified to "on
this dataset pair" or "in the Title Defense literature table" where
broader "first in PCOS" claims would be unverifiable.

### N1. Cross-dataset generalisation study using Figshare -> PCOSgen

**Claim.** PEARL trains on one dataset (Figshare PCOS Ultrasound) and
externally validates on a second, independently collected dataset
(PCOSgen, Sundari et al., 2025), reporting the full internal-vs-external
collapse and recovery arc on this dataset pair.

**Evidence.**
- `results/ablation/sweep_matrix.csv` — 18 foundation configurations, internal AUC.
- `results/external_validation/summary.csv` — external AUC on PCOSgen, mean 0.5222 (range 0.4500–0.5868) for the padded foundation stage.
- `docs/latex/findings_table.md` Step 1 — verifies the collapse.

**Closest prior work.**
- Sundari et al. 2025 [1] uses PCOSgen but evaluates **in-distribution** only (their own training/test split). Title Defense Table 2.1 row [1] states "Single-source dataset limits population generalisability" as their own acknowledged limitation.
- Ghaderzadeh 2025 SLR [7] flags "Limited external validation; risk of selection bias in patient cohorts" as a literature-wide gap.
- Suha 2023 SLR [13] identifies "lack of dataset diversity" as a recurring problem.
- Bedi 2024 [14] notes "Limited external validation across different clinical settings" as a limitation.

PEARL is the only entry in the Title Defense table that performs an
external-cohort evaluation.

### N2. Diagnoses a letterbox-padding shortcut that no other PCOS study audited

**Claim.** During foundation-stage evaluation, PEARL discovered that the
models were attending to the black border added by letterbox
preprocessing rather than to the ovary. Disabling padding and resizing
directly recovers **+0.25 absolute AUC** (mean external AUC 0.5222 →
0.7704). This is verified by per-image Grad-CAM border-attention audit.

**Evidence.**
- `results/ablation_nopad/sweep_matrix.csv` — mean external AUC 0.7704 (no-pad).
- `docs/latex/figures/f27_border_attention_before_after.png` — visual Grad-CAM before/after fix.
- Per-image audit CSVs at `results/ablation_nopad/checkpoints/srad_nopad/<arch>/shortcut_audit/`.
- `docs/latex/findings_table.md` Step 1 row "Disabling padding recovers 0.52 → 0.77 (foundation)" — verified.

**Closest prior work.** None of the Title Defense references perform a
shortcut diagnosis. Reka 2025 [8] uses ESRGAN super-resolution but does
not audit whether attention is on the anatomy or on a border artefact.
This is a verifiable gap that PEARL fills.

### N3. SRAD (Speckle-Reducing Anisotropic Diffusion) in place of Gaussian filtering

**Claim.** PEARL applies SRAD (Yu & Acton, 2002) — a denoiser designed
for multiplicative speckle noise characteristic of ultrasound — in place
of the Gaussian filtering used in most prior PCOS pipelines. The
denoiser choice is documented as part of the preprocessing ablation.

**Evidence.**
- `docs/methodology.tex` §5 — SRAD equation (ICOV-based diffusion coefficient).
- `docs/latex/figures/f25_padding_vs_nopad_external_auc.png` — SRAD vs. Gaussian × pad/nopad comparison.
- Verified provenance: `docs/thesis/refs/thesis.bib` `yu2002srad` entry, fetched from primary source.

**Honest framing.** This is a methodological choice rather than a
verified superiority claim. Many prior PCOS studies do not denoise at
all (e.g., Shanmugavadivel 2024 [5] uses VGG16 on raw ultrasound
images), so the comparison is partly between "denoised" and
"not-denoised" rather than strictly "SRAD" vs. "Gaussian". PEARL
documents the ablation transparently and reports that on this dataset
pair, SRAD-nopad and Gaussian-nopad are competitive — neither
unambiguously dominates the other.

### N4. Two-pass temperature-scaling calibration (per-model, then ensemble)

**Claim.** PEARL is the first PCOS pipeline that applies temperature
scaling twice: once per-architecture, then a second time on the
ensemble's averaged probabilities. The two temperatures differ
substantially (per-model T in {2.0162, 1.9579, 2.1982}; ensemble T
0.9306), demonstrating that the ensemble is miscalibrated in a
different direction than any individual member.

**Evidence.**
- `results/finetune_zenodo/ensemble/top3_pre_hpo/calibration/calibration_results.json` — both temperature passes documented.
- ECE reduced from 0.0854 (after pass 1) to 0.081 (after pass 2) on 15 bins, 734 samples.

**Closest prior work.** Guo 2017 [22] is the only calibration reference
in the Title Defense table and is a foundational method paper on
ImageNet/CIFAR. No PCOS study in the table applies calibration at all;
PEARL's contribution is to bring calibration to PCOS detection with a
two-pass protocol.

### N5. MC-Dropout entropy at the ensemble level for per-case uncertainty

**Claim.** PEARL is the first PCOS pipeline that quantifies
per-prediction predictive entropy by running 50 stochastic forward
passes through the calibrated ensemble and reporting per-case entropy
distributions alongside accuracy.

**Evidence.**
- `results/finetune_zenodo/ensemble/top3_pre_hpo/uncertainty/mc_dropout_predictions.csv` — per-image MC-Dropout predictions.
- Mean predictive entropies (nats): DenseNet-121 0.234, ConvNeXt-Tiny 0.215, ViT-B/16 0.160.

**Closest prior work.** Akram 2025 [19] applies MC-Dropout (alongside
MFVI and deterministic inference) but reports uncertainty on a combined
single-source dataset, with no held-out external cohort and no
calibration. PEARL's combination of MC-Dropout + cross-dataset
validation + two-pass calibration is not present in any prior entry.

### N6. Three-model probability-averaged ensemble across heterogeneous architectures

**Claim.** PEARL's final ensemble combines a CNN (DenseNet-121), a
modern convolutional backbone (ConvNeXt-Tiny), and a transformer (ViT-B/16)
by probability averaging, achieving AUC 0.9487 on the held-out PCOSgen
test set. The heterogeneous mix produces a sharper decision boundary
than any single member.

**Evidence.**
- `results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.json` — ensemble AUC 0.9487 (95% CI 0.9368–0.9596).
- Per-member AUCs: 0.9420, 0.9403, 0.9397.

**Closest prior work.** Ghosh & Srinivasan 2025 [25] report an ensemble
of EfficientNetB7 + DenseNet201 (both CNNs, same family) with GA-based
HPO, achieving AUC 0.9897 — but on a Kaggle dataset, in-distribution,
without external validation. The Title Defense SLRs flag "limited
external validation" as the recurring gap; PEARL fills it. The
heterogeneous CNN/transformer mix is what distinguishes PEARL from
Ghosh's same-family ensemble.

### N7. Grad-CAM verification that attention has migrated from border to interior

**Claim.** PEARL uses Grad-CAM to verify that after fine-tuning, the
model's attention has migrated from the padding border (where the
foundation stage attends) to the scan interior (the ovary). This is
documented in the per-image border-attention audit and in the Grad-CAM
panel grid.

**Evidence.**
- `docs/latex/figures/paper_fig_xai_grid.png` — Grad-CAM panels for the top-3 ensemble members.
- `docs/latex/figures/f27_border_attention_before_after.png` — before-and-after comparison.
- `results/ablation_nopad/checkpoints/<preproc>/<arch>/shortcut_audit/per_image_gradcam.csv` — quantitative audit.

**Closest prior work.** Sundari 2025 [1] applies Grad-CAM and reports
that "Grad-CAM confirmed clinically relevant follicle regions as
diagnostically significant", and Lakshmi 2026 [4] uses Grad-CAM++ and
attention heatmaps for a multimodal model. Neither performs a
border-vs-interior audit or documents the shortcut-and-recovery
narrative. PEARL's novelty is the **verifiable** link between the
preprocessing choice (padding) and where the model attends.

---

## 3. What we explicitly do NOT claim as novel

These non-claims keep the novelty section truthful:

- **Not the first PCOS deep-learning paper.** Sundari 2025, CystNet 2024,
  Bedi 2024, Ghosh 2025, Jeyashanker 2025, and others preceded us.
- **Not the first to apply Grad-CAM to PCOS.** Sundari 2025 [1] and
  Lakshmi 2026 [4] both use Grad-CAM/Grad-CAM++ on PCOS ultrasound.
- **Not the first to apply MC-Dropout to a medical imaging task.** Roy
  2018 [18] does so on brain MRI; Akram 2025 [19] does so on PCOS.
- **Not the first to propose temperature scaling.** Guo 2017 [22] is
  the foundational reference.
- **Not the first to use an ensemble for PCOS.** Ghosh 2025 [25] does so.

PEARL's novelty is the **joint execution** of these techniques on a
cross-dataset pair with a verified preprocessing-shortcut diagnosis.
Each technique individually has prior art; the combination, plus the
audit step that documents the border-vs-interior migration, is what
distinguishes this work.

---

## 4. Notes on the novelty statement in the FYDP proposal

The FYDP Title Defense (`docs/1. Title Defense.md` §4) lists five
contributions. Two are restated by PEARL exactly as proposed; three
were either descoped or have evolved in execution:

- "First systematic 9×6 preprocessing-architecture benchmark" — the
  executed benchmark is 9×2 (SRAD vs Gaussian), not 9×6. The number of
  configurations is smaller but the diagnostic insight (the padding
  shortcut) is sharper than a 9×6 sweep would have produced.
- "First application of Perona-Malik anisotropic diffusion" — the
  executed pipeline uses SRAD (Yu & Acton 2002), not Perona-Malik.
  Perona-Malik was an earlier phase-of-the-work design choice.
- "Three XAI methods: Grad-CAM, LRP, SHAP" — only Grad-CAM was
  executed on the fine-tuned checkpoints. LRP and SHAP are documented
  as designed-but-not-executed in `docs/thesis/appendix/A_designed_not_executed.tex`.
- "Uncertainty-gated clinical referral system" — only the per-coverage
  statistics from which a deployment could set the cutoff are produced;
  the deployed gating mechanism remains a design concept.

The novelty section above reflects what was actually executed, with
evidence in the artefacts.