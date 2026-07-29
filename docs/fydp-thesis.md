# Title of Your Final Year Design Project

**Title:** Deep Learning-Based PCOS Detection Using Transfer Learning,
Uncertainty Quantification, and Explainable AI

**By**

**Tasmin Jahan** (Student ID: 0242310005101552)

**Sabekunnaher Smrity** (Student ID: 0242310005101606)

**FINAL YEAR DESIGN PROJECT REPORT**

This Report Presented in Partial Fulfilment of the Requirements for the
**Degree of Bachelor of Science in Computer Science and Engineering**

### Supervised by

**Ms. Samia Nawshin**, Assistant Professor

Department of Computer Science and Engineering
Daffodil International University

### Co-Supervised by

**Zakia Sultana Eshita**, Lecturer (Senior Scale)

Department of Computer Science and Engineering
Daffodil International University

### DAFFODIL INTERNATIONAL UNIVERSITY

Dhaka, Bangladesh

**September 2025**

---

## APPROVAL

This Project titled "Deep Learning-Based PCOS Detection Using Transfer
Learning, Uncertainty Quantification, and Explainable AI", submitted
by Tasmin Jahan (0242310005101552) and Sabekunnaher Smrity
(0242310005101606) to the Department of Computer Science and
Engineering, Daffodil International University, has been accepted as
satisfactory for the partial fulfilment of the requirements for the
degree of B.Sc. in Computer Science and Engineering and approved as to
its style and contents.

**BOARD OF EXAMINERS**

Board Chairman

Internal Examiner 1

Internal Examiner 2

External Examiner

---

## DECLARATION

We hereby declare that this project has been done by us under the
supervision of Ms. Samia Nawshin, Assistant Professor, Department of
Computer Science and Engineering, Daffodil International University.
We also declare that neither this project nor any part of this project
has been submitted elsewhere for the award of any degree or diploma.

**Supervised by:** Ms. Samia Nawshin, Assistant Professor

**Co-Supervised by:** Zakia Sultana Eshita, Lecturer (Senior Scale)

**Submitted by:** Tasmin Jahan (0242310005101552), Sabekunnaher Smrity
(0242310005101606)

---

## ACKNOWLEDGEMENTS

This work would not have been possible without the support and
contributions of many individuals over the past two semesters. We are
deeply grateful to everyone who has assisted us in one way or another.

First, we express our heartfelt thanks and gratefulness to the Almighty
for His divine blessing making it possible for us to complete the
Final Year Design Project (FYDP) successfully.

We are grateful and wish to express our profound indebtedness to our
supervisor, Ms. Samia Nawshin, Assistant Professor, Department of
Computer Science and Engineering, Daffodil International University,
Dhaka, Bangladesh. Her deep knowledge and keen interest in the field of
medical-imaging deep learning enabled us to carry out this project.
Her endless patience, scholarly guidance, continual encouragement,
constant and energetic supervision, constructive criticism, valuable
advice, reading many inferior drafts, and correcting them at all stages
have made it possible to complete this project.

We would like to express our heartfelt gratitude to the Head of the
Department of Computer Science and Engineering for his kind help in
finishing our project, and also to other faculty members and the staff
of the Department of Computer Science and Engineering, Daffodil
International University.

We would like to thank our course-mates at Daffodil International
University who took part in the discussion while completing the
coursework.

Finally, we must acknowledge with due respect the constant support and
patience of our parents.

---

## ABSTRACT

Polycystic Ovary Syndrome (PCOS) is one of the most prevalent
endocrine disorders affecting women of reproductive age, with global
frequencies between 8% and 12% depending on the diagnostic criteria
applied. The clinical diagnostic pipeline currently relies on a
combination of hormonal blood panels and ovarian ultrasound, a process
that is time-consuming, operator-dependent, and inconsistently applied.
Deep-learning systems have been proposed as triage support, but most
prior work evaluates on a single source dataset and produces a single
SoftMax probability that is mathematically known to be overconfident
and uncalibrated.

This project (PEARL — Probabilistic Explainability with Adaptive
Reliability via transfer Learning) develops a transfer-learning-based
deep-learning pipeline for binary classification of transvaginal
ovarian ultrasound images into PCOS-positive and PCOS-negative. The
contributions are: (i) a 9-architecture × 2-preprocessing foundation
benchmark on the Figshare PCOS Ultrasound Dataset, followed by external
validation on the independent PCOSgen cohort; (ii) a quantitative
diagnosis of a letterbox-padding preprocessing shortcut that no other
PCOS study has audited, with a +0.25 absolute AUC recovery after the
shortcut is removed; (iii) fine-tuning of all 18 configurations on the
PCOSgen train pool and a three-model probability-averaged ensemble
(DenseNet-121, ConvNeXt-Tiny, ViT-B/16) that achieves external AUC
0.9487 with 95% CI [0.9368, 0.9596]; (iv) a two-pass temperature-
scaling calibration that reduces the ensemble Expected Calibration
Error from 0.0854 to 0.0810; (v) MC-Dropout uncertainty quantification
producing per-case predictive entropy with 50 stochastic forward
passes; and (vi) Grad-CAM-based verification that the model's
attention has migrated from the padding border to the scan interior
after fine-tuning. The recommended deployed model is the pre-HPO
probability-averaged ensemble.

**Keywords:** Polycystic Ovary Syndrome, ovarian ultrasound, transfer
learning, cross-dataset generalisation, Speckle-Reducing Anisotropic
Diffusion, calibration, MC-Dropout, Grad-CAM, uncertainty estimation,
ensemble learning.

---

# Table of Contents

Approval — i
Declaration — ii
Acknowledgements — iii
Abstract — iv
List of Figures — vii
List of Tables — viii

1. Introduction — 1
   1.1 Introduction — 1
   1.2 Motivation — 1
   1.3 Objectives — 1
   1.4 Methodology — 2
   1.5 Project Outcome — 2
   1.6 Organisation of the Report — 2
2. Background — 3
   2.1 Introduction — 3
   2.2 Literature Review — 3
   2.3 Gap Analysis — 5
   2.4 Summary — 6
3. Research Methodology — 7
   3.1 Methodology / Requirement Analysis — 7
   3.2 Detailed Methodology and Design — 8
   3.3 Project Plan — 9
   3.4 Task Allocation — 9
   3.5 Summary — 10
4. Implementation and Results — 11
   4.1 Environment Setup — 11
   4.2 Testing and Evaluation / Comparative Analysis — 11
   4.3 Results and Discussion — 12
   4.4 Summary — 14
5. Engineering Standards and Design Challenges — 15
   5.1 Compliance with Standards — 15
   5.2 Impact on Society, Environment, and Sustainability — 15
   5.3 Project Management and Financial Analysis — 16
   5.4 Complex Engineering Problem — 16
   5.5 Summary — 17
6. Conclusion — 18
   6.1 Summary — 18
   6.2 Limitations — 18
   6.3 Future Work — 18

References — 19

# List of Figures

Figure 1.1 Internal AUC across 18 foundation configurations — 1
Figure 1.2 External AUC across the same 18 foundation configurations — 1
Figure 2.1 18-run sweep matrix heatmap — 5
Figure 3.1 Grad-CAM border-attention diagnosis — 8
Figure 4.1 Combined ROC of top-3 ensemble members and ensemble — 12
Figure 4.2 Combined precision-recall curves — 12
Figure 4.3 Three-stage confusion-matrix progression — 12
Figure 4.4 Generalisation recovery arc — 12
Figure 4.5 Reliability diagrams (before/after two-pass calibration) — 13
Figure 4.6 MC-Dropout entropy per ensemble member — 13
Figure 4.7 Grad-CAM panel grid for top-3 ensemble members — 13
Figure 4.8 Ensemble metric bars (vs single models) — 13
Figure 4.9 Padding vs no-pad external AUC — 13
Figure 4.10 Sensitivity vs FPR at default threshold — 13
Figure 4.11 Per-class P/R/F1 at default threshold — 14
Figure 4.12 Threshold sweep — 14
Figure 4.13 Predictive entropy histogram — 14
Figure 4.14 Training curves for the three ensemble members — 14

# List of Tables

Table 2.1 Summary of Literature Reviewed — 4
Table 4.1 Headline metrics — 12
Table 4.2 Calibration before/after two-pass temperature scaling — 13
Table 4.3 MC-Dropout entropy summary — 13
Table 5.1 Mapping with Complex Engineering Problem — 16
Table 5.2 Mapping with Knowledge Profile — 16
Table 5.3 Mapping with Complex Engineering Activities — 17

---

## Chapter 1: Introduction

This chapter motivates the PCOS detection problem, states the
project objectives, summarises the methodology at a high level, lists
the project outcomes, and describes the chapter-by-chapter structure
of the report.

### 1.1 Introduction

Polycystic Ovary Syndrome (PCOS) is one of the most prevalent
endocrine disorders among women, with global frequencies between 8%
and 12% depending on the diagnostic criteria applied [1,4,7].
Despite its prevalence, most cases remain undetected because no
single test can definitively confirm PCOS; clinicians rely on a
combination of hormonal blood panels and ovarian ultrasound, a
process that is time-consuming and operator-dependent [5,11,13].
Ultrasound images are inherently difficult to interpret due to
speckle noise, low contrast, and variable acquisition technique [8].

Deep learning, particularly transfer-learned CNNs, has achieved
notable classification accuracy in automated PCOS detection,
frequently exceeding 95% on benchmark datasets [1,3,7,8,25]. However,
accuracy alone does not guarantee clinical trustworthiness. Current
models output a single SoftMax probability value that is
mathematically established to be overconfident and uncalibrated [22].
No prior PCOS study has formally quantified Expected Calibration
Error or applied MC-Dropout-based uncertainty quantification on an
external cohort. Furthermore, prior work evaluates architectures
against fixed preprocessing pipelines, leaving the interaction
between preprocessing choice and model performance systematically
unexplored [4].

[Figure 1.1: `phase-1/fig_01_internal_auc.png` — internal AUC across the 18 foundation configurations.]

[Figure 1.2: `phase-1/fig_02_external_auc.png` — external AUC on the same configurations showing the generalisation collapse.]

This project proposes a comprehensive, trust-oriented deep-learning
pipeline that incorporates calibration analysis, MC-Dropout
uncertainty quantification, an uncertainty-aware operating point, and
explainability, evaluated on the Figshare → PCOSgen dataset pair.

### 1.2 Motivation

The computational motivation is to close the gap between a model
that scores well in-distribution and one that is safe to deploy
across cohorts. The clinical motivation is to provide a triage tool
that flags PCOS suspicion without making treatment decisions, and to
report model confidence in a form that clinicians can interpret.

### 1.3 Objectives

The project pursues five objectives:

- **O1.** Benchmark nine pre-trained CNN architectures against two
  preprocessing pipelines on the Figshare PCOS Ultrasound Dataset and
  identify the optimal configuration.
- **O2.** Diagnose the letterbox-padding preprocessing shortcut and
  quantify the recovery in external AUC after the border is removed.
- **O3.** Fine-tune the foundation configurations on the PCOSgen
  external cohort and construct a probability-averaged ensemble of the
  top three architectures.
- **O4.** Apply two-pass temperature scaling to the ensemble and report
  Expected Calibration Error before and after.
- **O5.** Quantify per-case uncertainty using MC-Dropout and verify
  attention migration using Grad-CAM.

### 1.4 Methodology

The methodology is summarised in Chapter 3 and detailed in the
companion paper (`docs/latex/pearl.tex`). Briefly: nine timm
backbones are transfer-learned on the deduplicated Figshare corpus
under two preprocessing pipelines (SRAD-nopad and Gaussian-nopad),
producing 18 foundation checkpoints. These are externally evaluated
on PCOSgen. The 18 checkpoints are fine-tuned on the PCOSgen train
pool; the top three by external AUC are combined by probability
averaging. The ensemble is calibrated with two-pass temperature
scaling, evaluated for predictive uncertainty with 50 MC-Dropout
forward passes, and audited for attention migration with Grad-CAM.

### 1.5 Project Outcome

The recommended deployed model is the pre-HPO probability-averaged
ensemble, with external AUC **0.9487** (95% CI 0.9368–0.9596),
F1 **0.8665**, MCC **0.7884**, Brier **0.0981**, and ECE reduced
from 0.0854 to **0.0810** by two-pass temperature scaling. The
project delivers the artefacts, the verification scripts, and the
calibrated ensemble checkpoint under `results/finetune_zenodo/`.

### 1.6 Organisation of the Report

Chapter 2 surveys the related literature. Chapter 3 describes the
research methodology, the proposed design, and the functional and
non-functional requirements. Chapter 4 reports the implementation
and results. Chapter 5 discusses engineering standards, impact, and
project management. Chapter 6 concludes.

---

## Chapter 2: Background

This chapter places PEARL in the context of prior PCOS detection
work and adjacent literature on calibration, uncertainty
quantification, and explainability.

### 2.1 Introduction

The background study in the FYDP Title Defence covered 25 references
across six themes: clinical-feature ML, CNN transfer learning on
ultrasound, segmentation, XAI methodology, uncertainty
quantification, and post-hoc calibration. The two systematic
literature reviews [7,13] both identify the same literature-wide
gaps: over-reliance on a small set of public datasets, almost no
external validation, and explainability applied in only about 25%
of studies.

### 2.2 Literature Review

The full literature table is reproduced from the Title Defence
document. Selected representative works are summarised below.

**Table 2.1: Summary of Literature Reviewed.**

| # | Author(s) (Year) | Title (short) | Method | Key finding | Limitation |
|---|-------------------|---------------|--------|-------------|------------|
| [1] | Sundari et al. (2025) | Transfer-learning enhanced CNN for PCOS | Enhanced EfficientNet-B3 + attention + Grad-CAM | Acc 94.8%, Sens 93.2%, Spec 95.5% | Single-source dataset |
| [3] | Moral et al. (2024) | CystNet | Watershed + autoencoder + ensemble ML | Acc 96.54% (FC), 97.75% (ML) | Single source; no XAI |
| [4] | Lakshmi & Pushpa (2026) | Multimodal cross-attention for PCOS | ResNet50 + Transformer + cross-attention | Outperforms unimodal baselines | Gaussian filtering inadequate for speckle |
| [5] | Shanmugavadivel et al. (2024) | Dual-pipeline PCOS prognosis | LR/NB/SVM + VGG16 (clinical + US) | VGG16: 98.29% | Dual pipelines not fused |
| [7] | Ghaderzadeh et al. (2025) | SLR on AI in PCOS | PRISMA SLR | CNN models > 95% acc; only ~25% XAI | Overuse of Kaggle |
| [8] | Reka et al. (2025) | SAM + ESRGAN for PCOS | ESRGAN + SAM + CNN classifiers | VGG19: 99.31% | Limited external validation |
| [11] | Mahesswari & Maheswari (2024) | SmartScanPCOS | TOMIM + two-level RF + Shapash | 99.31% acc | Limited dataset |
| [13] | Suha & Islam (2023) | SLR on PCOS detection | PRISMA SLR | ML dominant; DL rising | Country-specific bias |
| [14] | Bedi et al. (2024) | AResUNet for PCOS | Adaptive bilateral + AResUNet | ~2% improvement | Limited external validation |
| [16] | Gal & Ghahramani (2016) | Dropout as Bayesian | MC-Dropout | Dropout ≈ variational inference | MC passes expensive |
| [17] | Selvaraju et al. (2017) | Grad-CAM | Gradient-based localisation | Applicable to all CNNs | Coarse heatmaps |
| [19] | Akram et al. (2025) | Bayesian extensions on PCOS | DenseNet-121 + MC-Dropout + MFVI | MC-Dropout 97.68% | No calibration analysis |
| [22] | Guo et al. (2017) | Calibration of modern NNs | Temperature scaling | TS most effective calibration | Restricted to classification |
| [25] | Ghosh & Srinivasan (2025) | Ensemble transfer learning for PCOS | EfficientNetB7 + DenseNet201 + GA | Acc 99.58%, AUC 98.97% | Kaggle only; no external cohort |

The adjacent literature on calibration [22], uncertainty
quantification [16,18,19], and XAI [17,21,23] is borrowed from the
broader deep-learning community, since these tools were not developed
for PCOS specifically.

#### 2.2.1 Similar Applications

Within the Title Defence table, the closest comparable systems are
Ghosh & Srinivasan 2025 [25] (ensemble, in-distribution only),
Akram et al. 2025 [19] (MC-Dropout, single source), and Sundari et
al. 2025 [1] (Grad-CAM, in-distribution). Outside PCOS, calibrated
CNN ensembles with MC-Dropout are common in dermatology
classification and chest-radiograph triage; the same design pattern
applies here.

### 2.3 Gap Analysis

The literature review surfaces four gaps that PEARL is positioned to
address:

1. **No external cohort.** Every CNN transfer-learning study in the
   table uses a single source. PEARL trains on Figshare and
   externally validates on PCOSgen.
2. **No preprocessing-architecture ablation.** Prior work uses fixed
   preprocessing and varies only the architecture. PEARL runs a 9 × 2
   sweep and quantifies the letterbox-padding shortcut.
3. **No calibration in PCOS.** Guo 2017 [22] established the
   miscalibration of modern NNs on ImageNet; PEARL applies
   temperature scaling to PCOS detection with a two-pass protocol.
4. **No preprocessing-shortcut diagnosis.** None of the prior PCOS
   studies audited whether the model attends to the padding border
   or to the ovary. PEARL does, and uses Grad-CAM to verify the
   migration.

[Figure 2.1: `phase-1/fig_07_sweep_matrix.png` — 18-run sweep matrix.]

### 2.4 Summary

PEARL closes the four identified gaps. It does not claim to be the
first PCOS deep-learning paper (Sundari 2025, CystNet 2024, and
others preceded it), nor the first to use Grad-CAM, MC-Dropout, or
temperature scaling in general; each of these has prior art in the
Title Defence table. PEARL's novelty is the joint execution on a
cross-dataset pair with a verified preprocessing-shortcut diagnosis.

---

## Chapter 3: Research Methodology

This chapter describes the research design, the proposed
methodology, the functional and non-functional requirements, the
data flow, and the project plan.

### 3.1 Methodology / Requirement Analysis & Design Specification

#### 3.1.1 Overview

PEARL is a research-oriented project: a quantitative experimental
study with controlled variables. The dataset, preprocessing,
training, calibration, uncertainty, and explainability layers are
treated as separate components of a single pipeline. The pipeline
operates on a single dataset pair (Figshare → PCOSgen) and produces
a calibrated ensemble checkpoint as its output.

#### 3.1.2 Proposed Methodology / System Design

The pipeline has six stages:

1. **Ingest.** Download Figshare and PCOSgen.
2. **Deduplicate.** MD5 audit; remove byte-level duplicates.
3. **Split.** Stratified 80/10/10 on Figshare; 2,560/640 train/val
   on PCOSgen; 1,468 held-out test.
4. **Preprocess.** SRAD-nopad and Gaussian-nopad pipelines.
5. **Train.** Transfer-learn nine architectures; fine-tune on
   PCOSgen; ensemble the top three.
6. **Calibrate + Explain.** Two-pass temperature scaling; MC-Dropout
   entropy; Grad-CAM panels.

**Figure 3.1: `phase-1/fig_06_border_attention.png` — Grad-CAM
border-attention diagnosis is the entry point to the no-pad
preprocessing choice.**

#### 3.1.3 Functional and Non-functional Requirements

**Functional.** The system ingests a directory of ultrasound images,
applies the preprocessing pipeline selected at the configuration
level, returns a probability of PCOS-positive, an uncertainty
estimate (predictive entropy), and a Grad-CAM attention map. It
persists the artefacts under `results/`.

**Non-functional.** The system runs on a single RTX 4070 SUPER
12 GB. The full pipeline runs in under 6 hours wall clock. All
seeds are recorded. Every numerical claim in this report is
traceable to an artefact.

### 3.2 Detailed Methodology and Design

Alternative preprocessing pipelines considered: Perona-Malik
anisotropic diffusion (used in earlier-phase design; not
multiplicative-speckle-aware), bilateral filtering (edge-preserving
but slower), and wavelet denoising (computationally costly, less
common in deep-learning pipelines). SRAD was selected because it is
designed for multiplicative speckle noise characteristic of
ultrasound, and because the ablation showed SRAD-nopad and
Gaussian-nopad are competitive on this dataset pair, with SRAD
marginally preferred.

### 3.3 Project Plan

The project plan was executed in three phases over the FYDP cycle:

- **Phase A — Foundation.** Data audit, deduplication, 9 × 2 sweep,
  external evaluation, padding-shortcut diagnosis.
- **Phase B — Fine-tune.** Fine-tune 18 configurations on PCOSgen;
  rank by external AUC; ensemble top 3.
- **Phase C — Calibrate and explain.** Two-pass calibration;
  MC-Dropout entropy; Grad-CAM panel.

### 3.4 Task Allocation

| Task | Lead | Support |
|------|------|---------|
| Dataset sourcing & dedup | Tasmin | Sabekunnaher |
| SRAD / Gaussian implementation | Sabekunnaher | Tasmin |
| Foundation 9 × 2 sweep | Tasmin | Sabekunnaher |
| External evaluation | Sabekunnaher | Tasmin |
| Border-attention audit | Tasmin | — |
| Fine-tune on PCOSgen | Tasmin | Sabekunnaher |
| Ensemble construction | Tasmin | Sabekunnaher |
| Calibration | Sabekunnaher | — |
| MC-Dropout | Tasmin | — |
| Grad-CAM | Sabekunnaher | Tasmin |
| Writing and verification | Tasmin | Sabekunnaher |

### 3.5 Summary

The methodology is a controlled experimental design with six
sequential stages. Each stage produces an artefact that the next
stage consumes, and every numerical claim in this report traces to
an artefact under `results/`. The alternative denoisers and
preprocessing choices are documented for reproducibility.

---

## Chapter 4: Implementation and Results

This chapter reports the implementation, the evaluation procedure,
and the headline results.

### 4.1 Environment Setup

- **Hardware.** NVIDIA RTX 4070 SUPER, 12 GB; one workstation.
- **OS.** Ubuntu 22.04 LTS.
- **Python.** 3.10+.
- **PyTorch.** 2.12.0 with CUDA 13.0.
- **timm.** 1.0.27 (transfer-learning backbones).
- **scikit-learn.** 1.9.0 (splits, metrics).
- **OpenCV.** 4.13.0.92 (preprocessing).
- **Optuna.** 4.9.0 (HPO).
- **NumPy / SciPy / matplotlib.** 2.4 / 1.17 / 3.10.

Total wall clock for foundation + fine-tune is approximately 5–6
hours. The reproducibility appendix (`docs/thesis/appendix/D_reproducibility.tex`)
gives the exact command-line invocations.

### 4.2 Testing and Evaluation / Comparative Analysis

The reported model is evaluated on the held-out PCOSgen test set
(n = 1,468) which is never seen during training or fine-tuning.
Metrics: AUC-ROC, F1, MCC, Brier, ECE, sensitivity, specificity.
Bootstrap 95% confidence intervals are reported for the primary
metric (AUC). Comparative baselines:

- **Foundation stage (no fine-tune).** Mean external AUC 0.5222
  padded, 0.7704 no-pad.
- **Single best fine-tuned model.** DenseNet-121 + SRAD-nopad,
  external AUC 0.9420.
- **Three-model ensemble (pre-HPO).** External AUC **0.9487**.
- **Three-model ensemble (post-HPO).** External AUC 0.9408.

The pre-HPO ensemble is the recommended deployed model.

### 4.3 Results and Discussion

#### 4.3.1 Headline metrics

**Table 4.1: Headline metrics on the held-out PCOSgen test set
(n = 1,468).**

| Metric | Value |
|--------|-------|
| AUC-ROC (95% CI) | **0.9487** (0.9368–0.9596) |
| F1 | **0.8665** |
| MCC | **0.7884** |
| Brier | **0.0981** |
| ECE before pass 1 | 0.0854 |
| ECE after pass 2 | **0.0810** |

#### 4.3.2 Three-stage confusion matrices

[Figure 4.3: `phase-1/fig_10_confusion_stages.png` — three-stage
confusion-matrix progression: foundation zero-shot, fine-tuned best
single model, pre-HPO ensemble.]

The foundation stage fails in the screening-relevant direction: it
predicts PCOS for almost every image because the Figshare training
distribution is 3.92:1 PCOS-heavy. After fine-tuning, the
false-positive count drops sharply; the ensemble's specificity
improves from 0.041 to 0.822 while keeping sensitivity above 0.99.

#### 4.3.3 Internal and external AUC curves

[Figure 4.1: `phase-1/fig_08_combined_roc.png` — ROC curves of the
three ensemble members and the pre-HPO ensemble.]

[Figure 4.2: `phase-1/fig_09_combined_pr.png` — PR curves; the
ensemble reaches average precision 0.8702.]

[Figure 4.4: `phase-1/fig_11_recovery.png` — generalisation recovery
arc from foundation zero-shot (0.45–0.59) to fine-tune (0.93) to
ensemble (0.9487).]

#### 4.3.4 Calibration

[Figure 4.5: `phase-1/fig_12_calibration.png` — reliability diagrams
before and after two-pass temperature scaling.]

**Table 4.2: Calibration before/after two-pass temperature
scaling.**

| Stage | ECE | Brier | NLL |
|-------|-----|-------|-----|
| Pre-calibration (per-model T = 1) | 0.0854 | 0.1306 | 0.40 |
| After pass 1 (per-model T) | 0.0854 | 0.1059 | 0.34 |
| After pass 2 (ensemble T = 0.93) | **0.0810** | **0.0981** | 0.32 |

Per-model temperatures are [2.0162, 1.9579, 2.1982]; the ensemble
temperature is 0.9306. The two passes calibrate the ensemble in
directions that are not aligned, motivating the two-pass protocol.

#### 4.3.5 MC-Dropout uncertainty

50 stochastic forward passes per test image produce per-case
predictive entropy. The distributions differ across architectures.

**Table 4.3: MC-Dropout entropy summary (nats).**

| Architecture | Mean entropy | Median entropy |
|--------------|--------------|----------------|
| DenseNet-121 | 0.234 | 0.151 |
| ConvNeXt-Tiny | 0.215 | 0.139 |
| ViT-B/16 | 0.160 | 0.089 |

[Figure 4.6: `phase-1/fig_13_uncertainty.png` — MC-Dropout entropy
per ensemble member.]

[Figure 4.13: `phase-1/fig_18_uncertainty_hist.png` — predictive
entropy histogram for correct vs incorrect predictions.]

#### 4.3.6 Grad-CAM verification

[Figure 4.7: `phase-1/fig_14_xai_grid.png` — Grad-CAM panel grid for
the top-3 ensemble members.]

After fine-tuning, the model's attention has migrated from the
padding border to the scan interior. The padding-shortcut diagnosis
is therefore closed by the fine-tune stage.

#### 4.3.7 Per-class precision/recall, threshold sweep, training curves

[Figure 4.8: `phase-1/fig_15_ensemble_metrics.png` — ensemble vs
single-model metric bars.]

[Figure 4.9: `phase-1/fig_05_padding_vs_nopad.png` — padding vs
no-pad external AUC (+0.25 absolute gain).]

[Figure 4.10: `phase-1/fig_04_sens_vs_fpr.png` — sensitivity vs FPR
at default 0.5 threshold.]

[Figure 4.11: `phase-1/fig_16_per_class_pr.png` — per-class
precision, recall, F1.]

[Figure 4.12: `phase-1/fig_17_threshold_sweep.png` — threshold sweep
(sens/spec/F1 vs threshold).]

[Figure 4.14: `phase-1/fig_19_training_curves.png` — training curves
for the three ensemble members.]

The operating-point comparison shows that the default 0.5 threshold
produces specificity between 0.20 and 0.45 across the 18 foundation
configurations. The recomputed operating points for the pre-HPO
ensemble are: default 0.5 → sensitivity 0.9927, specificity 0.8215;
Youden-J (0.6198) → sensitivity 0.9909, specificity 0.8324;
sensitivity-targeted (0.7597) → sensitivity 0.9500, specificity
0.8553.

### 4.4 Summary

The pre-HPO probability-averaged ensemble achieves external AUC
0.9487 with F1 0.8665 and MCC 0.7884 on the held-out PCOSgen test
set. The two-pass temperature scaling reduces ECE from 0.0854 to
0.0810. MC-Dropout entropy is well separated across architectures
and supports a coverage-accuracy referral design. Grad-CAM confirms
that the model's attention has moved from the padding border to the
scan interior. The padding-shortcut diagnosis is the most original
finding.

---

## Chapter 5: Engineering Standards and Design Challenges

This chapter covers engineering standards, societal/environmental
impact, project management, and the complex-engineering-problem
mapping required by the FYDP rubric.

### 5.1 Compliance with Standards

#### 5.1.1 Software Standards

The implementation uses PyTorch as the deep-learning framework, NumPy
and SciPy for numerical work, scikit-learn for splits and metrics, and
timm for backbones. The codebase is organised under `src/` and
`scripts/`; configurations are YAML; experiments are tracked under
`results/`. Git is used for version control with deterministic seed
documentation.

#### 5.1.2 Hardware Standards

The single-GPU target is NVIDIA RTX 4070 SUPER (12 GB). All
configuration settings (batch size, gradient accumulation) are chosen
to remain within this envelope. CUDA seeding is best-effort;
deterministic algorithms are enabled where available.

#### 5.1.3 Communication Standards

Artefacts are persisted as JSON / CSV / YAML. Figures are PNG. The
report and thesis use IEEE citation format consistent with the FYDP
Title Defence.

### 5.2 Impact on Society, Environment, and Sustainability

#### 5.2.1 Impact on Life

The pipeline is intended as a triage step, not a diagnostic
authority. PCOS detection at scale reduces time-to-suspicion for
women in under-served settings where ultrasound expertise is
limited.

#### 5.2.2 Impact on Society and Environment

A single-GPU workstation consumes on the order of 250 W during
training. The full pipeline runs in under 6 hours; total energy
consumed is under 1.5 kWh per execution. Cloud deployment is
optional and would scale proportionally.

#### 5.2.3 Ethical Aspects

The training data are public, de-identified ultrasound images.
PCOSgen is used under the published terms. Patient-level grouping
is not available in PCOSgen; therefore the held-out split cannot be
verified as patient-independent and we recommend nested
cross-validation as future work.

#### 5.2.4 Sustainability Plan

The codebase and YAML configurations are checked into the
repository. Future re-runs can reproduce the headline numbers from
`results/finetune_zenodo/`. The companion paper and thesis are the
primary documentation artefacts.

### 5.3 Project Management and Financial Analysis

The project is research-only. No direct financial cost to the
university beyond existing compute. The full 6-hour wall clock is
within the FYDP student-hour budget. In an alternative budget
scenario, paid cloud compute (e.g., AWS p3.2xlarge) would cost on
the order of USD 5–10 per full re-run.

### 5.4 Complex Engineering Problem

The PEARL pipeline touches all seven Complex Problem Solving
attributes and all eight Engineering Practice knowledge areas.

**Table 5.1: Mapping with Complex Engineering Problem.**

| EP1 Knowledge | EP2 Conflicting Reqs | EP3 Analysis Depth | EP4 Familiarity | EP5 Applicable Codes | EP6 Stakeholder | EP7 Interdependence |
|---------------|----------------------|--------------------|-----------------|----------------------|------------------|----------------------|
| Deep learning, statistics, medical-imaging | Accuracy vs calibration vs interpretability | Formal ECE / reliability-diagram analysis | PCOS imaging literature | Rotterdam criteria; DICOM | Clinicians, patients, regulators | Cross-cohort evaluation with audit trail |

**Table 5.2: Mapping with Knowledge Profile.**

| K1 Natural Sci | K2 Math | K3 Eng Fund | K4 Specialist | K5 Eng Design | K6 Eng Practice | K7 Comprehension | K8 Research Lit |
|----------------|---------|--------------|----------------|----------------|------------------|-------------------|------------------|
| — | Statistical cal, ECE | Software eng | Deep learning for medical imaging | Pipeline + ablation design | Reproducibility | Trade-offs across objectives | SLRs [7,13], Guo [22], Gal [16] |

**Table 5.3: Mapping with Complex Engineering Activities.**

| EA1 Resources | EA2 Interaction | EA3 Innovation | EA4 Society/Env | EA5 Familiarity |
|---------------|-----------------|----------------|------------------|------------------|
| Single-GPU workstation | Supervisor + committee | Padding-shortcut diagnosis; two-pass calibration | Reduced time-to-suspicion | PCOS literature and adjacent deep-learning |

### 5.5 Summary

PEARL conforms to the engineering standards expected of an
FYDP-grade implementation and addresses the FYDP rubric's complex
engineering problem mapping in full. The societal and ethical
limitations of the system are documented honestly, including the
unverifiable patient-level grouping on PCOSgen.

---

## Chapter 6: Conclusion

### 6.1 Summary

PEARL is a transfer-learning-based deep-learning pipeline for
binary classification of transvaginal ovarian ultrasound images into
PCOS-positive and PCOS-negative. The recommended deployed model is
the pre-HPO probability-averaged ensemble, with external AUC 0.9487
on the held-out PCOSgen test set. The pipeline also delivers a
two-pass temperature-scaling calibration that reduces ECE from
0.0854 to 0.0810, MC-Dropout per-case predictive entropy, and
Grad-CAM verification of attention migration.

The most original finding is the letterbox-padding preprocessing
shortcut: foundation configurations attend to the black border
rather than to the ovary, and disabling padding recovers +0.25
absolute external AUC. This is a concrete, quantitative improvement
that no prior PCOS study has reported.

### 6.2 Limitations

- **No patient-level grouping.** PCOSgen does not document a
  verified patient-independent grouping, so the held-out split
  cannot be guaranteed patient-independent. The bootstrap 95% CI
  on AUC is reported, but nested cross-validation is recommended
  as future work.
- **Single held-out cohort.** The cross-dataset evaluation uses one
  external cohort (PCOSgen). A second external cohort would
  strengthen the generalisation claim.
- **No prospective trial.** No prospective clinical evaluation has
  been performed.
- **Reduced preprocessing grid.** The 9 × 2 ablation replaces the
  originally proposed 9 × 6 grid to fit the single-GPU budget.

### 6.3 Future Work

- **Second external cohort.** Extend the evaluation to a
  second publicly released PCOS ultrasound dataset when one
  becomes available.
- **Nested cross-validation.** Implement 5-fold patient-level
  cross-validation on a unified Figshare + PCOSgen corpus,
  provided a patient-independent grouping can be established.
- **Uncertainty-gated referral.** A deployed triage tool with a
  defined entropy cutoff and a target coverage could be evaluated
  prospectively in a clinical setting.
- **Federated multi-centre training.** With IRB approval, a
  federated training scheme across multiple hospitals would
  reduce the single-source bias identified by both SLRs [7,13].
- **Multimodal fusion.** Hormonal panels and ultrasound could be
  fused by cross-attention, following Lakshmi & Pushpa 2026 [4].

---

## References

[1] M. S. Sundari, N. V. Sailaja, D. Swapna, V. C. Jadala, and K.
Durga, "Transfer learning enhanced CNN model for integrative
ultrasound and biomarker-based diagnosis of polycystic ovarian
disease," *Sci. Rep.*, vol. 15, p. 34519, Oct. 2025, doi:
10.1038/s41598-025-17711-w.

[2] M. Agirsoy and M. A. Oehlschlaeger, "A machine learning approach
for non-invasive PCOS diagnosis from ultrasound and clinical
features," *Sci. Rep.*, vol. 15, Art. no. 33638, Sep. 2025, doi:
10.1038/s41598-025-10453-9.

[3] P. Moral, D. Mustafi, A. Mustafi, and S. K. Sahana, "CystNet: An
AI driven model for PCOS detection using multilevel thresholding
of ultrasound images," *Sci. Rep.*, vol. 14, p. 25012, Oct. 2024,
doi: 10.1038/s41598-024-75964-3.

[4] V. Lakshmi and B. Pushpa, "Explainable multimodal deep learning
using cross-attention fusion of ultrasound and clinical features
for PCOS classification," *Discov. Comput.*, vol. 29, no. 1, p. 7,
2026, doi: 10.1007/s10791-025-09901-x.

[5] K. Shanmugavadivel, M. S. Murali Dhar, T. R. Mahesh, T.
Al-Shehari, N. A. Alsadhan, and T. E. Yimer, "Optimized polycystic
ovarian disease prognosis and classification using AI based
computational approaches on multi-modality data," *BMC Med.
Inform. Decis. Mak.*, vol. 24, no. 1, p. 281, Oct. 2024, doi:
10.1186/s12911-024-02688-9.

[6] P. B. Patil, Rashmi M., Natesha B. V., and R. D. Shetty,
"Explainable ensemble-based machine learning model for polycystic
ovary syndrome detection using hybrid feature selection," *Int. J.
Inf. Technol.*, 2025, doi: 10.1007/s41870-025-03044-4.

[7] C. Salehnasab, M. Ghaderzadeh, and A. Garavand, "Artificial
intelligence in polycystic ovary syndrome: a systematic review of
diagnostic and predictive applications," *BMC Med. Inform. Decis.
Mak.*, vol. 25, p. 427, Nov. 2025, doi: 10.1186/s12911-025-03255-6.

[8] S. Reka, T. S. Praba, M. Prasanna, K. R. Sri Preethaa, and M.
Shyamala Devi, "Automated high precision PCOS detection through a
segment anything model on super resolution ultrasound ovary
images," *Sci. Rep.*, vol. 15, p. 16832, May 2025, doi:
10.1038/s41598-025-01744-2.

[9] P. Jeyashanker, A. G. V. G. Sundaram, P. Sadagopan, A. Yahya,
R. Samikannu, I. A. Badruddin, S. Kamangar, and M. G. Shukur,
"Advanced holographic convolutional dense networks and Tangent
runner optimization for enhanced polycystic ovarian disease
classification," *Sci. Rep.*, vol. 15, Art. no. 15719, May 2025,
doi: 10.1038/s41598-025-98873-5.

[10] B. Poorani and R. Khilar, "An innovative approach for PCO
morphology segmentation using a novel MOT-SF technique," *Discov.
Comput.*, vol. 27, p. 27, 2024, doi: 10.1007/s10791-024-09458-1.

[11] G. U. Mahesswari and P. U. Maheswari, "SmartScanPCOS: A
feature-driven approach to cutting-edge prediction of polycystic
ovary syndrome using machine learning and explainable artificial
intelligence," *Heliyon*, vol. 10, no. 20, p. e39205, Oct. 2024,
doi: 10.1016/j.heliyon.2024.e39205.

[12] E. Nsugbe, "An artificial intelligence-based decision support
system for early diagnosis of polycystic ovaries syndrome,"
*Healthc. Anal.*, vol. 3, p. 100164, Nov. 2023, doi:
10.1016/j.health.2023.100164.

[13] S. A. Suha and M. N. Islam, "A systematic review and future
research agenda on detection of polycystic ovary syndrome (PCOS)
with computer-aided techniques," *Heliyon*, vol. 9, no. 10, p.
e20524, Oct. 2023, doi: 10.1016/j.heliyon.2023.e20524.

[14] P. Bedi, S. K. Das, S. Gupta, S. Saha, and S. Singh, "An
attention residual U-Net-based deep learning model for detection
of polycystic ovary syndrome," *Decis. Anal. J.*, vol. 10, p.
100386, 2024.

[15] J. Lim et al., "Radial pulse wave parameters for PCOS
screening using machine learning," *BMC Complement. Med. Ther.*,
vol. 23, no. 1, p. 397, 2023.

[16] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation:
Representing model uncertainty in deep learning," *Proc. 33rd Int.
Conf. Mach. Learn. (ICML)*, pp. 1050–1059, 2016.

[17] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh,
and D. Batra, "Grad-CAM: Visual explanations from deep networks
via gradient-based localization," *Proc. IEEE Int. Conf. Comput.
Vis. (ICCV)*, pp. 618–626, 2017.

[18] A. G. Roy, S. Conjeti, D. Sheet, A. Katouzian, N. Navab, and
C. Wachinger, "Bayesian fully convolutional networks for
uncertainty-aware segmentation," *Proc. MICCAI*, LNCS 11070, pp.
30–38, 2018.

[19] A. Akram, M. A. Khan, A. Alqahtani, A. Alharbi, and M. S.
Alsubai, "Bayesian deep learning for polycystic ovary syndrome
detection," *Sci. Rep.*, vol. 15, Art. no. 11248, 2025.

[20] J. Zhang, S. A. Bargal, Z. Lin, J. Brandt, X. Shen, and S.
Sclaroff, "Top-down neural attention by excitation backprop,"
*Int. J. Comput. Vis.*, vol. 126, no. 10, pp. 1084–1102, 2018.

[21] W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R.
Müller, "Evaluating the visualization of what a deep neural
network has learned," *IEEE Trans. Neural Netw. Learn. Syst.*,
vol. 28, no. 11, pp. 2660–2673, 2017.

[22] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On
calibration of modern neural networks," *Proc. 34th Int. Conf.
Mach. Learn. (ICML)*, pp. 1321–1330, 2017.

[23] E. Tjoa and C. Guan, "A survey on explainable artificial
intelligence (XAI): Toward medical XAI," *IEEE Trans. Neural
Netw. Learn. Syst.*, vol. 32, no. 11, pp. 4793–4813, 2021.

[24] Y. Fan, S. Li, and H. Wang, "Ocys-Net: A lightweight
polycystic ovary syndrome classification network with efficient
channel attention," *IEEE Access*, vol. 11, pp. 78901–78912, 2023.

[25] A. Ghosh and K. Srinivasan, "EffiDenseGenOp: Ensemble transfer
learning with hyperparameter tuning using genetic algorithm
optimization for PCOS detection from ultrasound sonography
images," *IEEE Access*, vol. 13, 2025, doi:
10.1109/ACCESS.2025.3549888.

[26] A. Indirani, "PCOS Dataset," *figshare*, 2024. [Online].
Available: https://doi.org/10.6084/m9.figshare.27682557.v1

[27] Y. Yu and S. T. Acton, "Speckle reducing anisotropic
diffusion," *IEEE Trans. Image Process.*, vol. 11, no. 11, pp.
1260–1270, 2002.

[28] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama,
"Optuna: A next-generation hyperparameter optimization framework,"
*Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, pp.
2623–2631, 2019.

[29] Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop
Group, "Revised 2003 consensus on diagnostic criteria and
long-term health risks related to polycystic ovary syndrome,"
*Fertil. Steril.*, vol. 81, no. 1, pp. 19–25, 2004.

[30] H. J. Teede et al., "Recommendations from the international
evidence-based guideline for the assessment and management of
polycystic ovary syndrome," *Fertil. Steril.*, vol. 110, no. 3,
pp. 364–379, 2018.