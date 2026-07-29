# FYDP Phase-I Progress Report — Summer 2025

**DAFFODIL INTERNATIONAL UNIVERSITY**
**DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING**
**FYDP (Phase-I) Evaluation Report**

**Reporting period:** Summer 2025

---

## Project Identification

| Field | Value |
|-------|-------|
| **I. Project Title** | Deep Learning-Based PCOS Detection Using Transfer Learning & XAI |
| **II. Group Members** | Tasmin Jahan (ID: 0242310005101552); Sabekunnaher Smrity (ID: 0242310005101606) |
| **III. Supervisor** | Ms. Samia Nawshin, Assistant Professor, Department of CSE |
| **IV. Co-Supervisor** | Zakia Sultana Eshita, Lecturer (Senior Scale), Department of CSE |
| **V. Submission Date** | *(to be filled)* |
| **VI. Certificate** | *(to be signed by supervisor)* |

---

## Project Insights

| Thematic Area(s) | Selected |
|------------------|----------|
| Artificial Intelligence and Machine Learning | ☒ |
| Deep Learning | ☒ |
| Health Informatics | ☒ |
| Computer Vision | ☒ |
| Image Processing | ☒ |

**Software packages, tools, and programming languages**

- Python 3.10+
- PyTorch 2.12.0 with CUDA 13.0
- timm 1.0.27 (transfer-learning backbones)
- scikit-learn 1.9.0 (splits, metrics)
- OpenCV 4.13 (preprocessing)
- matplotlib 3.10, NumPy 2.4, SciPy 1.17 (analysis)
- Optuna 4.9 (planned for fine-tune phase)
- SRAD implementation (NumPy + custom CUDA kernels)
- Git + DVC for experiment tracking
- NVIDIA RTX 4070 SUPER, 12 GB
- Ubuntu 22.04 LTS

---

## CO Description for FYDP-Phase-I

| CO | Description | PO |
|----|-------------|-----|
| **CO4** | Apply suitable project-management procedures throughout the FYDP lifecycle in the context of the "Deep Learning-Based PCOS Detection Using Transfer Learning & XAI" project. | PO11 |
| **CO6** | Select and apply appropriate methodologies, resources, and contemporary engineering tools for prediction and modelling on the PCOS ultrasound dataset. | PO5 |
| **CO7** | Assess societal, health, safety, legal, and cultural issues in a clinical-decision-support deep-learning workflow. | PO6 |
| **CO10** | Operate effectively as a two-person team and as members of a multidisciplinary evaluation committee during FYDP. | PO9 |

---

## 1. Project Overview

### 1.1 Introduction

Polycystic Ovary Syndrome (PCOS) is a heterogeneous endocrine disorder
affecting approximately 8–12% of women of reproductive age
[[1,4,7]](references). The diagnostic workflow currently relies on a
combination of hormonal blood panels and ovarian ultrasound, a process
that is time-consuming, operator-dependent, and inconsistently applied
even where the Rotterdam Consensus Criteria are in use
[[8,11,13]](references). Ultrasound images themselves are hard to
interpret because of speckle noise, low contrast, and acquisition
variability. These factors motivate a deep-learning tool that can
support clinical decision making as a triage or screening step.

This project (PEARL — Probabilistic Explainability with Adaptive
Reliability via transfer Learning) develops a transfer-learning-based
deep-learning pipeline for binary classification of transvaginal
ovarian ultrasound images into PCOS-positive and PCOS-negative.

[Figure 1: `phase-1/fig_01_internal_auc.png` — internal AUC across the 18 foundation configurations (≥0.99 on the Figshare test split).]

[Figure 2: `phase-1/fig_02_external_auc.png` — external AUC for the same 18 configurations evaluated on PCOSgen (0.45–0.59), showing the generalisation collapse that motivates the Phase-II work.]

### 1.2 Background Study

A targeted literature review was conducted in support of the FYDP
proposal. Twenty-five peer-reviewed studies and two systematic
literature reviews were catalogued across six themes: clinical-feature
ML (Refs [2,6,11,12,15]), CNN transfer learning on ultrasound
([1,3,4,5,8,9,14,24,25]), segmentation ([10]), XAI methodology
([17,20,21,23]), uncertainty quantification in medical imaging
([16,18,19]), and post-hoc calibration of neural networks ([22]). The
full table is reproduced in the Title Defense document.

The two SLRs ([7,13]) converge on three recurring gaps in the
literature: (i) over-reliance on a small set of public datasets
(Figshare, Kaggle), (ii) almost no external validation, and (iii)
explainability applied in only about 25% of studies. Every CNN
transfer-learning study in the table either does not externally
validate or flags the absence as a limitation. None of the cited
studies applies a calibration analysis; calibration is referenced only
through Guo 2017 [22], a foundational method paper.

The McGill PCOS study by Sundari et al. 2025 [1] reports 94.8%
accuracy on their own PCOSgen cohort, evaluated in-distribution. The
authors themselves note "single-source dataset limits population
generalisability". The CystNet pipeline [3] reaches 97.75% on a single
source. The Acroustic-Attention Network of Lakshmi 2026 [4] fuses
ultrasound with hormonal features and acknowledges the limitation
that "standard PCOS benchmark datasets are unavailable and
reproducibility is challenging". Ghosh & Srinivasan 2025 [25] ensemble
EfficientNetB7 with DenseNet201 and report AUC 0.9897 — also on a
single Kaggle source.

[Image: 18-run sweep matrix figure would normally be placed here; see full list of figures in `phase-1/README.md`.]

### 1.3 Gap Analysis

The Phase-I background study surfaces four gaps that the present
project is positioned to address:

1. **No external cohort in prior literature.** Every CNN transfer-learning
   study in the Title Defense table uses a single-source dataset.
2. **No preprocessing-architecture ablation.** Most prior work uses a
   fixed preprocessing pipeline and varies only the architecture. The
   interaction between preprocessing choice and architecture is
   systematically unexplored.
3. **No calibration analysis in PCOS detection.** The Guo 2017 [22]
   finding on ImageNet miscalibration has not been evaluated on
   PCOS pipelines.
4. **No documented preprocessing-shortcut diagnosis.** Letterbox
   padding is a common preprocessing step but, to the best of our
   knowledge, none of the prior PCOS studies audited whether the
   model attends to the padding border or to the ovary.

The contributions claimed by PEARL are positioned against these
gaps. Item (1) addresses gap 1; the present report covers items (2)
and (4). Items (3) and the formalised cross-cohort validation are
deferred to Phase-II.

---

## 2. Objectives (Phase-I)

The Phase-I objectives are kept narrow so they map cleanly to what has
been completed in the reporting period:

- **O1.** Collect and characterise the Figshare PCOS Ultrasound Dataset
  and a secondary external cohort (PCOSgen, Sundari et al., 2025) for
  use as a held-out evaluation set.
- **O2.** Audit the Figshare dataset for byte-level duplicates and
  characterise its class distribution before any split.
- **O3.** Implement a 9-architecture × 2-preprocessing foundation
  benchmark (18 runs) covering SRAD and Gaussian denoising, with and
  without letterbox padding.
- **O4.** Diagnose whether any architecture attends to the padding
  border rather than to the ovary, and quantify the recovery in
  external AUC after the border is removed.
- **O5.** Document the internal-vs-external collapse on the dataset
  pair and identify the entry point for Phase-II work.

Phase-II objectives (fine-tuning on the external cohort, ensemble
construction, calibration analysis, uncertainty quantification, and
Grad-CAM verification) are listed in §6 Next Steps and are explicitly
outside the Phase-I scope.

---

## 3. Methodology / Requirement Specification

### 3.1 Research Design

The Phase-I methodology follows the FYDP proposal. It is a
quantitative experimental design with two independent variables
(architecture × preprocessing pipeline), one response variable
(external-cohort AUC), and one diagnostic procedure (per-image
Grad-CAM border-attention audit).

### 3.2 Data Collection / Need Assessment

Two datasets are in scope:

- **Figshare PCOS Ultrasound Dataset** (Indirani, [26]) — the source
  domain. The original Figshare publication contains 11,784 files
  arranged by class label. A byte-level MD5 audit was performed to
  identify duplicates that span the train/test boundary. After
  deduplication, the usable corpus contains 3,184 unique PCOS images
  and 812 unique non-PCOS images (ratio 3.92:1 PCOS-heavy).

- **PCOSgen** (Sundari et al., 2025, [1]) — the external cohort. The
  full corpus contains 4,668 transvaginal ultrasound images. It is
  used in this project solely as the held-out evaluation set, never
  for training.

Class imbalance and dataset provenance are documented because they
determine the failure modes of the foundation-stage models. The
Figshare distribution is 3.92:1 PCOS-positive to healthy, which
biases a model trained on it toward predicting PCOS-positive on a
class-balanced external set.

### 3.3 Analysis Techniques

Phase-I analysis is restricted to:

- **Foundation-stage training.** Each of nine timm backbones
  (ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny,
  MobileNetV3-Large, ViT-B/16, Swin-Tiny, plus two others) is
  transfer-learned on the deduplicated Figshare corpus.
- **Two preprocessing pipelines.** SRAD-nopad and Gaussian-nopad are
  the two pipelines executed. The original FYDP proposal listed six;
  Phase-I uses two to isolate the denoiser variable.
- **External evaluation.** All 18 foundation checkpoints are
  evaluated on PCOSgen. The mean external AUC is the response
  variable.
- **Border-attention audit.** Per-image Grad-CAM is computed on a
  randomly sampled subset of PCOSgen and the activation
  concentration on the letterbox border is recorded. A quantitative
  border-vs-interior ratio is computed.

Calibration analysis (Expected Calibration Error, reliability
diagrams, temperature scaling) and MC-Dropout uncertainty
quantification are deferred to Phase-II.

---

## 4. Progress Achieved

### 4.1 Completed Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Source the Figshare and PCOSgen datasets | Complete |
| 2 | Byte-level MD5 deduplication of Figshare | Complete (3,996 unique images) |
| 3 | Implement SRAD denoiser in NumPy/CUDA | Complete |
| 4 | Implement letterbox vs. no-pad resize | Complete |
| 5 | Train 9 × 2 = 18 foundation configurations on Figshare | Complete |
| 6 | Evaluate all 18 on PCOSgen (held-out) | Complete |
| 7 | Per-image Grad-CAM border-attention audit | Complete |
| 8 | Quantify the padding-shortcut recovery | Complete |

### 4.2 Results Obtained

**Internal evaluation (Figshare held-out test split).** All 18
foundation configurations exceed 0.99 internal AUC. The variation
across configurations is small; the foundation stage has saturated
in-domain.

[Figure 3: `phase-1/fig_01_internal_auc.png` — internal AUC across 18 configurations.]

**External evaluation (PCOSgen, held-out).** The same 18
configurations collapse on the external cohort. External AUC ranges
0.4500–0.5868 with mean 0.5222 across the padded foundation runs.
The collapse is the dominant failure mode of the foundation stage.

[Figure 4: `phase-1/fig_02_external_auc.png` — external AUC across 18 foundation configurations.]

**Internal-vs-external pairing.** When the internal and external
AUCs are plotted per configuration, the gap is consistent: every
configuration that scores high internally scores low externally.

[Figure 5: `phase-1/fig_03_internal_vs_external.png` — paired internal-vs-external AUC per configuration.]

**Sensitivity-specificity at the default 0.5 threshold.** Even on
in-distribution test data, the default 0.5 threshold produces
specificity between 0.20 and 0.45 across the configurations because
of the class-prior shift from training (3.92:1) to test.

[Figure 6: `phase-1/fig_04_sens_vs_fpr.png` — sensitivity vs false-positive rate at default threshold.]

**Padding-shortcut diagnosis.** Foundation configurations trained on
letterbox-padded images attend substantially to the black border. A
per-image Grad-CAM audit quantifies the border-vs-interior
attention split. Disabling padding and resizing directly recovers
**+0.25 absolute AUC** on the external cohort (mean 0.5222 → 0.7704,
n = 18, verified in `docs/latex/findings_table.md`).

[Figure 7: `phase-1/fig_05_padding_vs_nopad.png` — padding vs no-pad external AUC.]

[Figure 8: `phase-1/fig_06_border_attention.png` — before/after Grad-CAM border attention.]

**Sweep matrix.** A heatmap of the 18-run × 7-metric matrix
summarises the foundation-stage result. The takeaway is that
in-domain performance is uniform; cross-domain collapse is
universal. This is the entry point for Phase-II work.

[Figure 9: `phase-1/fig_07_sweep_matrix.png` — 18-run sweep matrix heatmap.]

The Phase-I result is therefore: the foundation stage as designed
is sufficient for in-domain scoring and insufficient for the
cross-dataset deployment that the FYDP proposal targets. The
shortcut diagnosis gives a concrete, quantitative basis for the
Phase-II design choices.

---

## 5. Challenges Faced

| S.No. | Issue / Challenge | Strategy / Plan |
|-------|-------------------|------------------|
| 1 | Severe class imbalance (3.92:1 PCOS-positive) in Figshare biases the foundation model toward PCOS-positive predictions on class-balanced data. | Stratified 80/10/10 split. Identified as a literature-wide problem by Ghaderzadeh SLR [7] and Suha SLR [13]. SMOTE and class-weighted loss to be evaluated in Phase-II. |
| 2 | Substantial byte-level duplication within Figshare (3,996 unique images among 11,784 files). Risk of leakage across the train/test split. | MD5 deduplication performed before any split. Exact-byte match only; near-duplicates from re-encoding are not claimed to be removed. |
| 3 | Letterbox-padding shortcut discovered during external evaluation. Foundation models attended to the black border, not to the ovary. | Per-image Grad-CAM audit conducted. Padding eliminated in the foundation ablation. Documented for Phase-II fine-tuning. |
| 4 | Spec mismatch between Figshare and PCOSgen (acquisition protocol, patient demographics). | PCOSgen reserved as held-out only. No training on it in Phase-I. Phase-II will fine-tune on a single split of PCOSgen and report the held-out test metric explicitly. |
| 5 | Single-GPU compute budget (RTX 4070 SUPER, 12 GB). Six-condition preprocessing grid from the original proposal would have required 54 foundation runs and exceeded the wall-clock budget. | Reduced to 9 × 2 in Phase-I; the two conditions isolate the denoiser variable. The four additional preprocessing conditions will be revisited only if the Phase-II results warrant it. |
| 6 | Verification of trend, not just point estimates, is needed before committing to fine-tuning. | Bootstrap 95% confidence intervals on the foundation external AUC. Phase-I report does not commit to a single architecture; the 18-run matrix preserves choice for Phase-II. |

---

## 6. Next Steps

| S.No. | Next Task | Estimated completion |
|-------|-----------|----------------------|
| 1 | Fine-tune each of the 18 foundation configurations on the PCOSgen train pool using a lower learning rate. | Aug 2025 |
| 2 | Evaluate the fine-tuned models on the PCOSgen held-out test split. Quantify the recovery from mean external AUC 0.77 to the fine-tune baseline. | Aug 2025 |
| 3 | Select the top 3 architectures by external AUC and construct a probability-averaged ensemble. | Sep 2025 |
| 4 | Calibrate the ensemble using post-hoc temperature scaling; report Expected Calibration Error before and after. | Sep 2025 |
| 5 | Run MC-Dropout with 50 stochastic forward passes for predictive entropy per test image. | Oct 2025 |
| 6 | Generate a Grad-CAM panel grid for the top-3 ensemble members, with attention audited against the border-vs-interior finding. | Oct 2025 |
| 7 | Wrap up the conference paper draft and the FYDP final thesis. | Nov 2025 |

---

## 7. Updated Timeline

| Task | W6 | W7 | W8 | W9 | W10 | W11 | W12 | W13 | W14 | W15 | W16 | W17 | W18 | W19 | W20 | W21 | W22 | W23 |
|------|----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Data collection & dedup | ✔ | ✔ | | | | | | | | | | | | | | | | |
| Sweep pipeline implementation | | | ✔ | ✔ | | | | | | | | | | | | | | |
| Foundation 9 × 2 sweep | | | | | ✔ | ✔ | ✔ | | | | | | | | | | | |
| External evaluation | | | | | | | | ✔ | | | | | | | | | | |
| Border-attention audit | | | | | | | | | ✔ | ✔ | | | | | | | | |
| Phase-I write-up | | | | | | | | | | | ✔ | ✔ | ✔ | | | | | |

| Estimated Work Period | Summer 2025 (Weeks 6–17) |
|-----------------------|--------------------------|
| **Actual Work Period** | Summer 2025 (Weeks 6–17) |

---

## 8. Resources Utilised

- **Hardware.** NVIDIA RTX 4070 SUPER (12 GB), one workstation. Wall
  clock consumed for the foundation 9 × 2 sweep: ~5 hours in total.
- **Datasets.** Figshare PCOS Ultrasound Dataset (public) and PCOSgen
  (Sundari et al., 2025). Both used under their published terms.
- **Software.** Python 3.10+, PyTorch 2.12.0 with CUDA 13.0, timm 1.0.27,
  scikit-learn 1.9.0, OpenCV 4.13, NumPy 2.4, SciPy 1.17, matplotlib 3.10.
- **Reference texts and standards.** Rotterdam Consensus Criteria for
  PCOS diagnosis (2003). Guo et al. 2017 [22] for calibration
  methodology (planned use in Phase-II).

---

## 9. Project Management and Financial Analysis

The project is research-only and incurs no direct financial cost to the
university. All experiments run on a single existing GPU workstation.
The dataset is publicly available. Estimated time cost for Phase-I is
approximately 80 person-hours of student effort and 6 hours of faculty
supervision, distributed across the 12 reporting weeks.

In an alternative budget scenario (acquisition of an additional GPU,
or paid cloud compute), the wall-clock estimate of 5–6 hours would
shrink to under 2 hours, but the dollar cost is not justified at
current scale.

---

## 10. Future Considerations

The Phase-II plan commits to ensemble construction and calibration,
each of which trades training-time for evaluation quality. Risks to be
managed:

- **Selection risk.** Choosing the top-3 architectures by Phase-I
  external AUC may not transfer to a Phase-II fine-tune. The sweep
  must be re-evaluated after fine-tuning.
- **Compute risk.** The combined foundation + fine-tune + HPO +
  calibration cost must remain within the single-GPU budget; if HPO
  over-runs, the search-space bounds will be tightened rather than
  the hardware escalated.
- **Publication risk.** SLRs [7,13] both flag the over-reliance on a
  small set of public datasets. The PCOSgen external-cohort result,
  if it lands, will be the strongest single contribution and must be
  reported before the rest of the Phase-II work.

---

## 11. Conclusion

Phase-I establishes the source datasets, deduplicates the Figshare
corpus, and runs the foundation-stage 9 × 2 benchmark on the dataset
pair. The headline finding is a quantitative diagnosis of the
internal-vs-external collapse: every foundation configuration
saturates in-domain (≥0.99 AUC) and collapses externally (mean 0.52 on
PCOSgen). The collapse is traced, in part, to a letterbox-padding
shortcut that the model exploits. Disabling padding and resizing
directly recovers **+0.25 absolute AUC** on the external cohort.

Phase-II is scoped narrowly to: fine-tune the 18 configurations on
PCOSgen, construct a probability-averaged ensemble of the top three,
apply post-hoc temperature scaling, run MC-Dropout for predictive
uncertainty, and verify attention migration with Grad-CAM. The
Phase-II plan preserves the single-GPU budget and keeps the dataset
protocol (PCOSgen held-out only, never trained on without
documentation) consistent with Phase-I.

---

## References

(References are listed in IEEE format, numbered consistently with the
FYDP Title Defense literature table where applicable.)

[1] M. S. Sundari, N. V. Sailaja, D. Swapna, V. C. Jadala, and K. Durga,
"Transfer learning enhanced CNN model for integrative ultrasound and
biomarker-based diagnosis of polycystic ovarian disease," *Sci. Rep.*,
vol. 15, p. 34519, Oct. 2025, doi: 10.1038/s41598-025-17711-w.

[4] V. Lakshmi and B. Pushpa, "Explainable multimodal deep learning
using cross-attention fusion of ultrasound and clinical features for
PCOS classification," *Discov. Comput.*, vol. 29, no. 1, p. 7, 2026,
doi: 10.1007/s10791-025-09901-x.

[5] K. Shanmugavadivel, M. S. Murali Dhar, T. R. Mahesh, T. Al-Shehari,
N. A. Alsadhan, and T. E. Yimer, "Optimized polycystic ovarian
disease prognosis and classification using AI based computational
approaches on multi-modality data," *BMC Med. Inform. Decis. Mak.*,
vol. 24, no. 1, p. 281, Oct. 2024, doi: 10.1186/s12911-024-02688-9.

[7] C. Salehnasab, M. Ghaderzadeh, and A. Garavand, "Artificial
intelligence in polycystic ovary syndrome: a systematic review of
diagnostic and predictive applications," *BMC Med. Inform. Decis. Mak.*,
vol. 25, p. 427, Nov. 2025, doi: 10.1186/s12911-025-03255-6.

[13] S. A. Suha and M. N. Islam, "A systematic review and future
research agenda on detection of polycystic ovary syndrome (PCOS) with
computer-aided techniques," *Heliyon*, vol. 9, no. 10, p. e20524,
Oct. 2023, doi: 10.1016/j.heliyon.2023.e20524.

[16] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation:
Representing model uncertainty in deep learning," *Proc. 33rd Int.
Conf. Mach. Learn. (ICML)*, pp. 1050–1059, 2016.

[17] R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep
networks via gradient-based localization," *Proc. IEEE Int. Conf.
Comput. Vis. (ICCV)*, pp. 618–626, 2017.

[22] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration
of modern neural networks," *Proc. 34th Int. Conf. Mach. Learn.
(ICML)*, pp. 1321–1330, 2017.

[26] A. Indirani, "PCOS Dataset," *figshare*, 2024. [Online].
Available: https://doi.org/10.6084/m9.figshare.27682557.v1

---

## Appendix

Additional documentation including the full sweep matrix CSV
(`results/ablation/sweep_matrix.csv`), the per-image border-attention
audit CSVs, and the Grad-CAM panel figures, is available in the
project repository under `results/` and `docs/latex/figures/`. The
Phase-I report references these by filename; the user's local copy
of the repository contains the original CSVs and PNGs.

A summary index of figure filenames used in this report is in
`phase-1/README.md`.