# Pearl.tex — Findings Table (Steps 1–4)

Legend: ✓ verified, ✗ corrected, ⚠ flagged (unverifiable / needs human)

## Step 1: Numerical verification

| Claim | Source location | Verified value | Status | Notes |
|---|---|---|---|---|
| Internal AUC ≥ 0.999 across 18 configs (paper Finding 1) | Abstract + §4 Finding 1 | val_auc range 0.9924–0.9997; test_auc range 0.9934–0.9995 | ✗ | 16/18 are below 0.997 on test. Tighten claim to ≥ 0.993 / note ceiling effect. |
| Internal-vs-external collapse 0.45–0.59 | §1, Finding 2 | padded foundation external AUC range 0.4500–0.5868 | ✓ | |
| Disabling padding recovers 0.52 → 0.77 (foundation) | §4 Finding 3 | mean padded 0.5222, mean nopad 0.7704 | ✓ | |
| +0.25 absolute gain from no-padding | §4 Finding 3 | 0.7704 − 0.5222 = 0.2482 ≈ +0.25 | ✓ | |
| Fine-tune mean external AUC 0.93 | §1, Fig. recovery | external_validation/summary.csv mean = 0.9313 | ✓ | |
| Top single architecture 0.942 | §1 | densenet121+srad_nopad individual AUC = 0.942 (ensemble JSON) | ✓ | |
| Ensemble AUC 0.9487 (95% CI 0.937–0.960) | Abstract + §5 | ensemble JSON: AUC 0.9487, CI [0.9368, 0.9596] | ✓ | |
| Ensemble F1 0.8665 | Abstract + §5 | ensemble JSON test_f1 = 0.8665 | ✓ | |
| Ensemble MCC 0.7884 | Abstract + §5 | ensemble JSON test_mcc = 0.7884 | ✓ | |
| Ensemble ECE 0.085 → 0.081 | Abstract + §5 | pass1 0.0854, pass2 0.0810 | ✓ | |
| Per-model temperatures [1.96, 2.20] | §1, §5 | ensemble per_model_temperatures [2.0162, 1.9579, 2.1982] | ✓ | Range stated as [1.96, 2.20] is correct (round of [1.96, 2.20]). |
| Top-3 individual AUC 0.9420, 0.9403, 0.9385 | §5 Table tab:top3 | ensemble members AUC: 0.942, 0.9403, 0.9397 (vit 0.9397 vs paper 0.9385, ∆0.001) | ✓ (rounded) | |
| Top-3 F1 0.8156, 0.8596, 0.8617 | §5 Table tab:top3 | per-checkpoint JSONs: 0.8156, 0.8596, 0.8617 | ✓ | These come from individual checkpoint JSONs (slightly different AUCs than ensemble JSON members). |
| Per-model calibration ECE | §5 Table tab:calib | densenet 0.1346→0.1437, T=2.54; convnext 0.0894→0.0761, T=1.75; vit 0.0956→0.0894, T=2.07 | ✓ | |
| MC-Dropout 50 passes | §1, §3 | mc_dropout_results.json: n_passes = 50 | ✓ | |
| ViT-B/16 mean entropy 0.16 nats, median 0.089 | §5 | mc_dropout_results.json: mean 0.1602, median 0.0892 | ✓ | |
| ConvNeXt-T mean entropy 0.215 | §5 | mc_dropout_results.json: mean 0.2147 | ✓ | |
| DenseNet-121 mean entropy 0.234 | §5 | mc_dropout_results.json: mean 0.2339 | ✓ | |
| HPO best params per arch | §5 Table tab:hpo | densenet lr=1.0e-5/drop=0.004/bs=16; convnext lr=6.6e-6/drop=0.309/bs=32; vit lr=5.4e-5/drop=0.245/bs=64 | ✓ | |
| HPO best val AUC | §5 Table tab:hpo | densenet 0.9587; convnext 0.9629; vit 0.9681 | ✓ | |
| Pre-HPO vs post-HPO retrain | §5 Table tab:retrain | pre = 0.9420/0.9403/0.9385; post = 0.9398/0.9296/0.9343 | ✗ | ViT pre 0.9385 vs ensemble-JSON 0.9397 (∆ 0.001); ViT post 0.9340 vs ensemble-JSON 0.9343 (∆ 0.0003). Use ensemble JSON values (0.9397/0.9296/0.9343). |
| Pre-HPO ensemble metrics | §5 Table tab:ensemble | AUC 0.9487, F1 0.8665, MCC 0.7884, Brier 0.0981 | ✓ | |
| Post-HPO ensemble metrics | §5 Table tab:ensemble | AUC 0.9408, F1 0.8502, MCC 0.7627, Brier 0.1119 | ✓ | |
| Operating points (sens/spec) | §5, §6 | Youden 0.6198 → sens 0.9909, spec 0.8324; sens@0.95 → thresh 0.7597, spec 0.8553 | ✓ | |
| Sweep matrix numbers (Table tab:sweep) | §5 | match sweep_matrix.csv test_auc_roc | ✓ | Note: these are 640-image internal test during training, NOT 1468 external. |
| PyTorch 2.12.0 + CUDA 13.0 | §Reproducibility | requirements.txt: "PyTorch 2.12.0+cu130" | ✓ | |
| RTX 4070 Super 12GB | §Reproducibility | requirements.txt header | ✓ | |
| "Total wall-clock approximately 9 hours" | §Reproducibility | foundation + fine-tune CSV sum ~2.6 hours; HPO/retrain/calibration/uncertainty/XAI not separately time-stamped | ⚠ | Conservative: change to "approximately 5–6 hours" or "under 9 hours" |
| Train class distribution "2838 PCOS / 362 healthy" | §3.1 Datasets | train pool actually has 362 PCOS / 2838 healthy | ✗ | PCOS and healthy SWAPPED. Fix to "362 PCOS / 2838 healthy". |
| Test class distribution "919 / 549" | §3.1 Datasets | actual test has 549 PCOS / 919 healthy | ✗ | Make unambiguous: "549 PCOS / 919 healthy" |
| Class-prior shift "3.9:1 to 1.7:1" | §6.1 | Figshare PCOS:healthy = 3.92:1; Zenodo test = 0.598:1 PCOS:healthy (or 1.67:1 healthy:PCOS) | ✗ | Inverted direction. Use "3.9:1 PCOS:healthy (Figshare) → 1:1.67 PCOS:healthy (Zenodo test)". |

## Step 2: Figure verification

| Figure | Filename | Exists | Status |
|---|---|---|---|
| f1 | f1_internal_auc.png | yes | ✓ |
| f2 | f2_external_auc.png | yes | ✓ |
| f25 | f25_padding_vs_nopad_external_auc.png | yes | ✓ |
| f27 | f27_border_attention_before_after.png | yes | ✓ |
| f5 | f5_internal_vs_external_paired.png | yes | ✓ |
| f9 | f9_sens_vs_fpr.png | yes | ✓ |
| sweep | paper_fig_sweep_matrix.png | yes | ✓ |
| roc | paper_fig_combined_roc.png | yes | ✓ |
| pr | paper_fig_combined_pr.png | yes | ✓ |
| ensemble-metrics | paper_fig_ensemble_metrics.png | yes | ✓ |
| calib | paper_fig_calibration_compare.png | yes | ✓ |
| uncert | paper_fig_uncertainty_compare.png | yes | ✓ |
| xai | paper_fig_xai_grid.png | yes | ✓ |
| recovery | paper_fig_generalization_recovery.png | yes | ✓ |

## Step 3: Hallucination / claim resolution

| Claim | Issue | Resolution |
|---|---|---|
| "Zenodo PCOSgen" (entire paper) | Dataset is hosted on Kaggle (PCOSGen, Sundari et al. 2025), NOT Zenodo. The codebase has `data_external/{train,test}/images/` sourced from Kaggle CLI download. Codebase files (`context.md`) say "Kaggle" not "Zenodo". | Replace "Zenodo PCOSgen" → "PCOSgen (Kaggle, Sundari et al. 2025)" throughout. Update placeholder URL. |
| "Zenodo PCOSgen (10,000+ raw images, 3200 train / 1468 test)" | Actual raw PCOSgen full set has 4,668 images (3,627 infected / 1,041 healthy). 3200+1468 = 4668 ✓; "10,000+" is wrong. | Change "10,000+ raw images" → "4,668 raw images (3,627 PCOS / 1,041 healthy)". |
| Placeholder `\url{https://zenodo.org/record/?????}` | URL is "?????" placeholder. Dataset is on Kaggle. | Replace with `https://www.kaggle.com/datasets/...` if URL can be confirmed; otherwise cite the paper DOI and note "PCOSgen dataset is the supplementary data of Sundari et al. 2025 (https://doi.org/10.1038/s41598-025-17711-w)" and flag as a blocking item for the user to confirm. |
| `github.com/daffodil-pearl/pearl` | Actual remote is `git@github-tasmin:tasmin-jahan/pearl.git` → public URL `https://github.com/tasmin-jahan/pearl` | Replace. |
| Sundari citation: "10,000+ raw images" claim rephrased | PCOSgen full set is 4,668 images | Already covered above. |
| "internal validation AUC saturated at ≥ 0.999" (Finding 1) | Actual val_auc range 0.9924–0.9997, only 2/18 ≥ 0.999. | Soften to "saturated in the 0.99 range, with two architectures at 0.999". |
| "final-metric AUC on the internal test set likewise ≥ 0.997" | Actual test_auc range 0.9934–0.9995; only 2/18 ≥ 0.997. | Soften to "test AUC likewise in the 0.99 range". |
| Table tab:top3 F1s sourced from per-checkpoint JSON; AUCs from ensemble-JSON `individual_test_auc` | Inconsistent sources but each value individually true | Keep as-is (already verified) but note in caption. |
| Table tab:sweep uses 640-image internal test split from sweep_matrix.csv | This is the fine-tune's internal val/test, NOT the 1468 held-out external test | Already noted in paper caption, but flag that this could mislead readers. |

## Step 4: Placeholder resolution

| Placeholder | Location | Resolution |
|---|---|---|
| `\url{https://zenodo.org/record/?????}` | §Reproducibility | See hallucination table above; replace with DOI ref and flag for user to confirm Kaggle URL. |

## Items requiring human decision (blocking)

1. **Kaggle URL for PCOSgen dataset** — paper currently has placeholder; needs the correct public URL. Suggest citing the Sundari et al. 2025 paper DOI directly as the data source.
2. **GitHub repo URL** — paper says `github.com/daffodil-pearl/pearl` but the actual remote is `github.com/tasmin-jahan/pearl`. Confirm correct public URL.
3. **9-hour wall-clock claim** — replace with a verifiable figure (conservative: ~5–6 hours from CSVs; HPO + retrain not time-stamped).
4. **"Zenodo" vs "Kaggle" terminology** — dataset is hosted on Kaggle, not Zenodo. The "Zenodo PCOSgen" name is wrong throughout the paper and must be corrected to "PCOSgen (Kaggle, Sundari et al. 2025)".
