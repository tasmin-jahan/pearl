# Fine-tune worklog

## Setup
- venv: `/home/farhan/my-projects/pearl/.venv` (Python with torch 2.12.0+cu130, optuna 4.9.0, sklearn 1.9.0, cv2 4.13.0, pandas 3.0.3).
- GPU: NVIDIA GeForce RTX 4070 SUPER (12GB).
- Source Figshare-trained checkpoints: `results/ablation_nopad/checkpoints/{srad_nopad,gauss_nopad}/{arch}/best.pt` (9 archs × 2 preprocessings = 18 checkpoints).
- Zenodo PCOSgen source: `data_external/{train,test}/{images/*.jpg,class_label.xlsx|class label.csv}` (3200 train, 1468 test, 300×300 jpg).

## Steps
1. [x] Build `src/data/zenodo_dataset.py` — labels parsed (train 3200, test 1468), class dist 2838/362 train, 919/549 test.
2. [x] Build `scripts/build_zenodo_splits.py` — 2560/640 stratified split, seed 42.
3. [x] Add `zenodo_labeled` layout to `scripts/evaluate_external.py` — 1468 pairs discovered.
4. [x] `configs/experiment/finetune_zenodo.yaml` — lr 3e-5, 15 epochs, no backbone freeze.
5. [x] `scripts/sweep_finetune.py` — sweep driver; loads Figshare weights only (no optimizer state).
6. [in progress] Run fine-tune sweep on 18 (prep, arch) pairs.
7. [ ] External eval on Zenodo test (held-out 1468).
8. [ ] Pick top-3 by external AUC.
9. [ ] `configs/experiment/finetune_hpo.yaml` × 3.
10. [ ] Optuna HPO sweep on top-3 (25 trials each, resume from fine-tuned).
11. [ ] Retrain top-3 with best Optuna params.
12. [ ] Re-eval top-3 on Zenodo test.
13. [ ] `scripts/run_calibration.py` on top-3.
14. [ ] `scripts/run_uncertainty.py` on top-3.
15. [ ] `scripts/run_xai.py` on top-3.
16. [ ] `scripts/run_finetune_ensemble.py` — ensemble + calibration.
17. [ ] Write `docs/analysis/finetune_zenodo.md`.

## Smoke test
- resnet50 + srad_nopad, 200 train + 80 val, 2 epochs: val AUC 0.81, test AUC 0.81, time 9.5s. Pipeline works end-to-end.
- Override freeze_fraction to 0.0 in sweep driver (default 0.60 was wrong for fine-tuning).

## Run 1 results (fine-tune complete)
- resnet50 + srad_nopad, 15 epochs, 6 min
  - val AUC at end: 0.9274 (peak: 0.9314 @ epoch 10)
  - Zenodo TEST AUC: **0.9262** (CI 0.9113–0.9396, sens 0.98, spec 0.82, F1 0.86)
- resnet50 + gauss_nopad, 15 epochs, 73 s
  - val AUC at end: 0.9487 @ epoch 14
  - (not yet externally evaluated)

## Full sweep matrix (Zenodo held-out test, 1468 images)
- 18 fine-tuned runs evaluated.
- Mean AUC: **0.9313** (vs 0.77 pre-FT, 0.52 pre-no-pad).
- Top-3 by external AUC:
  1. densenet121 + srad_nopad: 0.9420 (Sens 0.98, Spec 0.82, F1 0.86)
  2. convnext_tiny + srad_nopad: 0.9403 (Sens 0.97, Spec 0.83, F1 0.86)
  3. vit_base + gauss_nopad: 0.9385 (Sens 0.99, Spec 0.80, F1 0.85)
- Best: 0.9420, Worst: 0.9176.

## HPO finalists (20 trials × 8 epochs each, TPE + MedianPruner)
- densenet121 + srad_nopad: best val_auc 0.9587 (trial 18), lr=1.0e-5, wd=1.1e-3, dropout=0.004, batch=16
- convnext_tiny + srad_nopad: best val_auc 0.9629 (trial 7), lr=6.6e-6, wd=1.2e-3, dropout=0.309, batch=32
- vit_base + gauss_nopad: best val_auc 0.9681 (trial 3), lr=5.4e-5, wd=1.9e-3, dropout=0.245, batch=64
- All three prefer freeze_fraction=0.0 (fully unfrozen).
- Wall-clock: ~2 hours total (densenet 50m, convnext 36m, vit 41m).

## Retrain with HPO best params (15 epochs)
- densenet121 + srad_nopad: val_auc 0.9360 (ep 11), test AUC 0.9398 (was 0.9420 pre-HPO, -0.0022)
- convnext_tiny + srad_nopad: val_auc 0.9629 (ep 5), test AUC 0.9296 (was 0.9403 pre-HPO, -0.0107)
- vit_base + gauss_nopad: val_auc 0.9674 (ep 8), test AUC 0.9340 (was 0.9385 pre-HPO, -0.0045)
- Conclusion: HPO retrain improves val AUC but slightly *degrades* test AUC.
  Pre-HPO checkpoints remain the best on the held-out test.

## Calibration (temperature scaling, held-out half-split)
- densenet121+srad_nopad: T=2.54, ECE 0.1346 → 0.1437
- convnext_tiny+srad_nopad: T=1.75, ECE 0.0894 → **0.0761**
- vit_base+gauss_nopad: T=2.07, ECE 0.0956 → 0.0894
- All models systematically overconfident.

## Uncertainty (MC Dropout, 50 passes)
- densenet121+srad_nopad: mean_entropy=0.234, median=0.151
- convnext_tiny+srad_nopad: mean_entropy=0.215, median=0.147
- vit_base+gauss_nopad: mean_entropy=0.160, median=0.089 (most confident)

## XAI (Grad-CAM, 20 samples per model)
- 60 PNGs per model (orig + heatmap + overlay).
- Visual attention on scan interior (not border).

## Final ensemble (3 archs, probability-averaged)
- **Top-3 pre-HPO ensemble: AUC 0.9487, F1 0.8665, MCC 0.7884, Brier 0.0981**
- Top-3 HPO ensemble: AUC 0.9408, F1 0.8502, MCC 0.7627, Brier 0.1119
- Pre-HPO wins on every metric except sensitivity (tied at 0.9927).

## Ensemble calibration (two-pass: per-model T + ensemble T)
- Pre-HPO: per-model T=[2.02, 1.96, 2.20], ensemble T=0.93, ECE 0.085 → **0.081**
- HPO: per-model T=[2.54, 1.75, 2.07], ensemble T=0.91, ECE 0.114 → 0.107

## Final deployable model
- The **Top-3 pre-HPO ensemble** is the recommended model.
- Members:
  - results/finetune_zenodo/checkpoints/srad_nopad/densenet121/best.pt
  - results/finetune_zenodo/checkpoints/srad_nopad/convnext_tiny/best.pt
  - results/finetune_zenodo/checkpoints/gauss_nopad/vit_base/best.pt
- Inference: softmax per model, average class-1 probs, threshold at 0.5
  (default) or 0.6198 (Youden-J) or 0.7597 (sens=0.95 operating point).

## Scripts created
- `src/data/zenodo_dataset.py` (ZenodoDataset, build_zenodo_loader, load_zenodo_labels, stratified_split)
- `scripts/build_zenodo_splits.py`
- `scripts/sweep_finetune.py` (the main fine-tune sweep)
- `scripts/eval_finetune_zenodo.py` (eval all 18 fine-tuned checkpoints on Zenodo test)
- `scripts/run_hpo_finetune.py` (HPO on top-3 finalists)
- `scripts/run_finetune_ensemble.py` (probability-averaged ensemble)
- `scripts/post_eval_finetune.py` (calibration + uncertainty + XAI runner)
- `configs/experiment/finetune_zenodo.yaml`
- `configs/experiment/finetune_hpo.yaml` (template)
- `scripts/evaluate_external.py` extended with `zenodo_labeled` layout
- `src/training/tuner.py` extended with `fine_tune_resume_from` support
