# Thesis Manuscript — Provenance and Source-of-Truth

This directory contains a thesis manuscript expanding the PEARL conference paper
for Daffodil International University final-year design / MS-style submission.
The conference paper at `docs/latex/pearl.tex` remains the standalone
submission and is not modified by this manuscript.

## Source hierarchy

When facts in the manuscript disagree across the source documents, this
hierarchy decides which wins:

1. **Result artefacts** (authoritative). All numeric claims must be
   traceable to one of these paths:
   - `results/finetune_zenodo/ensemble/top3_pre_hpo/external_validation/pcosgen.json`
   - `results/finetune_zenodo/ensemble/top3_pre_hpo/calibration/calibration_results.json`
   - `results/finetune_zenodo/checkpoints/{srad,gauss}_nopad/<arch>/final_metrics.json`
   - `results/finetune_hpo/{srad_nopad_densenet121,srad_nopad_convnext_tiny,gauss_nopad_vit_base}/*/best_params.yaml`
   - `results/ablation*/sweep_matrix.csv`
   - `results/finetune_zenodo/checkpoints/{srad,gauss}_nopad/<arch>/external_validation/pcosgen.csv`
   - `results/finetune_zenodo/checkpoints/{srad,gauss}_nopad/<arch>/uncertainty/mc_dropout_results.json`
   - `data_external/zenodo_splits/{train,val,test}.json`
   - `data_external/figshare/` (md5-dedupped)
2. **Conference paper** (`docs/latex/pearl.tex`, rewritten session). Treated
   as a peer-reviewed snapshot of what was run.
3. **Title Defense proposal** (`docs/Title Defense.md`). Source for the
   research questions, objectives, and the literature-review table. Scope
   claims that were later revised are documented in Appendix C.
4. **Methodology doc** (`docs/methodology.tex`, v3 pipeline spec).
   Treated as the *intended* design. Where the executed run deviated
   (e.g. freeze_fraction, EMA/SWA, k-fold CV), the manuscript follows
   the artefacts and footnotes the deviation; the design rationale is
   preserved in Appendix A.

## What this manuscript is

A multi-file LaTeX thesis that:
- absorbs every section of `methodology.tex` (full math + reproducibility),
- incorporates the results from `pearl.tex` (verified against JSON),
- contextualises both against the research questions in `Title Defense.md`,
- is structured so each chapter compiles standalone in Overleaf
  (via `subfiles` + `bibunits`),
- cites only references that appear in `Title Defense.md`'s literature
  table or that were independently verified from a primary source.

## What this manuscript is not

- Not a research journal article. It is a thesis chapter that documents
  the entire pipeline, including design choices that were later revised.
- Not a replacement for the conference paper. `pearl.tex` remains the
  concise submission; this thesis is the long-form documentation.
- Not an authoritative source for any number. Every number in the
  manuscript can be reproduced from the result artefacts listed above.

## File map

```
docs/thesis/
  thesis.tex            Master file
  preamble.tex          Shared packages and macros
  abstract.tex          Abstract (also \chapter*)
  acknowledgements.tex
  abbreviations.tex
  README.md             Overleaf + standalone-chapter instructions
  chapters/             10 chapters (independent-compilable)
  appendix/             5 appendices
  figures/              Symlinks to docs/latex/figures/*.png
  tables/               Generated fragments + numbers_master.tex
  refs/thesis.bib       Single bibliography; per-chapter bibunits
```