# PEARL Thesis Manuscript

Multi-file LaTeX thesis for the PEARL project, written for Daffodil
International University (DIU) submission. The thesis absorbs the
methodology from `docs/methodology.tex`, the context from
`docs/Title Defense.md`, and the results from `docs/latex/pearl.tex`,
without modifying any of those three source documents.

The companion conference paper `docs/latex/pearl.tex` is preserved as a
standalone submission; this thesis is the longer-form, multi-chapter
treatment of the same study.

## Directory Tree

```
docs/thesis/
  thesis.tex                    -- master file (compiles everything)
  preamble.tex                  -- shared packages, macros, biblatex config
  abstract.tex                  -- frontmatter abstract
  acknowledgements.tex          -- frontmatter acknowledgements
  abbreviations.tex             -- list of acronyms
  PROVENANCE.md                 -- source-of-truth hierarchy

  chapters/
    01_introduction.tex
    02_background.tex
    03_methodology_overview.tex
    04_data.tex
    05_preprocessing.tex
    06_models_training.tex
    07_uncertainty_calibration.tex
    08_results.tex
    09_discussion.tex
    10_conclusion.tex

  appendix/
    A_designed_not_executed.tex
    B_hpo_search_space.tex
    C_scope_evolution.tex
    D_reproducibility.tex
    E_intended_vs_executed.tex

  figures/                      -- symlinks to ../latex/figures/ + new ones
  tables/                       -- generated fragments + numbers_master.tex
  refs/thesis.bib               -- single bibliography
```

## Compilation

### Master thesis (Overleaf or local)

The project compiles with `latexmk` + `biber`. The bibliography must be
re-run after the first pass to resolve all references:

```
latexmk -pdf -interaction=nonstopmode thesis.tex
biber thesis
latexmk -pdf -interaction=nonstopmode thesis.tex
latexmk -pdf -interaction=nonstopmode thesis.tex
```

`latexmk` with the `-pdf` flag invokes `biber` automatically when needed;
running `biber thesis` explicitly is only required if `latexmk` cannot
locate the biber binary on `$PATH`.

On Overleaf:

1. Zip the `docs/thesis/` directory (without `build/` if present).
2. Upload as a new project. Set the main file to `thesis.tex`.
3. The Compiler dropdown should be set to `latexmk` (default) and the
   Bibliography dropdown should be set to `biber`.

### Standalone chapters

Each chapter uses the `subfiles` package and can be opened and
compiled on its own in Overleaf:

```
\documentclass[../preamble.tex]{subfiles}
\begin{document}
\chapter{...}
...
\end{document}
```

The chapter preamble in each file (`\documentclass[../preamble.tex]{subfiles}`)
already provides the right root path, so opening any `chapters/0X_*.tex`
or `appendix/*.tex` file in Overleaf compiles that single file with the
shared preamble and bibliography.

When a chapter is compiled standalone, citations resolve against the
same `refs/thesis.bib` file because the `subfiles` preamble points at
`../preamble.tex`, which sets the bibliography path relative to the
master `thesis.tex` directory.

## Source-of-truth Hierarchy

Numbers and configurations in the thesis are taken from artefacts in
descending order of authority:

1. JSON/CSV under `results/` -- the executed run.
2. `docs/latex/pearl.tex` -- the verified conference paper.
3. `docs/Title Defense.md` -- the FYDP proposal and RQs.
4. `docs/methodology.tex` -- the v3 specification (most divergent).

When the v3 specification disagrees with the executed run, the executed
run wins and the divergence is recorded in `appendix/E_intended_vs_executed.tex`.
Where the proposal diverges from the executed run, the deviation is
recorded in `appendix/C_scope_evolution.tex`.

## Figures

The `figures/` directory contains symbolic links into
`docs/latex/figures/` so the conference paper and the thesis share a
single source of truth for every image. To regenerate the four
thesis-only figures (`thesis_per_class_pr.png`,
`thesis_threshold_sweep.png`, `thesis_uncertainty_hist.png`,
`thesis_training_curves.png`), run:

```
python scripts/thesis_figures.py
```

The generated table fragments under `tables/` are produced by the
same script. The thesis chapters `\input` these fragments rather than
hard-coding numbers, which prevents transcription drift when the
underlying JSON changes.

## Bibliography Anti-Hallucination Policy

The seed bibliography is the literature-review table in
`docs/Title Defense.md`, numbered `[1]` through `[25]`. Every entry
was copied into `refs/thesis.bib` with complete bibliographic fields
and a `% provenance:` comment that records the source line.

Any reference required by the thesis beyond those twenty-five entries
was added only after independent verification. The four additions
(`yu2002srad`, `akiba2019optuna`, `rotterdam2003`, `teede2018`) all
have provenance comments in the bib file.

Every `\cite{key}` in the thesis resolves to an entry in
`refs/thesis.bib`. A chapter that compiles clean in Overleaf is by
construction free of unverified citations.

## Verification

A small set of checks was run before the directory was packaged:

- `latexmk -pdf` against every chapter and the master file.
- `biber --tool thesis` to confirm every key resolves.
- A figure-existence scan over every `\includegraphics{...}` argument.
- A Unicode audit on the chapter and appendix files.
- A numeric diff of `tables/numbers_master.tex` against
  `docs/latex/findings_table.md` and the headlined JSON files.

If you regenerate artefacts under `results/` and want to refresh the
embedded numbers, run `scripts/thesis_figures.py` and recompile.

## Caveats

- `thesis.tex` includes chapters and appendices in numeric order.
  Reordering requires updating the labels in `\cref` and `\autoref`
  calls throughout the manuscript.
- The standalone-chapter compile path uses the same bibliography file
  as the master thesis. Editing `refs/thesis.bib` therefore affects
  every chapter on the next compile.
- The thesis deliberately records the gap between what was designed
  (`methodology.tex`) and what was executed (the JSON artefacts).
  Re-enabling any of the descoped features listed in
  `appendix/A_designed_not_executed.tex` requires a code change to
  the trainer scaffold, not just a configuration edit.