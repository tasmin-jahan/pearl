#!/usr/bin/env python3
"""Unified figure and table generator for the PEARL project.

Dispatches generation of:
  1. Paper figures (docs/latex/figures/)
  2. Thesis figures and LaTeX tables (docs/thesis/figures/, docs/thesis/tables/)
  3. Analysis figures (docs/analysis/figures/)

Usage::

    # Generate everything
    python scripts/generate_figures.py

    # Generate only paper figures
    python scripts/generate_figures.py --only paper

    # Generate only thesis figures & tables
    python scripts/generate_figures.py --only thesis

    # Generate only analysis figures
    python scripts/generate_figures.py --only analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PEARL figures and tables.")
    parser.add_argument(
        "--only",
        choices=("all", "paper", "thesis", "analysis"),
        default="all",
        help="Target section of figures to generate (default: all).",
    )
    args = parser.parse_args(argv)

    print(f"=== PEARL Figure & Table Generation (Target: {args.only}) ===")

    if args.only in ("all", "paper"):
        print("\n--- Generating Paper Figures (docs/latex/figures/) ---")
        try:
            from scripts._figures_lib.make_paper_figures import main as run_paper
            run_paper()
        except Exception as e:
            print(f"[ERROR] Failed to generate paper figures: {e}")

    if args.only in ("all", "thesis"):
        print("\n--- Generating Thesis Figures & Tables (docs/thesis/) ---")
        try:
            from scripts._figures_lib.thesis_figures import main as run_thesis
            run_thesis()
        except Exception as e:
            print(f"[ERROR] Failed to generate thesis figures: {e}")

    if args.only in ("all", "analysis"):
        print("\n--- Generating Analysis Figures (docs/analysis/figures/) ---")
        try:
            from scripts._figures_lib.generate_analysis_figures import run_all_analysis_figures
            run_all_analysis_figures()
        except Exception as e:
            print(f"[ERROR] Failed to generate analysis figures: {e}")

    print("\n=== Figure generation finished. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
