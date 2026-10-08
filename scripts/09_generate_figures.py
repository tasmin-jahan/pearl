#!/usr/bin/env python3
"""Stage 09: Generate All Figures and LaTeX Tables for PEARL.

Calls the unified figure generator to render:
  1. Paper figures (docs/latex/figures/)
  2. Thesis figures and tables (docs/thesis/figures/, docs/thesis/tables/)
  3. Analysis figures (docs/analysis/figures/)

Usage::

    python scripts/09_generate_figures.py
    python scripts/09_generate_figures.py --only paper
    python scripts/09_generate_figures.py --only thesis
    python scripts/09_generate_figures.py --only analysis
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_figures import main

if __name__ == "__main__":
    sys.exit(main())
