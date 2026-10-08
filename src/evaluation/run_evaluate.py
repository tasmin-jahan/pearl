"""
CLI wrapper for src.evaluation.evaluate.

Enables invoking via ``python -m src.evaluation.run_evaluate`` as documented in
the project README and methodology specifications.
"""
import sys
from src.evaluation.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
