#!/usr/bin/env python3
"""
Placeholder CLI entry point for the Section 7 empirical validation pipeline.

The actual pipeline (cross-dataset FI trajectories, predictive validity,
hysteresis stability, aggregation benefit) currently only exists in
`Reliability_of_Fatigue_Index.ipynb` in this directory. This script is not
yet a working port of that notebook -- see the README.md in this directory
before relying on it.
"""

import sys

if __name__ == "__main__":
    sys.exit(
        "run_validation.py is not yet implemented as a script.\n"
        "Use Reliability_of_Fatigue_Index.ipynb in this directory instead, "
        "or port its cells here using `fatigue` for signal/FI computation "
        "(see experiments/rq2.py for the pattern)."
    )
