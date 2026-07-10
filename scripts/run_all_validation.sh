#!/usr/bin/env bash
# Runs the Section 7 empirical validation stress experiments via rq2.py
# across all four datasets. For the notebook-based RQ1 analyses
# (temporal dynamics, predictive validity, hysteresis stability, weight
# sensitivity), see experiments/empirical_validation/*.ipynb directly --
# they are not yet wired into a CLI script.
set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/rq2.py \
    --datasets hotpot,triviaqa,squad,natural_questions \
    --run_context --run_positional \
    --outdir rq2_outputs/validation \
    "$@"
