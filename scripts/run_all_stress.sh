#!/usr/bin/env bash
# Runs the Section 5.2 architectural stress ablations: context-length,
# positional sensitivity, and FP16-vs-4-bit precision, on Falcon-7B-Instruct
# (the paper's default for these experiments).
set -euo pipefail
cd "$(dirname "$0")/.."

python experiments/rq2.py \
    --model tiiuae/falcon-7b-instruct \
    --datasets hotpot \
    --run_context --run_positional --run_precision \
    --outdir rq2_outputs/stress \
    "$@"
