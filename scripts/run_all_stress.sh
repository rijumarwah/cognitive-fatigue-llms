#!/usr/bin/env bash
# Run the architectural stress ablations (RQ2) across all four datasets with
# the paper's default calibration (configs/default.yaml). Edit the arrays
# below to change datasets, sample size, or model.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS=("hotpot" "triviaqa" "squad" "natural_questions")
SAMPLE_SIZE="${SAMPLE_SIZE:-20}"
MODEL="${MODEL:-facebook/opt-2.7b}"
SEEDS="${SEEDS:-123,2027}"

for ds in "${DATASETS[@]}"; do
    echo "=== architectural stress: ${ds} ==="
    python experiments/architectural_stress/run_stress.py \
        --datasets "${ds}" \
        --sample_size "${SAMPLE_SIZE}" \
        --model "${MODEL}" \
        --seeds "${SEEDS}" \
        --run_context --run_positional --run_precision
done
