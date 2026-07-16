#!/usr/bin/env bash
# Run the empirical validation analyses (RQ3) across all four datasets with
# the paper's default calibration (configs/default.yaml). Edit the arrays
# below to change datasets, sample size, or model.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS=("hotpot" "triviaqa" "squad" "natural_questions")
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
MODEL="${MODEL:-facebook/opt-2.7b}"
SEEDS="${SEEDS:-123,2027}"

for ds in "${DATASETS[@]}"; do
    echo "=== empirical validation: ${ds} ==="
    python experiments/empirical_validation/run_validation.py \
        --datasets "${ds}" \
        --sample_size "${SAMPLE_SIZE}" \
        --model "${MODEL}" \
        --seeds "${SEEDS}" \
        --precision fp
done
