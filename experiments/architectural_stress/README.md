# Architectural stress experiments (RQ2, paper Sec. 5.2)

`run_stress.py` measures how the Fatigue Index and its component signals
respond to three architectural stressors, using the harness in
`experiments/common.py` and FI math from `fatigue/` (calibration: paper
Table 6, `configs/default.yaml`).

## Ablations

- **Context length** (`context_length_stress`) — fixed evidence position
  ("middle"), sweeping the amount of filler text around it
  (`CONTEXT_SHORT` / `CONTEXT_MEDIUM` / `CONTEXT_LONG` token targets).
- **Positional sensitivity** (`positional_sensitivity_profile`) — fixed
  filler length, sweeping the evidence position (`front` / `middle` / `end`)
  to test whether FI tracks loss of attention to evidence as it moves away
  from the end of the context.
- **Precision** (`precision_quantization_ablation`) — the same prompts and
  seeds run under FP16/FP32 vs. 4-bit NF4 (`bitsandbytes`), comparing entropy
  collapse, repetition, EM/F1, and FI between precisions. Requires a CUDA GPU
  for the 4-bit path.

## Usage

```bash
python run_stress.py \
    --datasets hotpot,triviaqa \
    --sample_size 20 \
    --run_context --run_positional --run_precision
```

Key flags (see `main()` / `argparse` block in `run_stress.py` for the full
list): `--model` (default `facebook/opt-2.7b`), `--seeds`, `--max_new_tokens`,
`--top_p`, `--top_k`, `--temperature`, `--outdir` (default
`results/architectural_stress/`).

Per-example JSON results, per-run signal plots, and aggregate plots (mean ±
std across examples, grouped by stressor level) are written to `--outdir`.

## Notebook

`Stress Test across models of different Sizes and Families.ipynb` is the
exploratory notebook behind the paper's model-scaling stress figures. It
predates the `fatigue`/`experiments.common` package split and computes FI
inline; see the top of the notebook for a note on its calibration relative to
`configs/default.yaml`.

