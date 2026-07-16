# Empirical validation experiments (RQ3, paper Sec. 7)

`run_validation.py` runs four reliability analyses on natural (unstressed)
generations, using the harness in `experiments/common.py` and FI math from
`fatigue/` (calibration: paper Table 6, `configs/default.yaml`).

## Analyses

- **E1 — Temporal trajectory.** Mean FI (± std across items) over decoding
  steps, to check that FI rises with generation length rather than staying
  flat. Plotted to `e1_temporal_fi.png`.
- **E2 — Predictive validity.** Spearman correlation between mean FI and two
  behavioral outcomes — repetition ratio and task failure (`1 - F1`) —
  computed over the full generation and over a first-`K`-step window
  (`--first_k`, default 20).
- **E3 — Aggregation value.** AUROC of FI, and of each individual normalized
  signal (`phi_A`, `phi_E`, `phi_D`) alone, at predicting item-level failure
  (F1 at or below the sample median). Tests whether combining the three
  signals outperforms any single one.
- **E4 — Hysteresis stability.** Mean number of alert-state flips under
  two-threshold hysteresis (`fatigue.hysteresis.apply_hysteresis`) vs. a
  naive single-threshold rule, on the same FI trajectories.

Spearman and AUROC are computed with small NumPy-only implementations (see
`spearman()` / `auroc()` in `run_validation.py`) so the script has no hard
SciPy/scikit-learn dependency; both are cross-checked against SciPy/sklearn in
`tests/` when those packages are available.

## Usage

```bash
python run_validation.py --datasets hotpot --sample_size 50 --precision fp
```

Writes a `validation_report.json` (all four analyses, per dataset) plus
per-dataset plots to `--outdir` (default `results/empirical_validation/`).

## Notebooks

`Reliability_of_Fatigue_Index.ipynb` and `Additional_analyses.ipynb` are the
exploratory notebooks behind the paper's Sec. 7 figures (including the
OPT-2.7B results). They predate the `fatigue`/`experiments.common` package
split and compute FI inline; see the top of each notebook for a note on their
calibration relative to `configs/default.yaml`.

