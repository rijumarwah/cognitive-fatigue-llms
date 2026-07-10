# Code audit notes

This file documents what was found and changed when packaging this repo as
an installable library, so the discrepancies below are visible before
anyone re-runs experiments or cites numbers from this artifact.

## What was empty / non-functional (before this pass)

The following were 1-byte placeholder files with no implementation, despite
being referenced by the README and by Appendix F of the paper as containing
"implementations of all three inference-time signals... the FI aggregation
pipeline":

- `fatigue/__init__.py`, `attention.py`, `drift.py`, `entropy.py`,
  `hysteresis.py`, `index.py`, `normalize.py`
- `utils/logging.py`, `utils/metrics.py`, `utils/seeds.py`
- `requirements.txt`
- `configs/default.yaml`, `fp16.yaml`, `nf4.yaml`, `stress.yaml`
- `scripts/run_all_stress.sh`, `run_all_validation.sh`
- `experiments/architectural_stress/context_length.py`
- `experiments/architectural_stress/README.md`
- `experiments/empirical_validation/run_validation.py`
- `experiments/empirical_validation/README.md`

Nothing in the repo imported from `fatigue/` or `utils/` -- confirmed via
grep before any changes were made. All real logic lived in
`experiments/helper.py`, `experiments/rq2.py`, and three Jupyter notebooks.

## Three divergent Fatigue Index implementations (before this pass)

The paper's Eq. 1 / Section 6.2 defines fixed, non-adaptive normalization
maps:

    phi_A(A_t) = 1 - clip(A_t, 0, 1)
    phi_E(E_t) = asymmetric band deviation, denominators H_low / beta
    phi_D(D_t) = clip(D_t / kappa, 0, 1)

with weights (0.40, 0.35, 0.25), entropy band [3.8, 5.0], hysteresis
(0.50, 0.40) -- all "fixed from a small preliminary pass and then frozen."

Three different implementations existed, none matching the paper exactly:

1. **`experiments/rq2.py`** (the script that actually produced
   `rq2_results/*.json`): weights, entropy band, and hysteresis thresholds
   matched Table 6, but `phi_A` and `phi_D` used **per-run adaptive**
   normalization -- min-max scaling against that specific run's own early
   window / own max value -- rather than a fixed calibration. This is not
   the same algorithm as the paper describes, and it also means every
   sequence's attention/drift signal gets rescaled to occupy [0,1] within
   itself, which could make trajectories look more similar across
   datasets/models than the underlying raw signals actually are. If this is
   the code that produced Figure 5 / Tables 2-5, this is worth revisiting
   before treating those numbers as final.

2. **`Reliability_of_Fatigue_Index.ipynb`**: `phi_A` matched the paper
   exactly (fixed). `phi_D` used a fixed cap (`DRIFT_CAP = 20.0`), also
   consistent with the paper's description. But `ENTROPY_BAND = (3.0, 4.5)`
   and hysteresis thresholds `(0.60, 0.45)` did not match Table 6
   `([3.8, 5.0]` and `(0.50, 0.40)`).

3. **`Additional_analyses.ipynb`**: same `phi_A`/`phi_E` as (2), but
   `DRIFT_CAP` appears as **both 20.0 and 150.0** in different cells of the
   same notebook, with a code comment ("Now divides by 40...") that matches
   neither value.

## What this packaging pass did

- Implemented the fixed, paper-faithful `phi_A`/`phi_E`/`phi_D`, `FI_t`
  aggregation, and hysteresis in `fatigue/normalize.py`, `fatigue/index.py`,
  `fatigue/hysteresis.py`, matching Eq. 1 / Section 6.2 / Table 6 exactly.
- `kappa` (drift cap) and `beta` (entropy high-side denominator) are **not
  published** in Table 6. This package defaults to `kappa=20.0`, `beta=5.0`
  (from the one notebook cell whose values look calibration-intended rather
  than experimental). **Confirm these against whatever actually produced
  your reported results before trusting downstream numbers**, and consider
  adding the true values to Table 6 in the paper.
- `experiments/rq2.py` now imports its raw-signal extraction and FI/
  hysteresis computation from `fatigue/` instead of maintaining its own
  copies; verified the refactored script produces output identical to
  calling `fatigue.index.compute_fi_series` directly.
- `experiments/helper.py` is superseded by `rq2.py` (it never computed FI at
  all) and is now marked deprecated rather than refactored in place.
- The notebooks were **not** modified -- notebook edits are riskier to do
  blind and the two RQ1 notebooks are the closest to the paper's stated
  formulas already. If you want them reconciled to `fatigue/` too, that's a
  separate pass; at minimum, fix the internal `DRIFT_CAP` 20.0 vs 150.0
  inconsistency in `Additional_analyses.ipynb`.
- Filled in `requirements.txt`, `configs/*.yaml` (now actually loadable via
  `fatigue.config.from_yaml`), and added `pyproject.toml` so `pip install -e .`
  installs a real package.
- Added `tests/` (17 tests, all passing) covering the normalization
  formulas, FI aggregation/boundedness, hysteresis, and raw signal
  extraction against hand-computed values.

## Other things worth knowing

- **No KV-cache reuse.** `FatigueMonitor`/`run_fatigue_experiment` re-run a
  full forward pass over the growing sequence at every step
  (`use_cache=False`) to get attentions at each step. This is what produced
  the paper's results, but it's O(n^2) in generated tokens and will be slow
  to reproduce at the reported scale (27,405 sequences). Worth optimizing
  if you re-run large sweeps.
- The empty top-level `utils/` directory (distinct from the new
  `fatigue/utils/`) was removed since nothing imported it and its name was
  confusing next to the real one.
- The README previously ended mid-code-block and referenced a `figures/`
  directory that doesn't exist in this repo. Rewritten to match reality.
