# Architectural stress (Section 5.2, Section 7.5)

- `context_length.py` -- currently a stub. Context-length and positional
  sensitivity stress tests (Fig. 2, Fig. 3) are implemented in
  `../rq2.py` (`--run_context` / `--run_positional`), which uses `fatigue/`
  for signal computation; that is the current source of truth.
- `Stress Test across models of different Sizes and Families.ipynb` --
  the model-size / instruction-tuning scaling experiment (Fig. 6, Fig. 7,
  Table 7): runs FI across 9 models from 1B-13B, base vs. instruct.

**Before running:** this notebook defines its own `get_entropy`/`get_drift`
inline rather than importing from `fatigue/`. It's self-consistent (unlike
the two empirical_validation notebooks, it doesn't compute the aggregated
FI at all, so there's no phi_A/phi_D discrepancy to worry about here) but
worth wiring to `fatigue/` if you extend it.
