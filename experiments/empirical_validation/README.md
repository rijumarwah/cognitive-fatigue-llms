# Empirical validation (Section 7)

Reproduces the Section 7 results: cross-dataset FI behavior (Fig. 5, Table 2),
predictive validity vs. repetition (Table 3), online hysteresis stability
(Table 4), and benefit of signal aggregation (Table 5).

- `Reliability_of_Fatigue_Index.ipynb` -- main RQ1 pipeline: runs FI over
  HotpotQA/TriviaQA/SQuAD, computes Spearman correlations, AUROC, hysteresis
  flip counts, weight-sensitivity landscape (Fig. 8), and a negative control
  (shuffled prompts).
- `Additional_analyses.ipynb` -- tri-signal specificity check and a
  repetition-ratio correlation patch.
- `run_validation.py` -- currently a stub; the pipeline above only exists in
  notebook form. If you need a scriptable CLI version, port the cells in
  `Reliability_of_Fatigue_Index.ipynb` here, importing signal/FI computation
  from `fatigue/` rather than redefining it (see `../rq2.py` for the pattern).

**Before running:** the calibration constants hardcoded in these notebooks
(`ENTROPY_BAND`, `DRIFT_CAP`, hysteresis thresholds) currently differ from
both the paper's Table 6 and from `fatigue/config.py`'s defaults -- see
`../../CODE_AUDIT_NOTES.md`. Decide which calibration is authoritative
before treating notebook output as reproducing the paper's numbers, or pass
the same `FatigueConfig` used elsewhere by importing from `fatigue`.
