# Cognitive Fatigue in Autoregressive Transformers

This repository contains code accompanying the paper:

**Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement**
(Accepted as a Main Track paper at ICML 2026)

**Project homepage:** https://rijumarwah.github.io/llmfatigue/

**Try out interactive version on Google Colab:** https://colab.research.google.com/drive/19BTb6mKJfm_tb24CaizOJE8BoIvV3nje?usp=sharing

We introduce the **Fatigue Index (FI)**, a lightweight, model-agnostic, inference-time
diagnostic for long-horizon generation degradation. FI aggregates three per-token
signals — attention to the prompt, embedding drift, and next-token entropy
deviation — into a single bounded score in `[0, 1]`:

```
FI_t = w_A * phi_A(A_t) + w_E * phi_E(E_t) + w_D * phi_D(D_t)
```

See Sections 5–7 and Appendix A of the paper for the full formalization,
axioms, and calibration.

## Repository structure

```
fatigue/                       Core FI package (signals, normalization, aggregation)
  config.py                      FatigueConfig — frozen calibration constants (Table 6)
  attention.py                   A_t: attention-to-prompt / evidence-attention
  drift.py                       D_t: embedding drift
  entropy.py                     E_t: next-token entropy
  normalize.py                   phi_A, phi_E, phi_D normalization maps (Sec. 6.2)
  index.py                       FI aggregation, smoothing, FatigueMonitor (online use)
  hysteresis.py                  Two-threshold alerting (Sec. 7.3)

utils/                         Shared evaluation utilities (not part of FI math)
  metrics.py                     EM / F1, repetition ratio
  seeds.py                       Reproducibility helpers

experiments/
  common.py                      Shared harness: dataset loaders, model loaders,
                                  prompt construction, per-token generation + probing,
                                  plotting. All FI math is imported from `fatigue/`.
  empirical_validation/          RQ3 (paper Sec. 7): reliability of FI
    run_validation.py              Temporal trajectory, predictive validity (Spearman),
                                    aggregation value (AUROC vs. single signals),
                                    hysteresis stability
    *.ipynb                        Exploratory notebooks behind the paper's Sec. 7 figures
  architectural_stress/          RQ2 (paper Sec. 5.2): FI under architectural stress
    run_stress.py                  Context-length, positional-sensitivity, and
                                    FP16-vs-4-bit-NF4 precision ablations
    *.ipynb                        Exploratory notebook behind the paper's stress figures

configs/                       YAML calibration + run configs (default.yaml = Table 6)
scripts/                       Shell wrappers that sweep the two experiment drivers
data/                          Dataset READMEs; sampled JSONL files are generated here
examples_rq2_outputs/          Example output artifacts (JSON + plots) from a past run
tests/                         Unit tests for the fatigue package (no GPU required)
```

| Paper element              | Repository location                                    |
|-----------------------------|--------------------------------------------------------|
| FI definition (Eq. 1)       | `fatigue/index.py`                                      |
| Normalization maps (§6.2)   | `fatigue/normalize.py`                                   |
| Calibration (Table 6)       | `fatigue/config.py`, `configs/default.yaml`             |
| §7 Empirical Validation     | `experiments/empirical_validation/`                      |
| §5.2 Architectural Stress   | `experiments/architectural_stress/`                      |

## Setup

```bash
pip install -r requirements.txt
```

The `fatigue` package itself has no heavy dependencies (just NumPy and, optionally,
PyYAML for config loading) and can also be installed standalone:

```bash
pip install -e .
```

`bitsandbytes` (4-bit NF4 quantization) requires a CUDA GPU; the FP16/FP32 path
runs on CPU or GPU. Datasets (HotpotQA, TriviaQA, SQuAD, Natural Questions) are
downloaded automatically from the Hugging Face Hub on first use and cached as
JSONL samples under `data/`.

## Quick start: the FI package

```python
from fatigue import FatigueMonitor, fatigue_index_series

# Online, step-by-step (e.g. during generation)
monitor = FatigueMonitor()
for a_t, e_t, d_t in signal_stream:       # one triple per decoding step
    fi, alert = monitor.update(a_t, e_t, d_t)

# Post hoc, over a full generation's probe series
fi_trajectory = fatigue_index_series(attention, drift, entropy)
```

All calibration constants default to the paper's Table 6 values
(`fatigue.DEFAULT_CONFIG`). To use a different calibration, pass a
`FatigueConfig` — e.g. `FatigueConfig.from_yaml("configs/nf4.yaml")` — to any
function in the package.

## Running the experiments

```bash
# Architectural stress (RQ2, paper Sec. 5.2)
python experiments/architectural_stress/run_stress.py \
    --datasets hotpot --run_context --run_positional --run_precision

# Empirical validation (RQ3, paper Sec. 7)
python experiments/empirical_validation/run_validation.py \
    --datasets hotpot --sample_size 50 --precision fp
```

Or sweep the default configuration across all datasets:

```bash
bash scripts/run_all_stress.sh
bash scripts/run_all_validation.sh
```

Both drivers write JSON reports and plots to `results/<experiment_name>/`.

## Tests

```bash
pytest tests/
```

The suite covers the normalization maps, the FI aggregation axioms
(monotonicity, boundedness, compositionality), hysteresis, the online
`FatigueMonitor` against the batch computation, and the evaluation metrics —
all without requiring a GPU or model weights. Two tests exercise `torch`
tensor inputs directly and are skipped automatically if PyTorch is not
installed.

## Citation

If you use this code, please cite the paper (see the project homepage above
for the BibTeX entry once available).

## License

MIT — see `LICENSE`.
