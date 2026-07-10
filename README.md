# Cognitive Fatigue in Autoregressive Language Models

Code accompanying the paper:

**Cognitive Fatigue in Autoregressive Language Models: Formalization and Measurement**
(Accepted as a Main Track Paper at ICML 2026)

**Project Homepage:** https://rijumarwah.github.io/llmfatigue/
**Try it on Colab:** https://colab.research.google.com/drive/19BTb6mKJfm_tb24CaizOJE8BoIvV3nje?usp=sharing

We introduce the Fatigue Index (FI), an online diagnostic that measures
long-horizon degradation in language models using prompt-directed
attention, embedding drift, and entropy deviation.

> **See `CODE_AUDIT_NOTES.md`** for a record of what was inconsistent
> between this repo and the paper before packaging, and what was fixed.
> In particular: `fatigue/` previously had no implementation, and the drift
> cap (kappa) / high-entropy denominator (beta) used here are not published
> in the paper's Table 6 -- confirm they match your original runs.
>
> **See `USAGE.md`** for install + quick-start instructions.

## Repository structure

- `fatigue/` -- installable library: FI definition, normalization maps,
  raw signal extraction, hysteresis, and a `FatigueMonitor` helper that
  drives a HuggingFace model end-to-end.
- `experiments/rq2.py` -- context-length / positional / precision stress
  experiments (Section 5.2), using `fatigue/` for signal computation.
- `experiments/empirical_validation/` -- notebooks for Section 7
  (temporal dynamics, predictive validity, hysteresis stability).
- `experiments/architectural_stress/` -- notebook for the model-size /
  instruction-tuning scaling experiment (Section 7.5, Figure 6).
- `configs/` -- FI calibration + experiment presets (default / fp16 / nf4 /
  stress), loadable via `fatigue.config.from_yaml`.
- `rq2_results/` -- JSON/PNG outputs from prior `rq2.py` runs.
- `tests/` -- unit tests for the `fatigue` package.

| Paper section              | Repository location                                    |
|-----------------------------|---------------------------------------------------------|
| FI definition (Eq. 1)       | `fatigue/index.py`                                       |
| Normalization maps (S6.2)   | `fatigue/normalize.py`                                    |
| Raw signals (S3.1, S6.1)    | `fatigue/attention.py`, `drift.py`, `entropy.py`           |
| Hysteresis (S7.3)           | `fatigue/hysteresis.py`                                    |
| Calibration (Table 6)       | `fatigue/config.py`, `configs/*.yaml`                       |
| S5.2 Architectural stress   | `experiments/rq2.py`, `experiments/architectural_stress/`  |
| S7 Empirical validation     | `experiments/empirical_validation/`                         |

## Setup

```bash
pip install -e .                    # core library only (numpy, torch)
pip install -e ".[experiments]"     # + transformers/datasets/etc to run real experiments
```

## Quick start: computing FI from precomputed signals

```python
from fatigue import compute_fi_series, apply_hysteresis

fi_series = compute_fi_series(attention_series, drift_series, entropy_series)
alerts = apply_hysteresis(fi_series)
```

## Quick start: driving a model end-to-end

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fatigue import FatigueConfig
from fatigue.monitor import FatigueMonitor

model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

monitor = FatigueMonitor(model, tokenizer, config=FatigueConfig())
trace = monitor.run(prompt="...", evidence="...", max_new_tokens=120)

print(trace.fi_series)   # FI_t trajectory
print(trace.pred)        # generated text
```

## Running the stress experiments (Section 5.2)

```bash
python experiments/rq2.py --run_context --run_positional --run_precision \
    --datasets hotpot,squad --subset_size 5
```

See `python experiments/rq2.py --help` for the full CLI (dataset selection,
model override, seeds, precision options).

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```
