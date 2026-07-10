# Usage

## 1. Install

```bash
pip install -e .                    # core library: numpy, torch — just the FI math
pip install -e ".[experiments]"     # + transformers/datasets/etc — needed to run real models
```

## 2. Compute FI from signals you already have

```python
from fatigue import compute_fi_series, apply_hysteresis

# attention_series, drift_series, entropy_series: one value per decoding step
fi_series = compute_fi_series(attention_series, drift_series, entropy_series)
alerts = apply_hysteresis(fi_series)
```

## 3. Run FI monitoring on a real HF model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fatigue import FatigueConfig
from fatigue.monitor import FatigueMonitor

model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

monitor = FatigueMonitor(model, tokenizer, config=FatigueConfig())
trace = monitor.run(prompt="Your question here...", evidence="...", max_new_tokens=120)

trace.fi_series   # the FI_t trajectory
trace.pred        # generated text
trace.entropy, trace.drift, trace.attention  # raw signals if you want them
```

Requires a GPU (or patience on CPU) and downloads the model from Hugging Face.

## 4. Reproduce the paper's stress experiments

```bash
python experiments/rq2.py --run_context --run_positional --run_precision \
    --datasets hotpot --subset_size 5 --model tiiuae/falcon-7b-instruct
```

Or use the prepared scripts:

```bash
bash scripts/run_all_stress.sh
bash scripts/run_all_validation.sh
```

## 5. Run tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Before trusting new numbers out of this

See `CODE_AUDIT_NOTES.md` for the full list, but the two that matter most:

- Confirm `drift_kappa=20.0` and `entropy_beta=5.0` in `fatigue/config.py`
  match what actually produced the paper's reported results — these
  weren't published in Table 6, so they were inferred from the notebook
  cell that looked calibration-intended.
- `experiments/empirical_validation/*.ipynb` were not modified during
  packaging and still use a different entropy band and hysteresis
  thresholds than the paper. Reconcile before re-running them for a
  revision.
