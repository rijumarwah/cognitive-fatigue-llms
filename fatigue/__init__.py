"""Cognitive Fatigue in Autoregressive Transformers -- Fatigue Index (FI).

A lightweight, model-agnostic, inference-time diagnostic for long-horizon
generation degradation. FI aggregates three per-token signals -- attention to
the prompt, embedding drift, and next-token entropy deviation -- into a single
bounded score in [0, 1].

Quick start
-----------
>>> from fatigue import FatigueMonitor
>>> monitor = FatigueMonitor()
>>> for a_t, e_t, d_t in signal_stream:          # per decoding step
...     fi, alert = monitor.update(a_t, e_t, d_t)

Or, post hoc over a full generation:

>>> from fatigue import fatigue_index_series
>>> fi_traj = fatigue_index_series(attention, drift, entropy)

See the paper (ICML 2026) for the formalization and axioms.
"""

from .config import DEFAULT_CONFIG, FatigueConfig
from .attention import prompt_attention, evidence_attention, prompt_span
from .drift import embedding_drift
from .entropy import token_entropy
from .normalize import (
    phi_attention, phi_entropy, phi_drift,
    phi_attention_series, phi_entropy_series, phi_drift_series,
)
from .index import (
    fatigue_index, fatigue_index_series, mean_fi, smooth_series, FatigueMonitor,
)
from .hysteresis import apply_hysteresis, count_flips, naive_alerts

__version__ = "1.0.0"

__all__ = [
    "FatigueConfig", "DEFAULT_CONFIG",
    "prompt_attention", "evidence_attention", "prompt_span",
    "embedding_drift", "token_entropy",
    "phi_attention", "phi_entropy", "phi_drift",
    "phi_attention_series", "phi_entropy_series", "phi_drift_series",
    "fatigue_index", "fatigue_index_series", "mean_fi", "smooth_series",
    "FatigueMonitor",
    "apply_hysteresis", "count_flips", "naive_alerts",
    "__version__",
]

