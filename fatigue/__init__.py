"""
fatigue: reference implementation of the Fatigue Index (FI) from

    Cognitive Fatigue in Autoregressive Transformers: Formalization
    and Measurement (Marwah et al., ICML 2026)

Quick start
-----------
    from fatigue import FatigueConfig, compute_fi_series, apply_hysteresis

    fi_series = compute_fi_series(attn_series, drift_series, entropy_series)
    alerts = apply_hysteresis(fi_series)

See `fatigue.monitor.FatigueMonitor` for an end-to-end helper that drives a
HuggingFace `transformers` decoder-only model and computes FI online during
generation.
"""

from .config import DEFAULT_CONFIG, FatigueConfig, from_yaml
from .attention import get_evidence_attention, get_prompt_attention, prompt_span
from .drift import get_embedding_drift
from .entropy import get_entropy
from .normalize import (
    phi_attention,
    phi_attention_series,
    phi_drift,
    phi_drift_series,
    phi_entropy,
    phi_entropy_series,
)
from .index import compute_fi_series, fi_at_step, fi_components
from .hysteresis import apply_hysteresis, count_flips, naive_threshold

__version__ = "0.1.0"

__all__ = [
    "FatigueConfig",
    "DEFAULT_CONFIG",
    "from_yaml",
    "get_prompt_attention",
    "get_evidence_attention",
    "prompt_span",
    "get_embedding_drift",
    "get_entropy",
    "phi_attention",
    "phi_entropy",
    "phi_drift",
    "phi_attention_series",
    "phi_entropy_series",
    "phi_drift_series",
    "fi_at_step",
    "fi_components",
    "compute_fi_series",
    "apply_hysteresis",
    "naive_threshold",
    "count_flips",
]
