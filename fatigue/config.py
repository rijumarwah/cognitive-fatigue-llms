"""
Frozen calibration for the Fatigue Index (FI).

Values here mirror Table 6 ("Fixed Defaults and Calibration") in the paper:

    Cognitive Fatigue in Autoregressive Transformers
    (Marwah et al., ICML 2026)

Two parameters referenced by the paper's normalization formulas in §6.2 —
the drift cap `kappa` and the high-entropy-side denominator `beta` — are
described in the text as "fixed from a small preliminary pass and then
frozen for all reported experiments" but their *numeric values* are not
listed in Table 6. This package ships defaults for both (kappa=20.0,
beta=5.0), taken from the values used in `Reliability_of_Fatigue_Index.ipynb`,
which is the notebook whose phi_A / phi_D implementation matches the paper's
formulas exactly (fixed, non-adaptive normalization).

IMPORTANT: If these do not match the values actually used to produce the
numbers reported in the paper (Tables 2-5, Figures 2-7), replace them here
before regenerating any results, and update Table 6 in the paper accordingly
so the artifact and the paper stay in sync. Do not tune these per-run.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FatigueConfig:
    # --- Signal weights (Eq. 1). Must sum to 1 (Axiom A5 / Theorem E.1). ---
    weight_attention: float = 0.40
    weight_entropy: float = 0.35
    weight_drift: float = 0.25

    # --- Entropy healthy band [H_low, H_high] (nats), §6.2 ---
    entropy_band_low: float = 3.8
    entropy_band_high: float = 5.0

    # --- phi_E high-side denominator beta (§6.2). NOT published in Table 6;
    #     see module docstring. ---
    entropy_beta: float = 5.0

    # --- phi_D drift cap kappa (§6.2). NOT published in Table 6;
    #     see module docstring. ---
    drift_kappa: float = 20.0

    # --- Prompt slice size K used for prompt-attention signal, Eq. 2 ---
    prompt_slice_k: int = 64

    # --- Probe frequency: compute signals every N decoding steps ---
    probe_every: int = 2

    # --- FI smoothing window (moving average), §6.3 / §7.3 ---
    smoothing_window: int = 5

    # --- Hysteresis activation / deactivation thresholds, §7.3 / Table 4 ---
    hysteresis_high: float = 0.50
    hysteresis_low: float = 0.40

    # --- Decoding / context settings, Table 6 ---
    max_context_tokens: int = 2048
    max_new_tokens: int = 120
    top_p: float = 0.95
    temperature: float = 1.0
    top_k: int = 0

    # --- Seeds used in the paper's robustness checks (Appendix C.1) ---
    seeds: tuple = (123, 2027)

    def __post_init__(self) -> None:
        total = self.weight_attention + self.weight_entropy + self.weight_drift
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"FI weights must sum to 1 (Axiom A5); got {total} "
                f"(wA={self.weight_attention}, wE={self.weight_entropy}, "
                f"wD={self.weight_drift})"
            )
        if self.entropy_band_low >= self.entropy_band_high:
            raise ValueError("entropy_band_low must be < entropy_band_high")
        if self.drift_kappa <= 0:
            raise ValueError("drift_kappa must be > 0")
        if self.entropy_beta <= 0:
            raise ValueError("entropy_beta must be > 0")
        if not (0.0 <= self.hysteresis_low < self.hysteresis_high <= 1.0):
            raise ValueError(
                "hysteresis thresholds must satisfy 0 <= low < high <= 1"
            )


#: Default configuration, matching Table 6 exactly (module docstring caveats apply).
DEFAULT_CONFIG = FatigueConfig()


def from_yaml(path: str) -> "FatigueConfig":
    """
    Load a FatigueConfig from one of the configs/*.yaml files.

    Only the fields that map onto FatigueConfig are consumed; keys like
    `model_name`, `precision`, `context_lengths_tokens`, etc. are experiment
    metadata, not part of the FI calibration itself, and are ignored here.
    Requires PyYAML (`pip install pyyaml`, or the `experiments` extra).
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "from_yaml requires PyYAML. Install with `pip install pyyaml` "
            "or `pip install cognitive-fatigue[experiments]`."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    kwargs = {}
    weights = raw.get("weights", {})
    if "attention" in weights:
        kwargs["weight_attention"] = float(weights["attention"])
    if "entropy" in weights:
        kwargs["weight_entropy"] = float(weights["entropy"])
    if "drift" in weights:
        kwargs["weight_drift"] = float(weights["drift"])

    band = raw.get("entropy_band", {})
    if "low" in band:
        kwargs["entropy_band_low"] = float(band["low"])
    if "high" in band:
        kwargs["entropy_band_high"] = float(band["high"])

    if "entropy_beta" in raw:
        kwargs["entropy_beta"] = float(raw["entropy_beta"])
    if "drift_kappa" in raw:
        kwargs["drift_kappa"] = float(raw["drift_kappa"])
    if "prompt_slice_k" in raw:
        kwargs["prompt_slice_k"] = int(raw["prompt_slice_k"])
    if "probe_every" in raw:
        kwargs["probe_every"] = int(raw["probe_every"])
    if "smoothing_window" in raw:
        kwargs["smoothing_window"] = int(raw["smoothing_window"])

    hysteresis = raw.get("hysteresis", {})
    if "high" in hysteresis:
        kwargs["hysteresis_high"] = float(hysteresis["high"])
    if "low" in hysteresis:
        kwargs["hysteresis_low"] = float(hysteresis["low"])

    decoding = raw.get("decoding", {})
    if "top_p" in decoding:
        kwargs["top_p"] = float(decoding["top_p"])
    if "temperature" in decoding:
        kwargs["temperature"] = float(decoding["temperature"])
    if "top_k" in decoding:
        kwargs["top_k"] = int(decoding["top_k"])

    if "max_context_tokens" in raw:
        kwargs["max_context_tokens"] = int(raw["max_context_tokens"])
    if "max_new_tokens" in raw:
        kwargs["max_new_tokens"] = int(raw["max_new_tokens"])
    if "seeds" in raw:
        kwargs["seeds"] = tuple(raw["seeds"])

    return FatigueConfig(**kwargs)
