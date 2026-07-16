"""Calibration constants for the Fatigue Index (FI).

All defaults are the frozen values reported in the paper (Appendix A, Table 6).
They are selected once from a small preliminary pass and then held fixed for
every experiment; nothing here is tuned per dataset or per item.

The two normalization-shape parameters that Table 6 does not tabulate
explicitly -- the entropy high-side slope ``beta`` and the drift cap ``kappa``
-- are the values used to produce the paper's reported results and are exposed
here so they can be inspected and, if needed, overridden from a YAML config.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass(frozen=True)
class FatigueConfig:
    # --- Signal probes -----------------------------------------------------
    prompt_slice_k: int = 64          # K: prompt tokens used for attention-to-prompt
    probe_every: int = 2              # probe frequency (tokens)

    # --- Aggregation weights (w_A >= w_E >= w_D, sum to 1) ------------------
    w_attention: float = 0.40         # w_A
    w_entropy: float = 0.35           # w_E
    w_drift: float = 0.25             # w_D

    # --- Normalization maps (phi_A, phi_E, phi_D) --------------------------
    entropy_band_low: float = 3.8     # H_l  (healthy entropy band, lower edge)
    entropy_band_high: float = 5.0    # H_u  (healthy entropy band, upper edge)
    entropy_beta: float = 5.0         # beta (high-side slope for phi_E)
    drift_cap: float = 20.0           # kappa (drift saturation cap for phi_D)

    # --- Online smoothing + hysteresis alerting ----------------------------
    smooth_window: int = 5            # L: FI smoothing window
    hysteresis_high: float = 0.50     # theta   (activation threshold)
    hysteresis_low: float = 0.40      # theta_low (deactivation threshold)

    def __post_init__(self) -> None:
        total = self.w_attention + self.w_entropy + self.w_drift
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FI weights must sum to 1.0 (got {total:.4f})")
        if self.entropy_band_low > self.entropy_band_high:
            raise ValueError("entropy_band_low must be <= entropy_band_high")
        if self.hysteresis_low > self.hysteresis_high:
            raise ValueError("hysteresis_low must be <= hysteresis_high")
        if self.drift_cap <= 0 or self.entropy_beta <= 0:
            raise ValueError("drift_cap and entropy_beta must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FatigueConfig":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "FatigueConfig":
        import yaml  # local import so PyYAML is only needed when loading configs
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        # Accept either a flat mapping or a nested {"fatigue": {...}} block.
        if "fatigue" in data and isinstance(data["fatigue"], dict):
            data = data["fatigue"]
        return cls.from_dict(data)


#: Module-level default used across the package when no config is passed.
DEFAULT_CONFIG = FatigueConfig()
