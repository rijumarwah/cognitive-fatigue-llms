"""
Normalization maps phi_A, phi_E, phi_D (paper Section 6.2).

These are FIXED, non-adaptive, monotone transforms of the raw per-token
signals into unitless penalties in [0, 1]. Calibration constants (entropy
band, beta, kappa) come from a single frozen `FatigueConfig` and are never
re-derived from the sequence being scored -- that per-run adaptivity is
exactly what Axiom A2 (Scale Invariance) and the "one-time calibration"
procedure in Section 6.3 rule out. If you need per-run min/max normalization
for some other purpose, keep it out of this module.

    phi_A(A_t) = 1 - clip(A_t, 0, 1)

    phi_E(E_t) = clip( (H_low - E_t) / H_low       if E_t < H_low
                        0                            if H_low <= E_t <= H_high
                        (E_t - H_high) / beta        if E_t > H_high  , 0, 1)

    phi_D(D_t) = clip(D_t / kappa, 0, 1)
"""

from typing import List, Sequence

from .config import DEFAULT_CONFIG, FatigueConfig


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, x))


def phi_attention(a_t: float, config: FatigueConfig = DEFAULT_CONFIG) -> float:
    """phi_A(A_t) = 1 - clip(A_t, 0, 1). Higher penalty = less attention to prompt."""
    return 1.0 - _clip(float(a_t), 0.0, 1.0)


def phi_entropy(e_t: float, config: FatigueConfig = DEFAULT_CONFIG) -> float:
    """phi_E(E_t): 0 inside the healthy band, else scaled deviation from the band edge."""
    lo, hi = config.entropy_band_low, config.entropy_band_high
    e_t = float(e_t)
    if e_t < lo:
        dev = (lo - e_t) / lo
    elif e_t > hi:
        dev = (e_t - hi) / config.entropy_beta
    else:
        dev = 0.0
    return _clip(dev, 0.0, 1.0)


def phi_drift(d_t: float, config: FatigueConfig = DEFAULT_CONFIG) -> float:
    """phi_D(D_t) = clip(D_t / kappa, 0, 1)."""
    return _clip(float(d_t) / config.drift_kappa, 0.0, 1.0)


def phi_attention_series(
    attn_series: Sequence[float], config: FatigueConfig = DEFAULT_CONFIG
) -> List[float]:
    return [phi_attention(a, config) for a in attn_series]


def phi_entropy_series(
    entropy_series: Sequence[float], config: FatigueConfig = DEFAULT_CONFIG
) -> List[float]:
    return [phi_entropy(e, config) for e in entropy_series]


def phi_drift_series(
    drift_series: Sequence[float], config: FatigueConfig = DEFAULT_CONFIG
) -> List[float]:
    return [phi_drift(d, config) for d in drift_series]
