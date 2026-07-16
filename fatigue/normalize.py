"""Normalization maps phi_A, phi_E, phi_D (paper Sec. 6.2).

Each raw signal is mapped to a unit-free penalty in [0, 1] by a fixed, monotone
transform. The maps are *stateless* -- they use only the frozen calibration
constants in :class:`fatigue.config.FatigueConfig`, never per-run statistics --
so the Fatigue Index sits on a constant, comparable scale across runs and
models (axioms A2/A3).

    phi_A(A_t) = 1 - clip(A_t, 0, 1)

                { (H_l - E_t) / H_l      if E_t < H_l
    phi_E(E_t) = { 0                      if H_l <= E_t <= H_u
                { (E_t - H_u) / beta      if E_t > H_u     (result clipped to [0, 1])

    phi_D(D_t) = clip(D_t / kappa, 0, 1)

Higher penalty => more fatigue.
"""

from __future__ import annotations

from typing import List, Sequence

from .config import DEFAULT_CONFIG, FatigueConfig


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def phi_attention(a_t: float, config: FatigueConfig = DEFAULT_CONFIG) -> float:
    """Attention penalty: low prompt attention => high penalty."""
    return 1.0 - _clip01(a_t)


def phi_entropy(e_t: float, config: FatigueConfig = DEFAULT_CONFIG) -> float:
    """Entropy penalty: deviation outside the healthy band [H_l, H_u]."""
    lo = config.entropy_band_low
    hi = config.entropy_band_high
    if e_t < lo:
        dev = (lo - e_t) / lo
    elif e_t > hi:
        dev = (e_t - hi) / config.entropy_beta
    else:
        dev = 0.0
    return _clip01(dev)


def phi_drift(d_t: float, config: FatigueConfig = DEFAULT_CONFIG) -> float:
    """Drift penalty: distance scaled by the fixed cap kappa, saturating at 1."""
    return _clip01(d_t / config.drift_cap)


def phi_attention_series(series: Sequence[float],
                         config: FatigueConfig = DEFAULT_CONFIG) -> List[float]:
    return [phi_attention(v, config) for v in series]


def phi_entropy_series(series: Sequence[float],
                       config: FatigueConfig = DEFAULT_CONFIG) -> List[float]:
    return [phi_entropy(v, config) for v in series]


def phi_drift_series(series: Sequence[float],
                     config: FatigueConfig = DEFAULT_CONFIG) -> List[float]:
    return [phi_drift(v, config) for v in series]
