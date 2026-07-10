"""
The Fatigue Index (FI): definition, aggregation, and online series
computation (paper Eq. 1, Section 3.2, Section 6.2-6.3).

    FI_t = w_A * phi_A(A_t) + w_E * phi_E(E_t) + w_D * phi_D(D_t)

wA + wE + wD = 1 (enforced by FatigueConfig). FI_t is in [0, 1]
(Axiom A3, Boundedness) since each phi_* term is clipped to [0, 1].

`compute_fi_series` mirrors the online computation used for the
paper's reported results: signals are converted step-by-step to
FI_t via the fixed phi_* maps and then smoothed with a short moving
average window (Table 6: smoothing_window=5) to suppress single-step
jitter before hysteresis-based alerting is applied on top (see
`fatigue.hysteresis`).

The initial forward pass over the prompt (step 0) is used only to
establish the drift anchor h_0 (so D_0 = 0 by construction) and is not
itself scored as a "generated token" -- by default it is excluded from
the returned FI trajectory, matching the "generated token position"
x-axis used in Figure 5 of the paper.
"""

from typing import List, Sequence

import numpy as np

from .config import DEFAULT_CONFIG, FatigueConfig
from .normalize import phi_attention, phi_drift, phi_entropy


def fi_at_step(
    a_t: float, e_t: float, d_t: float, config: FatigueConfig = DEFAULT_CONFIG
) -> float:
    """Instantaneous (unsmoothed) Fatigue Index at a single decoding step (Eq. 1)."""
    return (
        config.weight_attention * phi_attention(a_t, config)
        + config.weight_entropy * phi_entropy(e_t, config)
        + config.weight_drift * phi_drift(d_t, config)
    )


def _smooth(series: Sequence[float], window: int) -> List[float]:
    """Causal moving average with window `window` (matches Section 6.3)."""
    if not series:
        return []
    w = max(1, int(window))
    if w == 1:
        return list(series)
    out = []
    for i in range(len(series)):
        start = max(0, i - w + 1)
        out.append(float(np.mean(series[start : i + 1])))
    return out


def compute_fi_series(
    attn_series: Sequence[float],
    drift_series: Sequence[float],
    entropy_series: Sequence[float],
    config: FatigueConfig = DEFAULT_CONFIG,
    drop_anchor_step: bool = True,
    smooth: bool = True,
) -> List[float]:
    """
    Compute the (optionally smoothed) FI_t trajectory from raw per-step
    signal series of equal or near-equal length.

    Parameters
    ----------
    attn_series, drift_series, entropy_series:
        Raw signal values A_t, D_t, E_t, one per probed decoding step,
        step 0 being the initial prompt-conditioned forward pass.
    drop_anchor_step:
        If True (default), excludes step 0 from the returned trajectory,
        since D_0 = 0 by construction and is not a generated token.
    smooth:
        If True (default), applies the Table-6 moving-average window.
    """
    a = list(attn_series)
    d = list(drift_series)
    e = list(entropy_series)
    if drop_anchor_step:
        a, d, e = a[1:], d[1:], e[1:]
    n = min(len(a), len(d), len(e))
    if n == 0:
        return []
    fi = [fi_at_step(a[i], e[i], d[i], config) for i in range(n)]
    return _smooth(fi, config.smoothing_window) if smooth else fi


def fi_components(
    a_t: float, e_t: float, d_t: float, config: FatigueConfig = DEFAULT_CONFIG
) -> dict:
    """Per-signal weighted contributions to FI_t, for attribution (Axiom A5)."""
    pa, pe, pd = phi_attention(a_t, config), phi_entropy(e_t, config), phi_drift(d_t, config)
    return {
        "phi_A": pa,
        "phi_E": pe,
        "phi_D": pd,
        "contrib_A": config.weight_attention * pa,
        "contrib_E": config.weight_entropy * pe,
        "contrib_D": config.weight_drift * pd,
        "FI": config.weight_attention * pa
        + config.weight_entropy * pe
        + config.weight_drift * pd,
    }
