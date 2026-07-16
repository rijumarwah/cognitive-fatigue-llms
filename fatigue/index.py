"""Fatigue Index (FI) aggregation (paper Eq. 1).

    FI_t = w_A * phi_A(A_t) + w_E * phi_E(E_t) + w_D * phi_D(D_t)

with w_A + w_E + w_D = 1. The FI is a linear aggregator over the three
normalized signals, chosen for interpretability and online use: it exposes
per-signal contributions and supports simple stabilizers such as smoothing and
hysteresis.

This module offers two entry points:

* :func:`fatigue_index_series` -- vectorized, for post-hoc analysis of a full
  generation (the reference-step signal at index 0 is dropped by default).
* :class:`FatigueMonitor` -- an online, step-by-step monitor for real-time
  deployment that maintains the smoothing buffer and hysteresis alert state.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Sequence

from .config import DEFAULT_CONFIG, FatigueConfig
from .normalize import phi_attention, phi_drift, phi_entropy


def fatigue_index(a_t: float, e_t: float, d_t: float,
                  config: FatigueConfig = DEFAULT_CONFIG) -> float:
    """Instantaneous FI from one triple of raw signals (attention, entropy, drift)."""
    return (config.w_attention * phi_attention(a_t, config)
            + config.w_entropy * phi_entropy(e_t, config)
            + config.w_drift * phi_drift(d_t, config))


def _drop_reference(series: Sequence[float]) -> List[float]:
    """Drop the step-0 reference sample (prompt forward) from a probe series."""
    if not series:
        return []
    return list(series[1:]) if len(series) > 1 else []


def smooth_series(series: Sequence[float], window: int) -> List[float]:
    """Trailing moving average over ``window`` samples (no look-ahead)."""
    if not series:
        return []
    w = max(1, int(window))
    if w == 1:
        return list(series)
    out: List[float] = []
    running = 0.0
    buf: Deque[float] = deque(maxlen=w)
    for v in series:
        if len(buf) == w:
            running -= buf[0]
        buf.append(v)
        running += v
        out.append(running / len(buf))
    return out


def fatigue_index_series(attention: Sequence[float],
                         drift: Sequence[float],
                         entropy: Sequence[float],
                         config: FatigueConfig = DEFAULT_CONFIG,
                         skip_reference: bool = True,
                         smooth: bool = True) -> List[float]:
    """Compute the FI trajectory from full signal series.

    Parameters
    ----------
    attention, drift, entropy:
        Per-probe raw signal series (as returned by a generation run). The
        first sample corresponds to the initial prompt forward and is dropped
        when ``skip_reference`` is set.
    config:
        Calibration constants (defaults to the paper's Table 6 values).
    skip_reference:
        Drop the step-0 reference sample before aggregating.
    smooth:
        Apply the length-``L`` trailing moving average used for online alerting.
    """
    a = _drop_reference(attention) if skip_reference else list(attention)
    d = _drop_reference(drift) if skip_reference else list(drift)
    e = _drop_reference(entropy) if skip_reference else list(entropy)
    n = min(len(a), len(d), len(e))
    if n == 0:
        return []
    fi = [fatigue_index(a[i], e[i], d[i], config) for i in range(n)]
    return smooth_series(fi, config.smooth_window) if smooth else fi


def mean_fi(fi_series: Sequence[float]) -> float:
    """Mean of an FI trajectory (0.0 for an empty series)."""
    if not fi_series:
        return 0.0
    return float(sum(fi_series) / len(fi_series))


class FatigueMonitor:
    """Online Fatigue Index monitor for real-time generation.

    Feed one triple of raw signals per decoding step via :meth:`update`; the
    monitor returns the smoothed FI and a boolean alert governed by hysteresis
    (activate above ``theta``, deactivate below ``theta_low``). Intended for
    deployment as a lightweight runtime diagnostic.

    Example
    -------
    >>> mon = FatigueMonitor()
    >>> fi, alert = mon.update(a_t=0.006, e_t=2.1, d_t=15.0)
    """

    def __init__(self, config: FatigueConfig = DEFAULT_CONFIG):
        self.config = config
        self._buffer: Deque[float] = deque(maxlen=max(1, int(config.smooth_window)))
        self._alert: bool = False
        self.history: List[float] = []

    def update(self, a_t: float, e_t: float, d_t: float):
        """Ingest one step; return ``(smoothed_fi, alert)``."""
        raw = fatigue_index(a_t, e_t, d_t, self.config)
        self._buffer.append(raw)
        fi = sum(self._buffer) / len(self._buffer)
        if not self._alert and fi >= self.config.hysteresis_high:
            self._alert = True
        elif self._alert and fi <= self.config.hysteresis_low:
            self._alert = False
        self.history.append(fi)
        return fi, self._alert

    @property
    def alert(self) -> bool:
        return self._alert

    def reset(self) -> None:
        self._buffer.clear()
        self._alert = False
        self.history = []
