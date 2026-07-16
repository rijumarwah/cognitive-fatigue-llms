"""Hysteresis-based alerting for the Fatigue Index (paper Sec. 7.3).

A naive single threshold on FI flips the alert state on every small oscillation.
Hysteresis uses two thresholds - activate when FI rises above ``theta`` and
deactivate only when it falls below ``theta_low`` - which suppresses jitter
while preserving genuine onset detection.
"""

from __future__ import annotations

from typing import List, Sequence

from .config import DEFAULT_CONFIG, FatigueConfig


def apply_hysteresis(series: Sequence[float],
                     config: FatigueConfig = DEFAULT_CONFIG,
                     high: float = None,
                     low: float = None) -> List[int]:
    """Return the per-step alert state (0/1) for an FI series under hysteresis.

    ``high`` / ``low`` override the config thresholds when provided.
    """
    if not series:
        return []
    hi = config.hysteresis_high if high is None else high
    lo = config.hysteresis_low if low is None else low
    state = 0
    out: List[int] = []
    for v in series:
        if state == 0 and v >= hi:
            state = 1
        elif state == 1 and v <= lo:
            state = 0
        out.append(state)
    return out


def count_flips(alert_states: Sequence[int]) -> int:
    """Number of transitions (0<->1) in an alert-state sequence."""
    if not alert_states:
        return 0
    return sum(1 for i in range(1, len(alert_states)) if alert_states[i] != alert_states[i - 1])


def naive_alerts(series: Sequence[float], threshold: float) -> List[int]:
    """Baseline single-threshold alerting (for comparison against hysteresis)."""
    if not series:
        return []
    return [1 if v >= threshold else 0 for v in series]
