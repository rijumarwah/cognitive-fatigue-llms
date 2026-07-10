"""
Hysteresis-based alerting on top of a Fatigue Index trajectory
(paper Section 7.3, Table 4).

A two-threshold state machine: an alert activates once FI crosses the high
(activation) threshold and stays active until FI drops below the low
(deactivation) threshold. This suppresses transient jitter around a single
naive threshold without changing the underlying signal definitions.
"""

from typing import List, Sequence

from .config import DEFAULT_CONFIG, FatigueConfig


def apply_hysteresis(
    fi_series: Sequence[float], config: FatigueConfig = DEFAULT_CONFIG
) -> List[int]:
    """Returns a 0/1 alert-state series, one entry per FI value in fi_series."""
    high, low = config.hysteresis_high, config.hysteresis_low
    state = 0
    out: List[int] = []
    for v in fi_series:
        if state == 0 and v >= high:
            state = 1
        elif state == 1 and v <= low:
            state = 0
        out.append(state)
    return out


def count_flips(alert_series: Sequence[int]) -> int:
    """Number of times the alert state toggles within a single generation."""
    if not alert_series:
        return 0
    flips = 0
    prev = alert_series[0]
    for s in alert_series[1:]:
        if s != prev:
            flips += 1
        prev = s
    return flips


def naive_threshold(
    fi_series: Sequence[float], threshold: float = DEFAULT_CONFIG.hysteresis_high
) -> List[int]:
    """Single-threshold alert rule, used only as the baseline in Table 4."""
    return [1 if v >= threshold else 0 for v in fi_series]
