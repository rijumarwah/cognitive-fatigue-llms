"""Unit tests for the Fatigue Index package.

These exercise the pure math (normalization maps, aggregation, hysteresis,
axioms, metrics) with no model or GPU required.
"""

import math

import numpy as np
import pytest

from fatigue import (
    FatigueConfig, DEFAULT_CONFIG,
    phi_attention, phi_entropy, phi_drift,
    fatigue_index, fatigue_index_series, mean_fi,
    apply_hysteresis, count_flips, FatigueMonitor,
    prompt_attention, embedding_drift, token_entropy,
)
from utils.metrics import em_f1, repetition_ratio

try:
    import torch  # noqa: F401
    _no_torch = False
except Exception:
    _no_torch = True


# ---- Config ---------------------------------------------------------------
def test_default_weights_sum_to_one():
    c = DEFAULT_CONFIG
    assert abs(c.w_attention + c.w_entropy + c.w_drift - 1.0) < 1e-9

def test_default_matches_paper_table6():
    c = DEFAULT_CONFIG
    assert (c.w_attention, c.w_entropy, c.w_drift) == (0.40, 0.35, 0.25)
    assert (c.entropy_band_low, c.entropy_band_high) == (3.8, 5.0)
    assert (c.hysteresis_high, c.hysteresis_low) == (0.50, 0.40)
    assert c.prompt_slice_k == 64 and c.smooth_window == 5 and c.probe_every == 2

def test_bad_weights_rejected():
    with pytest.raises(ValueError):
        FatigueConfig(w_attention=0.5, w_entropy=0.5, w_drift=0.5)


# ---- Normalization maps (bounded, monotone, correct piecewise) ------------
def test_phi_outputs_bounded():
    for v in [-5, 0, 0.003, 0.5, 1, 10]:
        assert 0.0 <= phi_attention(v) <= 1.0
        assert 0.0 <= phi_drift(v) <= 1.0
    for e in [0, 2, 3.8, 4.4, 5.0, 8, 20]:
        assert 0.0 <= phi_entropy(e) <= 1.0

def test_phi_entropy_zero_in_band():
    assert phi_entropy(3.8) == 0.0
    assert phi_entropy(4.4) == 0.0
    assert phi_entropy(5.0) == 0.0

def test_phi_entropy_piecewise_values():
    # low side divides by H_l, high side divides by beta
    assert phi_entropy(0.0) == pytest.approx(min(1.0, 3.8 / 3.8))
    assert phi_entropy(2.6) == pytest.approx((3.8 - 2.6) / 3.8)
    assert phi_entropy(7.5) == pytest.approx((7.5 - 5.0) / 5.0)

def test_phi_attention_formula():
    assert phi_attention(0.0) == 1.0
    assert phi_attention(1.0) == 0.0
    assert phi_attention(0.25) == pytest.approx(0.75)

def test_phi_drift_saturates_at_cap():
    assert phi_drift(0.0) == 0.0
    assert phi_drift(DEFAULT_CONFIG.drift_cap) == pytest.approx(1.0)
    assert phi_drift(10 * DEFAULT_CONFIG.drift_cap) == 1.0


# ---- Axioms ---------------------------------------------------------------
def test_axiom_monotonicity():
    base = fatigue_index(a_t=0.5, e_t=4.4, d_t=5.0)
    # lower attention -> higher fatigue
    assert fatigue_index(a_t=0.2, e_t=4.4, d_t=5.0) >= base
    # larger entropy deviation -> higher fatigue
    assert fatigue_index(a_t=0.5, e_t=8.0, d_t=5.0) >= base
    # larger drift -> higher fatigue
    assert fatigue_index(a_t=0.5, e_t=4.4, d_t=18.0) >= base

def test_axiom_boundedness():
    for a in (0.0, 0.5, 1.0):
        for e in (0.0, 4.4, 12.0):
            for d in (0.0, 10.0, 50.0):
                assert 0.0 <= fatigue_index(a, e, d) <= 1.0

def test_axiom_compositionality_weights():
    a, e, d = 0.3, 6.0, 10.0
    expected = (0.40 * phi_attention(a) + 0.35 * phi_entropy(e) + 0.25 * phi_drift(d))
    assert fatigue_index(a, e, d) == pytest.approx(expected)


# ---- Series + smoothing ---------------------------------------------------
def test_series_drops_reference_and_smooths():
    attn = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    drift = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    ent = [4.4, 4.4, 3.0, 2.0, 1.0, 0.5]
    fi = fatigue_index_series(attn, drift, ent)
    assert len(fi) == 5  # one reference sample dropped
    assert all(0.0 <= v <= 1.0 for v in fi)

def test_series_rises_as_signals_worsen():
    n = 20
    attn = list(np.linspace(0.9, 0.0, n))     # attention decays
    drift = list(np.linspace(0.0, 40.0, n))   # drift grows
    ent = list(np.linspace(4.4, 0.2, n))      # entropy collapses
    fi = fatigue_index_series(attn, drift, ent, smooth=False)
    assert fi[-1] > fi[0]

def test_mean_fi_empty():
    assert mean_fi([]) == 0.0


# ---- Hysteresis -----------------------------------------------------------
def test_hysteresis_reduces_flips():
    series = [0.3, 0.55, 0.45, 0.52, 0.44, 0.51, 0.3]  # jitter around threshold
    naive = [1 if v >= 0.50 else 0 for v in series]
    hyst = apply_hysteresis(series)
    assert count_flips(hyst) <= count_flips(naive)

def test_hysteresis_state_machine():
    # rises above 0.50 -> on; stays on until below 0.40
    series = [0.2, 0.6, 0.45, 0.41, 0.39, 0.5]
    assert apply_hysteresis(series) == [0, 1, 1, 1, 0, 1]


# ---- Online monitor matches batch computation -----------------------------
def test_monitor_matches_series():
    rng = np.random.default_rng(0)
    n = 30
    attn = list(rng.random(n))
    drift = list(rng.random(n) * 30)
    ent = list(rng.random(n) * 8)
    # batch (no reference drop, with smoothing) vs online monitor
    batch = fatigue_index_series(attn, drift, ent, skip_reference=False, smooth=True)
    mon = FatigueMonitor()
    online = [mon.update(attn[i], ent[i], drift[i])[0] for i in range(n)]
    assert np.allclose(batch, online, atol=1e-9)


# ---- Signal extractors ----------------------------------------------------
def test_prompt_attention_shape_guard():
    assert prompt_attention(None, [0, 1]) == 0.0
    attn = np.ones((4, 6, 6), dtype=np.float32)
    val = prompt_attention(attn, [0, 1, 2])
    assert val == pytest.approx(1.0)

@pytest.mark.skipif(_no_torch, reason="torch not installed")
def test_token_entropy_uniform():
    import torch
    vocab = 100
    logits = torch.zeros(1, 1, vocab)  # uniform -> entropy = ln(vocab)
    assert token_entropy(logits) == pytest.approx(math.log(vocab), abs=1e-4)

@pytest.mark.skipif(_no_torch, reason="torch not installed")
def test_embedding_drift_zero_and_positive():
    import torch
    h = torch.ones(8)
    assert embedding_drift(h, h) == 0.0
    assert embedding_drift(2 * h, h) == pytest.approx(math.sqrt(8))


# ---- Metrics --------------------------------------------------------------
def test_em_f1_basic():
    assert em_f1("the cat", "the cat") == (1, 1.0)
    em, f1 = em_f1("the cat sat", "the cat")
    assert em == 0 and 0.0 < f1 < 1.0

def test_repetition_ratio():
    assert repetition_ratio("a a a a a a", n=3) > 0.0
    assert repetition_ratio("one two three four five", n=3) == 0.0
