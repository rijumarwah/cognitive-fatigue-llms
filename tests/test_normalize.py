import math

from fatigue.config import FatigueConfig
from fatigue.normalize import phi_attention, phi_drift, phi_entropy


def test_phi_attention_bounds_and_matches_paper_formula():
    cfg = FatigueConfig()
    assert phi_attention(1.0, cfg) == 0.0   # full attention -> no penalty
    assert phi_attention(0.0, cfg) == 1.0   # zero attention -> full penalty
    assert math.isclose(phi_attention(0.3, cfg), 0.7, rel_tol=1e-9)
    # out-of-range inputs get clipped, not raise
    assert phi_attention(-5.0, cfg) == 1.0
    assert phi_attention(5.0, cfg) == 0.0


def test_phi_entropy_zero_inside_healthy_band():
    cfg = FatigueConfig()
    assert phi_entropy(cfg.entropy_band_low, cfg) == 0.0
    assert phi_entropy(cfg.entropy_band_high, cfg) == 0.0
    assert phi_entropy((cfg.entropy_band_low + cfg.entropy_band_high) / 2, cfg) == 0.0


def test_phi_entropy_below_band_uses_H_low_denominator():
    cfg = FatigueConfig(entropy_band_low=3.8, entropy_band_high=5.0)
    # E_t = 0 -> dev = (3.8 - 0) / 3.8 = 1.0
    assert math.isclose(phi_entropy(0.0, cfg), 1.0, rel_tol=1e-9)
    # E_t = 1.9 -> dev = (3.8-1.9)/3.8 = 0.5
    assert math.isclose(phi_entropy(1.9, cfg), 0.5, rel_tol=1e-6)


def test_phi_entropy_above_band_uses_beta_denominator():
    cfg = FatigueConfig(entropy_band_low=3.8, entropy_band_high=5.0, entropy_beta=5.0)
    # E_t = 7.5 -> dev = (7.5-5.0)/5.0 = 0.5
    assert math.isclose(phi_entropy(7.5, cfg), 0.5, rel_tol=1e-6)
    # far above band clips to 1.0
    assert phi_entropy(100.0, cfg) == 1.0


def test_phi_drift_fixed_cap_not_adaptive():
    cfg = FatigueConfig(drift_kappa=20.0)
    assert math.isclose(phi_drift(10.0, cfg), 0.5, rel_tol=1e-9)
    assert phi_drift(0.0, cfg) == 0.0
    assert phi_drift(1000.0, cfg) == 1.0  # clipped, cap is fixed regardless of series
