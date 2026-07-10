import math

from fatigue.config import FatigueConfig
from fatigue.index import compute_fi_series, fi_at_step, fi_components


def test_weights_must_sum_to_one():
    import pytest

    with pytest.raises(ValueError):
        FatigueConfig(weight_attention=0.5, weight_entropy=0.5, weight_drift=0.5)


def test_fi_at_step_is_weighted_sum_of_phis():
    cfg = FatigueConfig()
    # attention=1.0 (no penalty), entropy in-band (no penalty), drift=0 (no penalty)
    fi = fi_at_step(a_t=1.0, e_t=(cfg.entropy_band_low + cfg.entropy_band_high) / 2, d_t=0.0, config=cfg)
    assert math.isclose(fi, 0.0, abs_tol=1e-9)

    # worst case: no attention, entropy far outside band, drift saturated
    fi_bad = fi_at_step(a_t=0.0, e_t=0.0, d_t=1e6, config=cfg)
    assert math.isclose(fi_bad, 1.0, abs_tol=1e-9)  # wA+wE+wD = 1, all phis = 1


def test_fi_bounded_in_unit_interval():
    cfg = FatigueConfig()
    for a in [-1, 0, 0.3, 0.7, 1, 2]:
        for e in [-5, 0, 3.8, 4.4, 5.0, 20]:
            for d in [-1, 0, 10, 20, 1000]:
                fi = fi_at_step(a, e, d, cfg)
                assert 0.0 <= fi <= 1.0, (a, e, d, fi)


def test_compute_fi_series_drops_anchor_step_by_default():
    cfg = FatigueConfig(smoothing_window=1)  # disable smoothing for a clean check
    attn = [1.0, 1.0, 0.0]     # step0=anchor, step1, step2
    drift = [0.0, 0.0, 20.0]
    entropy = [4.0, 4.0, 4.0]  # inside band throughout
    series = compute_fi_series(attn, drift, entropy, cfg, drop_anchor_step=True, smooth=False)
    assert len(series) == 2  # anchor step dropped
    assert math.isclose(series[0], 0.0, abs_tol=1e-9)
    assert math.isclose(series[1], cfg.weight_attention * 1.0 + cfg.weight_drift * 1.0, abs_tol=1e-9)


def test_fi_components_attribution_matches_fi_at_step():
    cfg = FatigueConfig()
    comp = fi_components(0.5, 4.5, 5.0, cfg)
    total = comp["contrib_A"] + comp["contrib_E"] + comp["contrib_D"]
    assert math.isclose(total, comp["FI"], abs_tol=1e-9)
    assert math.isclose(comp["FI"], fi_at_step(0.5, 4.5, 5.0, cfg), abs_tol=1e-9)
