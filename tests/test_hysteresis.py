from fatigue.config import FatigueConfig
from fatigue.hysteresis import apply_hysteresis, count_flips, naive_threshold


def test_hysteresis_reduces_flips_vs_naive_threshold():
    cfg = FatigueConfig(hysteresis_high=0.50, hysteresis_low=0.40)
    # oscillates right around 0.5 -- naive single threshold flips a lot
    fi = [0.51, 0.49, 0.52, 0.48, 0.51, 0.49, 0.52, 0.30, 0.60, 0.35]
    naive = naive_threshold(fi, threshold=cfg.hysteresis_high)
    hyst = apply_hysteresis(fi, cfg)
    assert count_flips(hyst) <= count_flips(naive)


def test_hysteresis_activation_and_deactivation():
    cfg = FatigueConfig(hysteresis_high=0.50, hysteresis_low=0.40)
    fi = [0.1, 0.6, 0.45, 0.41, 0.39, 0.1]
    alerts = apply_hysteresis(fi, cfg)
    assert alerts == [0, 1, 1, 1, 0, 0]
