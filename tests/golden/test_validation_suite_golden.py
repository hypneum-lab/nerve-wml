import numpy as np

from nerve_wml.methodology.hsic_cknna import cknna, hsic_debiased


def test_hsic_debiased_golden_value():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((300, 8))
    y = x + 0.2 * rng.standard_normal((300, 8))
    val = hsic_debiased(x, y)
    # Frozen 2026-05-19: numpy default_rng is stable across versions.
    assert abs(val - 8.5320) < 1e-3, val


def test_cknna_golden_value():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((300, 8))
    y = x + 0.2 * rng.standard_normal((300, 8))
    val = cknna(x, y, k=10)
    # Frozen 2026-05-19.
    assert abs(val - 0.7157) < 1e-3, val
