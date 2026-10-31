"""Synthetic-data tests for the sigma_seed extractor (no real result used)."""
import random

import pytest

from experiments.n10d.sigma_seed import DEV_SEEDS, MIN_K, required_seeds, sigma_seed


def _recs(sigma, n_eval=500, seed=1, extra_arm=None):
    rng = random.Random(seed)
    out = [
        {"seed": s, "condition": "asym", "arm": "nocomm",
         "success": min(1, max(0, 0.25 + rng.gauss(0, sigma))), "n_eval": n_eval}
        for s in DEV_SEEDS
    ]
    if extra_arm:
        out.append({"seed": 1, "condition": "asym", "arm": extra_arm, "success": 0.9, "n_eval": n_eval})
    return out


def test_recovers_known_sigma():
    est = sum(sigma_seed(_recs(0.05, seed=i)) for i in range(200)) / 200
    assert 0.03 < est < 0.06


def test_zero_noise_gives_zero_after_binomial_correction():
    flat = [{**r, "success": 0.25} for r in _recs(0.0)]
    assert sigma_seed(flat) == 0.0


def test_wrong_seed_set_rejected():
    r = _recs(0.02)
    r[0]["seed"] = 1
    with pytest.raises(ValueError):
        sigma_seed(r)


def test_other_arms_ignored():
    assert sigma_seed(_recs(0.02, extra_arm="eng")) == sigma_seed(_recs(0.02))


def test_required_seeds_floor_and_monotone():
    assert required_seeds(0.0, 0.25, 500) == MIN_K
    assert required_seeds(0.30, 0.25, 500) > required_seeds(0.15, 0.25, 500) >= MIN_K
