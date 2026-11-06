"""Synthetic-data tests for the sigma_seed extractor (no real result used)."""
import json
import random

import pytest

from experiments.n10d.sigma_seed import (
    DEV_SEEDS,
    MIN_K,
    main,
    pilot_n_eval,
    required_seeds,
    sigma_d_bound,
    sigma_seed,
    sign_test_power,
    sign_test_threshold,
)


def _recs(sigma, n_eval=500, seed=1, extra_arm=None):
    rng = random.Random(seed)
    out = [
        {"seed": s, "condition": "asym", "arm": "nocomm",
         "success": min(1, max(0, 0.25 + rng.gauss(0, sigma))), "n_eval": n_eval}
        for s in DEV_SEEDS
    ]
    if extra_arm:
        out.append({"seed": 1, "condition": "asym", "arm": extra_arm,
                    "success": 0.9, "n_eval": n_eval})
    return out


def _eng(sigma, centre=0.8, n_eval=500, seed=2):
    rng = random.Random(seed)
    return [
        {"seed": s, "condition": "asym", "arm": "eng",
         "success": min(1, max(0, centre + rng.gauss(0, sigma))), "n_eval": n_eval}
        for s in DEV_SEEDS
    ]


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


def test_sign_test_table():
    # One-sided alpha = 0.05; ties count as failures.
    assert {k: sign_test_threshold(k) for k in (5, 8, 10, 12)} == {
        5: 5, 8: 7, 10: 9, 12: 10}
    table = {(8, 0.8): 0.503, (8, 0.9): 0.813, (10, 0.8): 0.376,
             (10, 0.9): 0.736, (12, 0.8): 0.558, (12, 0.9): 0.889}
    for (k, q), power in table.items():
        assert sign_test_power(k, q) == pytest.approx(power, abs=1e-3)
    assert sign_test_power(5, 0.9) == pytest.approx(0.9**5)


def test_required_seeds_floor_and_monotone():
    assert MIN_K == 8
    assert required_seeds(0.0) == MIN_K
    assert required_seeds(0.20) == 8
    assert required_seeds(0.25) == 13
    assert required_seeds(0.30) == 16
    assert required_seeds(0.40) >= required_seeds(0.30) >= required_seeds(0.15)


def test_required_seeds_rejects_hopeless_spread():
    with pytest.raises(ValueError):
        required_seeds(5.0)


def test_sigma_d_bound_is_blind_to_the_contrast():
    base = _recs(0.02) + _eng(0.10)
    shifted = _recs(0.02) + [{**r, "success": r["success"] - 0.3} for r in _eng(0.10)]
    assert sigma_d_bound(base) == pytest.approx(sigma_d_bound(shifted))
    assert sigma_d_bound(base) > 0


def test_cli_reads_n_eval_from_filtered_rows(tmp_path):
    # Foreign row first (other arm, n_eval=1): before the A3 fix the CLI took
    # recs[0]["n_eval"] and sized k on n_eval=1.
    recs = [{"seed": 1, "condition": "asym", "arm": "learned",
             "success": 0.9, "n_eval": 1}]
    recs += [{**r, "success": 0.25} for r in _recs(0.0)]
    recs += [{**r, "success": 0.9} for r in _eng(0.0)]
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(recs))
    assert pilot_n_eval(recs, ("nocomm", "eng")) == 500
    out = main([str(path)])
    assert out.startswith("n_eval=500 ") and out.endswith(f"k={MIN_K}")


def test_heterogeneous_n_eval_rejected():
    r = _recs(0.02) + _eng(0.05)
    r[3]["n_eval"] = 100
    with pytest.raises(ValueError):
        pilot_n_eval(r)
    r = _recs(0.02) + _eng(0.05, n_eval=100)
    with pytest.raises(ValueError):
        pilot_n_eval(r, ("nocomm", "eng"))
