"""Synthetic-data tests for the sigma_seed extractor (no real result used)."""
import json
import random

import pytest

from experiments.n10d.sigma_seed import (
    DEV_SEEDS,
    MIN_K,
    main,
    required_seeds,
    required_seeds_from_sigma_d,
    sigma_seed,
    sign_test_critical,
    sign_test_power,
    sign_test_seeds,
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


def _write(tmp_path, recs):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(recs))
    return str(path)


def test_cli_reads_n_eval_of_filtered_rows(tmp_path):
    # Bug of 6e682cc: n_eval came from recs[0]; a leading eng row at n_eval=1 gave k=48.
    # Since TBD-14 (2026-11-06) the CLI prints no k: k does not come from the pilot.
    flat = [{**r, "success": 0.25} for r in _recs(0.0)]
    lead = {"seed": 9600, "condition": "asym", "arm": "eng", "success": 1.0, "n_eval": 1}
    out = main([_write(tmp_path, [lead, *flat])])
    assert "n_eval=500" in out and "k=" not in out


def test_cli_rejects_heterogeneous_filtered_n_eval(tmp_path):
    recs = _recs(0.02)
    recs[3]["n_eval"] = 100
    with pytest.raises(ValueError):
        main([_write(tmp_path, recs)])


def test_pilot_reads_nocomm_only():
    # TBD-14 closed 2026-11-06: the pilot never reads eng.
    eng = [{**r, "arm": "eng"} for r in _recs(0.05, seed=3)]
    both = _recs(0.0) + eng
    assert sigma_seed(both) == sigma_seed(_recs(0.0))
    for arm in ("eng", "learned"):
        with pytest.raises(ValueError):
            sigma_seed(both, arm=arm)


def test_obsolete_k_mean_and_sign_test():
    # k_mean is obsolete (TBD-13); kept for the report, value unchanged.
    assert required_seeds_from_sigma_d(0.30) == 9  # no factor 2 when sigma_D is direct
    assert required_seeds_from_sigma_d(0.0) == MIN_K
    assert sign_test_critical(5) == 5 and sign_test_critical(4) is None
    assert sign_test_power(5, 0.9) == pytest.approx(0.9**5)
    k = sign_test_seeds(0.9)
    assert k > 5 and sign_test_power(k, 0.9) >= 0.90
