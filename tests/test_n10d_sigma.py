"""Synthetic-data tests for the sigma_seed extractor (no real result used)."""

import json
import math
import random

import pytest

from experiments.n10d.sigma_seed import (
    DEV_SEEDS,
    main,
    sigma_seed,
    sign_test_critical,
    sign_test_power,
)


def _recs(sigma, n_eval=500, seed=1, extra_arm=None):
    rng = random.Random(seed)
    out = [
        {
            "seed": s,
            "condition": "asym",
            "arm": "nocomm",
            "success": min(1, max(0, 0.25 + rng.gauss(0, sigma))),
            "n_eval": n_eval,
        }
        for s in DEV_SEEDS
    ]
    if extra_arm:
        out.append(
            {"seed": 1, "condition": "asym", "arm": extra_arm, "success": 0.9, "n_eval": n_eval}
        )
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


def test_sign_test_table():
    # One-sided alpha = 0.05; ties count as failures (from PR #115).
    assert {k: sign_test_critical(k) for k in (5, 8, 10, 12)} == {5: 5, 8: 7, 10: 9, 12: 10}
    table = {
        (8, 0.8): 0.503,
        (8, 0.9): 0.813,
        (10, 0.8): 0.376,
        (10, 0.9): 0.736,
        (12, 0.8): 0.558,
        (12, 0.9): 0.889,
    }
    for (k, q), power in table.items():
        assert sign_test_power(k, q) == pytest.approx(power, abs=1e-3)
    assert sign_test_power(5, 0.9) == pytest.approx(0.9**5)
    assert sign_test_critical(4) is None and sign_test_power(4, 0.9) == 0.0


def test_sign_test_rows_of_the_sourced_analysis():
    # analysis-4-points-2026-11-06.md, point 3: m_k, size, power at q = 0.8 / 0.9.
    rows = {8: (7, 0.035, 0.503, 0.813), 16: (12, 0.038, 0.798, 0.983)}
    rows[18] = (13, 0.048, 0.867, 0.994)
    for k, (m, size, p8, p9) in rows.items():
        assert sign_test_critical(k) == m
        assert sum(math.comb(k, i) for i in range(m, k + 1)) / 2**k == pytest.approx(size, abs=1e-3)
        assert sign_test_power(k, 0.8) == pytest.approx(p8, abs=1e-3)
        assert sign_test_power(k, 0.9) == pytest.approx(p9, abs=1e-3)
    assert sign_test_critical(20) == 15  # K_CONFIRM = 20 (TBD-13, section 9)


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


def test_no_pilot_sized_k_or_paired_read_out():
    # Sourced analysis 2026-11-06, points 1 and 3: k does not come from the
    # pilot and the paired sd is a post-unblinding sensitivity analysis only.
    import experiments.n10d.sigma_seed as mod

    for name in ("MIN_K", "required_seeds", "required_seeds_from_sigma_d", "sigma_d"):
        assert not hasattr(mod, name)
    assert mod.READABLE_ARMS == ("nocomm",)
