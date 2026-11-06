"""Synthetic-data tests for the sigma_seed extractor (no real result used)."""

import json
import random
import statistics
from statistics import NormalDist

import pytest

from experiments.n10d.sigma_seed import (
    DEV_SEEDS,
    MIN_K,
    main,
    pilot_n_eval,
    required_seeds,
    required_seeds_from_sigma_d,
    sigma_d,
    sigma_seed,
    sign_test_power,
    sign_test_seeds,
    sign_test_threshold,
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


def _eng(sigma, centre=0.8, n_eval=500, seed=2):
    rng = random.Random(seed)
    return [
        {
            "seed": s,
            "condition": "asym",
            "arm": "eng",
            "success": min(1, max(0, centre + rng.gauss(0, sigma))),
            "n_eval": n_eval,
        }
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
    assert {k: sign_test_threshold(k) for k in (5, 8, 10, 12)} == {5: 5, 8: 7, 10: 9, 12: 10}
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


def _correlated(n_eval=500):
    # Shared seed effect u_s on both arms, small eng-only jitter: the two
    # arms are strongly correlated, so their sd sum overstates sd(D).
    u = [-0.21, -0.14, -0.07, 0.0, 0.04, 0.08, 0.13, 0.17]
    jit = [0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01, 0.0]
    noc = [
        {"seed": s, "condition": "asym", "arm": "nocomm", "success": 0.30 + a, "n_eval": n_eval}
        for s, a in zip(DEV_SEEDS, u, strict=True)
    ]
    eng = [
        {"seed": s, "condition": "asym", "arm": "eng", "success": 0.65 + a + b, "n_eval": n_eval}
        for s, a, b in zip(DEV_SEEDS, u, jit, strict=True)
    ]
    return noc + eng


def test_paired_sigma_d_beats_the_bound_on_correlated_seeds():
    recs = _correlated()
    arm = {a: [r["success"] for r in recs if r["arm"] == a] for a in ("eng", "nocomm")}
    bound = statistics.stdev(arm["eng"]) + statistics.stdev(arm["nocomm"])
    diffs = [e - n for e, n in zip(arm["eng"], arm["nocomm"], strict=True)]
    assert sigma_d(recs) == pytest.approx(statistics.stdev(diffs))
    assert bound == pytest.approx(0.2623, abs=1e-3)
    assert sigma_d(recs) == pytest.approx(0.0131, abs=1e-3)
    assert required_seeds(bound) == 13
    assert required_seeds(sigma_d(recs)) == MIN_K


def test_sigma_d_is_blind_to_a_constant_shift():
    base = _recs(0.02) + _eng(0.10)
    shifted = _recs(0.02) + [{**r, "success": r["success"] - 0.3} for r in _eng(0.10)]
    assert sigma_d(base) == pytest.approx(sigma_d(shifted))
    assert sigma_d(base) > 0


def test_unpaired_seeds_rejected():
    recs = _correlated()
    eng_9603 = next(r for r in recs if r["arm"] == "eng" and r["seed"] == 9603)
    eng_9603["seed"] = 9608
    with pytest.raises(ValueError):
        sigma_d(recs)
    recs = [r for r in _correlated() if not (r["arm"] == "eng" and r["seed"] == 9603)]
    with pytest.raises(ValueError):
        sigma_d(recs)


def test_per_seed_n_eval_mismatch_rejected():
    recs = _correlated()
    recs[-1]["n_eval"] = 499
    with pytest.raises(ValueError, match="n_eval"):
        sigma_d(recs)


def test_cli_reads_n_eval_from_filtered_rows(tmp_path):
    # Foreign row first (other arm, n_eval=1): before the A3 fix the CLI took
    # recs[0]["n_eval"] and sized k on n_eval=1.
    recs = [{"seed": 1, "condition": "asym", "arm": "learned", "success": 0.9, "n_eval": 1}]
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


# Tests carried over from PR #100 (e5fbe88), adapted to the paired read-out
# of the PI decision 2026-11-06 16:55 (protocol section 8, reconciliation).


def _write(tmp_path, recs):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(recs))
    return str(path)


def test_cli_refuses_stray_eng_row_with_other_n_eval(tmp_path):
    # Bug of 6e682cc: n_eval came from recs[0]; a leading eng row at n_eval=1
    # gave k=48. With the paired read the stray row is an eng record, so the
    # CLI must stop instead of sizing k on it.
    lead = {"seed": 9600, "condition": "asym", "arm": "eng", "success": 1.0, "n_eval": 1}
    recs = [lead, *[{**r, "success": 0.25} for r in _recs(0.0)], *_eng(0.0)]
    with pytest.raises(ValueError):
        main([_write(tmp_path, recs)])


def test_cli_rejects_heterogeneous_filtered_n_eval(tmp_path):
    recs = _recs(0.02) + _eng(0.05)
    recs[3]["n_eval"] = 100
    with pytest.raises(ValueError, match="n_eval"):
        main([_write(tmp_path, recs)])


def test_readable_arms_guard():
    # PR #100 read one arm per call; the paired read keeps the guard that no
    # arm other than nocomm and eng is ever read.
    both = _recs(0.0) + _eng(0.05, seed=3)
    assert sigma_seed(both) == sigma_seed(_recs(0.0))
    for arm in ("learned", "msg_shuffle"):
        with pytest.raises(ValueError):
            pilot_n_eval(both, (arm,))


def test_seed_rules_for_paired_d_and_sign_test():
    assert required_seeds_from_sigma_d(0.30) == 9  # no factor 2 when sigma_D is direct
    assert required_seeds_from_sigma_d(0.0) == MIN_K
    assert sign_test_threshold(5) == 5
    with pytest.raises(ValueError):
        sign_test_threshold(4)  # 4/4 positive has p = 1/16 > 0.05
    assert sign_test_power(5, 0.9) == pytest.approx(0.9**5)
    k = sign_test_seeds(0.9)
    assert k > 5 and sign_test_power(k, 0.9) >= 0.90
    assert {q: sign_test_seeds(q) for q in (0.80, 0.90, 0.95)} == {0.80: 23, 0.90: 11, 0.95: 8}


def test_sign_test_k_dominates_the_paired_mean_k():
    # Under D_i ~ N(0.30, sigma_D) the paired-mean floor never binds.
    for i in range(51):
        sd = i / 100
        q = 1.0 if sd == 0 else NormalDist().cdf(0.30 / sd)
        assert required_seeds(sd) == sign_test_seeds(q)
        assert required_seeds(sd) >= required_seeds_from_sigma_d(sd)
