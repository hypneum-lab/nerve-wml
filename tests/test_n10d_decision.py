"""Synthetic-data tests for the N10-D decision rules (no real result used)."""
import random

import pytest

from experiments.n10d.decision import (
    bootstrap_ci,
    combine_backends,
    decide_asym,
    sign_test_rejects,
    sym_gate,
    sym_leak_status,
)
from experiments.n10d.spec import SYM_EPSILON, SYM_LEAK_DELTA

K = 19


def _noisy(values, sd=0.01, seed=0):
    rng = random.Random(seed)
    return [v + rng.gauss(0, sd) for v in values]


def test_constants_of_tbd15():
    assert SYM_EPSILON == 0.05 and SYM_LEAK_DELTA == 0.10


def test_ci_brackets_the_mean_and_is_reproducible():
    d = _noisy([0.75] * 15 + [0.0] * 4)
    lo, hi = bootstrap_ci(d, n_boot=2000)
    assert lo < sum(d) / len(d) < hi
    assert bootstrap_ci(d, n_boot=2000) == (lo, hi)


def test_asym_case_a():
    d = _noisy([0.75] * 17 + [0.02] * 2)
    assert sign_test_rejects(d)
    assert decide_asym(d, n_boot=2000) == "A"


def test_asym_rejection_b_when_upper_below_margin():
    assert decide_asym(_noisy([0.15] * K, sd=0.02), n_boot=2000) == "B"


def test_asym_b_requires_the_sign_test():
    # Sourced analysis, point 4 (a): U < margin but only 10/19 positive seeds,
    # so D > 0 is not shown and the status is "non concluant", not B.
    d = [0.30] * 10 + [-0.05] * 9
    assert sum(d) / len(d) >= 0.10
    assert bootstrap_ci(d, n_boot=2000)[1] < 0.30
    assert not sign_test_rejects(d)
    assert decide_asym(d, n_boot=2000) == "non concluant"


def test_backend_combination_propagates_identical_verdicts():
    for s in ("A", "B", "C"):
        assert combine_backends(s, s) == s
    for pair in [("A", "B"), ("B", "C"), ("A", "C"), ("A", "invalid")]:
        assert combine_backends(*pair) == "non concluant"
        assert combine_backends(*reversed(pair)) == "non concluant"
    for s in ("non concluant", "invalid", "non mesurable"):
        assert combine_backends(s, s) == "non concluant"


def test_asym_case_c_below_floor():
    assert decide_asym(_noisy([0.02] * K), n_boot=2000) == "C"


def test_asym_non_concluant_when_ci_straddles_margin():
    d = _noisy([0.75] * 10 + [0.0] * 9)
    lo, hi = bootstrap_ci(d, n_boot=2000)
    assert lo < 0.30 <= hi
    assert decide_asym(d, n_boot=2000) == "non concluant"


def test_asym_a_requires_the_sign_test():
    # 13/19 positive: the sign test does not reject (critical count 14),
    # although the CI lower bound clears the margin.
    d = [0.95 + 0.001 * i for i in range(13)] + [-0.01] * 6
    assert bootstrap_ci(d, n_boot=2000)[0] >= 0.30
    assert not sign_test_rejects(d)
    assert decide_asym(d, n_boot=2000) == "non concluant"


def test_sym_gate_on_eng_only():
    assert sym_gate([0.96] * K) and not sym_gate([0.94] * K)


def test_sym_case_d_when_ci_lower_clears_delta():
    # Example of protocol section 5: eng_sym 1.0, nocomm_sym 0.625.
    eng = [1.0] * K
    assert sym_leak_status(eng, _noisy([0.625] * K), n_boot=2000) == "D"


def test_sym_small_gap_is_no_leak_signal():
    eng = [1.0] * K
    assert sym_leak_status(eng, _noisy([0.98] * K, sd=0.005), n_boot=2000) == (
        "no leak signal")


def test_sym_not_measurable_when_eng_sym_low():
    assert sym_leak_status([0.90] * K, [0.30] * K, n_boot=2000) == "not measurable"


def test_sym_requires_pairing():
    with pytest.raises(ValueError):
        sym_leak_status([1.0] * K, [1.0] * (K - 1))
