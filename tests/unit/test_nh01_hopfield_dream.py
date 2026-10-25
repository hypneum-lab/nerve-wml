"""Implementation checks for the NH-01 pilot (pre-registered, binding).

Pre-reg: ``docs/milestones/2026-10-25-nh01-hopfield-dream-vs-downscale.md``.
These tests contain no empirical number: they check identities only
(J(0) = J, spectral form = pattern-space closed form, shrink invariance at
T = 0, estimator on synthetic curves).
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.nh01_hopfield_dream import run as nh


def _patterns(p: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random((p, n)) < 0.5, -1.0, 1.0)


def test_dreaming_kernel_at_t_zero_is_hebb() -> None:
    xi = _patterns(5, 40, 1)
    J = nh.hebbian_kernel(xi)
    assert np.allclose(nh.dreaming_kernel(J, 0.0), J, atol=1e-12)
    # continuity: a tiny t stays close to J
    assert np.allclose(nh.dreaming_kernel(J, 1e-9), J, atol=1e-7)


@pytest.mark.parametrize("t", [0.3, 1.0, 10.0, 100.0])
def test_spectral_form_matches_two_pattern_closed_form(t: float) -> None:
    """Two patterns: (1+t)(1+tC)^-1 has the explicit 2x2 closed form."""
    n = 32
    xi = _patterns(2, n, 7)
    q = float(xi[0] @ xi[1] / n)
    det = (1.0 + t) ** 2 - (t * q) ** 2
    m = (1.0 + t) / det * np.array([[1.0 + t, -t * q], [-t * q, 1.0 + t]])
    J_closed = sum(m[a, b] * np.outer(xi[a], xi[b]) for a in range(2) for b in range(2)) / n
    J = nh.hebbian_kernel(xi)
    assert np.allclose(nh.dreaming_kernel(J, t), J_closed, atol=1e-10)
    assert np.allclose(nh.dreaming_kernel_pattern_space(xi, t), J_closed, atol=1e-10)


def test_spectral_form_matches_pattern_space_many_patterns() -> None:
    xi = _patterns(12, 64, 3)
    J = nh.hebbian_kernel(xi)
    for t in (0.3, 3.0, 30.0):
        assert np.allclose(nh.dreaming_kernel(J, t), nh.dreaming_kernel_pattern_space(xi, t), atol=1e-9)


def test_infinite_time_limit_is_projector() -> None:
    xi = _patterns(6, 48, 11)
    J = nh.hebbian_kernel(xi)
    proj = xi.T @ np.linalg.inv(xi @ xi.T) @ xi  # orthogonal projector on span(xi)
    assert np.allclose(nh.dreaming_kernel(J, 1e8), proj, atol=1e-6)


def test_matched_shrink_equalises_frobenius_norm() -> None:
    xi = _patterns(10, 64, 5)
    J0 = nh.offdiag(nh.hebbian_kernel(xi))
    Jt = nh.offdiag(nh.dreaming_kernel(nh.hebbian_kernel(xi), 10.0))
    c_fro, c_spec = nh.matched_shrink_factors(J0, Jt)
    assert np.isclose(np.linalg.norm(c_fro * J0), np.linalg.norm(Jt))
    assert 0.0 < c_fro < 1.0  # dreaming reduces the synaptic budget at alpha = 10/64
    assert c_spec > 0.0


def test_uniform_shrink_is_inert_at_zero_temperature() -> None:
    """Theorem 1 of the pre-reg: sign dynamics are invariant under c > 0."""
    xi = _patterns(8, 64, 9)
    J0 = nh.offdiag(nh.hebbian_kernel(xi))
    cues = nh.make_cues(xi, np.arange(8), 0.10, np.random.default_rng(1))
    S_a, sw_a, _ = nh.run_zero_temperature(J0, cues, np.random.default_rng(2))
    S_b, sw_b, _ = nh.run_zero_temperature(0.37 * J0, cues, np.random.default_rng(2))
    assert sw_a == sw_b
    assert np.array_equal(S_a, S_b)


def test_make_cues_flips_exact_count() -> None:
    xi = _patterns(4, 50, 2)
    cues = nh.make_cues(xi, np.arange(4), 0.10, np.random.default_rng(0))
    assert np.all((cues != xi).sum(axis=1) == 5)
    cues0 = nh.make_cues(xi, np.arange(4), 0.0, np.random.default_rng(0))
    assert np.array_equal(cues0, xi)


def test_capacity_estimator_cases() -> None:
    alphas = np.array([0.1, 0.2, 0.3, 0.4])
    assert nh.capacity_from_curve(alphas, np.array([1.0, 1.0, 0.0, 0.0])) == (0.25, "ok")
    assert nh.capacity_from_curve(alphas, np.array([1.0, 0.75, 0.25, 0.0])) == (0.25, "ok")
    assert nh.capacity_from_curve(alphas, np.array([1.0, 1.0, 1.0, 1.0])) == (0.4, "censored")
    assert nh.capacity_from_curve(alphas, np.array([0.4, 0.2, 0.0, 0.0])) == (0.0, "below_grid")
    # last alpha with r >= 0.5 rules, even after a transient dip
    a, flag = nh.capacity_from_curve(alphas, np.array([1.0, 0.4, 0.6, 0.0]))
    assert flag == "ok" and np.isclose(a, 0.3 + (0.6 - 0.5) / 0.6 * 0.1)


def test_classify_partitions_probes() -> None:
    m_cued = np.array([0.95, 0.2, 0.1])
    m_all = np.array([[0.95, 0.0], [0.2, 0.93], [0.1, 0.3]])
    out = nh.classify(m_cued, m_all, np.array([0, 0, 0]), 0.9)
    assert out["retrieved_frac"] == pytest.approx(1 / 3)
    assert out["other_stored_frac"] == pytest.approx(1 / 3)
    assert out["spurious_frac"] == pytest.approx(1 / 3)


def test_glauber_runs_and_returns_shapes() -> None:
    xi = _patterns(4, 32, 4)
    J0 = nh.offdiag(nh.hebbian_kernel(xi))
    cues = nh.make_cues(xi, np.arange(4), 0.10, np.random.default_rng(1))
    S, m = nh.run_glauber(J0, cues, 5.0, xi, np.random.default_rng(3), sweeps=6, avg_last=2)
    assert S.shape == (4, 32) and m.shape == (4, 4)
    assert np.all(np.abs(S) == 1.0)
    assert np.all(np.abs(m) <= 1.0 + 1e-12)
