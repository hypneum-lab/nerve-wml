"""Implementation checks of the P1bis-ext and P8c-nocap drivers (smoke seeds only).

Requires numba (pinned in experiments/hopfield_dream_conf/requirements-hopfield-conf.txt,
not in uv.lock); skipped when numba is absent.  No empirical number is asserted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("numba")

from experiments.hopfield_dream_conf import p1bis_ext, p8c_nocap  # noqa: E402


def _rand_patterns(p, n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(p, n)) * 2 - 1).astype(np.float32)


def test_kernel_t0_is_hebb_and_symmetric():
    n = 256
    X = _rand_patterns(40, n)
    J0 = p8c_nocap.hebb(X, n)
    lam, Y = p8c_nocap.kernel_factor(X, n)
    Jt = p8c_nocap.kernel_coupling(lam, Y, n, 0.0)
    assert np.allclose(Jt, J0, atol=1e-4)
    Jinf = p8c_nocap.kernel_coupling(lam, Y, n, math.inf)
    assert np.allclose(Jinf, Jinf.T, atol=1e-5)
    assert np.all(np.diag(Jinf) == 0)


def test_pseudo_inverse_stores_patterns_exactly():
    n = 256
    X = _rand_patterns(60, n, seed=1)
    lam, Y = p8c_nocap.kernel_factor(X, n)
    Jinf = p8c_nocap.kernel_coupling(lam, Y, n, math.inf)
    H = X @ Jinf
    assert np.all(X * H > 0)


def test_relax_batch_reaches_fixed_points_and_keeps_input():
    n = 256
    X = _rand_patterns(20, n, seed=2)
    J = p8c_nocap.hebb(X, n)
    S0 = _rand_patterns(8, n, seed=3)
    keep = S0.copy()
    S, nc, _ = p8c_nocap.relax(S0, J, 12345)
    assert nc == 0
    assert np.array_equal(S0, keep)
    assert np.all(S * (S @ J) >= 0)


def test_unlearn_matches_dense_update():
    n = 128
    X = _rand_patterns(10, n, seed=4)
    J = p8c_nocap.hebb(X, n)
    S = _rand_patterns(5, n, seed=5)
    ref = J - np.float32(0.01) * (S.T @ S)
    np.fill_diagonal(ref, 0.0)
    p8c_nocap.unlearn(J, S, 0.01, rows=32)
    assert np.allclose(J, ref, atol=1e-6)


def test_confirmatory_seed_guard():
    with pytest.raises(SystemExit):
        p8c_nocap.main(["--out", "/nonexistent", "--seeds", "32001"])
    with pytest.raises(SystemExit):
        p1bis_ext.main(["--out", "/nonexistent", "--seeds", "31001"])


def test_seed_namespaces_disjoint_from_pilots_and_smoke():
    pilots = set(range(9505, 9510)) | set(range(9511, 9516)) | set(range(9601, 9621)) \
        | set(range(9611, 9620))
    for s in p1bis_ext.CONF_SEEDS + p8c_nocap.CONF_SEEDS:
        assert s not in pilots and s < p1bis_ext.SMOKE_MIN_SEED
    assert not set(p1bis_ext.CONF_SEEDS) & set(p8c_nocap.CONF_SEEDS)


def test_p1bis_cell_is_deterministic(tmp_path):
    stop = str(tmp_path / "STOP")
    a = p1bis_ext.cell((200, 0.15, 99990, stop))
    b = p1bis_ext.cell((200, 0.15, 99990, stop))

    def strip(rows):
        return [{k: v for k, v in r.items() if k != "secs"} for r in rows]

    assert strip(a["rows"]) == strip(b["rows"])
    arms = {(r["f"], r["arm"]) for r in a["rows"]}
    assert {(f, arm) for f in p1bis_ext.FRACS for arm in "bdg"} <= arms


@pytest.mark.slow
def test_p8c_cell_runs_and_is_deterministic(tmp_path):
    stop = str(tmp_path / "STOP")
    a = p8c_nocap.run_cell(1024, 0.4, 99991, 600.0, stop)
    b = p8c_nocap.run_cell(1024, 0.4, 99991, 600.0, stop)

    def strip(rows):
        return [{k: v for k, v in r.items() if "secs" not in k} for r in rows]

    assert strip(a[0]) == strip(b[0])
    kinds = {(r["arm"], r.get("kind")) for r in a[0]}
    assert ("stoch", "grid") in kinds and ("stoch", "matched") in kinds
    assert a[1]["status"] == "complete"
