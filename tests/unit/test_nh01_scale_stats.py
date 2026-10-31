"""Extended-real bootstrap (F1), seed ordering (F2), import and version checks (F3, F4)."""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments.nh01_hopfield_dream import provenance as pv
from experiments.nh01_hopfield_dream import run as nh
from experiments.nh01_hopfield_dream import scale_stats as ss

SEEDS = list(range(10))


def test_finite_data_identical_to_historical_helper() -> None:
    x = np.random.default_rng(3).normal(3.5, 0.2, 10)
    assert ss.bootstrap_mean_ci_ext(x, SEEDS) == nh.bootstrap_mean_ci(x)


def test_historical_helper_unchanged_drops_inf() -> None:
    assert nh.bootstrap_mean_ci(np.array([1.0] * 9 + [math.inf])) == (1.0, 1.0, 1.0)


def test_codex_counterexample_censored_seed_is_not_dropped() -> None:
    """9 seeds with S = 1 and one censored seed (HIGH bound +inf)."""
    mean, lo, hi = ss.bootstrap_mean_ci_ext([1.0] * 9 + [math.inf], SEEDS)
    assert mean == math.inf and lo == 1.0 and hi == math.inf
    assert not hi < 1.1  # C3' cannot « hold » on this completion


def test_negative_infinity_bound_of_d() -> None:
    mean, lo, hi = ss.bootstrap_mean_ci_ext([0.5] * 9 + [-math.inf], SEEDS)
    assert mean == -math.inf and lo == -math.inf and hi == 0.5
    assert not lo >= 3  # an infinite bound fails every strict test it meets


def test_mixed_infinities_and_nan_rejected() -> None:
    with pytest.raises(ValueError):
        ss.bootstrap_mean_ci_ext([math.inf, -math.inf, 1.0], [0, 1, 2])
    with pytest.raises(ValueError):
        ss.bootstrap_mean_ci_ext([math.nan, 1.0], [0, 1])
    with pytest.raises(ValueError):
        ss.bootstrap_mean_ci_ext([1.0, 2.0], [0, 0])


def test_invariant_to_input_order() -> None:
    """The second-review counter-example: the result must not depend on order."""
    ratios = [2.802, 2.902, 3.002, 3.052, 3.102, 3.122, 3.172, 3.202, 3.302, 3.502]
    ref = ss.bootstrap_mean_ci_ext(ratios, SEEDS)
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(10)
        got = ss.bootstrap_mean_ci_ext([ratios[i] for i in perm], [SEEDS[i] for i in perm])
        assert got == ref
    with_inf = [*ratios[1:], math.inf]
    ids = [*SEEDS[1:], 0]
    assert ss.bootstrap_mean_ci_ext(with_inf, ids) == ss.bootstrap_mean_ci_ext(
        [math.inf, *ratios[1:]], SEEDS
    )


def test_size_seeded_streams_differ() -> None:
    x = np.linspace(1.0, 2.0, 10)
    a = ss.bootstrap_mean_ci_ext(x, SEEDS, seed=(nh.BOOT_SEED, 1000))
    b = ss.bootstrap_mean_ci_ext(x, SEEDS, seed=(nh.BOOT_SEED, 4000))
    assert a[0] == b[0] and a[1:] != b[1:]


def test_version_check_uses_first_three_fields() -> None:
    ok = (3, 12, 15, "final", 0)
    assert pv.version_ok(ok)
    assert not pv.version_ok((3, 12, 14, "final", 0))
    assert not pv.version_ok((3, 12, 15, "candidate", 1))
    assert sys.version_info != tuple(sys.version_info[:3])  # literal tuple compare is wrong


def _ctx(tmp_path: Path, repo: Path, hashes: dict) -> pv.ImportContext:
    return pv.ImportContext(
        repo_root=repo,
        site_packages=tmp_path / "site",
        stdlib_root=tmp_path / "std",
        tag_hashes=hashes,
        out_dir=tmp_path / "prov",
    )


def test_classification_and_hash_check(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (tmp_path / "site").mkdir()
    (tmp_path / "std").mkdir()
    good = repo / "pkg" / "a.py"
    good.write_text("x = 1\n")
    env = tmp_path / "site" / "lib.py"
    env.write_text("")
    std = tmp_path / "std" / "json.py"
    std.write_text("")
    mods = {
        "pkg.a": types.SimpleNamespace(__file__=str(good)),
        "lib": types.SimpleNamespace(__file__=str(env)),
        "json": types.SimpleNamespace(__file__=str(std)),
        "builtin": types.SimpleNamespace(__file__=None),
    }
    h = pv._sha256(good)
    rep = pv.check_loaded_modules(_ctx(tmp_path, repo, {"pkg/a.py": h}), mods)
    assert rep["violations"] == [] and rep["environment"] == 1 and rep["stdlib"] == 1
    rep = pv.check_loaded_modules(_ctx(tmp_path, repo, {"pkg/a.py": "0" * 64}), mods)
    assert rep["violations"][0]["reason"] == "hash differs"
    rep = pv.check_loaded_modules(_ctx(tmp_path, repo, {}), mods)
    assert rep["violations"][0]["reason"] == "not in tag"


def _late_import_task(path: str) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location("late_helper_nh01", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["late_helper_nh01"] = mod
    spec.loader.exec_module(mod)
    return mod.VALUE


def _real_ctx(out_dir: Path) -> pv.ImportContext:
    import sysconfig

    repo = Path(nh.__file__).resolve().parents[2]
    hashes = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve().is_relative_to(repo) and Path(f).resolve().is_file():
            p = Path(f).resolve()
            hashes[p.relative_to(repo).as_posix()] = pv._sha256(p)
    return pv.ImportContext(
        repo_root=repo,
        site_packages=Path(sysconfig.get_paths()["purelib"]),
        stdlib_root=Path(sysconfig.get_paths()["stdlib"]),
        tag_hashes=hashes,
        out_dir=out_dir,
    )


def _worker(args: tuple) -> tuple[list, list]:
    path, ctx = args
    before = sorted(v["module"] for v in pv.check_loaded_modules(ctx)["violations"])
    try:
        pv.run_checked(_late_import_task, path, ctx)
    except pv.ImportCheckError:
        pass
    line = next(ctx.out_dir.glob("imports-*.jsonl")).read_text().splitlines()[-1]
    import json

    after = sorted(v["module"] for v in json.loads(line)["violations"])
    return before, after


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs fork")
def test_late_import_in_a_real_worker_is_caught(tmp_path: Path) -> None:
    helper = tmp_path / "elsewhere" / "helper.py"
    helper.parent.mkdir()
    helper.write_text("VALUE = 7\n")
    ctx = _real_ctx(tmp_path / "prov")
    with mp.get_context("fork").Pool(1) as pool:
        ((before, after),) = pool.map(_worker, [(str(helper), ctx)])
    assert "late_helper_nh01" not in before
    assert set(after) - set(before) == {"late_helper_nh01"}
    assert len(list((tmp_path / "prov").glob("imports-*.jsonl"))) == 1
