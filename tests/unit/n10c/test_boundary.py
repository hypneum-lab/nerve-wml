"""Execution boundary: train_run / evaluate_* refuse registered budgets and
held-out levels without a guard-issued RegistrationContext."""
import pytest

pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.campaign import guards  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness import codes  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness import evaluate as ev  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness.train import train_run  # noqa: E402

MAC = {"system": "Darwin", "machine": "arm64", "backend": "metal"}
KX = {"system": "Linux", "machine": "x86_64", "backend": "cuda"}
KX_WHEEL = {"version": "0.10.7", "tags": [
    "cp310-cp310-manylinux_2_17_x86_64", "cp310-cp310-manylinux2014_x86_64"]}


@pytest.fixture
def ctx():
    return guards.make_context_for_tests()


def _runtime(monkeypatch, rt, wheel=None):
    monkeypatch.setattr(guards, "_runtime_probe", lambda: rt)
    monkeypatch.setattr(guards, "_procgen_probe", lambda: wheel or KX_WHEEL)


def _kw(**over):
    base = dict(arm="gtm", seed=0, condition="registered", total_pair_steps=50_000_000,
                settings=dict(codes.HOST_SETTINGS["macm1"]),
                config=dict(guards.REGISTERED_TRAIN_CONFIG), has_env_factory=False)
    return base | over


# -- training ---------------------------------------------------------------
@pytest.mark.parametrize("steps", [50_000_000, 10_000_000, 1_000_001, 100_000_000])
def test_train_run_refuses_registered_or_large_budget_without_context(tmp_path, steps):
    with pytest.raises(guards.GuardError, match="development mode|registered path"):
        train_run("gtm", 0, "registered", steps, tmp_path)
    assert not list(tmp_path.iterdir())  # nothing started


def test_train_run_direct_call_with_default_settings_is_refused(tmp_path):
    # defaults = full minibatch / off / one worker: the old unguarded bypass
    with pytest.raises(guards.GuardError):
        train_run("gtm", 73, "registered", 50_000_000, tmp_path, n_pairs=64)


def test_dev_mode_is_constrained_to_1m(monkeypatch):
    assert guards.check_train_boundary(None, **_kw(total_pair_steps=1_000_000)) is None
    assert guards.check_train_boundary(None, **_kw(total_pair_steps=64)) is None
    with pytest.raises(guards.GuardError):
        guards.check_train_boundary(None, **_kw(total_pair_steps=1_000_064))


def test_registered_train_requires_a_genuine_context():
    with pytest.raises(guards.GuardError, match="RegistrationContext"):
        guards.check_train_boundary(object(), **_kw())


def test_registered_train_accepts_the_registered_tuple_on_its_host(monkeypatch, ctx):
    for seed, host, rt in ((0, "macm1", MAC), (17, "macm1", MAC), (42, "macm1", MAC),
                           (73, "kxkm", KX), (101, "kxkm", KX), (131, "kxkm", KX)):
        _runtime(monkeypatch, rt)
        got = guards.check_train_boundary(
            ctx, **_kw(seed=seed, settings=dict(codes.HOST_SETTINGS[host])))
        assert got == host


@pytest.mark.parametrize("over,needle", [
    ({"seed": 7}, "registered seeds"),
    ({"total_pair_steps": 49_999_872}, "budget"),
    ({"total_pair_steps": 50_000_064}, "budget"),
    ({"condition": "closed", "arm": "mlp"}, "not run"),
    ({"condition": "closed", "arm": "atoc"}, "not run"),
    ({"settings": {"micro_batch_pairs": 512, "clear_cache": True, "n_workers": 8}},
     "per-host settings"),  # seed 0 is macM1: KXKM settings refused
    ({"settings": {"micro_batch_pairs": None, "clear_cache": False, "n_workers": 4}},
     "per-host settings"),
    ({"config": {**guards.REGISTERED_TRAIN_CONFIG, "n_pairs": 32}}, "n_pairs"),
    ({"config": {**guards.REGISTERED_TRAIN_CONFIG, "horizon": 64}}, "horizon"),
    ({"config": {**guards.REGISTERED_TRAIN_CONFIG, "log_every_pair_steps": 640_000}},
     "log_every"),
    ({"config": {**guards.REGISTERED_TRAIN_CONFIG, "minibatch_pairs": 512}},
     "minibatch_pairs"),
    ({"has_env_factory": True}, "env_factory"),
])
def test_registered_train_refuses_off_tuple_runs(monkeypatch, ctx, over, needle):
    _runtime(monkeypatch, MAC)
    with pytest.raises(guards.GuardError, match=needle):
        guards.check_train_boundary(ctx, **_kw(**over))


def test_registered_train_checks_the_actual_host_and_backend(monkeypatch, ctx):
    kxs = dict(codes.HOST_SETTINGS["kxkm"])
    for rt in (MAC, {**KX, "backend": "cpu"}, {**KX, "machine": "aarch64"}):
        _runtime(monkeypatch, rt)
        with pytest.raises(guards.GuardError, match="runtime"):
            guards.check_train_boundary(ctx, **_kw(seed=73, settings=kxs))
    _runtime(monkeypatch, KX)  # a macM1 seed on the KXKM box is refused too
    with pytest.raises(guards.GuardError, match="runtime"):
        guards.check_train_boundary(ctx, **_kw(seed=0))


def test_train_run_with_context_still_enforces_host_settings(tmp_path, monkeypatch, ctx):
    _runtime(monkeypatch, MAC)
    with pytest.raises(guards.GuardError, match="per-host settings"):
        train_run("gtm", 0, "registered", 50_000_000, tmp_path, registration=ctx,
                  micro_batch_pairs=256)


# -- evaluation -------------------------------------------------------------
@pytest.mark.parametrize("levels", [range(1000, 1200), range(990, 1001), range(1000, 1001),
                                    range(2000, 2004)])
def test_evaluate_refuses_held_out_without_context(levels):
    with pytest.raises(guards.GuardError, match="RegistrationContext"):
        ev.evaluate_policy(object(), "gtm", "registered", levels)
    with pytest.raises(guards.GuardError, match="RegistrationContext"):
        ev.evaluate_model(None, "gtm", "registered", levels, seed=0)
    with pytest.raises(guards.GuardError, match="RegistrationContext"):
        ev.cross_play_model(None, None, "gtm", "registered", 0, 17, levels)
    with pytest.raises(guards.GuardError, match="RegistrationContext"):
        ev.cross_play_policy(object(), object(), "gtm", "registered", levels)


def test_default_levels_are_held_out_so_they_need_the_context():
    with pytest.raises(guards.GuardError):
        ev.evaluate_model(None, "gtm", "registered", seed=0)  # default HELD_OUT


def test_dev_evaluation_is_limited_to_training_levels():
    assert guards.check_eval_boundary(None, range(0, 16), arm="gtm",
                                      condition="registered") is False
    with pytest.raises(guards.GuardError, match="training levels"):
        guards.check_eval_boundary(None, range(16, 40), arm="gtm", condition="registered")


def test_registered_evaluation_needs_kxkm_x86_build(monkeypatch, ctx):
    ok = dict(arm="gtm", condition="registered", seeds=(73,))
    _runtime(monkeypatch, KX)
    assert guards.check_eval_boundary(ctx, ev.HELD_OUT, **ok) is True
    _runtime(monkeypatch, MAC)  # macM1 / arm64 / Metal
    with pytest.raises(guards.GuardError, match="runtime"):
        guards.check_eval_boundary(ctx, ev.HELD_OUT, **ok)
    _runtime(monkeypatch, KX, {"version": "0.10.7", "tags": ["cp310-cp310-macosx_11_0_arm64"]})
    with pytest.raises(guards.GuardError, match="wheel"):
        guards.check_eval_boundary(ctx, ev.HELD_OUT, **ok)
    _runtime(monkeypatch, KX, {"version": "0.10.6", "tags": KX_WHEEL["tags"]})
    with pytest.raises(guards.GuardError, match="procgen 0.10.6"):
        guards.check_eval_boundary(ctx, ev.HELD_OUT, **ok)


@pytest.mark.parametrize("over,needle", [
    ({"levels": range(1000, 1100)}, "levels must be exactly"),
    ({"seeds": (7,)}, "registered seeds"),
    ({"condition": "closed", "arm": "mlp"}, "registered cell"),
    ({"has_env_factory": True}, "env_factory"),
    ({"seeds": (0, 42), "cross": True}, "fixed pairs"),
])
def test_registered_evaluation_refuses_off_protocol(monkeypatch, ctx, over, needle):
    _runtime(monkeypatch, KX)
    kw = dict(arm="gtm", condition="registered", seeds=(73,), levels=ev.HELD_OUT) | over
    levels = kw.pop("levels")
    with pytest.raises(guards.GuardError, match=needle):
        guards.check_eval_boundary(ctx, levels, **kw)


def test_cross_play_fixed_pairs_are_accepted(monkeypatch, ctx):
    _runtime(monkeypatch, KX)
    for pair in codes.CROSS_PAIRS:
        assert guards.check_eval_boundary(ctx, ev.HELD_OUT, arm="gtm", condition="registered",
                                          seeds=pair, cross=True) is True
