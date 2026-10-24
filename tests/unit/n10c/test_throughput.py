"""Development allowance: smoke runs <= 1M steps, <= 3 per arm, ledgered."""
import json
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness import (  # noqa: E402
    devledger,
)
from experiments.procgen_multiagent_gtm_comm.harness import throughput as th  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness import train as tr  # noqa: E402


class _Scripted:
    def __init__(self, rand_seed):
        self.t = -1

    def act(self, actions):
        self.t += 1

    def observe(self):
        coin = self.t in {0}
        img = np.full((1, 64, 64, 3), (self.t + 1) % 256, dtype=np.uint8)
        return np.array([10.0 if coin else 0.0]), {"rgb": img}, np.array([coin])

    def close(self):
        pass


def test_committed_ledger_matches_the_disclosed_smoke_runs():
    ledger = json.loads(th.SMOKE_LEDGER.read_text())
    sm = json.loads((th.PKG / "manifest" / "smoke_runs.json").read_text())
    smokes = [r for r in ledger["runs"] if r["kind"] == "smoke"]
    assert len(smokes) == len(sm["runs"]) == 8  # 7 + the fresh gtm crash (PATH error)
    counts = th.smoke_counts(th.SMOKE_LEDGER)
    assert counts["gtm"] == 3 and all(v <= th.SMOKE_MAX_PER_ARM for v in counts.values())
    # history is kept: the pre-rule throughput / profiling runs are disclosed
    legacy = [r for r in ledger["runs"] if r["kind"] == "legacy_measurement"]
    assert len(legacy) == 6 and all(r["policy_updates"] for r in legacy)
    assert {r["arm"] for r in legacy} == {"gtm", "mlp", "nocomm"}
    assert devledger.remaining(["gtm", "mlp", "atoc"], th.SMOKE_LEDGER) == {
        "gtm": 1, "mlp": 2, "atoc": 2}


def test_reserve_enforces_step_cap_and_three_per_arm(tmp_path):
    led = tmp_path / "ledger.json"
    with pytest.raises(th.SmokeLimitError, match="1,000,000"):
        th.reserve_smoke("gtm", "registered", 1_000_001, led)
    assert not led.exists()  # a refused request is not accounted
    for _ in range(3):
        th.reserve_smoke("mlp", "registered", 1_000_000, led)
    with pytest.raises(th.SmokeLimitError, match="3 of 3"):
        th.reserve_smoke("mlp", "closed", 64, led)
    th.reserve_smoke("gtm", "registered", 64, led)  # other arms are unaffected
    assert th.smoke_counts(led) == {"mlp": 3, "gtm": 1}
    with pytest.raises(ValueError):
        th.reserve_smoke("nope", "registered", 64, led)


def test_ledger_persists_across_processes_and_counts_crashes(tmp_path):
    led = tmp_path / "ledger.json"
    boom = lambda rand_seed: (_ for _ in ()).throw(RuntimeError("env"))  # noqa: E731
    e = th.run_smoke("nocomm", "registered", tmp_path / "r", total=8, n_pairs=2,
                     horizon=4, log_every=8, env_factory=boom, ledger=led)
    assert e["outcome"] == "crashed"
    data = json.loads(led.read_text())
    assert [(r["arm"], r["outcome"]) for r in data["runs"]] == [("nocomm", "crashed")]
    assert th.smoke_counts(led) == {"nocomm": 1}  # the crash still counts


def test_run_smoke_refuses_before_starting_anything(tmp_path):
    led = tmp_path / "ledger.json"
    called = []
    with pytest.raises(th.SmokeLimitError):
        th.run_smoke("gtm", "registered", tmp_path / "r", total=2_000_000,
                     env_factory=lambda s: called.append(1), ledger=led)
    assert called == [] and not led.exists() and not (tmp_path / "r").exists()
    for _ in range(4):
        th.reserve_smoke("gtm", "registered", 8, led)
    with pytest.raises(th.SmokeLimitError):
        th.run_smoke("gtm", "closed", tmp_path / "r", total=8, n_pairs=2, horizon=4,
                     log_every=8, env_factory=_Scripted, ledger=led)
    assert not (tmp_path / "r").exists()


def test_run_smoke_runs_and_records_completed(tmp_path):
    led = tmp_path / "ledger.json"
    e = th.run_smoke("mlp", "registered", tmp_path / "r", total=8, n_pairs=2, horizon=4,
                     log_every=8, env_factory=_Scripted, ledger=led)
    assert e["outcome"] == "completed" and e["pair_steps_done"] == 8
    assert json.loads(led.read_text())["runs"][0]["outcome"] == "completed"


def test_smoke_cli_rejects_more_than_1m_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["throughput", "smoke", "--runs-dir", str(tmp_path),
                                      "--out", str(tmp_path / "o.json"),
                                      "--steps", "2000000", "--ledger", str(tmp_path / "l.json")])
    monkeypatch.setattr(th, "run_smoke", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(SystemExit):
        th.main()


def test_throughput_measurement_is_capped_to_the_development_mode():
    with pytest.raises(ValueError, match="development mode"):
        th.measure_throughput("gtm", n_updates=200)  # 1,638,400 pair-steps


def test_smoke_training_goes_through_the_dev_cap(tmp_path):
    with pytest.raises(tr.guards.GuardError):
        tr.train_run("mlp", th.SMOKE_SEED, "registered", 2_000_000, tmp_path)


# ---- one ledger for every real-environment development path (round 2) ----
@pytest.fixture
def real_env(monkeypatch):
    """Make train_run's real-Procgen branch (env_factory None) use a scripted env."""
    orig = tr.VecCoopEnv

    def make(*a, **k):
        if k.get("env_factory") is None:
            k["env_factory"] = _Scripted
        return orig(*a, **k)

    monkeypatch.setattr(tr, "VecCoopEnv", make)


def _dev(led, arm="mlp", **kw):
    return tr.train_run(arm, 5, "registered", 8, kw.pop("out"), n_pairs=2, horizon=4,
                        log_every_pair_steps=8, ckpt_every_pair_steps=8, dev_ledger=led, **kw)


def test_direct_dev_train_run_is_ledgered_and_counts(tmp_path, real_env):
    led = tmp_path / "l.json"
    for i in range(3):
        _dev(led, out=tmp_path / f"o{i}")
    runs = json.loads(led.read_text())["runs"]
    expected = [("dev_train", "mlp", "completed")] * 3
    assert [(r["kind"], r["arm"], r["outcome"]) for r in runs] == expected
    with pytest.raises(th.SmokeLimitError, match="3 of 3"):
        _dev(led, out=tmp_path / "o4")  # repeated direct calls are no longer unaccounted
    assert not (tmp_path / "o4").exists()
    _dev(led, arm="gtm", out=tmp_path / "g")  # other arms keep their own allowance
    assert th.smoke_counts(led) == {"mlp": 3, "gtm": 1}


def test_direct_dev_train_run_crash_still_counts(tmp_path, monkeypatch):
    led = tmp_path / "l.json"
    orig = tr.VecCoopEnv

    def boom(*a, **k):
        raise RuntimeError("procgen exploded")

    monkeypatch.setattr(tr, "VecCoopEnv", boom)
    with pytest.raises(RuntimeError):
        _dev(led, out=tmp_path / "o")
    assert th.smoke_counts(led) == {"mlp": 1}
    assert json.loads(led.read_text())["runs"][0]["outcome"] == "crashed"
    monkeypatch.setattr(tr, "VecCoopEnv", orig)


def test_scripted_env_runs_are_not_accounted(tmp_path):
    led = tmp_path / "l.json"
    tr.train_run("mlp", 5, "registered", 8, tmp_path / "o", env_factory=_Scripted,
                 n_pairs=2, horizon=4, log_every_pair_steps=8, ckpt_every_pair_steps=8,
                 dev_ledger=led)
    assert not led.exists()


def test_throughput_is_ledgered_as_measurement_on_a_frozen_policy(tmp_path, real_env):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from experiments.procgen_multiagent_gtm_comm.harness.codes import init_key_int
    from experiments.procgen_multiagent_gtm_comm.harness.policy import PairPolicy

    led = tmp_path / "l.json"
    out = tmp_path / "o"
    rec = tr.train_run("mlp", th.SMOKE_SEED, "registered", 16, out, n_pairs=2, horizon=4,
                       log_every_pair_steps=8, ckpt_every_pair_steps=16, frozen_policy=True,
                       dev_ledger=led)
    assert rec.optimizer_steps > 0  # the full update cost was paid ...
    init = PairPolicy("mlp", mx.random.key(init_key_int(th.SMOKE_SEED, "mlp", "registered")))
    saved = mx.load(str(out / rec.last_checkpoint))
    for k, v in tree_flatten(init.parameters()):  # ... and the policy stayed random
        assert mx.array_equal(saved[k], v), k
    r = json.loads(led.read_text())["runs"][0]
    assert r["kind"] == "measurement" and r["policy_updates"] is False
    for i in range(5):  # measurements are ledgered but not counted
        tr.train_run("mlp", 5, "registered", 8, tmp_path / f"m{i}", n_pairs=2, horizon=4,
                     log_every_pair_steps=8, ckpt_every_pair_steps=8, frozen_policy=True,
                     dev_ledger=led)
    assert th.smoke_counts(led) == {}
    assert len(json.loads(led.read_text())["runs"]) == 6


def test_measure_throughput_real_env_path_uses_the_ledger(tmp_path, real_env, monkeypatch):
    led = tmp_path / "l.json"
    monkeypatch.setattr(th, "eval_proxy_seconds", lambda *a, **k: {"label": "PROXY"})
    out = th.measure_throughput("gtm", n_updates=2, n_pairs=2, horizon=4, skip=0, ledger=led)
    assert out["pair_steps"] == 16 and out["training_failure"] is False
    r = json.loads(led.read_text())["runs"]
    assert [(x["kind"], x["arm"], x["steps"], x["outcome"]) for x in r] == [
        ("measurement", "gtm", 16, "completed")]
    assert th.smoke_counts(led) == {}


def test_smoke_with_real_env_makes_exactly_one_ledger_entry(tmp_path, real_env):
    led = tmp_path / "l.json"
    e = th.run_smoke("gtm", "registered", tmp_path / "r", total=8, n_pairs=2, horizon=4,
                     log_every=8, ledger=led)
    assert e["outcome"] == "completed"
    r = json.loads(led.read_text())["runs"]
    assert len(r) == 1 and r[0]["kind"] == "smoke" and r[0]["outcome"] == "completed"


def test_reservations_cannot_be_forged_or_reused_for_another_run(tmp_path, real_env):
    led = tmp_path / "l.json"
    with pytest.raises(th.SmokeLimitError):
        devledger.Reservation(0, "mlp", "registered", 8, "smoke", led)
    res = devledger.reserve("mlp", "registered", 8, kind="smoke", path=led)
    with pytest.raises(th.SmokeLimitError, match="does not cover"):
        _dev(led, arm="gtm", out=tmp_path / "a", dev_reservation=res)
    with pytest.raises(th.SmokeLimitError, match="not a genuine"):
        _dev(led, out=tmp_path / "b", dev_reservation=object())
    meas = devledger.reserve("mlp", "registered", 8, kind="measurement", path=led)
    with pytest.raises(th.SmokeLimitError, match="kind"):
        _dev(led, out=tmp_path / "c", dev_reservation=meas)  # learning run on a measurement slot


def test_registered_runs_never_use_the_dev_ledger_paths(tmp_path):
    from experiments.procgen_multiagent_gtm_comm.campaign import guards

    ctx = guards.make_context_for_tests()
    with pytest.raises(guards.GuardError):
        tr.train_run("gtm", 0, "registered", 50_000_000, tmp_path, registration=ctx,
                     frozen_policy=True)


def test_per_arm_allowance_single_constant_gtm_four_others_three(tmp_path):
    assert devledger.ARM_ALLOWANCE == {"gtm": 4}
    assert [devledger.allowance(a) for a in ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")] == [
        4, 3, 3, 3, 3, 3]
    led = tmp_path / "l.json"
    for _ in range(4):
        th.reserve_smoke("gtm", "registered", 64, led)
    with pytest.raises(th.SmokeLimitError, match="4 of 4"):
        th.reserve_smoke("gtm", "closed", 64, led)
    for _ in range(3):
        devledger.reserve("dial", "registered", 64, kind="dev_train", path=led)
    with pytest.raises(th.SmokeLimitError, match="3 of 3"):
        devledger.reserve("dial", "closed", 64, kind="smoke", path=led)
    assert devledger.remaining(["gtm", "dial", "eng"], led) == {"gtm": 0, "dial": 0, "eng": 3}
    assert json.loads(th.SMOKE_LEDGER.read_text())["arm_allowance"] == {"gtm": 4}
