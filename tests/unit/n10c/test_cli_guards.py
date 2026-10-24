"""CLI entry guards: registered budget / held-out levels need credentials."""
import sys

import pytest

pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness import (  # noqa: E402
    evaluate as ev,
)
from experiments.procgen_multiagent_gtm_comm.harness import train as tr  # noqa: E402


def _train_argv(steps, seed="0"):
    return ["train", "--arm", "gtm", "--seed", seed, "--steps", str(steps),
            "--out", "/nonexistent/out"]


def _eval_argv(start, stop):
    return ["evaluate", "--arm", "gtm", "--seed", "0", "--out", "/nonexistent/e.json",
            "--levels", str(start), str(stop)]


def test_train_cli_refuses_registered_budget_without_credentials(monkeypatch):
    called = []
    monkeypatch.setattr(tr, "train_run", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", _train_argv(50_000_000))
    with pytest.raises(SystemExit) as e:
        tr.main()
    assert "refused" in str(e.value.code) and called == []


def test_train_cli_dev_budget_is_allowed(monkeypatch, capsys):
    rec = tr.RunRecord(arm="gtm", seed=0, condition="registered", codes={},
                       total_pair_steps=64)
    calls = []
    monkeypatch.setattr(tr, "train_run", lambda *a, **k: calls.append(a) or rec)
    monkeypatch.setattr(sys, "argv", _train_argv(1_000_000))
    tr.main()
    assert len(calls) == 1


def test_evaluate_cli_refuses_held_out_levels_without_credentials(monkeypatch):
    called = []
    monkeypatch.setattr(ev, "evaluate_model", lambda *a, **k: called.append(1))
    for lv in [(1000, 1200), (990, 1001)]:
        monkeypatch.setattr(sys, "argv", _eval_argv(*lv))
        with pytest.raises(SystemExit) as e:
            ev.main()
        assert "refused" in str(e.value.code)
    assert called == []


def test_evaluate_cli_training_levels_allowed(monkeypatch, tmp_path):
    class R:
        def to_json(self):
            return "{}"
    monkeypatch.setattr(ev, "evaluate_model", lambda *a, **k: R())
    monkeypatch.setattr(ev, "asdict", lambda r: {})
    out = tmp_path / "e.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--arm", "gtm", "--seed", "0",
                                      "--out", str(out), "--levels", "0", "16"])
    ev.main()
    assert out.exists()


def test_cli_with_bad_credentials_still_refused(monkeypatch):
    called = []
    monkeypatch.setattr(ev, "evaluate_model", lambda *a, **k: called.append(1))
    monkeypatch.setattr(sys, "argv", _eval_argv(1000, 1200) + [
        "--manifest-doi", "10.17605/OSF.IO/XXXXX", "--manifest", "/x",
        "--manifest-sha256", "0", "--expect-commit", "1" * 40])
    with pytest.raises(SystemExit):
        ev.main()
    assert called == []


@pytest.mark.parametrize("flag", [["--micro-batch-pairs", "256"], ["--clear-cache"],
                                  ["--n-workers", "2"]])
def test_train_cli_registered_budget_refuses_setting_overrides(monkeypatch, flag):
    called = []
    monkeypatch.setattr(tr, "train_run", lambda *a, **k: called.append(1))
    from experiments.procgen_multiagent_gtm_comm.campaign import guards
    monkeypatch.setattr(guards, "enforce_cli", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", _train_argv(50_000_000) + ["--host", "kxkm"] + flag)
    with pytest.raises(SystemExit) as e:
        tr.main()
    assert "registered" in str(e.value.code) and called == []


def test_train_cli_registered_budget_uses_host_settings(monkeypatch):
    from experiments.procgen_multiagent_gtm_comm.campaign import guards
    from experiments.procgen_multiagent_gtm_comm.harness import codes

    got = []
    ctx = guards.make_context_for_tests()
    monkeypatch.setattr(guards, "enforce_cli", lambda *a, **k: ctx)
    monkeypatch.setattr(
        tr, "train_run",
        lambda *a, **k: got.append(k) or tr.RunRecord(
            arm="gtm", seed=0, condition="registered", codes={}, total_pair_steps=64))
    for host, seed in (("kxkm", "73"), ("macm1", "0")):
        monkeypatch.setattr(sys, "argv", _train_argv(50_000_000, seed) + ["--host", host])
        tr.main()
        assert {k: got[-1][k] for k in codes.HOST_SETTINGS[host]} == codes.HOST_SETTINGS[host]
        assert got[-1]["registration"] is ctx  # the execution boundary re-checks it


def test_train_cli_registered_budget_requires_host(monkeypatch):
    from experiments.procgen_multiagent_gtm_comm.campaign import guards

    monkeypatch.setattr(guards, "enforce_cli", lambda *a, **k: None)
    monkeypatch.setattr(tr, "train_run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", _train_argv(50_000_000))
    with pytest.raises(SystemExit) as e:
        tr.main()
    assert "--host" in str(e.value.code)


@pytest.mark.parametrize("extra,argv_steps,seed,host,needle", [
    ([], 50_000_000, "0", "kxkm", "fixed host"),  # seed 0 is a macM1 seed
    ([], 50_000_000, "73", "macm1", "fixed host"),  # seed 73 is a KXKM seed
    ([], 50_000_000, "7", "macm1", "registered seeds"),
    ([], 50_000_001, "0", "macm1", "exact registered budget"),
    ([], 100_000_000, "0", "macm1", "exact registered budget"),
    ([], 2_000_000, "0", "macm1", "exact registered budget"),  # neither dev nor 50M
    (["--n-pairs", "32"], 50_000_000, "0", "macm1", "n_pairs"),
    (["--horizon", "64"], 50_000_000, "0", "macm1", "horizon"),
    (["--log-every", "500000"], 50_000_000, "0", "macm1", "log_every"),
    (["--ckpt-every", "2000000"], 50_000_000, "0", "macm1", "ckpt_every"),
    (["--condition", "closed", "--arm", "mlp"], 50_000_000, "0", "macm1", "not run"),
])
def test_train_cli_registered_tuple_and_config_are_validated(
        monkeypatch, extra, argv_steps, seed, host, needle):
    from experiments.procgen_multiagent_gtm_comm.campaign import guards

    called = []
    monkeypatch.setattr(guards, "enforce_cli",
                        lambda *a, **k: guards.make_context_for_tests())
    monkeypatch.setattr(tr, "train_run", lambda *a, **k: called.append(1))
    argv = _train_argv(argv_steps, seed)
    if "--arm" in extra:  # replace the default arm instead of duplicating it
        i = argv.index("--arm")
        del argv[i:i + 2]
    monkeypatch.setattr(sys, "argv", argv + ["--host", host] + extra)
    with pytest.raises(SystemExit) as e:
        tr.main()
    assert needle in str(e.value.code) and "refused" in str(e.value.code)
    assert called == []


def test_evaluate_cli_held_out_levels_must_be_exactly_1000_1200(monkeypatch, tmp_path):
    from experiments.procgen_multiagent_gtm_comm.campaign import guards

    called = []
    monkeypatch.setattr(guards, "enforce_cli",
                        lambda *a, **k: guards.make_context_for_tests())
    monkeypatch.setattr(ev, "evaluate_model", lambda *a, **k: called.append(1))
    for lv in [(1000, 1100), (990, 1200), (1000, 1201)]:
        monkeypatch.setattr(sys, "argv", _eval_argv(*lv))
        with pytest.raises(SystemExit):
            ev.main()
    assert called == []
    monkeypatch.setattr(ev, "asdict", lambda r: {})
    out = []

    class R:
        def to_json(self):
            return "{}"

    monkeypatch.setattr(ev, "evaluate_model",
                        lambda *a, **k: out.append(k["registration"]) or R())
    monkeypatch.setattr(sys, "argv", ["evaluate", "--arm", "gtm", "--seed", "0",
                                      "--out", str(tmp_path / "e.json"),
                                      "--levels", "1000", "1200"])
    ev.main()
    assert len(out) == 1 and out[0] is not None
