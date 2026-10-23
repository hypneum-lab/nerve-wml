"""CLI entry guards: registered budget / held-out levels need credentials."""
import sys

import pytest

pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness import (  # noqa: E402
    evaluate as ev,
)
from experiments.procgen_multiagent_gtm_comm.harness import train as tr  # noqa: E402


def _train_argv(steps):
    return ["train", "--arm", "gtm", "--seed", "0", "--steps", str(steps),
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
    monkeypatch.setattr(guards, "enforce_cli", lambda *a, **k: None)
    monkeypatch.setattr(
        tr, "train_run",
        lambda *a, **k: got.append(k) or tr.RunRecord(
            arm="gtm", seed=0, condition="registered", codes={}, total_pair_steps=64))
    for host in ("kxkm", "macm1"):
        monkeypatch.setattr(sys, "argv", _train_argv(50_000_000) + ["--host", host])
        tr.main()
        assert {k: got[-1][k] for k in codes.HOST_SETTINGS[host]} == codes.HOST_SETTINGS[host]


def test_train_cli_registered_budget_requires_host(monkeypatch):
    from experiments.procgen_multiagent_gtm_comm.campaign import guards

    monkeypatch.setattr(guards, "enforce_cli", lambda *a, **k: None)
    monkeypatch.setattr(tr, "train_run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", _train_argv(50_000_000))
    with pytest.raises(SystemExit) as e:
        tr.main()
    assert "--host" in str(e.value.code)
