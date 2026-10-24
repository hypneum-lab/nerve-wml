"""Development allowance: smoke runs <= 1M steps, <= 3 per arm, ledgered."""
import json
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")

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
    assert len(ledger["runs"]) == len(sm["runs"]) == 7
    counts = th.smoke_counts(th.SMOKE_LEDGER)
    assert counts["gtm"] == 2 and all(v <= th.SMOKE_MAX_PER_ARM for v in counts.values())


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
    for _ in range(3):
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
