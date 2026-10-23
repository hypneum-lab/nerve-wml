"""Throughput / smoke / stage-2 manifest tests (scripted envs, tiny budgets)."""
import json
import re

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness import (  # noqa: E402
    manifest,
    throughput,
)


class _Scripted:
    def __init__(self, rand_seed, coin_steps):
        self.coin_steps, self.t = coin_steps, -1

    def act(self, actions):
        self.t += 1

    def observe(self):
        coin = self.t in self.coin_steps
        img = np.full((1, 64, 64, 3), (self.t + 1) % 256, dtype=np.uint8)
        return np.array([10.0 if coin else 0.0]), {"rgb": img}, np.array([coin])

    def close(self):
        pass


def _factory():
    created = []

    def f(rand_seed):
        env = _Scripted(rand_seed, {0} if len(created) % 2 == 0 else {1})
        created.append(env)
        return env

    return f


def test_measure_throughput_reports_rates(tmp_path):
    res = throughput.measure_throughput(
        "nocomm", 5, env_factory=_factory(), n_pairs=2, horizon=4,
        eval_levels=range(0, 2), skip=1,
    )
    assert res["pair_steps"] == 5 * 8 and not res["training_failure"]
    assert res["pair_steps_per_s"] > 0
    assert res["agent_samples_per_s"] == pytest.approx(2 * res["pair_steps_per_s"])
    assert res["eval_proxy"]["levels"] == [0, 2]
    assert "PROXY" in res["eval_proxy"]["label"]
    assert res["eval_proxy"]["seconds_per_model_2x200"] > 0
    assert "uptime_before" in res


def test_smoke_records_outcome_and_crash(tmp_path):
    ok = throughput.run_smoke("gtm", "registered", tmp_path, total=16,
                              env_factory=_factory(), n_pairs=2, horizon=4,
                              log_every=8)
    assert ok["outcome"] == "completed" and ok["pair_steps_done"] == 16
    assert ok["all_losses_finite"] and ok["using_stub"] is False

    def boom(seed):
        raise RuntimeError("env exploded")

    bad = throughput.run_smoke("mlp", "closed", tmp_path, total=16,
                               env_factory=boom, n_pairs=2, horizon=4,
                               log_every=8)
    assert bad["outcome"] == "crashed" and "env exploded" in bad["error"]


def test_smoke_plan_within_registered_caps():
    from collections import Counter

    per_arm = Counter(a for a, _ in throughput.SMOKE_RUNS)
    assert max(per_arm.values()) <= 3
    assert throughput.SMOKE_PAIR_STEPS <= 1_000_000
    assert len(throughput.SMOKE_RUNS) == 7


def _manifest(tmp_path, **kw):
    out = tmp_path / "stage2-manifest-DRAFT.md"
    manifest.write_manifest(out, **kw)
    return out.read_text()


def test_manifest_contains_required_items(tmp_path):
    text = _manifest(tmp_path)
    assert "DRAFT — to be filed on OSF by the PI as stage 2 before any run" in text
    heads = re.findall(r"^## (.+)$", text, re.M)
    for n in ("1.", "2.", "3.", "4."):
        assert any(h.startswith(n) for h in heads), n
    for needle in ("ARM_CODES", "CONDITION_CODES", "Caveflyer-coop",
                   "Maze-coop", "not run", "SHA-256", "5e1dbf3", "Qt 5.15.19",
                   "K = 4", "A = 16", "pool", "lr 5e-4", "Smoke runs",
                   "Artefact archiving", "100.78.191.52", "post-hoc"):
        assert needle in text, needle
    for arm in ("gtm", "mlp", "atoc", "dial", "eng", "nocomm"):
        assert re.search(rf"\| {arm} \| [\d,]+ \|", text), arm


def test_manifest_never_mentions_heldout_results(tmp_path):
    import inspect

    assert max(throughput.TRAIN_LEVELS) < 1000
    assert "HELD_OUT" not in inspect.getsource(throughput)
    assert "HELD_OUT" not in inspect.getsource(manifest)
    with pytest.raises(ValueError, match="training levels"):
        throughput.eval_proxy_seconds("gtm", levels=range(1000, 1002))
    d = tmp_path / "m"
    d.mkdir()
    (d / "throughput.json").write_text(json.dumps({"results": {"gtm": {
        "eval_proxy": {"levels": [1000, 1200]}}}}))
    with pytest.raises(ValueError, match="training levels"):
        manifest.write_manifest(d / "out.md")


def test_manifest_flags_estimate_above_planning_figure(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "throughput.json").write_text(json.dumps({"measured_on": "t", "results": {
        a: {"pair_steps_per_s": 500.0, "agent_samples_per_s": 1000.0,
            "steady_s_per_update_cycle": 16.0, "pair_steps_per_s_incl_warmup": 480.0,
            "uptime_before": "up", "uptime_after": "up",
            "eval_proxy": {"levels": [0, 16], "seconds_per_model_2x200": 100.0,
                           "label": "PROXY"}}
        for a in ("gtm", "nocomm")}}))
    out = d / "out.md"
    manifest.write_manifest(out)
    text = out.read_text()
    assert "EXCEEDS the amendment's planning figure" in text
    est = manifest.compute_estimate(
        {"gtm": {"pair_steps_per_s": 500.0, "eval_proxy": {"seconds_per_model_2x200": 100.0}},
         "nocomm": {"pair_steps_per_s": 1000.0, "eval_proxy": {"seconds_per_model_2x200": 50.0}}})
    assert est["hours_per_run_gtm"] == pytest.approx(50e6 / 500 / 3600 + 100 / 3600)
    assert est["total_hours"] > 0
