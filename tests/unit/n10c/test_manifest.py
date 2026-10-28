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
                              log_every=8, ledger=tmp_path / "ledger.json")
    assert ok["outcome"] == "completed" and ok["pair_steps_done"] == 16
    assert ok["all_losses_finite"] and ok["using_stub"] is False

    def boom(seed):
        raise RuntimeError("env exploded")

    bad = throughput.run_smoke("mlp", "closed", tmp_path, total=16,
                               env_factory=boom, n_pairs=2, horizon=4,
                               log_every=8, ledger=tmp_path / "ledger.json")
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
    assert "**Status: DRAFT — not frozen (see freeze rule" in text
    assert "filed on OSF" not in text and "Immutable once filed" not in text
    assert "Immutable once frozen" in text
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
    assert est["hours_per_run_gtm"] == pytest.approx(50e6 / 500 / 3600)  # eval runs on KXKM
    assert est["total_hours"] > 0


def test_seed_host_mapping_is_frozen():
    from experiments.procgen_multiagent_gtm_comm.harness import codes

    assert codes.SEEDS == (0, 17, 42, 73, 101, 131)
    assert codes.SEED_HOST == {0: "macm1", 17: "macm1", 42: "macm1",
                               73: "kxkm", 101: "kxkm", 131: "kxkm"}
    assert set(codes.SEED_HOST) == set(codes.SEEDS)
    # three seeds per host
    assert sorted(codes.SEED_HOST.values()) == ["kxkm"] * 3 + ["macm1"] * 3


def test_manifest_records_both_hosts_and_builds(tmp_path):
    text = _manifest(tmp_path)
    for needle in (
            "seeds {0, 17, 42, 73, 101, 131}",
            "macM1: seeds {0, 17, 42}", "KXKM-AI: seeds {73, 101, 131}",
            # both Procgen builds, patch SHA and wheel hash
            "df9bbfa1d454ee51cbea1b8af47fbb1966a79f4a075357ba2a4c95acca834e7f",
            "4b594d14e42f0f2166e59a9e294477b906eb99c90c3343b570b7124cbc865f53",
            "procgen-0.10.7-cp310-cp310-manylinux_2_17_x86_64",
            "NOT bit-identical",
            # versions and KXKM settings
            "mlx-cuda-12", "0.31.1", "micro_batch_pairs = 512", "clear_cache = on",
            "n_workers",
            # nondeterminism statement
            "nondeterministic", "statistically, not bitwise",
            "will not retrace", "bench/kxkm/repeat",
            # evaluation host
            "ALL evaluations", "x86-64"):
        assert needle in text, needle
    assert "single build" not in text
    assert "has not been checked" not in text
    assert "seeds {0, 17, 42, 73, 101}." not in text


def test_manifest_counts_60_runs_and_two_hosts(tmp_path):
    text = _manifest(tmp_path)
    assert "60 runs" in text and "50 runs" not in text
    d = tmp_path / "m"
    d.mkdir()
    (d / "throughput.json").write_text(json.dumps({"measured_on": "t", "results": {
        "gtm": {"pair_steps_per_s": 775.8, "agent_samples_per_s": 1551.6,
                "steady_s_per_update_cycle": 10.5,
                "pair_steps_per_s_incl_warmup": 774.0,
                "uptime_before": "up", "uptime_after": "up",
                "eval_proxy": {"levels": [0, 16], "seconds_per_model_2x200": 402.0,
                               "label": "PROXY"}},
        "nocomm": {"pair_steps_per_s": 805.6, "agent_samples_per_s": 1611.2,
                   "steady_s_per_update_cycle": 10.1,
                   "pair_steps_per_s_incl_warmup": 800.0,
                   "uptime_before": "up", "uptime_after": "up",
                   "eval_proxy": {"levels": [0, 16], "seconds_per_model_2x200": 219.0,
                                  "label": "PROXY"}}}}))
    manifest.write_manifest(d / "out.md")
    out = (d / "out.md").read_text()
    assert "macM1" in out and "bottleneck" in out
    assert "324 h" in out or "323 h" in out  # 30 x 10.77 h on KXKM


def test_compute_estimate_splits_hosts():
    res = {"gtm": {"pair_steps_per_s": 775.8,
                   "eval_proxy": {"seconds_per_model_2x200": 402.0}},
           "nocomm": {"pair_steps_per_s": 805.6,
                      "eval_proxy": {"seconds_per_model_2x200": 219.0}}}
    e = manifest.compute_estimate(res)
    # macM1: 30 runs (6 gtm + 24 others), training only
    mac = 6 * 50e6 / 775.8 / 3600 + 24 * 50e6 / 805.6 / 3600
    assert e["macm1_hours"] == pytest.approx(mac)
    assert e["kxkm_train_hours"] == pytest.approx(30 * 50e6 / 1290 / 3600)
    # evaluation of all 60 models on KXKM
    assert e["kxkm_eval_hours"] == pytest.approx(60 * 402.0 / 3600)
    # cross-seed cross-play: 10 cells x 6 fixed pairs, half an evaluation each
    assert e["kxkm_cross_hours"] == pytest.approx(60 * 0.5 * 402.0 / 3600)
    kx = e["kxkm_train_hours"] + e["kxkm_eval_hours"] + e["kxkm_cross_hours"]
    assert e["kxkm_total_hours"] == pytest.approx(kx)
    assert e["wall_hours"] == pytest.approx(max(e["macm1_hours"], kx))
    assert e["bottleneck_host"] == "macm1"
    assert e["total_hours"] == pytest.approx(e["macm1_hours"] + kx)


def test_manifest_text_includes_cross_play_cost_and_smoke_ledger(tmp_path):
    (tmp_path / "throughput.json").write_text(json.dumps({"measured_on": "t", "results": {
        a: {"pair_steps_per_s": 800.0, "agent_samples_per_s": 1600.0,
            "steady_s_per_update_cycle": 10.0, "pair_steps_per_s_incl_warmup": 780.0,
            "uptime_before": "up", "uptime_after": "up",
            "eval_proxy": {"levels": [0, 16], "seconds_per_model_2x200": 100.0,
                           "label": "PROXY"}}
        for a in ("gtm", "nocomm")}}))
    (tmp_path / "smoke_ledger.json").write_text(
        json.dumps({"max_steps": 1_000_000, "max_per_arm": 3,
                    "arm_allowance": {"gtm": 4}, "runs": [
            {"arm": "gtm", "condition": "registered"}]}))
    text = _manifest(tmp_path)
    assert "cross-seed cross-play probe" in text and "60 passes" in text
    assert "60 cross-play passes" in text
    assert "Development ledger" in text and "cap 3 per arm" in text
    assert "ledger lists 1 counted run(s) but" in text
    assert "(gtm 4, PI decision" in text and "cap: 3 per arm," not in text


def test_gtm_gradient_deviation_no_longer_disclosed(tmp_path):
    text = _manifest(tmp_path)
    assert "differs from the GTMBridge" not in text
    assert "GTMBridge straight-through one-hot" in text


def test_frozen_status_line_and_no_draft_word(tmp_path, monkeypatch):
    from tests.unit.n10c import _trunks
    _trunks.pin_into_guards(monkeypatch, _trunks.make_all(tmp_path / "trunks"))
    text = _manifest(tmp_path, freeze_tag="n10c-freeze-20261024")
    assert re.search(r"^\W*Status: FROZEN \(tag n10c-freeze-20261024\)",
                     text, re.M)
    assert "DRAFT" not in text
    with pytest.raises(ValueError):
        _manifest(tmp_path, freeze_tag="bad tag")


def test_manifest_references_procgen_build_evidence(tmp_path):
    text = _manifest(tmp_path)
    d = "experiments/procgen_multiagent_gtm_comm/bench/procgen-builds"
    assert f"{d}/README.md" in text
    for f in ("frames.py", "frames2.py", "frames_kxkm.json", "frames_macm1.json",
              "h_kxkm.txt", "h_macm1.txt"):
        assert re.search(rf"\| `{re.escape(f)}` \| `[0-9a-f]{{64}}` \|", text), f


def test_manifest_records_n_workers_per_host(tmp_path):
    text = _manifest(tmp_path)
    assert re.search(r"macM1: .*n_workers = 1 ", text)
    assert re.search(r"KXKM-AI: .*n_workers = 8 ", text)


def test_manifest_records_archive_target_and_pull(tmp_path):
    text = _manifest(tmp_path)
    assert "<PATH>" not in text
    assert "electron@100.78.191.52:/home/electron/archives/n10c-fixc/" in text
    assert "pull_archive.sh" in text and "10.2.0.237" in text


def test_manifest_discloses_the_throughput_learning_deviation(tmp_path):
    import shutil

    shutil.copy(throughput.SMOKE_LEDGER, tmp_path / "smoke_ledger.json")
    shutil.copy(throughput.PKG / "manifest" / "smoke_runs.json", tmp_path / "smoke_runs.json")
    text = _manifest(tmp_path)
    assert "throughput_learning" in text and "Declared deviation" in text
    assert "6 of them ran on 2026-10-23" in text and "legacy" not in text
    assert "ledger lists" not in text  # ledger and smoke_runs.json agree


# -- section 5: fix C (freeze plan C3 / C4) -----------------------------------
def test_manifest_section_5_fix_c_with_all_pinned_trunks(tmp_path, monkeypatch):
    from experiments.procgen_multiagent_gtm_comm.campaign import guards
    from tests.unit.n10c import _trunks

    shas = _trunks.make_all(tmp_path / "trunks")
    _trunks.pin_into_guards(monkeypatch, shas)
    text = _manifest(tmp_path, freeze_tag="n10c-freeze-20261101")
    assert "## 5. Amendment 2026-10-25" in text
    assert "Registered preset `fix-c-coin-pretrain`" in text
    assert "MAX_LIVENESS_RESTARTS = 1" in text
    for n, h in shas.items():
        assert f"`{n}`" in text and h in text
    assert text.count("`prc-") == 10
    assert "amendment-2026-10-25.md" in text and "Not yet pinned" not in text
    assert "| kxkm |" in text and "| 40 |" in text
    assert guards.REGISTERED_TRAIN_OPTIONS in text


def test_frozen_manifest_needs_all_16_trunks(tmp_path, monkeypatch):
    from tests.unit.n10c import _trunks

    with pytest.raises(ValueError, match="all 16 pinned trunks"):
        _manifest(tmp_path, freeze_tag="n10c-freeze-20261101")
    draft = _manifest(tmp_path)  # a draft may be incomplete, and says so
    assert "Not yet pinned" in draft
    shas = _trunks.make_all(tmp_path / "trunks")
    _trunks.pin_into_guards(monkeypatch, shas)
    (tmp_path / "trunks" / "seed-73" / "trunk.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="SHA-256|disagree"):
        _manifest(tmp_path, freeze_tag="n10c-freeze-20261101")


def test_fixc_compute_estimate():
    est = manifest.fixc_compute_estimate()
    assert 520 < est["macm1"] < 560 and 400 < est["kxkm"] < 425


def test_throughput_passes_preset_trunk_and_host_settings(monkeypatch):
    seen = {}

    def fake_train_run(*a, **kw):
        seen.update(kw)
        import time
        kw["_rollout_hook"](0, None)
        time.sleep(0.01)
        kw["_rollout_hook"](1, None)
        from types import SimpleNamespace
        return SimpleNamespace(pair_steps_done=8, training_failure=False)

    import experiments.procgen_multiagent_gtm_comm.harness.train as tr
    monkeypatch.setattr(tr, "train_run", fake_train_run)
    monkeypatch.setattr(throughput, "eval_proxy_seconds", lambda *a, **k: None)
    opts = tr.PRESETS["fix-c-coin-pretrain"]
    out = throughput.measure_throughput(
        "gtm", 2, env_factory=_factory(), n_pairs=2, horizon=2, skip=0,
        options=opts, init_trunk="T", micro_batch_pairs=512, clear_cache=True)
    assert seen["options"] is opts and seen["init_trunk"] == "T"
    assert (seen["micro_batch_pairs"], seen["clear_cache"]) == (512, True)
    assert out["preset"] == "fix-c-coin-pretrain"
