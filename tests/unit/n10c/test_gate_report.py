"""Learnability-gate verdict (amendment 2026-10-25); numpy-free, runs in CI."""
import json

from experiments.procgen_multiagent_gtm_comm.harness.gate_report import (
    combined_verdict,
    report,
    verdict,
)

M = 1_000_000


def _run(final=0.10, coin=(0.1, 0.4), flagged_late=False, abort=False,
         done=M, total=M, n_last=100, probes=10, last_step=M):
    live = {"block3_pos_frac": 0.3, "live_h_frac": 0.6, "logit_std": 0.1,
            "value_std": 0.1, "flagged": False}
    probes_l = [live | {"step": 0, "flagged": True}]  # step 0 is ignored
    probes_l += [live | {"step": 100_000 * (k + 1)} for k in range(probes)]
    if flagged_late:
        probes_l[-1]["flagged"] = True
    return {
        "total_pair_steps": total, "pair_steps_done": done,
        "training_failure": abort, "liveness_abort": abort,
        "success_curve": [[40_000, 0.0], [last_step, final]],
        "success_curve_n": [12, n_last],
        "coin_rate_curve": [[40_000, coin[0]], [last_step, coin[1]]],
        "entropy_curve": [[8192, 2.7, 2.6], [M, 1.5, 0.2]],
        "liveness": probes_l, "liveness_flagged": flagged_late,
        "wall_time_s": 1200.0,
    }


def test_gate_passes_only_when_all_criteria_hold():
    assert verdict(_run()) == (True, [])
    for bad in (dict(final=0.04), dict(coin=(0.3, 0.3)), dict(flagged_late=True),
                dict(abort=True), dict(done=400_000), dict(total=40_000, done=40_000,
                                                           last_step=40_000),
                dict(n_last=37), dict(probes=0), dict(final=float("nan")),
                dict(coin=(0.1, float("nan")))):
        assert not verdict(_run(**bad))[0], bad


def test_combined_verdict_needs_both_arms(tmp_path):
    d = tmp_path / "fix-c-coin-pretrain-gtm"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(_run()))
    ok, why = combined_verdict(tmp_path)
    assert not ok and "eng: missing" in why[0]
    e = tmp_path / "fix-c-coin-pretrain-eng"
    e.mkdir()
    (e / "run.json").write_text(json.dumps(_run(final=0.01)))
    assert not combined_verdict(tmp_path)[0]
    (e / "run.json").write_text(json.dumps(_run()))
    assert combined_verdict(tmp_path) == (True, [])
    txt = report(tmp_path, "kxkm")
    assert "| kxkm | fix-c-coin-pretrain-gtm" in txt and "Combined gate" in txt
