"""Learnability-gate verdict (amendment 2026-10-25); numpy-free, runs in CI."""
import json

from experiments.procgen_multiagent_gtm_comm.harness.gate_report import (
    report,
    verdict,
)


def _run(final=0.10, coin=(0.1, 0.4), flagged_late=False, abort=False,
         done=1_000_000):
    live = {"block3_pos_frac": 0.3, "live_h_frac": 0.6, "logit_std": 0.1,
            "value_std": 0.1, "flagged": False}
    return {
        "total_pair_steps": 1_000_000, "pair_steps_done": done,
        "success_curve": [[40_000, 0.0], [1_000_000, final]],
        "coin_rate_curve": [[40_000, coin[0]], [1_000_000, coin[1]]],
        "entropy_curve": [[8192, 2.7, 2.6], [1_000_000, 1.5, 0.2]],
        "liveness": [live | {"step": 0, "flagged": True},  # step 0 ignored
                     live | {"step": 100_000, "flagged": flagged_late}],
        "liveness_flagged": flagged_late, "liveness_abort": abort,
        "wall_time_s": 1200.0,
    }


def test_gate_passes_only_when_all_criteria_hold():
    assert verdict(_run()) == (True, [])
    assert not verdict(_run(final=0.04))[0]  # below 2x random (0.05)
    assert not verdict(_run(coin=(0.3, 0.3)))[0]
    assert not verdict(_run(flagged_late=True))[0]
    assert not verdict(_run(abort=True))[0]
    assert not verdict(_run(done=400_000))[0]


def test_report_renders_runs(tmp_path):
    d = tmp_path / "fix-c-coin-pretrain-gtm"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(_run()))
    txt = report(tmp_path)
    assert "fix-c-coin-pretrain-gtm" in txt and "PASS" in txt
