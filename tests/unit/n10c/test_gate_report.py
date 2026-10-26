"""Learnability-gate verdict (amendment 2026-10-25); numpy-free, runs in CI."""
import copy
import json
from pathlib import Path

import pytest

from experiments.procgen_multiagent_gtm_comm.harness.gate_report import (
    TABLE_HEADER,
    combined_verdict,
    load_index,
    report,
    table_rows,
    verdict,
)

M = 1_000_000
REPO = Path(__file__).resolve().parents[3]
CELLS = REPO / "docs" / "superpowers" / "research" / "2026-10-25-n10c-gate"
AMENDMENT = (REPO / "docs" / "milestones"
             / "n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md")


def _run(final=0.10, coin=(0.1, 0.4), flagged_late=False, abort=False,
         done=M, total=M, n_last=100, probes=10, last_step=M, arm="gtm",
         preset="fix-c-coin-pretrain", seed=9101, condition="registered"):
    live = {"block3_pos_frac": 0.3, "live_h_frac": 0.6, "logit_std": 0.1,
            "value_std": 0.1, "flagged": False}
    probes_l = [live | {"step": 0, "flagged": True}]  # step 0 is ignored
    # first update ending at or after each 100k (8192-step updates)
    probes_l += [live | {"step": min(M, -(-100_000 * (k + 1) // 8192) * 8192)}
                 for k in range(probes)]
    if flagged_late:
        probes_l[-1]["flagged"] = True
    return {
        "arm": arm, "seed": seed, "condition": condition,
        "train_options": {"preset": preset, "liveness": "abort",
                          "liveness_every": 100_000},
        "total_pair_steps": total, "pair_steps_done": done,
        "training_failure": abort, "liveness_abort": abort,
        "success_curve": [[40_000, 0.0], [last_step, final]],
        "success_curve_n": [12, n_last],
        "coin_rate_curve": [[40_000, coin[0]], [last_step, coin[1]]],
        "entropy_curve": [[8192, 2.7, 2.6], [M, 1.5, 0.2]],
        "liveness": probes_l, "liveness_flagged": flagged_late,
        "wall_time_s": 1200.0,
    }


def _ok(r, **kw):
    return verdict(r, **({"arm": "gtm"} | kw))[0]


def test_gate_passes_only_when_all_criteria_hold():
    assert verdict(_run(), arm="gtm") == (True, [])
    for bad in (dict(final=0.04), dict(coin=(0.3, 0.3)), dict(flagged_late=True),
                dict(abort=True), dict(done=400_000), dict(total=40_000, done=40_000,
                                                           last_step=40_000),
                dict(n_last=37), dict(probes=0), dict(probes=9),
                dict(final=float("nan")), dict(coin=(0.1, float("nan")))):
        assert not _ok(_run(**bad)), bad


@pytest.mark.parametrize("field,value", [
    ("arm", "eng"), ("arm", "nocomm"), ("preset", "fix-a-coin"),
    ("seed", 999), ("condition", "closed")])
def test_identity_must_match_the_expectation(field, value):
    assert not _ok(_run(**{field: value}))


def test_codex_round2_counterexample_is_rejected():
    """nocomm arm, preset A, seed 999, condition closed, ten duplicate
    step-1 probes and a coin curve ending at 80k: per-run and combined FAIL."""
    r = _run(arm="nocomm", preset="fix-a-coin", seed=999, condition="closed")
    r["liveness"] = r["liveness"][:1] + [r["liveness"][1] | {"step": 1}
                                         for _ in range(10)]
    r["coin_rate_curve"] = [[40_000, 0.1], [80_000, 0.4]]
    ok, why = verdict(r, arm="gtm")
    assert not ok
    text = "; ".join(why)
    for needle in ("arm", "preset", "seed", "condition", "schedule",
                   "coin-rate curve: ends at 80,000"):
        assert needle in text, needle


def test_probe_coverage_rejections():
    r = _run()
    dup = copy.deepcopy(r)
    dup["liveness"][5] = dict(dup["liveness"][4])  # duplicate, one missing
    gap = copy.deepcopy(r)
    del gap["liveness"][3]
    extra = copy.deepcopy(r)
    extra["liveness"].insert(3, dict(extra["liveness"][2]) | {"step": 210_000})
    no0 = copy.deepcopy(r)
    no0["liveness"] = no0["liveness"][1:]
    unsorted = copy.deepcopy(r)
    unsorted["liveness"][1:] = unsorted["liveness"][:0:-1]
    beyond = copy.deepcopy(r)  # last probe past the completed 1M budget
    beyond["liveness"][-1]["step"] = 1_099_999
    for bad in (dup, gap, extra, no0, unsorted, beyond):
        assert not _ok(bad)


def test_flag_is_recomputed_from_metrics():
    for k, v in (("block3_pos_frac", 0.0), ("live_h_frac", 0.005),
                 ("logit_std", 0.0), ("value_std", 1e-7),
                 ("value_std", float("nan")), ("logit_std", None)):
        r = _run()
        r["liveness"][4][k] = v  # recorded flag stays False
        assert not _ok(r), (k, v)


def test_curves_must_be_increasing_and_end_at_1m():
    r = _run()
    r["coin_rate_curve"] = [[40_000, 0.1], [40_000, 0.2], [M, 0.4]]
    assert not _ok(r)
    r = _run()
    r["success_curve"] = [[M, 0.1], [M, 0.1]]
    r["success_curve_n"] = [100, 100]
    assert not _ok(r)
    r = _run()
    r["success_curve_n"] = [100]  # length differs from the curve
    assert not _ok(r)


def test_combined_verdict_needs_both_arms(tmp_path):
    d = tmp_path / "fix-c-coin-pretrain-gtm"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(_run()))
    ok, why = combined_verdict(tmp_path)
    assert not ok and "eng: missing" in why[0]
    e = tmp_path / "fix-c-coin-pretrain-eng"
    e.mkdir()
    (e / "run.json").write_text(json.dumps(_run(final=0.01, arm="eng")))
    assert not combined_verdict(tmp_path)[0]
    (e / "run.json").write_text(json.dumps(_run(arm="gtm")))  # wrong arm
    assert not combined_verdict(tmp_path)[0]
    (e / "run.json").write_text(json.dumps(_run(arm="eng")))
    assert combined_verdict(tmp_path) == (True, [])
    assert not combined_verdict(tmp_path, seed=9102)[0]
    txt = report(tmp_path, "kxkm")
    assert "| kxkm | fix-c-coin-pretrain-gtm" in txt and "Combined gate" in txt
    assert "FAIL" not in txt


def test_committed_gate_cells_match_index_and_amendment():
    idx, bad = load_index(CELLS / "index.json")
    assert bad == []
    rows = table_rows(CELLS / "index.json")
    assert len(rows) == len(idx["runs"])
    text = AMENDMENT.read_text()
    assert TABLE_HEADER in text
    for row in rows:
        assert row in text, row


def test_index_detects_tampering(tmp_path):
    import shutil
    dst = tmp_path / "cells"
    shutil.copytree(CELLS, dst)
    e = json.loads((dst / "index.json").read_text())["runs"][0]
    f = dst / e["path"]
    f.write_text(f.read_text().replace('"arm": "', '"arm": "x', 1))
    assert load_index(dst / "index.json")[1]
    with pytest.raises(ValueError):
        table_rows(dst / "index.json")


def test_committed_records_are_ledgered_and_within_the_caps():
    from experiments.procgen_multiagent_gtm_comm.harness import devledger
    led = {r["id"]: r for r in devledger.read()["runs"]}
    idx = json.loads((CELLS / "index.json").read_text())
    for e in idx["runs"] + idx["pretrain"] + idx["adam_study"]:
        r = led[e["ledger_id"]]
        assert r["outcome"] == "completed" and r["seed"] == e["seed"]
        assert r["arm"] == e.get("arm", "trunk")
        assert e.get("host", "KXKM-AI") in r["note"]
    by = devledger.counts_by_kind()
    for arm, n in by.get("gate_dev", {}).items():
        assert n <= devledger.gate_allowance("gate_dev", arm)
    assert sum(by.get("pretrain_dev", {}).values()) <= (
        devledger.gate_allowance("pretrain_dev", "trunk"))


def test_committed_adam_rows_match_amendment():
    from experiments.procgen_multiagent_gtm_comm.harness.gate_report import (
        ADAM_HEADER,
        adam_rows,
    )
    rows, summary = adam_rows(CELLS / "index.json")
    assert len(rows) == 6  # 3 seeds x 2 arms, C vs C-bc
    text = AMENDMENT.read_text()
    for line in ADAM_HEADER.splitlines():
        assert "  " + line in text
    for row in rows:
        assert "  " + row in text, row
    assert summary in text


def test_bis_gate_runs_parse_and_have_their_allowance():
    from experiments.procgen_multiagent_gtm_comm.harness import devledger
    from experiments.procgen_multiagent_gtm_comm.harness.gate_report import (
        parse_run_dir,
    )
    for arm in ("gtm_mis", "gtm_warm", "eng_gtmrx", "vq"):
        assert parse_run_dir(f"fix-c-coin-pretrain-{arm}") == (
            "fix-c-coin-pretrain", arm, "")
        assert devledger.gate_allowance("gate_dev", arm) == 3  # PI 2026-10-26
        assert verdict(_run(arm=arm), arm=arm)[0]
    assert parse_run_dir("fix-c-coin-pretrain-bc-eng") == (
        "fix-c-coin-pretrain-bc", "eng", "")
    assert parse_run_dir("fix-c-coin-pretrain-gtm-macm1-r2") == (
        "fix-c-coin-pretrain", "gtm", "-macm1-r2")
