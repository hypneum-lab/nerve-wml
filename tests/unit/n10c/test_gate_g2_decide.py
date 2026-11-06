"""N10-C-bis gate A+ decision (harness/gate_g2_decide.py) on synthetic
records; numpy-free except the host-settings parity test."""
import json
import os
import re
import stat

import pytest

from experiments.procgen_multiagent_gtm_comm.harness import gate_g2_decide as g

M = 1_000_000
SHA = {9104: g.TRUNK_SHA[9104], 9105: g.TRUNK_SHA[9105], 9106: "a" * 64}
TRUNK_ARGS = ["--trunk-sha", f"9106={SHA[9106]}"]


def _rec(arm, seed, trunk, *, success=0.5, first=0.01, coin=(0.1, 0.5),
         abort=False, flagged=False, settings=None, steps=10 * M, last=None):
    host = g.SEED_HOST[trunk]
    mbp, cc, nw = settings or g.HOST_SETTINGS[host]
    live = {"block3_pos_frac": 0.3, "live_h_frac": 0.6, "logit_std": 0.1,
            "value_std": 0.1, "flagged": False}
    probes = [live | {"step": 0, "flagged": True}]  # step 0 is ignored
    probes += [live | {"step": min(steps, -(-100_000 * k // 8192) * 8192)}
               for k in range(1, steps // 100_000 + 1)]
    if flagged:
        probes[40]["flagged"] = True
    log_steps = list(range(200_000, steps + 1, 200_000))
    vals = [first] + [success] * (len(log_steps) - 1)
    if last is not None:
        vals[-1] = last
    return {
        "arm": arm, "seed": seed, "condition": "closed",
        "train_options": {"preset": "fix-c-coin-pretrain", "liveness": "abort",
                          "liveness_every": 100_000},
        "total_pair_steps": 10 * M, "pair_steps_done": steps,
        "using_stub": False, "training_failure": abort, "liveness_abort": abort,
        "init_trunk_sha256": SHA[trunk],
        "micro_batch_pairs": mbp, "clear_cache": cc, "n_workers": nw,
        "success_curve": [[s, v] for s, v in zip(log_steps, vals, strict=True)],
        "success_curve_n": [37] + [100] * (len(log_steps) - 1),
        "coin_rate_curve": [[200_000, coin[0]], [steps, coin[1]]],
        "liveness": probes,
    }


def _write(root, name, rec):
    d = root / name
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(rec))
    return d


def _seed_cells(root, s, eng=0.8, nocomm=0.5, rep=None, **kw):
    rep = nocomm + 0.02 if rep is None else rep
    _write(root, g.cell_dir(s, "eng", s, 1), _rec("eng", s, s, success=eng, **kw))
    _write(root, g.cell_dir(s, "nocomm", s, 1),
           _rec("nocomm", s, s, success=nocomm))
    _write(root, g.cell_dir(s + 10, "nocomm", s, 1),
           _rec("nocomm", s + 10, s, success=rep))


def _decide(*roots):
    return g.decide(list(roots), trunk_sha={9106: SHA[9106]})


@pytest.fixture
def two_hosts(tmp_path):
    kx, m1 = tmp_path / "kxkm", tmp_path / "macm1"
    kx.mkdir()
    m1.mkdir()
    return kx, m1


def test_pass(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104)
    _seed_cells(m1, 9105)
    _seed_cells(kx, 9106)
    cat, sealed = _decide(kx, m1)
    assert cat == "PASS"
    assert sealed["D_bar"] == pytest.approx(0.3)
    assert sealed["per_seed"]["9105"]["host"] == "macm1"


def test_threshold_is_inclusive_despite_float_error(two_hosts):
    kx, m1 = two_hosts
    for s, root in ((9104, kx), (9105, m1), (9106, kx)):
        _seed_cells(root, s, eng=0.7, nocomm=0.5)
    assert 0.7 - 0.5 < 0.2  # float: the tolerance is needed
    assert _decide(kx, m1)[0] == "PASS"


def test_fail(two_hosts):
    kx, m1 = two_hosts
    for s, root in ((9104, kx), (9105, m1), (9106, kx)):
        _seed_cells(root, s, eng=0.6, nocomm=0.5)
    cat, sealed = _decide(kx, m1)
    assert cat == "FAIL" and sealed["D_bar"] == pytest.approx(0.1)


def test_not_measurable_learning_after_one_rerun(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104)
    _seed_cells(m1, 9105)
    _seed_cells(kx, 9106)
    # nocomm of 9105: attempt 1 aborted on liveness, attempt 2 did not learn
    d = m1 / g.cell_dir(9105, "nocomm", 9105, 1)
    (d / "run.json").write_text(json.dumps(_rec("nocomm", 9105, 9105, abort=True)))
    _write(m1, g.cell_dir(9105, "nocomm", 9105, 2),
           _rec("nocomm", 9105, 9105, coin=(0.3, 0.3)))
    cat, sealed = _decide(kx, m1)
    assert cat == "NOT_MEASURABLE_LEARNING"
    cell = sealed["per_seed"]["9105"]["cells"]["nocomm"]
    assert cell["chosen"] is None
    assert "liveness_abort" in cell["attempts"][1]["status"]
    assert cell["attempts"][2]["status"] == "FAIL:coin_rate_rise"


def test_rerun_is_used_when_attempt_1_failed(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104)
    _seed_cells(m1, 9105)
    _seed_cells(kx, 9106)
    d = kx / g.cell_dir(9106, "eng", 9106, 1)
    (d / "run.json").unlink()  # killed run: no record
    _write(kx, g.cell_dir(9106, "eng", 9106, 2), _rec("eng", 9106, 9106, success=0.8))
    cat, sealed = _decide(kx, m1)
    assert cat == "PASS"
    assert sealed["per_seed"]["9106"]["cells"]["eng"]["chosen"] == 2


def test_eng_control_on_two_seeds_is_not_measurable(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104, eng=0.04, nocomm=0.02)  # below 0.05
    _seed_cells(m1, 9105, eng=0.30, nocomm=0.02, first=0.30, last=0.30)  # flat
    _seed_cells(kx, 9106, eng=0.8, nocomm=0.5)
    cat, sealed = _decide(kx, m1)
    assert cat == "NOT_MEASURABLE_ENG_CONTROL"
    assert sealed["eng_control_failed"] == [9104, 9105]


def test_eng_control_on_one_seed_goes_on(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104, eng=0.04, nocomm=0.02)
    _seed_cells(m1, 9105, eng=0.9, nocomm=0.5)
    _seed_cells(kx, 9106, eng=0.9, nocomm=0.5)
    assert _decide(kx, m1)[0] in ("PASS", "FAIL")


def test_inconclusive_noise(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104, rep=0.62)  # |D0| = 0.12
    _seed_cells(m1, 9105, rep=0.40)  # |D0| = 0.10, inclusive
    _seed_cells(kx, 9106)
    cat, sealed = _decide(kx, m1)
    assert cat == "INCONCLUSIVE_NOISE"
    assert sealed["noise_control_failed"] == [9104, 9105]


def test_missing_rep_counts_against_the_noise_control(two_hosts):
    kx, m1 = two_hosts
    for s, root in ((9104, kx), (9105, m1), (9106, kx)):
        _seed_cells(root, s)
    for s, root in ((9104, kx), (9105, m1)):
        (root / g.cell_dir(s + 10, "nocomm", s, 1) / "run.json").unlink()
    assert _decide(kx, m1)[0] == "INCONCLUSIVE_NOISE"


def test_rule_order_learning_first(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104, rep=0.9)
    _seed_cells(m1, 9105, rep=0.9)
    _seed_cells(kx, 9106)
    (kx / g.cell_dir(9106, "eng", 9106, 1) / "run.json").unlink()
    assert _decide(kx, m1)[0] == "NOT_MEASURABLE_LEARNING"


@pytest.mark.parametrize("change, name", [
    (dict(settings=(None, False, 1)), "settings"),
    (dict(abort=True), "liveness_abort"),
    (dict(flagged=True), "liveness_flagged"),
    (dict(steps=9_800_000), "steps"),
])
def test_check_cell_names_the_failed_condition(tmp_path, change, name):
    d = _write(tmp_path, "c", _rec("eng", 9104, 9104, **change))
    st = g.check_cell(d, arm="eng", seed=9104, trunk_sha=SHA[9104], host="kxkm")
    assert st.startswith("FAIL:") and name in st.split(":")[1].split(",")
    assert not re.search(r"\d\.\d", st)  # never a value


def test_check_cell_identity_and_trunk(tmp_path):
    d = _write(tmp_path, "c", _rec("nocomm", 9114, 9104))
    ok = dict(arm="nocomm", seed=9114, trunk_sha=SHA[9104], host="kxkm")
    assert g.check_cell(d, **ok) == "OK"
    assert g.check_cell(d, **(ok | {"seed": 9104})) == "FAIL:seed"
    assert g.check_cell(d, **(ok | {"trunk_sha": "b" * 64})) == "FAIL:trunk"
    assert g.check_cell(tmp_path / "none", **ok) == "FAIL:no_run_json"


def test_duplicate_cell_across_hosts_is_refused(two_hosts):
    kx, m1 = two_hosts
    _seed_cells(kx, 9104)
    _seed_cells(m1, 9104)
    with pytest.raises(g.DecideError, match="twice"):
        _decide(kx, m1)


def test_third_attempt_is_refused(two_hosts):
    kx, m1 = two_hosts
    for s, root in ((9104, kx), (9105, m1), (9106, kx)):
        _seed_cells(root, s)
    _write(kx, g.cell_dir(9104, "eng", 9104, 3), _rec("eng", 9104, 9104))
    with pytest.raises(g.DecideError, match="attempts"):
        _decide(kx, m1)


def test_trunk_sha_of_9106_is_required(two_hosts):
    kx, _ = two_hosts
    with pytest.raises(g.DecideError, match="9106"):
        g.decide([kx])


def test_cli_prints_only_the_category_and_seals_values(two_hosts, tmp_path,
                                                       capsys):
    kx, m1 = two_hosts
    for s, root in ((9104, kx), (9105, m1), (9106, kx)):
        _seed_cells(root, s)
    out = tmp_path / "sealed" / "g2.json"
    argv = ["decide", "--runs-root", str(kx), "--runs-root", str(m1),
            "--sealed-out", str(out), *TRUNK_ARGS]
    assert g.main(argv) == 0
    printed = capsys.readouterr()
    assert printed.out == "PASS\n" and printed.err == ""
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o400
    sealed = json.loads(out.read_text())
    assert sealed["category"] == "PASS"
    assert sealed["per_seed"]["9104"]["D"] == pytest.approx(0.3)
    # a second decision never overwrites the sealed file
    before = out.read_bytes()
    assert g.main(argv) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and "exists" in printed.err
    assert out.read_bytes() == before


def test_cli_check_cell(tmp_path, capsys):
    d = _write(tmp_path, g.cell_dir(9105, "eng", 9105, 1), _rec("eng", 9105, 9105))
    argv = ["check-cell", str(d), "--arm", "eng", "--seed", "9105",
            "--trunk-seed", "9105", "--host", "macm1"]
    assert g.main(argv) == 0 and capsys.readouterr().out == "OK\n"
    assert g.main([*argv[:-1], "kxkm"]) == 1
    assert capsys.readouterr().out == "FAIL:settings\n"
    assert g.main(["check-cell", str(d), "--arm", "eng", "--seed", "9106",
                   "--trunk-seed", "9106", "--host", "kxkm"]) == 2


def test_constants_match_the_harness():
    from experiments.procgen_multiagent_gtm_comm.harness import codes
    for host, (mbp, cc, nw) in g.HOST_SETTINGS.items():
        assert codes.HOST_SETTINGS[host] == {
            "micro_batch_pairs": mbp, "clear_cache": cc, "n_workers": nw}
    assert g.LAST_STEPS == [9_200_000, 9_400_000, 9_600_000, 9_800_000,
                            10_000_000]
    assert g.ENG_MIN_SUCCESS == 0.05
    from experiments.procgen_multiagent_gtm_comm.harness import devledger
    assert devledger.GATE_MAX_STEPS["gate_dev"] == g.STEPS
