"""N10-C-bis registry: arm codes, the 48-run queue (scope option bis-A),
registration-aware guards and the campaign runner's warm-start dependency.
The N10-C defaults are asserted unchanged."""
import json
from types import SimpleNamespace

import pytest

from experiments.procgen_multiagent_gtm_comm.campaign import guards
from experiments.procgen_multiagent_gtm_comm.campaign import queue as q
from experiments.procgen_multiagent_gtm_comm.campaign import run_campaign as rc
from experiments.procgen_multiagent_gtm_comm.harness.codes import (
    ARM_CODES,
    BIS_ARMS,
    HOST_SETTINGS,
    N10C_ARMS,
    SEED_HOST,
    SEED_HOST_BIS,
    SEEDS,
)

TOTAL = 64
MAC = {"system": "Darwin", "machine": "arm64", "backend": "metal"}
KX = {"system": "Linux", "machine": "x86_64", "backend": "cuda"}
KX_WHEEL = {"version": "0.10.7", "tags": [
    "cp310-cp310-manylinux_2_17_x86_64", "cp310-cp310-manylinux2014_x86_64"]}
BIS_TAG = "n10c-bis-freeze-20261106"


def _runtime(monkeypatch, rt, wheel=None):
    monkeypatch.setattr(guards, "_runtime_probe", lambda: rt)
    monkeypatch.setattr(guards, "_procgen_probe", lambda: wheel or KX_WHEEL)


# -- registry -------------------------------------------------------------------
def test_bis_codes_hosts_and_arm_sets():
    assert N10C_ARMS == ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
    assert BIS_ARMS == ("gtm_mis", "eng_gtmrx", "gtm_warm", "vq")
    assert [ARM_CODES[a] for a in N10C_ARMS] == [0, 1, 2, 3, 4, 5]
    assert [ARM_CODES[a] for a in BIS_ARMS] == [6, 7, 8, 9]
    assert SEED_HOST_BIS == {s: "kxkm" for s in SEEDS}
    assert SEED_HOST == {0: "macm1", 17: "macm1", 42: "macm1",
                         73: "kxkm", 101: "kxkm", 131: "kxkm"}


def test_bis_queue_is_48_runs_8_cells_all_on_kxkm_seed_major():
    runs = q.build_queue("n10c-bis")
    assert runs == q.build_queue_bis()
    assert len(runs) == q.N_RUNS_BIS == 48
    assert len({r.run_id for r in runs}) == 48
    cells = {(r.arm, r.condition) for r in runs}
    assert cells == {(a, c) for a in BIS_ARMS for c in ("registered", "closed")}
    assert q.cross_cells("n10c-bis") == sorted(cells, key=[
        (r.arm, r.condition) for r in runs].index)
    assert len(q.cross_cells("n10c-bis")) == 8
    assert all(r.host == "kxkm" and r.host_fixed == "kxkm" for r in runs)
    for s in SEEDS:
        assert sum(r.seed == s for r in runs) == 8
    # seed-major blocks in SEEDS order; closed run directly after registered
    assert [runs[8 * i].seed for i in range(6)] == list(SEEDS)
    for i in range(0, 48, 2):
        assert runs[i].arm == runs[i + 1].arm and runs[i].seed == runs[i + 1].seed
        assert (runs[i].condition, runs[i + 1].condition) == ("registered", "closed")
    assert len({runs[8 * i].arm for i in range(6)}) > 1  # rotation
    assert q.runs_for_host("kxkm", "n10c-bis") == runs
    assert q.runs_for_host("macm1", "n10c-bis") == []
    with pytest.raises(ValueError):
        q.build_queue("n10d")


def test_bis_queue_warm_dependency():
    runs = q.build_queue_bis()
    ids = [r.run_id for r in runs]
    for r in runs:
        if r.arm == "gtm_warm":
            assert r.warm_from == f"{r.condition}-eng_gtmrx-s{r.seed}"
            src = r.warm_source()
            assert src.arm == "eng_gtmrx" and (src.seed, src.condition) == (r.seed, r.condition)
            assert src in runs and ids.index(src.run_id) < ids.index(r.run_id)
            assert q.new_state(r)["warm_from"] == r.warm_from
        else:
            assert r.warm_from is None and r.warm_source() is None
            assert "warm_from" not in q.new_state(r)


def test_n10c_queue_and_guards_defaults_are_unchanged():
    runs = q.build_queue()
    assert runs == q.build_queue("n10c") and len(runs) == 60
    assert all(r.host_fixed is None and r.warm_from is None for r in runs)
    assert all(r.host == SEED_HOST[r.seed] for r in runs)
    assert runs[0] == q.RunSpec("gtm", 0, "registered")
    assert len(q.cross_cells()) == 10
    assert q.runs_for_host("macm1") == [r for r in runs if r.seed in (0, 17, 42)]
    assert "warm_from" not in q.new_state(runs[0])
    n = guards.registration()
    assert n.name == "n10c" and n.tag_prefix == "n10c-freeze-"
    assert n.amendment_path == guards.AMENDMENT_PATH
    assert n.manifest_name == "stage2-manifest.md"
    assert n.arms_by_condition == guards.ARMS_BY_CONDITION and n.seed_host == SEED_HOST
    assert guards.make_context_for_tests().registration == "n10c"
    assert guards.make_context_for_tests().pins()["amendment_path"] == guards.AMENDMENT_PATH


# -- guards -----------------------------------------------------------------------
def test_bis_registration_record_and_tag_names():
    b = guards.registration("n10c-bis")
    assert b.tag_prefix == "n10c-bis-freeze-"
    assert b.amendment_path == "docs/milestones/2026-10-25-n10c-bis-control-arms.md"
    assert b.manifest_name == "stage2-manifest-bis.md"
    assert b.arms_by_condition == {"registered": BIS_ARMS, "closed": BIS_ARMS}
    assert b.seed_host == SEED_HOST_BIS
    with pytest.raises(guards.GuardError):
        guards.registration("n10d")
    assert guards.valid_tag_name(BIS_TAG, "n10c-bis")
    assert not guards.valid_tag_name("n10c-freeze-20261106", "n10c-bis")
    assert not guards.valid_tag_name(BIS_TAG)  # default registration: N10-C
    assert guards.valid_tag_name("n10c-freeze-20261024")
    assert not guards.valid_tag_name("n10c-bis-freeze-20261399", "n10c-bis")


def test_registered_run_problems_per_registration():
    base = dict(seed=0, condition="registered", total_pair_steps=50_000_000,
                config=dict(guards.REGISTERED_TRAIN_CONFIG))
    kx, mac = dict(HOST_SETTINGS["kxkm"]), dict(HOST_SETTINGS["macm1"])
    # seed 0 is a KXKM seed in N10-C-bis: KXKM settings required
    assert guards.registered_run_problems(arm="gtm_mis", settings=kx,
                                          registration="n10c-bis", **base) == []
    bad = guards.registered_run_problems(arm="gtm_mis", settings=mac,
                                         registration="n10c-bis", **base)
    assert any("per-host settings" in e and "kxkm" in e for e in bad)
    for arm in BIS_ARMS:  # both conditions for every control arm
        for cond in ("registered", "closed"):
            assert guards.registered_run_problems(
                arm=arm, settings=kx, registration="n10c-bis",
                **{**base, "condition": cond}) == []
    # an N10-C arm is not an N10-C-bis cell and vice versa
    assert any("not run" in e for e in guards.registered_run_problems(
        arm="gtm", settings=kx, registration="n10c-bis", **base))
    assert any("not run" in e for e in guards.registered_run_problems(
        arm="gtm_mis", settings=mac, **base))
    assert guards.registered_run_problems(arm="gtm", settings=mac, **base) == []
    assert guards.registered_run_problems(
        arm="gtm", settings=mac, registration="n10d", **base) == ["unknown registration 'n10d'"]


def test_context_registration_drives_the_execution_boundary(monkeypatch):
    pytest.importorskip("mlx.core")  # evaluate imports mlx
    from experiments.procgen_multiagent_gtm_comm.harness.evaluate import HELD_OUT

    ctx_bis = guards.make_context_for_tests(registration="n10c-bis", tag=BIS_TAG)
    ctx_n10c = guards.make_context_for_tests()
    kw = dict(seed=0, condition="registered", total_pair_steps=50_000_000,
              settings=dict(HOST_SETTINGS["kxkm"]),
              config=dict(guards.REGISTERED_TRAIN_CONFIG), has_env_factory=False)
    _runtime(monkeypatch, KX)
    assert guards.check_train_boundary(ctx_bis, arm="gtm_mis", **kw) == "kxkm"
    assert guards.check_train_boundary(ctx_bis, arm="vq", **{**kw, "condition": "closed"}) == "kxkm"
    with pytest.raises(guards.GuardError, match="not run"):
        guards.check_train_boundary(ctx_n10c, arm="gtm_mis", **kw)
    with pytest.raises(guards.GuardError, match="not run"):
        guards.check_train_boundary(ctx_bis, arm="gtm", **kw)
    _runtime(monkeypatch, MAC)  # every N10-C-bis seed is a KXKM seed
    with pytest.raises(guards.GuardError, match="runtime"):
        guards.check_train_boundary(ctx_bis, arm="gtm_mis", **kw)
    _runtime(monkeypatch, KX)
    assert guards.check_eval_boundary(ctx_bis, HELD_OUT, arm="vq", condition="closed",
                                      seeds=(0,)) is True
    with pytest.raises(guards.GuardError, match="registered cell"):
        guards.check_eval_boundary(ctx_n10c, HELD_OUT, arm="vq", condition="closed",
                                   seeds=(0,))
    assert ctx_bis.pins()["registration"] == "n10c-bis"
    assert ctx_bis.pins()["amendment_path"] == guards.ADDENDUM_PATH
    with pytest.raises(guards.GuardError):
        guards.make_context_for_tests(registration="n10d")


def test_freeze_problems_use_the_bis_tag_prefix_and_addendum(tmp_path):
    bad = guards.guard_problems(repo=tmp_path, freeze_tag="n10c-freeze-20261106",
                                swhid="swh:1:rev:" + "0" * 40, registration="n10c-bis")
    assert any("n10c-bis-freeze-YYYYMMDD" in b for b in bad)
    assert not any("n10c-freeze-YYYYMMDD" in b for b in bad)


# -- campaign runner -----------------------------------------------------------------
def _rec(**kw):
    base = dict(training_failure=False, pair_steps_done=TOTAL,
                last_checkpoint=f"ckpt_{TOTAL}.safetensors", wall_time_s=1.0,
                failure_reason=None)
    base.update(kw)
    return SimpleNamespace(**base)


class _Eval:
    S, R_intact = 0.1, 0.5

    def to_json(self):
        return json.dumps({"S": self.S})


class _Cross:
    def __init__(self, ckpt0, ckpt1, arm, cond, seed0, seed1):
        self.R_cross = 0.25
        self.d = {"arm": arm, "condition": cond, "seed_role0": seed0,
                  "seed_role1": seed1, "R_cross": 0.25,
                  "untrained": ckpt0 is None or ckpt1 is None}

    def to_json(self):
        return json.dumps(self.d)


def _bis_campaign(tmp_path, train, **o):
    calls = {"train": [], "eval": [], "cross": []}

    def t(**kw):
        calls["train"].append(kw)
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        rec = train(kw)
        if rec.last_checkpoint:
            (kw["out_dir"] / rec.last_checkpoint).write_text(kw["arm"] + str(kw["seed"]))
        return rec

    def e(ckpt, arm, cond, seed, env_factory=None):
        calls["eval"].append((ckpt, arm, cond, seed))
        return _Eval()

    def x(ckpt0, ckpt1, arm, cond, seed0, seed1, env_factory=None):
        calls["cross"].append((arm, cond, seed0, seed1))
        return _Cross(ckpt0, ckpt1, arm, cond, seed0, seed1)

    o.setdefault("registration", guards.make_context_for_tests(
        registration="n10c-bis", tag=BIS_TAG))
    o.setdefault("registration_name", "n10c-bis")
    c = rc.Campaign(rc.Options(campaign_dir=tmp_path / "c", host="kxkm",
                               total_pair_steps=TOTAL, min_free_gb=8, **o),
                    train_fn=t, eval_fn=e, cross_fn=x, gpu_fn=lambda: None,
                    archive_fn=None, log=lambda *_: None)
    (tmp_path / "c").mkdir(exist_ok=True)
    return c, calls


def test_bis_campaign_runs_the_48_runs_warm_starting_from_the_source(tmp_path):
    c, calls = _bis_campaign(tmp_path, lambda kw: _rec())
    assert c.run() == rc.EXIT_OK  # no macM1 runs to await, 8 cells complete
    ids = [f"{k['condition']}-{k['arm']}-s{k['seed']}" for k in calls["train"]]
    assert ids == [r.run_id for r in q.build_queue_bis()]
    assert len(calls["eval"]) == 48 and len(calls["cross"]) == 8 * 6
    for kw in calls["train"]:
        spec = q.RunSpec(kw["arm"], kw["seed"], kw["condition"], host_fixed="kxkm")
        if kw["arm"] == "gtm_warm":
            src = q.RunSpec("eng_gtmrx", kw["seed"], kw["condition"], host_fixed="kxkm")
            assert kw["warm_start_from"] == c.run_dir(src) / f"ckpt_{TOTAL}.safetensors"
            st = c.store.load(spec)
            assert st["warm_source"]["run_id"] == src.run_id
            assert st["warm_source"]["checkpoint"] == f"ckpt_{TOTAL}.safetensors"
            assert st["warm_source"]["untrained"] is False
            assert st["warm_source"]["sha256"] == c.store.load(src)["last_checkpoint_sha256"]
        else:
            assert "warm_start_from" not in kw
    pins = json.loads((c.dir / "campaign.json").read_text())
    assert pins["registration"] == "n10c-bis" and pins["host"] == "kxkm"


def test_bis_campaign_failed_source_gives_last_finite_checkpoint_or_init(tmp_path):
    def train(kw):
        if (kw["arm"], kw["seed"]) == ("eng_gtmrx", 0):
            if kw["condition"] == "registered":  # no finite checkpoint at all
                return _rec(training_failure=True, last_checkpoint=None,
                            failure_reason="nan", pair_steps_done=8)
            return _rec(training_failure=True, last_checkpoint="ckpt_8.safetensors",
                        failure_reason="nan", pair_steps_done=16)
        return _rec()

    c, calls = _bis_campaign(tmp_path, train)
    assert c.run() == rc.EXIT_OK
    warm = {k["condition"]: k for k in calls["train"]
            if (k["arm"], k["seed"]) == ("gtm_warm", 0)}
    assert warm["registered"]["warm_start_from"] is None  # initialisation, flagged
    st = c.store.load(q.RunSpec("gtm_warm", 0, "registered", host_fixed="kxkm"))
    assert st["warm_source"]["untrained"] is True
    assert st["warm_source"]["status"] == q.TRAINING_FAILURE
    src = q.RunSpec("eng_gtmrx", 0, "closed", host_fixed="kxkm")
    assert warm["closed"]["warm_start_from"] == c.run_dir(src) / "ckpt_8.safetensors"
    st2 = c.store.load(q.RunSpec("gtm_warm", 0, "closed", host_fixed="kxkm"))
    assert st2["warm_source"]["untrained"] is False
    assert st2["warm_source"]["checkpoint"] == "ckpt_8.safetensors"


def test_bis_campaign_refuses_an_unfinished_warm_source(tmp_path):
    c, _ = _bis_campaign(tmp_path, lambda kw: _rec())
    spec = next(r for r in q.build_queue_bis() if r.arm == "gtm_warm")
    with pytest.raises(rc.GuardError, match="has not finished"):
        c.warm_source(spec)  # the source has not even started


def test_campaign_registration_consistency(tmp_path):
    with pytest.raises(rc.GuardError, match="registration context"):
        rc.Campaign(rc.Options(campaign_dir=tmp_path / "c", host="kxkm",
                               registration=guards.make_context_for_tests(),
                               registration_name="n10c-bis"))
    with pytest.raises(rc.GuardError, match="unknown registration"):
        rc.Campaign(rc.Options(campaign_dir=tmp_path / "c", host="kxkm", trial=True,
                               total_pair_steps=TOTAL, registration_name="n10d"))
    c = rc.Campaign(rc.Options(campaign_dir=tmp_path / "c", host="kxkm",
                               registration=guards.make_context_for_tests()))
    assert c.reg_name == "n10c" and c.queue_all() == q.build_queue()
    assert c.queue_host() == q.runs_for_host("kxkm") and c.cells() == q.cross_cells()


def test_dry_run_lists_the_bis_queue(tmp_path, capsys):
    code = rc.main(["--campaign-dir", str(tmp_path / "c"), "--dry-run",
                    "--registration", "n10c-bis", "--host", "kxkm"])
    out = capsys.readouterr().out
    assert code == rc.EXIT_OK
    assert "DRY RUN (n10c-bis)" in out and "48 runs" in out
    assert "registered-gtm_mis-s0" in out and "closed-vq-s131" in out
    code = rc.main(["--campaign-dir", str(tmp_path / "c"), "--dry-run", "--host", "kxkm"])
    out = capsys.readouterr().out
    assert code == rc.EXIT_OK and "DRY RUN (n10c)" in out and "30 runs" in out


def test_estimate_covers_the_bis_queue():
    thr = {"results": {a: {"pair_steps_per_s": 1000.0,
                           "eval_proxy": {"seconds_per_model_2x200": 0.0}}
                       for a in ("gtm", "nocomm")}}
    e = rc.estimate(thr, 3_600_000, "kxkm", "n10c-bis")
    assert len(e["per_run_hours"]) == 48 and e["total_hours"] == pytest.approx(48.0)
    assert len(rc.estimate(thr, 3_600_000, "kxkm")["per_run_hours"]) == 30
