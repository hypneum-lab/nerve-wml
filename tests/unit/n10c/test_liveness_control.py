"""Amendment 2026-10-25: liveness-abort restart rule and the declared
per-run-pretraining control in the campaign runner (fake train / eval
functions; no mlx, no procgen)."""
import json
from types import SimpleNamespace

import pytest

from experiments.procgen_multiagent_gtm_comm.campaign import guards
from experiments.procgen_multiagent_gtm_comm.campaign import queue as q
from experiments.procgen_multiagent_gtm_comm.campaign import run_campaign as rc

TOTAL = 64
KX = q.runs_for_host("kxkm")
FIRST = KX[0]


class _Eval:
    S, R_intact = 0.1, 0.5

    def to_json(self):
        return json.dumps({"S": self.S})


def _rec(abort=False, **kw):
    base = dict(training_failure=abort, liveness_abort=abort,
                pair_steps_done=TOTAL if not abort else TOTAL // 2,
                last_checkpoint=(f"ckpt_{TOTAL // 2}.safetensors" if abort
                                 else f"ckpt_{TOTAL}.safetensors"),
                wall_time_s=1.0,
                failure_reason="trunk liveness guard at update 3" if abort
                else None)
    base.update(kw)
    return SimpleNamespace(**base)


def _campaign(tmp_path, script, preset=None, monkeypatch=None):
    """``script(spec_tag, n_for_that_run, kw)`` returns a record or raises."""
    if preset is not None:
        monkeypatch.setattr(guards, "REGISTERED_TRAIN_OPTIONS", preset)
    calls: list[dict] = []
    count: dict[str, int] = {}

    def t(**kw):
        tag = f"{kw['condition']}-{kw['arm']}-s{kw['seed']}"
        if kw.get("per_run_trunk"):
            tag = q.CONTROL_PREFIX + tag
        count[tag] = count.get(tag, 0) + 1
        calls.append({**kw, "tag": tag})
        kw["out_dir"].mkdir(parents=True, exist_ok=True)
        rec = script(tag, count[tag], kw)
        if rec.last_checkpoint:
            (kw["out_dir"] / rec.last_checkpoint).write_text(f"{tag}-{count[tag]}")
        return rec

    evals = []

    def e(ckpt, arm, cond, seed, env_factory=None):
        evals.append(ckpt)
        return _Eval()

    c = rc.Campaign(
        rc.Options(campaign_dir=tmp_path / "c", host="kxkm",
                   total_pair_steps=TOTAL, min_free_gb=8,
                   registration=guards.make_context_for_tests()),
        train_fn=t, eval_fn=e, cross_fn=lambda *a, **k: None,
        gpu_fn=lambda: None, archive_fn=None, log=lambda *_: None)
    (tmp_path / "c").mkdir(exist_ok=True)
    return c, calls, evals


def _abort_first_once(tag, n, kw):
    return _rec(abort=(tag == FIRST.run_id and n == 1))


# -- liveness-abort rule -------------------------------------------------------
def test_first_liveness_abort_restarts_from_scratch_once(tmp_path):
    c, calls, evals = _campaign(tmp_path, _abort_first_once)
    c.pin({})
    c._process(FIRST, can_eval=True)
    st = c.store.load(FIRST)
    assert st["status"] == q.DONE
    assert st["liveness_restarts"] == 1 and st["infra_failures"] == 0
    assert [a["outcome"] for a in st["attempts"]] == ["liveness_abort", "trained"]
    assert [x["tag"] for x in calls] == [FIRST.run_id] * 2
    aside = c.dir / "runs" / f"{FIRST.run_id}.liveness-abort-1"
    assert (aside / f"ckpt_{TOTAL // 2}.safetensors").exists()  # kept, unused
    # from scratch: the second attempt starts in an empty run directory and
    # only its own final checkpoint is evaluated
    assert st["last_checkpoint"] == f"ckpt_{TOTAL}.safetensors"
    assert len(evals) == 1 and evals[0].read_text() == f"{FIRST.run_id}-2"
    assert calls[1]["restarts"] == 0  # infra counter untouched


def test_second_liveness_abort_is_a_final_training_failure(tmp_path):
    c, calls, evals = _campaign(
        tmp_path, lambda tag, n, kw: _rec(abort=tag == FIRST.run_id))
    c.pin({})
    c._process(FIRST, can_eval=True)
    st = c.store.load(FIRST)
    assert st["status"] == q.TRAINING_FAILURE and st["eval_done"]  # not replaced
    assert st["liveness_abort"] is True and st["liveness_restarts"] == 1
    assert st["failure_reason"].startswith("liveness abort after 1 restart")
    assert [a["outcome"] for a in st["attempts"]] == ["liveness_abort",
                                                     "training_failure"]
    assert len(calls) == 2  # never a third attempt
    # evaluated at the last finite checkpoint of the restarted attempt
    assert st["last_checkpoint"] == f"ckpt_{TOTAL // 2}.safetensors"
    assert evals[0].read_text() == f"{FIRST.run_id}-2"


def test_liveness_and_infra_restarts_are_counted_separately(tmp_path):
    def script(tag, n, kw):
        if tag == FIRST.run_id and n == 1:
            raise OSError("disk")
        return _rec(abort=(tag == FIRST.run_id and n == 2))

    c, calls, _ = _campaign(tmp_path, script)
    c.pin({})
    c._process(FIRST, can_eval=True)
    st = c.store.load(FIRST)
    assert st["status"] == q.DONE
    assert (st["infra_failures"], st["liveness_restarts"]) == (1, 1)
    assert [a["outcome"] for a in st["attempts"]] == [
        "infra_failed", "liveness_abort", "trained"]
    assert (c.dir / "runs" / f"{FIRST.run_id}.failed-1").exists()
    assert (c.dir / "runs" / f"{FIRST.run_id}.liveness-abort-1").exists()


def test_pending_liveness_restart_resumes_without_counting_infra(tmp_path):
    c, calls, _ = _campaign(tmp_path, lambda tag, n, kw: _rec())
    c.pin({})
    st = c.store.load(FIRST)
    st.update(status=q.LIVENESS_RESTART, liveness_restarts=1)
    c.store.save(st)
    st = c.store.recover_stale(FIRST)
    assert st["status"] == q.LIVENESS_RESTART and q.needs_training(st)
    c._process(FIRST, can_eval=True)
    st = c.store.load(FIRST)
    assert st["status"] == q.DONE and st["infra_failures"] == 0


def test_old_status_files_without_the_counter_still_restart_once(tmp_path):
    c, calls, _ = _campaign(tmp_path, _abort_first_once)
    c.pin({})
    st = c.store.load(FIRST)
    del st["liveness_restarts"]
    c.store.save(st)
    c._process(FIRST, can_eval=True)
    assert c.store.load(FIRST)["liveness_restarts"] == 1


# -- per-run-pretraining control -------------------------------------------
def test_control_queue_is_seed_101_with_own_trunk_ids():
    ctl = q.control_queue()
    assert len(ctl) == q.N_CONTROL_RUNS == 10
    assert {r.seed for r in ctl} == {q.PER_RUN_TRUNK_SEED} == {101}
    assert {r.host for r in ctl} == {"kxkm"}
    assert all(r.trunk == q.PER_RUN and r.run_id.startswith("prc-") for r in ctl)
    reg101 = [r for r in q.build_queue() if r.seed == 101]
    assert [(r.arm, r.condition) for r in ctl] == [
        (r.arm, r.condition) for r in reg101]
    ids = {r.run_id for r in q.build_queue()}
    assert not ids & {r.run_id for r in ctl}  # never in the 60 estimand runs
    assert len(q.build_queue()) == 60  # frozen list unchanged
    assert q.control_queue("n10c-bis") == []
    assert q.control_queue(enabled=False) == []
    assert ctl[0].run_id == f"prc-{ctl[0].condition}-{ctl[0].arm}-s101"


def test_frozen_preset_has_no_control_runs(tmp_path):
    c, calls, _ = _campaign(tmp_path, lambda tag, n, kw: _rec())
    assert c.control_all() == []
    assert c.run() == rc.EXIT_AWAITING_SYNC
    assert len(calls) == 30 and not any(x.get("per_run_trunk") for x in calls)


def test_control_runs_follow_the_registered_kxkm_runs(tmp_path, monkeypatch):
    c, calls, evals = _campaign(tmp_path, lambda tag, n, kw: _rec(),
                                preset="fix-c-coin-pretrain",
                                monkeypatch=monkeypatch)
    assert c.run() == rc.EXIT_AWAITING_SYNC  # macM1 runs still to sync
    tags = [x["tag"] for x in calls]
    assert len(tags) == 40
    assert tags[:30] == [s.run_id for s in KX]
    assert tags[30:] == [s.run_id for s in q.control_queue()]
    for x in calls[:30]:
        assert x["init_trunk"] == guards.trunk_path(c.dir, x["seed"])
        assert "per_run_trunk" not in x
    for x in calls[30:]:
        assert x["per_run_trunk"] is True
        assert x["init_trunk"] == guards.control_trunk_path(
            c.dir, x["arm"], x["condition"])
    assert len({x["init_trunk"] for x in calls[30:]}) == 10
    for s in q.control_queue():
        assert c.store.load(s)["status"] == q.DONE
        assert c.store.load(s)["trunk"] == q.PER_RUN
    # the control is part of completion, not of the cross-play cells
    assert all(not r.run_id.startswith("prc-")
               for a, cnd in c.cells() for r in q.build_queue()
               if (r.arm, r.condition) == (a, cnd))


def test_control_boundary_uses_the_per_run_pinned_trunk(monkeypatch):
    ctx = guards.make_context_for_tests()
    monkeypatch.setattr(guards, "_runtime_probe", lambda: dict(zip(
        ("system", "machine", "backend"), guards.HOST_RUNTIME["kxkm"],
        strict=True)))
    monkeypatch.setattr(guards, "_procgen_probe", lambda: {})
    monkeypatch.setattr(guards, "REGISTERED_TRAIN_OPTIONS", "fix-c-coin-pretrain")
    monkeypatch.setitem(guards.REGISTERED_TRUNK_SHA256, 101, "aa" * 32)
    monkeypatch.setitem(guards.REGISTERED_CONTROL_TRUNK_SHA256,
                        ("gtm", "registered"), "bb" * 32)
    kw = dict(arm="gtm", condition="registered", total_pair_steps=50_000_000,
              settings=dict(guards.HOST_SETTINGS["kxkm"]),
              config=dict(guards.REGISTERED_TRAIN_CONFIG), has_env_factory=False,
              options_preset="fix-c-coin-pretrain", pretrained_trunk=True)

    def problems(**extra):
        try:
            guards.check_train_boundary(ctx, **(kw | extra))
        except guards.GuardError as exc:
            return str(exc)
        return ""

    assert "trunk" not in problems(seed=101, init_trunk_sha256="aa" * 32)
    assert "trunk sha256" in problems(seed=101, init_trunk_sha256="aa" * 32,
                                      per_run_trunk=True)
    assert "trunk" not in problems(seed=101, init_trunk_sha256="bb" * 32,
                                   per_run_trunk=True)
    assert "seed 101 only" in problems(seed=73, init_trunk_sha256="bb" * 32,
                                       per_run_trunk=True)
    assert guards.expected_trunk_sha256(101, "gtm", "registered", True) == "bb" * 32
    assert guards.expected_trunk_sha256(101, "gtm", "registered") == "aa" * 32


def test_control_pretraining_boundary():
    ctx = guards.make_context_for_tests()
    with pytest.raises(guards.GuardError, match="control seed 101"):
        guards.check_pretrain_boundary(ctx, seed=73, total_steps=10**6,
                                       has_env=False, unit=("gtm", "registered"))
    with pytest.raises(guards.GuardError, match="not an N10-C run"):
        guards.check_pretrain_boundary(ctx, seed=101, total_steps=10**6,
                                       has_env=False, unit=("mlp", "closed"))


def test_control_trunk_paths_are_distinct_from_the_shared_one(tmp_path):
    paths = {guards.control_trunk_path(tmp_path, r.arm, r.condition)
             for r in q.control_queue()}
    assert len(paths) == 10 and guards.trunk_path(tmp_path, 101) not in paths


# -- interruption windows of the liveness transition (review round 5) -------
def test_interruption_after_persisting_the_abort_never_grants_a_second(
        tmp_path, monkeypatch):
    c, calls, _ = _campaign(tmp_path, _abort_first_once)
    c.pin({})
    real_move = rc.shutil.move
    boom = {"armed": True}

    def flaky_move(src, dst):
        if boom["armed"] and ".liveness-abort-" in str(dst):
            boom["armed"] = False
            raise KeyboardInterrupt  # killed right after the status save
        return real_move(src, dst)

    monkeypatch.setattr(rc.shutil, "move", flaky_move)
    with pytest.raises(KeyboardInterrupt):
        c._process(FIRST, can_eval=True)
    st = c.store.load(FIRST)
    assert st["status"] == q.LIVENESS_RESTART and st["liveness_restarts"] == 1
    assert (c.dir / "runs" / FIRST.run_id).exists()  # not moved yet
    c2, calls2, _ = _campaign(tmp_path, lambda tag, n, kw: _rec(abort=True))
    c2._process(FIRST, can_eval=True)  # resume: the restart already counted
    st = c2.store.load(FIRST)
    assert (c2.dir / "runs" / f"{FIRST.run_id}.liveness-abort-1").exists()
    assert st["infra_failures"] == 0 and len(calls2) == 1
    assert st["status"] == q.TRAINING_FAILURE and st["liveness_abort"] is True


def _interrupted_running(c, restarts):
    st = c.store.load(FIRST)
    st.update(status=q.RUNNING, liveness_restarts=restarts)
    c.store.save(st)
    rd = c.dir / "runs" / FIRST.run_id
    rd.mkdir(parents=True)
    (rd / f"ckpt_{TOTAL // 2}.safetensors").write_text("aborted")
    (rd / "run.json").write_text(json.dumps({
        "training_failure": True, "liveness_abort": True,
        "pair_steps_done": TOTAL // 2,
        "last_checkpoint": f"ckpt_{TOTAL // 2}.safetensors",
        "wall_time_s": 1.0, "failure_reason": "trunk liveness guard"}))


def test_abort_interrupted_before_the_status_save_is_recovered(tmp_path):
    c, calls, _ = _campaign(tmp_path, lambda tag, n, kw: _rec())
    c.pin({})
    _interrupted_running(c, restarts=0)
    c._process(FIRST, can_eval=True)
    st = c.store.load(FIRST)
    assert st["status"] == q.DONE and st["infra_failures"] == 0
    assert st["liveness_restarts"] == 1 and len(calls) == 1
    assert [a["outcome"] for a in st["attempts"]] == ["liveness_abort", "trained"]


def test_second_abort_interrupted_before_the_save_stays_final(tmp_path):
    c, calls, evals = _campaign(tmp_path, lambda tag, n, kw: _rec())
    c.pin({})
    _interrupted_running(c, restarts=1)
    c._process(FIRST, can_eval=True)
    st = c.store.load(FIRST)
    assert st["status"] == q.TRAINING_FAILURE and calls == []  # no 3rd attempt
    assert st["infra_failures"] == 0 and st["eval_done"]
    assert evals[0].read_text() == "aborted"
