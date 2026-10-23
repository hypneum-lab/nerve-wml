"""Campaign tests: fake train/eval functions, no mlx / procgen needed."""
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from experiments.procgen_multiagent_gtm_comm.campaign import queue as q
from experiments.procgen_multiagent_gtm_comm.campaign import run_campaign as rc

TOTAL = 64


SEEDS6 = (0, 17, 42, 73, 101, 131)
HOST_OF = {0: "macm1", 17: "macm1", 42: "macm1", 73: "kxkm", 101: "kxkm", 131: "kxkm"}


def test_queue_is_the_frozen_60_unique_runs():
    runs = q.build_queue()
    assert len(runs) == 60 and len({r.run_id for r in runs}) == 60
    reg = [r for r in runs if r.condition == "registered"]
    closed = [r for r in runs if r.condition == "closed"]
    assert len(reg) == 36 and len(closed) == 24
    assert {(r.arm, r.seed) for r in reg} == {
        (a, s) for a in ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
        for s in SEEDS6}
    assert {(r.arm, r.seed) for r in closed} == {
        (a, s) for a in ("gtm", "dial", "eng", "nocomm") for s in SEEDS6}
    assert q.SEEDS == SEEDS6 and q.N_RUNS == 60


def test_seed_to_host_mapping_exact_and_per_run_host_field():
    assert q.SEED_HOST == HOST_OF
    for r in q.build_queue():
        assert r.host == HOST_OF[r.seed]
        assert q.new_state(r)["host"] == HOST_OF[r.seed]
    # all arms and both conditions of a seed share its host
    for s in SEEDS6:
        assert len({r.host for r in q.build_queue() if r.seed == s}) == 1


def test_host_filtering_partitions_the_queue():
    mac, kx = q.runs_for_host("macm1"), q.runs_for_host("kxkm")
    assert len(mac) == 30 and len(kx) == 30
    assert {r.seed for r in mac} == {0, 17, 42}
    assert {r.seed for r in kx} == {73, 101, 131}
    assert set(mac) | set(kx) == set(q.build_queue()) and not set(mac) & set(kx)
    with pytest.raises(ValueError):
        q.runs_for_host("grosmac")


def test_queue_order_deterministic_and_balanced():
    assert q.build_queue() == q.build_queue()
    runs = q.build_queue()
    first10 = runs[:10]  # one complete seed block
    assert {r.seed for r in first10} == {0}
    assert sum(r.condition == "closed" for r in first10) == 4
    # arm rotation: the first run is not the same arm for every seed
    firsts = [runs[i * 10].arm for i in range(6)]
    assert len(set(firsts)) > 1


KX = q.runs_for_host("kxkm")
FIRST, SECOND = KX[0], KX[1]
FIRST_TAG = f"{FIRST.arm}/{FIRST.condition}/{FIRST.seed}"


def _is_first(kw):
    return (kw["arm"], kw["seed"], kw["condition"]) == (
        FIRST.arm, FIRST.seed, FIRST.condition)


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


def _campaign(tmp_path, train, eval_fn=None, gpu=None, archive=None,
              ckpts=True, mkdir=True, host="kxkm", cross_fn=None, **o):
    calls = {"train": [], "eval": [], "cross": []}

    def t(**kw):
        calls["train"].append(kw["arm"] + "/" + kw["condition"] + f"/{kw['seed']}")
        if mkdir:
            kw["out_dir"].mkdir(parents=True, exist_ok=True)
        if ckpts:
            (kw["out_dir"] / f"ckpt_{TOTAL}.safetensors").write_text("x")
        return train(len(calls["train"]), kw)

    def e(ckpt, arm, cond, seed, env_factory=None):
        calls["eval"].append((ckpt, arm))
        return (eval_fn or (lambda *a: _Eval()))(ckpt, arm, cond, seed)

    def x(ckpt0, ckpt1, arm, cond, seed0, seed1, env_factory=None):
        calls["cross"].append((ckpt0, ckpt1, arm, cond, seed0, seed1))
        return (cross_fn or (lambda *a: _Cross(*a)))(ckpt0, ckpt1, arm, cond, seed0, seed1)

    c = rc.Campaign(rc.Options(campaign_dir=tmp_path / "c", host=host, total_pair_steps=TOTAL,
                               min_free_gb=8, **o),
                    train_fn=t, eval_fn=e, cross_fn=x, gpu_fn=gpu or (lambda: None),
                    archive_fn=archive, log=lambda *_: None)
    (tmp_path / "c").mkdir(exist_ok=True)
    return c, calls


def test_full_campaign_runs_all_and_resume_skips_done(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec())
    assert c.run() == rc.EXIT_OK
    assert len(calls["train"]) == 30 and len(calls["eval"]) == 30
    assert all(c.store.load(s)["status"] == q.DONE for s in KX)
    c2, calls2 = _campaign(tmp_path, lambda n, kw: _rec())
    assert c2.run() == rc.EXIT_OK
    assert calls2["train"] == [] and calls2["eval"] == []


def test_resume_after_interruption_restarts_running_from_scratch(tmp_path):
    first = FIRST
    c, calls = _campaign(tmp_path, lambda n, kw: _rec())
    c.pin({})
    st = c.store.load(first)
    st["status"] = q.RUNNING
    c.store.save(st)
    (c.dir / "runs" / first.run_id).mkdir(parents=True)
    (c.dir / "runs" / first.run_id / "partial").write_text("p")
    assert c.run() == rc.EXIT_OK
    s = c.store.load(first)
    assert s["status"] == q.DONE and s["infra_failures"] == 1  # counted
    assert (c.dir / "runs" / f"{first.run_id}.failed-1" / "partial").exists()


def _boom_first(n, kw):
    if _is_first(kw):
        raise OSError("disk")
    return _rec()


def test_infra_restart_capped_at_two_then_training_failure(tmp_path):
    c, calls = _campaign(tmp_path, _boom_first)
    code = c.run()
    first = FIRST
    assert calls["train"].count(FIRST_TAG) == 3  # 1 + 2 restarts
    s = c.store.load(first)
    assert s["status"] == q.TRAINING_FAILURE and s["infra_failures"] == 3
    assert s["infra_exhausted"] and s["eval_done"]
    assert not q.needs_training(s)
    # evaluated at the last finite checkpoint kept from the failed attempts
    ckpt = calls["eval"][0][0]
    assert ckpt is not None and ckpt.resolve().exists()
    assert code == rc.EXIT_OK
    c2, calls2 = _campaign(tmp_path, _boom_first)
    c2.run()
    assert calls2["train"] == [] and calls2["eval"] == []


def test_infra_exhausted_without_checkpoint_evaluates_init(tmp_path):
    c, calls = _campaign(tmp_path, _boom_first, ckpts=False)
    c.run()
    assert calls["eval"][0][0] is None
    s = c.store.load(FIRST)
    assert s["status"] == q.TRAINING_FAILURE and s["eval_done"]


def test_tmp_checkpoint_never_picked(tmp_path):
    def t(n, kw):
        for name in ("ckpt_100.safetensors", "ckpt_200.tmp.safetensors"):
            (kw["out_dir"] / name).write_text("x")
        if _is_first(kw):
            raise OSError("disk")
        return _rec()
    c, calls = _campaign(tmp_path, t, ckpts=False)
    c.run()
    assert calls["eval"][0][0].name == "ckpt_100.safetensors"
    files = rc.archive_files(c.run_dir(FIRST), None)
    assert not any("tmp" in f.name for f in files)


def test_exhausted_before_run_dir_created_still_evaluates(tmp_path):
    c, calls = _campaign(tmp_path, _boom_first, ckpts=False, mkdir=False)
    assert c.run() == rc.EXIT_OK
    s = c.store.load(FIRST)
    assert s["eval_done"] and s["eval_failures"] == 0
    assert calls["eval"][0][0] is None
    assert (c.run_dir(FIRST) / "eval.json").exists()


def test_exhausted_checkpoint_found_in_earlier_failed_dir(tmp_path):
    made = {"n": 0}

    def t(n, kw):
        made["n"] += 1
        if _is_first(kw):
            if made["n"] > 1:  # later attempts leave no checkpoint
                for f in kw["out_dir"].glob("ckpt_*"):
                    f.unlink()
            raise OSError("disk")
        return _rec()
    c, calls = _campaign(tmp_path, t)
    c.run()
    ckpt = calls["eval"][0][0]
    assert ckpt is not None and ckpt.resolve().exists()
    assert ".failed-1" in str(ckpt.resolve())


def test_infra_failure_then_success(tmp_path):
    def flaky(n, kw):
        if n == 1:
            raise RuntimeError("procgen stub in use")
        return _rec()
    c, calls = _campaign(tmp_path, flaky)
    assert c.run() == rc.EXIT_OK
    assert calls["train"][0] == calls["train"][1]
    assert c.store.load(FIRST)["infra_failures"] == 1


def test_training_failure_not_retried_and_evaluated_untrained(tmp_path):
    def tf(n, kw):
        if n == 1:
            return _rec(training_failure=True, pair_steps_done=8,
                        last_checkpoint=None, failure_reason="nan")
        return _rec()
    c, calls = _campaign(tmp_path, tf)
    assert c.run() == rc.EXIT_OK
    first = FIRST
    s = c.store.load(first)
    assert s["status"] == q.TRAINING_FAILURE and s["eval_done"]
    assert calls["train"].count(FIRST_TAG) == 1
    assert calls["eval"][0][0] is None  # untrained init evaluated


def test_eval_only_after_training_and_eval_retry_does_not_retrain(tmp_path):
    state = {"n": 0}

    def ev(ckpt, arm, cond, seed):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("eval crash")
        return _Eval()
    c, calls = _campaign(tmp_path, lambda n, kw: _rec(), eval_fn=ev)
    assert c.run() == rc.EXIT_OK
    assert calls["train"].count(FIRST_TAG) == 1
    assert c.store.load(FIRST)["eval_failures"] == 1


def test_gpu_guard_stops_without_state_change(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec(), gpu=lambda: 2.0)
    assert c.run() == rc.EXIT_GPU
    assert calls["train"] == []
    assert c.store.load(FIRST)["status"] == q.PENDING


def test_stop_file(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec())
    (c.dir / "STOP").write_text("")
    assert c.run() == rc.EXIT_STOP and calls["train"] == []


def test_archive_called_and_failure_nonfatal(tmp_path):
    got = []

    def arch(rd, files, target, run_id):
        got.append((run_id, sorted(f.name for f in files)))
        if len(got) == 1:
            raise RuntimeError("rsync down")
    c, _ = _campaign(tmp_path, lambda n, kw: _rec(), archive=arch,
                     archive_to="u@h:/p")
    assert c.run() == rc.EXIT_OK
    assert len(got) == 30
    assert not c.store.load(FIRST)["archived"]
    assert c.store.load(SECOND)["archived"]
    assert f"ckpt_{TOTAL}.safetensors" in got[1][1]


def test_pin_refuses_other_manifest(tmp_path):
    c, _ = _campaign(tmp_path, lambda n, kw: _rec())
    c.pin({"manifest_sha256": "a", "commit": "b"})
    with pytest.raises(rc.GuardError):
        c.pin({"manifest_sha256": "other", "commit": "b"})


# ---- guards ---------------------------------------------------------

def _git_repo(tmp_path):
    """Code commit C, then a later commit adding docs/manifest.md naming C."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def g(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True,
                              capture_output=True, text=True).stdout.strip()
    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (repo / "code.py").write_text("x = 1\n")
    g("add", ".")
    g("commit", "-q", "-m", "code")
    code = g("rev-parse", "HEAD")
    md = repo / "experiments" / "procgen_multiagent_gtm_comm" / "manifest"
    md.mkdir(parents=True)
    (md / "stage2-manifest.md").write_text(MANIFEST_TMPL.format(code=code))
    g("add", ".")
    g("commit", "-q", "-m", "manifest")
    return repo, code, g


MANIFEST_TMPL = ("# Stage 2 manifest\n**Status: FILED on OSF**\n"
                 "DOI 10.17605/OSF.IO/A1B2C\nharness commit {code}\n")
DOI = "10.17605/OSF.IO/A1B2C"


def _ok(repo, code):
    m = repo / "experiments" / "procgen_multiagent_gtm_comm" / "manifest" / "stage2-manifest.md"
    return dict(repo=repo, manifest=m,
                manifest_sha256=hashlib.sha256(m.read_bytes()).hexdigest(),
                manifest_doi=DOI, expect_commit=code)


def test_guards_pass_when_only_docs_changed_since_commit(tmp_path):
    repo, code, _ = _git_repo(tmp_path)
    assert rc.guard_problems(**_ok(repo, code)) == []


def test_guard_refusals(tmp_path):
    repo, code, g = _git_repo(tmp_path)
    kw = _ok(repo, code)
    assert any("DOI" in p for p in rc.guard_problems(**{**kw, "manifest_doi": None}))
    assert any("DOI" in p for p in rc.guard_problems(
        **{**kw, "manifest_doi": "not-a-doi"}))
    assert any("SHA-256" in p for p in rc.guard_problems(
        **{**kw, "manifest_sha256": "0" * 64}))
    assert any("sha256" in p.lower() for p in rc.guard_problems(
        **{**kw, "manifest_sha256": None}))
    assert any("expect-commit" in p for p in rc.guard_problems(
        **{**kw, "expect_commit": None}))
    assert any("HEAD" in p for p in rc.guard_problems(
        **{**kw, "expect_commit": "1" * 40}))
    # code changed after the recorded commit -> refused
    (repo / "code.py").write_text("x = 2\n")
    g("commit", "-q", "-am", "code2")
    assert any("code changed" in p for p in rc.guard_problems(**kw))
    # dirty tree -> refused
    (repo / "dirty.txt").write_text("d")
    assert any("dirty" in p for p in rc.guard_problems(**kw))
    assert rc.normalize_doi("https://doi.org/" + DOI) == DOI


def test_guard_untracked_manifest_refused(tmp_path):
    repo, code, g = _git_repo(tmp_path)
    m = repo / "experiments" / "procgen_multiagent_gtm_comm" / "manifest" / "o.md"
    m.write_text(MANIFEST_TMPL.format(code=code))
    kw = {**_ok(repo, code), "manifest": m,
          "manifest_sha256": hashlib.sha256(m.read_bytes()).hexdigest()}
    probs = rc.guard_problems(**kw)
    assert any("not tracked" in p for p in probs)


@pytest.mark.parametrize("doi", [
    "10.17605/OSF.IO/XXXXX", "10.17605/OSF.IO/AAAAA", "10.1234/abcde",
    "10.17605/OSF.IO/AB", "not-a-doi", "10.17605/osf.io/a1b2c", ""])
def test_placeholder_dois_rejected(tmp_path, doi):
    repo, code, _ = _git_repo(tmp_path)
    assert rc.normalize_doi(doi) is None
    assert any("DOI" in p for p in rc.guard_problems(
        **{**_ok(repo, code), "manifest_doi": doi}))


def _rewrite_manifest(repo, g, text, code):
    m = repo / "experiments" / "procgen_multiagent_gtm_comm" / "manifest" / "stage2-manifest.md"
    m.write_text(text.format(code=code))
    g("commit", "-q", "-am", "edit manifest")
    return m


@pytest.mark.parametrize("text,needle", [
    ("# M\n**Status: DRAFT**\nDOI 10.17605/OSF.IO/A1B2C\n{code}\n", "DRAFT"),
    ("# M\nDOI 10.17605/OSF.IO/A1B2C\n{code}\n", "FILED"),
    ("# M\nStatus: FILED\nDOI 10.17605/OSF.IO/Z9Z9Z\n{code}\n", "DOI"),
    ("# M\nStatus: FILED\nDOI 10.17605/OSF.IO/A1B2C\n", "expect-commit"),
    ("# M\nStatus: FILED draft later\nDRAFT\n10.17605/OSF.IO/A1B2C {code}\n",
     "DRAFT"),
])
def test_manifest_content_binding(tmp_path, text, needle):
    repo, code, g = _git_repo(tmp_path)
    m = _rewrite_manifest(repo, g, text, code)
    kw = {**_ok(repo, code), "manifest": m,
          "manifest_sha256": hashlib.sha256(m.read_bytes()).hexdigest()}
    assert any(needle in p for p in rc.guard_problems(**kw))


def test_manifest_outside_manifest_dir_refused(tmp_path):
    repo, code, g = _git_repo(tmp_path)
    m = repo / "docs" / "m.md"
    m.parent.mkdir()
    m.write_text(MANIFEST_TMPL.format(code=code))
    g("add", ".")
    g("commit", "-q", "-m", "docs manifest")
    kw = {**_ok(repo, code), "manifest": m,
          "manifest_sha256": hashlib.sha256(m.read_bytes()).hexdigest()}
    assert any("must be under" in p for p in rc.guard_problems(**kw))


def test_intervening_code_commit_reverted_is_still_refused(tmp_path):
    repo, code, g = _git_repo(tmp_path)
    (repo / "code.py").write_text("x = 2\n")
    g("commit", "-q", "-am", "change")
    (repo / "code.py").write_text("x = 1\n")
    g("commit", "-q", "-am", "revert")  # net diff empty, history not
    assert any("code changed" in p for p in rc.guard_problems(**_ok(repo, code)))


@pytest.mark.parametrize("extra", [
    ["--manifest-doi", "10.17605/OSF.IO/XXXXX"],
    ["--manifest-sha256", "0" * 64],
    [],
])
def test_main_negative_guards_call_nothing(tmp_path, monkeypatch, extra):
    repo, code, _ = _git_repo(tmp_path)
    called = []
    monkeypatch.setattr(rc, "REPO", repo)
    monkeypatch.setattr(rc, "_default_train", lambda **kw: called.append("t"))
    monkeypatch.setattr(rc, "_default_eval", lambda *a, **k: called.append("e"))
    m = _ok(repo, code)
    argv = ["--campaign-dir", str(tmp_path / "c"), "--manifest", str(m["manifest"]),
            "--manifest-sha256", m["manifest_sha256"], "--manifest-doi", DOI,
            "--expect-commit", code]
    for i in range(0, len(extra), 2):
        argv[argv.index(extra[i]) + 1] = extra[i + 1]
    if not extra:
        argv += ["--total-pair-steps", "64"]  # trial flag missing
    assert rc.main(argv) == rc.EXIT_GUARD
    assert called == [] and not (tmp_path / "c").exists()


def test_pin_budget_and_trial_status(tmp_path):
    c, _ = _campaign(tmp_path, lambda n, kw: _rec(), trial=True)
    c.pin({"manifest_sha256": "a"})
    c2, _ = _campaign(tmp_path, lambda n, kw: _rec(), trial=False)
    with pytest.raises(rc.GuardError, match="trial"):
        c2.pin({"manifest_sha256": "a"})
    c3, _ = _campaign(tmp_path, lambda n, kw: _rec(), trial=True)
    c3.o.total_pair_steps = TOTAL * 2
    with pytest.raises(rc.GuardError, match="total_pair_steps"):
        c3.pin({"manifest_sha256": "a"})


def test_unpinned_existing_state_refused(tmp_path):
    c, _ = _campaign(tmp_path, lambda n, kw: _rec())
    c.store.save(c.store.load(FIRST))
    with pytest.raises(rc.GuardError, match="unpinned"):
        c.pin({})


def test_run_pins_budget_automatically(tmp_path):
    c, _ = _campaign(tmp_path, lambda n, kw: _rec())
    c.run()
    meta = json.loads((c.dir / "campaign.json").read_text())
    assert meta["total_pair_steps"] == TOTAL and meta["trial"] is False


def test_main_refuses_without_doi_and_trains_nothing(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(rc, "_default_train", lambda **kw: called.append(kw))
    code = rc.main(["--campaign-dir", str(tmp_path / "c")])
    assert code == rc.EXIT_GUARD and called == []
    assert not (tmp_path / "c").exists()


def test_dry_run_trains_nothing_and_estimates(tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(rc, "_default_train", lambda **kw: called.append(kw))
    thr = tmp_path / "t.json"
    thr.write_text(json.dumps({"results": {
        "gtm": {"pair_steps_per_s": 1000.0,
                "eval_proxy": {"seconds_per_model_2x200": 600.0}},
        "nocomm": {"pair_steps_per_s": 2000.0,
                   "eval_proxy": {"seconds_per_model_2x200": 300.0}}}}))
    code = rc.main(["--campaign-dir", str(tmp_path / "c"), "--dry-run",
                    "--throughput-json", str(thr)])
    out = capsys.readouterr().out
    assert code == 0 and called == []
    assert "60 runs" in out and "WOULD REFUSE" in out and "Estimated duration" in out
    assert not (tmp_path / "c").exists()
    e = rc.estimate(json.loads(thr.read_text()))
    # 12 gtm runs at 50M/1000/3600+600/3600 h, 48 others at nocomm rate
    gtm_h = 50e6 / 1000 / 3600 + 600 / 3600
    oth_h = 50e6 / 2000 / 3600 + 300 / 3600
    assert e["total_hours"] == pytest.approx(12 * gtm_h + 48 * oth_h)


def test_gpu_free_parsing_and_missing_smi(monkeypatch):
    monkeypatch.setattr(rc.platform, "system", lambda: "Linux")
    def ok(*a, **k):
        return SimpleNamespace(returncode=0, stdout="16384\n", stderr="")

    assert rc.gpu_free_gb(ok) == pytest.approx(16.0)

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    with pytest.raises(rc.GuardError):
        rc.gpu_free_gb(missing)
    monkeypatch.setattr(rc.platform, "system", lambda: "Darwin")
    assert rc.gpu_free_gb(missing) is None


# ---- hosts (seed -> host block factor, 2026-10-23) --------------------

def test_kxkm_campaign_touches_only_its_own_runs(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec(), host="kxkm")
    assert c.run() == rc.EXIT_OK
    assert len(calls["train"]) == 30 and len(calls["eval"]) == 30
    assert {int(t.rsplit("/", 1)[1]) for t in calls["train"]} == {73, 101, 131}
    # the other host's runs were never touched (no status file written)
    assert not any(c.store.path(s).exists() for s in q.runs_for_host("macm1"))


def test_macm1_trains_its_runs_and_never_evaluates(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec(), host="macm1")
    assert c.run() == rc.EXIT_OK
    assert len(calls["train"]) == 30 and calls["eval"] == []
    assert {int(t.rsplit("/", 1)[1]) for t in calls["train"]} == {0, 17, 42}
    for s in q.runs_for_host("macm1"):
        st = c.store.load(s)
        assert st["status"] == q.TRAINED and not st["eval_done"]
        assert st["host"] == "macm1"
    assert not any(c.store.path(s).exists() for s in q.runs_for_host("kxkm"))


def test_evaluation_refused_on_macm1(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec(), host="macm1")
    spec = q.runs_for_host("macm1")[0]
    st = c.store.load(spec)
    st.update(status=q.TRAINED, last_checkpoint=None)
    with pytest.raises(rc.GuardError, match="kxkm"):
        c.evaluate(spec, st)
    assert calls["eval"] == []


def test_options_reject_unknown_host(tmp_path):
    with pytest.raises(rc.GuardError):
        rc.Campaign(rc.Options(campaign_dir=tmp_path / "c", host="grosmac"))


def test_pin_records_host_and_refuses_other_host(tmp_path):
    c, _ = _campaign(tmp_path, lambda n, kw: _rec(), host="kxkm")
    c.pin({})
    assert json.loads((c.dir / "campaign.json").read_text())["host"] == "kxkm"
    c2, _ = _campaign(tmp_path, lambda n, kw: _rec(), host="macm1")
    with pytest.raises(rc.GuardError, match="host"):
        c2.pin({})


def _sync_fixture(tmp_path):
    """macM1 campaign dir with its 30 runs trained (fake checkpoints)."""
    c, _ = _campaign(tmp_path / "mac", lambda n, kw: _rec(), host="macm1")
    assert c.run() == rc.EXIT_OK
    return c


def test_sync_stages_trained_macm1_runs_for_kxkm(tmp_path):
    mac = _sync_fixture(tmp_path)
    sent = []

    def fake_rsync(src, target, jump):
        sent.append((sorted(p.name for p in src.rglob("*") if p.is_file()),
                     target, jump))

    done = rc.sync_runs(mac.dir, "u@kxkm:/camp", jump="electron@es",
                        rsync_fn=fake_rsync)
    assert len(done) == 30 and all(r.host == "macm1" for r in done)
    assert all(mac.store.load(s)["synced"] for s in q.runs_for_host("macm1"))
    # per run: the run dir (checkpoint) first, then the status file
    assert any(f"ckpt_{TOTAL}.safetensors" in names for names, _, _ in sent)
    assert all(j == "electron@es" for _, _, j in sent)
    # idempotent: nothing left to send
    assert rc.sync_runs(mac.dir, "u@kxkm:/camp", jump="j", rsync_fn=fake_rsync) == []


def test_sync_skips_untrained_and_failed_rsync_stays_unsynced(tmp_path):
    c, _ = _campaign(tmp_path / "mac", lambda n, kw: _rec(), host="macm1")
    c.pin({})
    first = q.runs_for_host("macm1")[0]
    st = c.store.load(first)
    st["status"] = q.RUNNING
    c.store.save(st)

    def boom(src, target, jump):
        raise RuntimeError("ssh down")

    assert rc.sync_runs(c.dir, "u@h:/p", jump="j", rsync_fn=boom) == []
    assert not c.store.load(first)["synced"]


def test_kxkm_evaluates_synced_macm1_runs_but_never_trains_them(tmp_path):
    mac = _sync_fixture(tmp_path)
    kx, calls = _campaign(tmp_path / "kx", lambda n, kw: _rec(), host="kxkm")
    # simulate the sync landing on KXKM: status files + checkpoints
    for s in q.runs_for_host("macm1")[:3]:
        st = mac.store.load(s)
        rd = kx.run_dir(s)
        rd.mkdir(parents=True)
        (rd / st["last_checkpoint"]).write_text("x")
        kx.store.save(dict(st, synced_from="macm1", synced=False))
    assert kx.run() == rc.EXIT_OK
    trained = {int(t.rsplit("/", 1)[1]) for t in calls["train"]}
    assert trained == {73, 101, 131}  # never trains macM1 seeds
    assert len(calls["eval"]) == 30 + 3
    evaluated = [s for s in q.runs_for_host("macm1") if kx.store.load(s)["eval_done"]]
    assert len(evaluated) == 3  # the other 27 await sync


def test_entry_points_are_guarded_by_main_check():
    import pathlib
    import re

    root = pathlib.Path(rc.__file__).resolve().parents[1]
    bad = []
    for f in root.rglob("*.py"):
        src = f.read_text()
        if re.search(r"^def main\(", src, re.M) and \
                not re.search(r'^if __name__ == "__main__":', src, re.M):
            bad.append(str(f.relative_to(root)))
    assert bad == []


def test_main_requires_host_and_sync_needs_args(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(rc, "_default_train", lambda **kw: called.append(kw))
    assert rc.main(["--campaign-dir", str(tmp_path / "c")]) == rc.EXIT_GUARD
    assert called == []
    with pytest.raises(SystemExit):
        rc.main(["sync"])  # missing required args


def test_dry_run_with_host_lists_only_that_host(tmp_path, capsys):
    code = rc.main(["--campaign-dir", str(tmp_path / "c"), "--dry-run",
                    "--host", "macm1"])
    out = capsys.readouterr().out
    assert code == 0 and "30 runs" in out and "s73" not in out and "s0" in out


# ---- freeze-tag guard mode (2026-10-23 22:17) -------------------------

def _frozen_repo(tmp_path, *, status=None, annotated=True, msg_sha=None, tag="n10c-freeze-t"):
    """Repo with origin (bare), code commit, manifest commit, freeze tag."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    repo = tmp_path / "frepo"
    repo.mkdir()

    def g(*a, check=True):
        return subprocess.run(["git", "-C", str(repo), *a], check=check,
                              capture_output=True, text=True).stdout.strip()
    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    g("remote", "add", "origin", str(origin))
    (repo / "code.py").write_text("x = 1\n")
    g("add", ".")
    g("commit", "-q", "-m", "code")
    md = repo / "experiments" / "procgen_multiagent_gtm_comm" / "manifest"
    md.mkdir(parents=True)
    m = md / "stage2-manifest.md"
    m.write_text(f"# Manifest\n**{status or f'Status: FROZEN (tag {tag})'}**\n")
    g("add", ".")
    g("commit", "-q", "-m", "manifest")
    sha = hashlib.sha256(m.read_bytes()).hexdigest()
    rel = "experiments/procgen_multiagent_gtm_comm/manifest/stage2-manifest.md"
    if annotated:
        g("tag", "-a", tag, "-m", f"freeze\n\nmanifest: {rel}\n"
          f"manifest-sha256: {msg_sha or sha}")
    else:
        g("tag", tag)
    g("push", "-q", "origin", "HEAD:refs/heads/main", f"refs/tags/{tag}")
    return repo, m, sha, tag, g


def _fkw(repo, m, sha, tag):
    return dict(repo=repo, manifest=m, manifest_sha256=sha, manifest_doi=None,
                expect_commit=None, freeze_tag=tag)


def test_freeze_tag_mode_passes(tmp_path):
    repo, m, sha, tag, _ = _frozen_repo(tmp_path)
    assert rc.guard_problems(**_fkw(repo, m, sha, tag)) == []


def test_freeze_tag_passes_with_docs_only_descendant(tmp_path):
    repo, m, sha, tag, g = _frozen_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "n.md").write_text("n")
    g("add", ".")
    g("commit", "-q", "-m", "docs")
    assert rc.guard_problems(**_fkw(repo, m, sha, tag)) == []
    (repo / "code.py").write_text("x = 2\n")
    g("commit", "-q", "-am", "code2")
    assert any("code changed" in p for p in rc.guard_problems(**_fkw(repo, m, sha, tag)))


def test_freeze_tag_negatives(tmp_path):
    repo, m, sha, tag, g = _frozen_repo(tmp_path)
    kw = _fkw(repo, m, sha, tag)
    assert any("does not exist" in p for p in rc.guard_problems(
        **{**kw, "freeze_tag": "nope"}))
    assert any("both" in p for p in rc.guard_problems(**{**kw, "manifest_doi": DOI}))
    assert any("SHA-256 mismatch" in p for p in rc.guard_problems(
        **{**kw, "manifest_sha256": "0" * 64}))
    (repo / "dirty.txt").write_text("d")
    assert any("dirty" in p for p in rc.guard_problems(**kw))


def test_freeze_lightweight_tag_refused(tmp_path):
    repo, m, sha, tag, _ = _frozen_repo(tmp_path, annotated=False)
    assert any("lightweight" in p for p in rc.guard_problems(**_fkw(repo, m, sha, tag)))


def test_freeze_tag_message_sha_mismatch_refused(tmp_path):
    repo, m, sha, tag, _ = _frozen_repo(tmp_path, msg_sha="a" * 64)
    assert any("tag message" in p for p in rc.guard_problems(**_fkw(repo, m, sha, tag)))


def test_freeze_tag_at_wrong_commit_refused(tmp_path):
    repo, m, sha, tag, g = _frozen_repo(tmp_path)
    g("reset", "-q", "--hard", "HEAD~1")  # HEAD no longer descends from the tag
    probs = rc.guard_problems(**_fkw(repo, m if m.exists() else m, sha, tag))
    assert probs and any("manifest" in p or "descendant" in p for p in probs)


def test_freeze_manifest_status_must_name_the_tag(tmp_path):
    repo, m, sha, tag, _ = _frozen_repo(tmp_path, status="Status: FROZEN (tag other)")
    assert any("FROZEN" in p for p in rc.guard_problems(**_fkw(repo, m, sha, tag)))
    repo2, m2, sha2, tag2, _ = _frozen_repo(tmp_path / "d", status="Status: DRAFT")
    assert any("DRAFT" in p for p in rc.guard_problems(**_fkw(repo2, m2, sha2, tag2)))


def test_filed_doi_mode_still_works_alongside(tmp_path):
    repo, code, _ = _git_repo(tmp_path)
    assert rc.guard_problems(**_ok(repo, code)) == []


@pytest.mark.parametrize("variant", ["missing", "lightweight", "sha", "dirty"])
def test_main_freeze_negatives_call_nothing(tmp_path, monkeypatch, variant):
    kw = {"annotated": variant != "lightweight",
          "msg_sha": "b" * 64 if variant == "sha" else None}
    repo, m, sha, tag, _ = _frozen_repo(tmp_path, **kw)
    if variant == "dirty":
        (repo / "x.txt").write_text("x")
    called = []
    monkeypatch.setattr(rc, "REPO", repo)
    monkeypatch.setattr(rc, "_default_train", lambda **k: called.append("t"))
    monkeypatch.setattr(rc, "_default_eval", lambda *a, **k: called.append("e"))
    argv = ["--campaign-dir", str(tmp_path / "c"), "--host", "kxkm",
            "--manifest", str(m), "--manifest-sha256", sha,
            "--freeze-tag", "ghost" if variant == "missing" else tag]
    assert rc.main(argv) == rc.EXIT_GUARD
    assert called == [] and not (tmp_path / "c").exists()


def test_main_freeze_mode_pins_tag_commit_and_swhid(tmp_path, monkeypatch):
    repo, m, sha, tag, g = _frozen_repo(tmp_path)
    called = []
    monkeypatch.setattr(rc, "REPO", repo)
    monkeypatch.setattr(rc.Campaign, "run", lambda self: called.append(1) or 0)
    swh = "swh:1:rev:" + g("rev-parse", "HEAD")
    argv = ["--campaign-dir", str(tmp_path / "c"), "--host", "macm1",
            "--manifest", str(m), "--manifest-sha256", sha,
            "--freeze-tag", tag, "--swhid", swh]
    assert rc.main(argv) == 0 and called == [1]
    meta = json.loads((tmp_path / "c" / "campaign.json").read_text())
    assert meta["freeze_tag"] == tag and meta["swhid"] == swh
    assert meta["freeze_commit"] == g("rev-parse", "HEAD") and meta["host"] == "macm1"
    assert rc.main(argv[:-1] + ["not-a-swhid"]) == rc.EXIT_GUARD


# ---- per-host registered implementation settings (single source) ------

def test_host_settings_are_the_registered_values():
    from experiments.procgen_multiagent_gtm_comm.harness import codes

    assert codes.HOST_SETTINGS == {
        "kxkm": {"micro_batch_pairs": 512, "clear_cache": True, "n_workers": 8},
        "macm1": {"micro_batch_pairs": None, "clear_cache": False, "n_workers": 1},
    }
    assert set(codes.HOST_SETTINGS) == set(codes.HOSTS)


@pytest.mark.parametrize("host", ["kxkm", "macm1"])
def test_train_run_receives_exactly_the_host_settings(tmp_path, host):
    from experiments.procgen_multiagent_gtm_comm.harness import codes

    seen = []

    def train(n, kw):
        seen.append({k: kw[k] for k in ("micro_batch_pairs", "clear_cache", "n_workers")})
        assert not ({"micro_batch_pairs", "clear_cache", "n_workers"} - set(kw))
        return _rec()

    c, calls = _campaign(tmp_path, train, host=host)
    assert c.run() == rc.EXIT_OK
    assert len(seen) == 30
    assert all(s == codes.HOST_SETTINGS[host] for s in seen)


def test_host_settings_recorded_in_status_file(tmp_path):
    from experiments.procgen_multiagent_gtm_comm.harness import codes

    c, _ = _campaign(tmp_path, lambda n, kw: _rec(), host="kxkm")
    assert c.run() == rc.EXIT_OK
    for s in KX:
        assert c.store.load(s)["train_settings"] == codes.HOST_SETTINGS["kxkm"]


@pytest.mark.parametrize("key", ["micro_batch_pairs", "clear_cache", "n_workers"])
def test_registered_run_refuses_setting_override(tmp_path, key):
    called = []
    with pytest.raises(rc.GuardError, match=key):
        c, _ = _campaign(tmp_path, lambda n, kw: called.append(1) or _rec(),
                         train_kwargs={key: 3})
        c.run()
    assert called == []


def test_trial_run_may_override_settings(tmp_path):
    seen = []
    c, _ = _campaign(tmp_path, lambda n, kw: seen.append(kw["n_workers"]) or _rec(),
                     trial=True, train_kwargs={"n_workers": 2})
    c.run()
    assert set(seen) == {2}


def test_campaign_and_manifest_agree_on_host_settings():
    from experiments.procgen_multiagent_gtm_comm.harness import codes, manifest

    assert set(manifest.HOST_TRAIN_SETTINGS) == set(codes.HOST_SETTINGS)
    for host, st in codes.HOST_SETTINGS.items():
        m = manifest.HOST_TRAIN_SETTINGS[host]
        assert m["n_workers"] == str(st["n_workers"])
        assert m["clear_cache"] == ("on" if st["clear_cache"] else "off")
        mb = st["micro_batch_pairs"]
        assert m["micro_batch_pairs"] == ("none (full minibatch)" if mb is None else str(mb))


# ---- cross-seed cross-play probe in the campaign (KXKM only) ----------

def _full_kx(tmp_path, **kw):
    """KXKM campaign whose 30 macM1 runs were synced in (trained, ckpts)."""
    mac = _sync_fixture(tmp_path)
    kx, calls = _campaign(tmp_path / "kx", lambda n, k: _rec(), host="kxkm", **kw)
    for s in q.runs_for_host("macm1"):
        st = mac.store.load(s)
        rd = kx.run_dir(s)
        rd.mkdir(parents=True)
        (rd / st["last_checkpoint"]).write_text("x")
        kx.store.save(dict(st, synced_from="macm1", synced=False))
    return kx, calls


def test_cross_probe_runs_for_every_complete_cell_in_fixed_pairing(tmp_path):
    from experiments.procgen_multiagent_gtm_comm.harness.evaluate import cross_pairs

    kx, calls = _full_kx(tmp_path)
    assert kx.run() == rc.EXIT_OK
    assert len(calls["eval"]) == 60
    # 10 cells (6 registered + 4 closed arms) x 6 seed pairs
    assert len(calls["cross"]) == 60
    cells = {(a, c) for _, _, a, c, _, _ in calls["cross"]}
    assert len(cells) == 10
    for arm, cond in cells:
        got = [(s0, s1) for _, _, a, c, s0, s1 in calls["cross"] if (a, c) == (arm, cond)]
        assert got == cross_pairs()
    # role 0 / role 1 checkpoints come from their own seed's run directory
    ck0, ck1, arm, cond, s0, s1 = calls["cross"][0]
    run0 = q.RunSpec(arm, s0, cond).run_id
    run1 = q.RunSpec(arm, s1, cond).run_id
    assert ck0.parent.name == run0 and ck1.parent.name == run1


def test_cross_probe_waits_until_every_seed_of_the_cell_is_evaluated(tmp_path):
    kx, calls = _campaign(tmp_path, lambda n, k: _rec(), host="kxkm")
    assert kx.run() == rc.EXIT_OK  # macM1 seeds never synced
    assert calls["cross"] == []


def test_cross_probe_partial_cell_is_skipped(tmp_path):
    kx, calls = _full_kx(tmp_path)
    victim = q.RunSpec("gtm", 0, "registered")
    kx.store.path(victim).unlink()  # one macM1 run not synced
    assert kx.run() == rc.EXIT_OK
    cells = {(a, c) for _, _, a, c, _, _ in calls["cross"]}
    assert ("gtm", "registered") not in cells and len(cells) == 9


def test_cross_probe_never_runs_on_macm1(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, k: _rec(), host="macm1")
    assert c.run() == rc.EXIT_OK
    assert calls["cross"] == []
    with pytest.raises(rc.GuardError, match="kxkm"):
        c.cross_pass()


def test_cross_probe_recorded_and_resumable(tmp_path):
    kx, calls = _full_kx(tmp_path)
    assert kx.run() == rc.EXIT_OK
    f = kx.dir / "cross_status" / "registered-gtm.json"
    st = json.loads(f.read_text())
    assert st["done"] is True and len(st["pairs"]) == 6
    assert st["pairs"]["0-17"]["done"] and st["pairs"]["0-17"]["R_cross"] == 0.25
    assert st["pairs"]["131-0"]["seed_role1"] == 0
    rec = json.loads((kx.dir / "cross" / "registered-gtm-s0-s17.json").read_text())
    assert rec["seed_role0"] == 0 and rec["seed_role1"] == 17
    # a restart repeats nothing
    n = len(calls["cross"])
    kx2, calls2 = _campaign(tmp_path / "kx", lambda n, k: _rec(), host="kxkm")
    kx2.dir = kx.dir
    kx2.store = kx.store
    kx2.cross_store = type(kx.cross_store)(kx.dir / "cross_status")
    assert kx2.run() == rc.EXIT_OK and calls2["cross"] == [] and n == 60


def test_cross_probe_failure_retried_then_flagged_and_done_pairs_kept(tmp_path):
    attempts = []

    def flaky(ckpt0, ckpt1, arm, cond, seed0, seed1):
        if (arm, cond, seed0) == ("gtm", "registered", 42):
            attempts.append(1)
            raise RuntimeError("boom")
        return _Cross(ckpt0, ckpt1, arm, cond, seed0, seed1)

    kx, calls = _full_kx(tmp_path, cross_fn=flaky)
    assert kx.run() == rc.EXIT_PROBLEMS
    assert len(attempts) == q.MAX_EVAL_ATTEMPTS
    st = json.loads((kx.dir / "cross_status" / "registered-gtm.json").read_text())
    assert st["done"] is False
    assert st["pairs"]["42-73"]["failures"] == q.MAX_EVAL_ATTEMPTS
    assert st["pairs"]["42-73"]["done"] is False
    assert all(p["done"] for k, p in st["pairs"].items() if k != "42-73")


def test_cross_probe_untrained_checkpoint_passed_as_none(tmp_path):
    kx, calls = _full_kx(tmp_path)
    victim = q.RunSpec("dial", 101, "closed")
    kx.pin({})
    st = kx.store.load(victim)  # kxkm run: training failure, no finite checkpoint
    st.update(status=q.TRAINING_FAILURE, last_checkpoint=None, eval_done=True,
              archived=True)
    kx.store.save(st)
    kx.run()
    hits = [c for c in calls["cross"] if c[2:4] == ("dial", "closed")]
    by = {(c[4], c[5]): c for c in hits}
    assert by[(101, 131)][0] is None and by[(101, 131)][1] is not None
    assert by[(73, 101)][1] is None
    rec = json.loads((kx.dir / "cross" / "closed-dial-s101-s131.json").read_text())
    assert rec["untrained"] is True
