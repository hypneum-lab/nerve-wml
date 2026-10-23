"""Campaign tests: fake train/eval functions, no mlx / procgen needed."""
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from experiments.procgen_multiagent_gtm_comm.campaign import queue as q
from experiments.procgen_multiagent_gtm_comm.campaign import run_campaign as rc

TOTAL = 64


def test_queue_is_the_frozen_50():
    runs = q.build_queue()
    assert len(runs) == 50 and len({r.run_id for r in runs}) == 50
    reg = [r for r in runs if r.condition == "registered"]
    closed = [r for r in runs if r.condition == "closed"]
    assert len(reg) == 30 and len(closed) == 20
    assert {(r.arm, r.seed) for r in reg} == {
        (a, s) for a in ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
        for s in (0, 17, 42, 73, 101)}
    assert {(r.arm, r.seed) for r in closed} == {
        (a, s) for a in ("gtm", "dial", "eng", "nocomm")
        for s in (0, 17, 42, 73, 101)}


def test_queue_order_deterministic_and_balanced():
    assert q.build_queue() == q.build_queue()
    runs = q.build_queue()
    first10 = runs[:10]  # one complete seed block
    assert {r.seed for r in first10} == {0}
    assert sum(r.condition == "closed" for r in first10) == 4
    # arm rotation: the first run is not the same arm for every seed
    firsts = [runs[i * 10].arm for i in range(5)]
    assert len(set(firsts)) > 1


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


def _campaign(tmp_path, train, eval_fn=None, gpu=None, archive=None,
              ckpts=True, mkdir=True, **o):
    calls = {"train": [], "eval": []}

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

    c = rc.Campaign(rc.Options(campaign_dir=tmp_path / "c", total_pair_steps=TOTAL,
                               min_free_gb=8, **o),
                    train_fn=t, eval_fn=e, gpu_fn=gpu or (lambda: None),
                    archive_fn=archive, log=lambda *_: None)
    (tmp_path / "c").mkdir(exist_ok=True)
    return c, calls


def test_full_campaign_runs_all_and_resume_skips_done(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec())
    assert c.run() == rc.EXIT_OK
    assert len(calls["train"]) == 50 and len(calls["eval"]) == 50
    assert all(c.store.load(s)["status"] == q.DONE for s in q.build_queue())
    c2, calls2 = _campaign(tmp_path, lambda n, kw: _rec())
    assert c2.run() == rc.EXIT_OK
    assert calls2["train"] == [] and calls2["eval"] == []


def test_resume_after_interruption_restarts_running_from_scratch(tmp_path):
    first = q.build_queue()[0]
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
    if kw["arm"] == "gtm" and kw["seed"] == 0 and kw["condition"] == "registered":
        raise OSError("disk")
    return _rec()


def test_infra_restart_capped_at_two_then_training_failure(tmp_path):
    c, calls = _campaign(tmp_path, _boom_first)
    code = c.run()
    first = q.build_queue()[0]
    assert calls["train"].count("gtm/registered/0") == 3  # 1 + 2 restarts
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
    s = c.store.load(q.build_queue()[0])
    assert s["status"] == q.TRAINING_FAILURE and s["eval_done"]


def test_tmp_checkpoint_never_picked(tmp_path):
    def t(n, kw):
        for name in ("ckpt_100.safetensors", "ckpt_200.tmp.safetensors"):
            (kw["out_dir"] / name).write_text("x")
        if kw["arm"] == "gtm" and kw["seed"] == 0 and kw["condition"] == "registered":
            raise OSError("disk")
        return _rec()
    c, calls = _campaign(tmp_path, t, ckpts=False)
    c.run()
    assert calls["eval"][0][0].name == "ckpt_100.safetensors"
    files = rc.archive_files(c.run_dir(q.build_queue()[0]), None)
    assert not any("tmp" in f.name for f in files)


def test_exhausted_before_run_dir_created_still_evaluates(tmp_path):
    c, calls = _campaign(tmp_path, _boom_first, ckpts=False, mkdir=False)
    assert c.run() == rc.EXIT_OK
    s = c.store.load(q.build_queue()[0])
    assert s["eval_done"] and s["eval_failures"] == 0
    assert calls["eval"][0][0] is None
    assert (c.run_dir(q.build_queue()[0]) / "eval.json").exists()


def test_exhausted_checkpoint_found_in_earlier_failed_dir(tmp_path):
    made = {"n": 0}

    def t(n, kw):
        made["n"] += 1
        if kw["arm"] == "gtm" and kw["seed"] == 0 and kw["condition"] == "registered":
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
    assert c.store.load(q.build_queue()[0])["infra_failures"] == 1


def test_training_failure_not_retried_and_evaluated_untrained(tmp_path):
    def tf(n, kw):
        if n == 1:
            return _rec(training_failure=True, pair_steps_done=8,
                        last_checkpoint=None, failure_reason="nan")
        return _rec()
    c, calls = _campaign(tmp_path, tf)
    assert c.run() == rc.EXIT_OK
    first = q.build_queue()[0]
    s = c.store.load(first)
    assert s["status"] == q.TRAINING_FAILURE and s["eval_done"]
    assert calls["train"].count("gtm/registered/0") == 1
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
    assert calls["train"].count("gtm/registered/0") == 1
    assert c.store.load(q.build_queue()[0])["eval_failures"] == 1


def test_gpu_guard_stops_without_state_change(tmp_path):
    c, calls = _campaign(tmp_path, lambda n, kw: _rec(), gpu=lambda: 2.0)
    assert c.run() == rc.EXIT_GPU
    assert calls["train"] == []
    assert c.store.load(q.build_queue()[0])["status"] == q.PENDING


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
    assert len(got) == 50
    assert not c.store.load(q.build_queue()[0])["archived"]
    assert c.store.load(q.build_queue()[1])["archived"]
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
    c.store.save(c.store.load(q.build_queue()[0]))
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
    assert "50 runs" in out and "WOULD REFUSE" in out and "Estimated duration" in out
    assert not (tmp_path / "c").exists()
    e = rc.estimate(json.loads(thr.read_text()))
    # 10 gtm runs at 50M/1000/3600+600/3600 h, 40 others at nocomm rate
    gtm_h = 50e6 / 1000 / 3600 + 600 / 3600
    oth_h = 50e6 / 2000 / 3600 + 300 / 3600
    assert e["total_hours"] == pytest.approx(10 * gtm_h + 40 * oth_h)


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
