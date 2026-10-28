"""Preparation of the fix-C freeze (freeze plan C1, C2, C5, C8; PI
decisions 2026-10-28): registration-driven freeze writer that never touches
the files frozen by n10c-freeze-20261024, the pre-freeze registered
pre-training gate and ledger kind, trunk pinning. No MLX, no GPU."""
import importlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.procgen_multiagent_gtm_comm.campaign import freeze_prepare as fp
from experiments.procgen_multiagent_gtm_comm.campaign import guards, pin_trunks
from experiments.procgen_multiagent_gtm_comm.harness import devledger
from tests.unit.n10c import _trunks

REPO = Path(__file__).resolve().parents[3]
TAG = "n10c-freeze-20261101"


# -- C1 / C2: registration-driven freeze writer --------------------------------
def test_n10c_registration_points_at_the_new_amendment_and_manifest():
    reg = guards.registration("n10c")
    assert reg.amendment_path.endswith("amendment-2026-10-25.md")
    assert reg.manifest_name == "stage2-manifest-fixc.md"
    assert guards.AMENDMENT_PATH_20261023 in guards.FROZEN_PATHS
    assert guards.MANIFEST_20261024 in guards.FROZEN_PATHS
    for rel in guards.FROZEN_PATHS:  # both exist in the repository
        assert (REPO / rel).is_file(), rel
    assert reg.manifest_name not in {Path(p).name for p in guards.FROZEN_PATHS}


def _repo_copy(tmp_path):
    repo = tmp_path / "repo"
    for rel in (guards.AMENDMENT_PATH, *guards.FROZEN_PATHS):
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO / rel).read_bytes())
    return repo


def _fake_manifest(out, tag):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# M\n**Status: FROZEN (tag {tag})**\n")


def test_freeze_prepare_writes_only_the_registration_files(tmp_path):
    repo = _repo_copy(tmp_path)
    frozen = {rel: (repo / rel).read_bytes() for rel in guards.FROZEN_PATHS}
    res = fp.prepare(repo, TAG, write_manifest=_fake_manifest)
    assert res["manifest"] == (guards.MANIFEST_DIR / "stage2-manifest-fixc.md").as_posix()
    amend = (repo / guards.AMENDMENT_PATH).read_text()
    assert guards.amendment_status_problem(amend, TAG) is None
    for rel, data in frozen.items():  # byte-identical
        assert (repo / rel).read_bytes() == data, rel


def test_freeze_prepare_refuses_frozen_paths_and_foreign_tags(tmp_path):
    repo = _repo_copy(tmp_path)
    with pytest.raises(ValueError, match="frozen by an earlier tag"):
        fp.prepare(repo, TAG, "stage2-manifest.md", write_manifest=_fake_manifest)
    with pytest.raises(ValueError, match="invalid freeze tag"):
        fp.prepare(repo, "n10c-bis-freeze-20261101", write_manifest=_fake_manifest)
    with pytest.raises(ValueError, match="no stage-2 manifest writer"):
        fp.prepare(repo, "n10c-bis-freeze-20261101", registration="n10c-bis")


# -- C5: pre-freeze registered pre-training gate --------------------------------
def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def pushed_repo(tmp_path, monkeypatch):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    repo = tmp_path / "w"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    led = repo / guards.MANIFEST_DIR / "smoke_ledger.json"
    led.parent.mkdir(parents=True)
    led.write_text("{}")
    (repo / "f.txt").write_text("x")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "c")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")
    monkeypatch.setattr(guards, "_runtime_probe", lambda: dict(zip(
        ("system", "machine", "backend"), guards.HOST_RUNTIME["kxkm"], strict=True)))
    return repo


def test_prefreeze_gate_accepts_the_registered_procedure(pushed_repo):
    info = guards.check_prefreeze_pretrain(seed=73, total_steps=10**6,
                                           has_env=False, repo=pushed_repo)
    assert info["host"] == "kxkm" and len(info["commit"]) == 40
    assert info["registered_preset"] == "fix-c-coin-pretrain"
    unit = guards.check_prefreeze_pretrain(seed=101, total_steps=10**6,
                                           has_env=False, unit=("gtm", "closed"),
                                           repo=pushed_repo)
    assert unit["unit"] == ["gtm", "closed"]
    # the development ledger written by these runs does not make it dirty
    led = pushed_repo / guards.MANIFEST_DIR / "smoke_ledger.json"
    led.write_text('{"runs": []}')
    (led.parent / "smoke_ledger.json.lock").write_text("")
    guards.check_prefreeze_pretrain(seed=73, total_steps=10**6, has_env=False,
                                    repo=pushed_repo)


@pytest.mark.parametrize("kw,needle", [
    (dict(seed=9101), "registered seeds"),
    (dict(total_steps=999_999), "budget"),
    (dict(has_env=True), "real Procgen"),
    (dict(seed=73, unit=("gtm", "registered")), "control seed"),
    (dict(seed=101, unit=("mlp", "closed")), "not an N10-C run"),
])
def test_prefreeze_gate_refusals(pushed_repo, kw, needle):
    base = dict(seed=73, total_steps=10**6, has_env=False, repo=pushed_repo)
    with pytest.raises(guards.GuardError, match=needle):
        guards.check_prefreeze_pretrain(**(base | kw))


def test_prefreeze_gate_needs_a_clean_pushed_tree_and_the_seed_host(pushed_repo,
                                                                     monkeypatch):
    (pushed_repo / "new.txt").write_text("dirty")
    with pytest.raises(guards.GuardError, match="dirty"):
        guards.check_prefreeze_pretrain(seed=73, total_steps=10**6,
                                        has_env=False, repo=pushed_repo)
    _git(pushed_repo, "add", ".")
    _git(pushed_repo, "commit", "-q", "-m", "local only")
    with pytest.raises(guards.GuardError, match="not on a branch of origin"):
        guards.check_prefreeze_pretrain(seed=73, total_steps=10**6,
                                        has_env=False, repo=pushed_repo)
    _git(pushed_repo, "push", "-q", "origin", "main")
    _git(pushed_repo, "fetch", "-q", "origin")
    with pytest.raises(guards.GuardError, match="runtime"):
        guards.check_prefreeze_pretrain(seed=0, total_steps=10**6,  # macM1 seed
                                        has_env=False, repo=pushed_repo)
    monkeypatch.setattr(guards, "REGISTERED_TRAIN_OPTIONS", "frozen-20261024")
    with pytest.raises(guards.GuardError, match="no pre-trained trunk"):
        guards.check_prefreeze_pretrain(seed=73, total_steps=10**6,
                                        has_env=False, repo=pushed_repo)


def test_prefreeze_ledger_one_usable_trunk_and_one_restart(tmp_path):
    led = tmp_path / "ledger.json"
    kw = dict(kind=devledger.PREFREEZE_KIND, path=led, seed=73)
    r1 = devledger.reserve("trunk", "single", 10**6, **kw)
    devledger.finish(r1, "liveness_abort")  # unusable
    r2 = devledger.reserve("trunk", "single", 10**6, **kw)  # the one restart
    devledger.finish(r2, "liveness_abort")
    with pytest.raises(devledger.SmokeLimitError, match="the PI decides"):
        devledger.reserve("trunk", "single", 10**6, **kw)
    # another seed or unit is independent; a usable trunk blocks any rerun
    u = dict(kind=devledger.PREFREEZE_KIND, path=led, seed=101)
    r3 = devledger.reserve("trunk", "single", 10**6, unit=("gtm", "closed"), **u)
    devledger.finish(r3, "completed")
    with pytest.raises(devledger.SmokeLimitError, match="usable trunk exists"):
        devledger.reserve("trunk", "single", 10**6, unit=["gtm", "closed"], **u)
    devledger.reserve("trunk", "single", 10**6, **u)  # shared trunk of 101
    assert devledger.read(led)["runs"][2]["unit"] == ["gtm", "closed"]
    with pytest.raises(ValueError):
        devledger.reserve("gtm", "registered", 10**6, kind=devledger.PREFREEZE_KIND,
                          path=led, seed=73)


# -- trunk pinning (phase M, step 3) -------------------------------------------
def test_pin_trunks_copies_checks_and_writes_the_pins(tmp_path, monkeypatch):
    src = tmp_path / "src"
    shas = _trunks.make_all(src)
    repo = tmp_path / "repo"
    pins_file = repo / pin_trunks.PINS_REL
    pins_file.parent.mkdir(parents=True)
    pins_file.write_text((REPO / pin_trunks.PINS_REL).read_text())
    first = sorted(src.iterdir())[:3]
    res = pin_trunks.pin(first, repo)
    assert len(res["missing"]) == 13
    res = pin_trunks.pin(sorted(src.iterdir()), repo)  # idempotent, complete
    assert res["missing"] == []
    root = repo / guards.MANIFEST_DIR / "trunks"
    for n, h in shas.items():
        assert json.loads((root / n / "pretrain.json").read_text())["trunk_sha256"] == h
    spec = importlib.util.spec_from_file_location("pins", pins_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SHARED == {s: shas[f"seed-{s}"] for s in guards.SEEDS}
    assert len(mod.CONTROL) == 10
    assert mod.CONTROL["closed/gtm"] == shas["seed-101-closed-gtm"]


@pytest.mark.parametrize("bad", [
    dict(usable=False), dict(host="macm1"),
    dict(settings={"micro_batch_pairs": None, "clear_cache": False})])
def test_pin_trunks_rejects_non_registered_records(tmp_path, bad):
    d = tmp_path / "seed-73"
    _trunks.make(d, 73, **bad)
    with pytest.raises(ValueError):
        pin_trunks.check_record(d)


def test_pin_trunks_rejects_a_changed_file(tmp_path):
    d = tmp_path / "seed-73"
    _trunks.make(d, 73)
    (d / "trunk.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA-256"):
        pin_trunks.check_record(d)


def test_guards_read_the_committed_pins():
    from experiments.procgen_multiagent_gtm_comm.campaign import trunk_pins
    assert guards.REGISTERED_TRUNK_SHA256 == dict(trunk_pins.SHARED)
    assert len(guards.REGISTERED_CONTROL_TRUNK_SHA256) == len(trunk_pins.CONTROL)


def test_kxkm_lanes_parse():
    for f in ("gpu_wait.sh", "pretrain_lane.sh", "gate_bis_lane.sh"):
        p = REPO / "experiments/procgen_multiagent_gtm_comm/campaign/kxkm" / f
        subprocess.run(["bash", "-n", str(p)], check=True)
