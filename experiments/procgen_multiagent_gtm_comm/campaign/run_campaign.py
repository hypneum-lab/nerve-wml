"""Sequential N10-C campaign runner (option D, 60 runs, one at a time per host).

    python -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign \
        --campaign-dir /ai/saisail/n10c-run/campaign --host {macm1,kxkm} \
        --freeze-tag n10c-freeze-YYYYMMDD --swhid swh:1:rev:<40 hex> \
        [--manifest PATH --manifest-sha256 SHA --expect-commit COMMIT] \
        [--manifest-doi 10.17605/OSF.IO/ID]   (optional metadata only) \
        [--archive-to user@host:/path] [--dry-run --throughput-json t.json]

    python -m ...run_campaign sync --campaign-dir D --to user@kxkm:CAMPAIGN_DIR

Hosts: macM1 trains seeds 0, 17, 42; KXKM trains 73, 101, 131. Evaluation
runs ONLY with --host kxkm (x86 Procgen build); macM1 runs end at status
"trained" and are evaluated on KXKM after `sync` copied their checkpoints.

Hard guards (campaign/guards.py, shared with the harness CLIs and the
execution boundary of ``train_run`` / ``evaluate_*``; all must pass, else
exit 2 before any training): the annotated freeze tag
``n10c-freeze-YYYYMMDD`` on origin is MANDATORY (a DOI is never required
and never a substitute); manifest and amendment are read from the peeled
tag commit and the tagged manifest hash must equal the tag message; a
Software Heritage id is required; git tree clean; HEAD == tag commit or a
descendant whose every intervening commit only touched docs/ or the
manifest folder. The tag, commit, SWHID and the GitHub push time (if
obtainable) are pinned in campaign.json at first start, with the budget
and trial flag. Before every run on Linux: nvidia-smi free memory >=
``--min-free-gb``. Other processes are never signalled or killed. After 3
infrastructure failures a run becomes a training failure (amendment:
evaluated at the last finite checkpoint, else at initialisation; never
replaced).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from ..harness.codes import HOST_SETTINGS
from . import queue as q
from .guards import (  # noqa: F401 - re-exported for callers and tests
    DEV_MAX_PAIR_STEPS,
    HEX40,
    REPO,
    SWHID_RE,
    GuardError,
    RegistrationContext,
    freeze_tag_info,
    guard_problems,
    issue_context,
    normalize_doi,
)

EXIT_OK, EXIT_ERROR, EXIT_GUARD, EXIT_GPU, EXIT_STOP, EXIT_PROBLEMS = 0, 1, 2, 3, 4, 5


def gpu_free_gb(runner: Callable[..., Any] = subprocess.run) -> float | None:
    """Free GPU memory in GiB via nvidia-smi; None when not applicable (non-Linux).

    On Linux a missing / failing nvidia-smi raises GuardError (refuse).
    """
    if platform.system() != "Linux":
        return None
    try:
        r = runner(["nvidia-smi", "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GuardError(f"nvidia-smi unavailable: {exc}") from exc
    if r.returncode != 0:
        raise GuardError(f"nvidia-smi failed: {r.stderr.strip()}")
    vals = [float(x) for x in r.stdout.split() if x.strip()]
    if not vals:
        raise GuardError("nvidia-smi returned no GPU")
    return vals[0] / 1024.0  # MiB -> GiB (GPU 0)


def estimate(throughput: dict, total: int = q.TOTAL_PAIR_STEPS,
             host: str | None = None) -> dict:
    """Duration estimate: gtm runs use the gtm rate, others the nocomm proxy
    (same convention as the manifest estimate); one 2x200 evaluation each."""
    res = throughput["results"]

    def hours(arm: str) -> float:
        r = res.get(arm) or res["nocomm"]
        return (total / r["pair_steps_per_s"]
                + r["eval_proxy"]["seconds_per_model_2x200"]) / 3600

    per_run = {s.run_id: hours(s.arm) for s in (q.runs_for_host(host) if host else q.build_queue())}
    tot = sum(per_run.values())
    return {"per_run_hours": per_run, "total_hours": tot, "total_days": tot / 24}


@dataclass
class Options:
    campaign_dir: Path
    host: str
    total_pair_steps: int = q.TOTAL_PAIR_STEPS
    min_free_gb: float = 8.0
    gpu_wait_s: float = 0.0
    archive_to: str | None = None
    max_concurrent: int = 1
    trial: bool = False  # non-registered budget; never the real campaign
    train_kwargs: dict | None = None  # test / smoke overrides (n_pairs, ...)
    env_factory: Callable[[int], Any] | None = None
    # context issued by campaign.guards (mandatory unless ``trial``)
    registration: RegistrationContext | None = None


def _default_train(registration=None, **kw):
    from experiments.procgen_multiagent_gtm_comm.harness.train import train_run
    return train_run(registration=registration, **kw)


def _default_eval(ckpt, arm, condition, seed, env_factory=None, registration=None):
    from experiments.procgen_multiagent_gtm_comm.harness.evaluate import (
        HELD_OUT,
        evaluate_model,
    )
    return evaluate_model(ckpt, arm, condition, HELD_OUT, seed=seed,
                          env_factory=env_factory, registration=registration)


def _default_cross(ckpt0, ckpt1, arm, condition, seed0, seed1, env_factory=None,
                   registration=None):
    from experiments.procgen_multiagent_gtm_comm.harness.evaluate import (
        HELD_OUT,
        cross_play_model,
    )
    return cross_play_model(ckpt0, ckpt1, arm, condition, seed0, seed1, HELD_OUT,
                            env_factory=env_factory, registration=registration)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


INFO_PINS = ("tag_push_time", "tag_push_time_note", "manifest_doi")  # fill-in only


class Campaign:
    def __init__(self, opts: Options, *, train_fn=None, eval_fn=None,
                 cross_fn=None, gpu_fn=None, archive_fn=None, log=print) -> None:
        if opts.host not in q.HOSTS:
            raise GuardError(f"unknown host {opts.host!r}; expected one of {q.HOSTS}")
        if not opts.trial:
            over = sorted(set(opts.train_kwargs or {}) & set(HOST_SETTINGS[opts.host]))
            if over:
                raise GuardError(
                    f"registered runs refuse overrides of the per-host "
                    f"implementation settings {over} (codes.HOST_SETTINGS)")
        if not opts.trial and opts.registration is None:
            raise GuardError(
                "a registered campaign needs the RegistrationContext issued by "
                "the freeze-tag guards (campaign.guards.require_guards)")
        if opts.trial and opts.total_pair_steps > DEV_MAX_PAIR_STEPS:
            raise GuardError(
                f"a non-registered trial is limited to {DEV_MAX_PAIR_STEPS:,} "
                "pair-steps (development mode)")
        self.o = opts
        self.dir = Path(opts.campaign_dir)
        self.store = q.StatusStore(self.dir / "status")
        reg = opts.registration
        self.train_fn = train_fn or partial(_default_train, registration=reg)
        self.eval_fn = eval_fn or partial(_default_eval, registration=reg)
        self.cross_fn = cross_fn or partial(_default_cross, registration=reg)
        self.cross_store = q.CrossStore(self.dir / "cross_status")
        self.gpu_fn = gpu_fn or gpu_free_gb
        self.archive_fn = archive_fn or rsync_archive
        self.log = log
        self._pinned = False

    # -- logging ----------------------------------------------------
    def event(self, **kw) -> None:
        kw = {"at": _now(), **kw}
        with open(self.dir / "attempts.jsonl", "a") as f:
            f.write(json.dumps(kw) + "\n")
        self.log(json.dumps(kw))

    # -- pinning ----------------------------------------------------
    def pin(self, pins: dict) -> None:
        """Record guard inputs, budget and trial flag; refuse other inputs."""
        pins = {**pins, "total_pair_steps": self.o.total_pair_steps,
                "trial": self.o.trial, "host": self.o.host}
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "campaign.json"
        if p.exists():
            old = json.loads(p.read_text())
            diff = {k: (old.get(k), v) for k, v in pins.items()
                    if old.get(k) != v and k not in INFO_PINS}
            if diff:
                raise GuardError(
                    f"campaign pinned to other inputs (trial/registered "
                    f"cannot be mixed, nor budget changed): {diff}")
            late = {k: pins[k] for k in INFO_PINS
                    if pins.get(k) and not old.get(k)}
            if late:  # e.g. the GitHub push time became obtainable later
                if "tag_push_time" in late:
                    late["tag_push_time_note"] = None
                p.write_text(json.dumps({**old, **late}, indent=2))
        else:
            own = [f for f in (self.dir / "status").glob("*.json")
                   if "synced_from" not in json.loads(f.read_text())]
            if own:  # status files synced in from macM1 are not own state
                raise GuardError("unpinned campaign dir already holds run state")
            p.write_text(json.dumps({**pins, "created": _now()}, indent=2))
        self._pinned = True

    # -- gpu --------------------------------------------------------
    def gpu_ok(self) -> tuple[bool, float | None]:
        deadline = time.time() + self.o.gpu_wait_s
        while True:
            free = self.gpu_fn()
            if free is None or free >= self.o.min_free_gb:
                return True, free
            if time.time() >= deadline:
                return False, free
            time.sleep(min(30.0, max(1.0, deadline - time.time())))

    # -- one run ----------------------------------------------------
    def run_dir(self, spec: q.RunSpec) -> Path:
        return self.dir / "runs" / spec.run_id

    def train(self, spec: q.RunSpec, st: dict) -> None:
        attempt = len(st["attempts"]) + 1
        rd = self.run_dir(spec)
        if rd.exists():  # restart from scratch; keep the failed partial aside
            aside = rd.with_name(f"{spec.run_id}.failed-{st['infra_failures']}")
            shutil.move(str(rd), str(aside))
        st.update(status=q.RUNNING, started_at=_now())
        self.store.save(st)
        self.event(run=spec.run_id, attempt=attempt, phase="train", state="start")
        # registered per-host implementation settings (single source:
        # codes.HOST_SETTINGS); only a non-registered trial may override them
        settings = dict(HOST_SETTINGS[self.o.host])
        kw = {**settings, **(self.o.train_kwargs or {})}
        st["train_settings"] = {k: kw[k] for k in settings}
        if self.o.env_factory is not None:
            kw["env_factory"] = self.o.env_factory
        try:
            rec = self.train_fn(
                arm=spec.arm, seed=spec.seed, condition=spec.condition,
                total_pair_steps=self.o.total_pair_steps, out_dir=rd,
                restarts=st["infra_failures"], **kw)
            if not rec.training_failure and rec.pair_steps_done != self.o.total_pair_steps:
                raise RuntimeError(
                    f"incomplete run: {rec.pair_steps_done}/{self.o.total_pair_steps}")
        except Exception as exc:  # noqa: BLE001 - any crash is infrastructure
            q.fail_attempt(st, f"{exc.__class__.__name__}: {exc}")
            self.store.save(st)
            self.event(run=spec.run_id, attempt=attempt, phase="train",
                       state="infra_failed", reason=st["failure_reason"],
                       infra_failures=st["infra_failures"])
            return
        st["last_checkpoint"] = rec.last_checkpoint
        st["last_checkpoint_sha256"] = checkpoint_sha256(rd, rec.last_checkpoint)
        st["finished_at"] = _now()
        if rec.training_failure:
            st.update(status=q.TRAINING_FAILURE, failure_reason=rec.failure_reason)
            outcome = "training_failure"
        else:
            st.update(status=q.TRAINED, failure_reason=None)
            outcome = "trained"
        st["attempts"].append({"n": attempt, "outcome": outcome, "at": _now(),
                               "wall_time_s": rec.wall_time_s,
                               "pair_steps_done": rec.pair_steps_done})
        self.store.save(st)
        self.event(run=spec.run_id, attempt=attempt, phase="train", state=outcome,
                   wall_time_s=rec.wall_time_s)

    def evaluate(self, spec: q.RunSpec, st: dict) -> None:
        if self.o.host != "kxkm":
            raise GuardError("evaluation runs only with --host kxkm "
                             "(x86 Procgen build); copy checkpoints with `sync`")
        rd = self.run_dir(spec)
        ckpt = rd / st["last_checkpoint"] if st["last_checkpoint"] else None
        self.event(run=spec.run_id, phase="eval", state="start",
                   ckpt=st["last_checkpoint"])
        try:  # held-out levels are touched only here, after training
            verify_checkpoint(rd, st["last_checkpoint"], st.get("last_checkpoint_sha256"))
            rec = self.eval_fn(ckpt, spec.arm, spec.condition, spec.seed,
                               env_factory=self.o.env_factory)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "eval.json").write_text(rec.to_json())
        except Exception as exc:  # noqa: BLE001
            st["eval_failures"] += 1
            self.store.save(st)
            self.event(run=spec.run_id, phase="eval", state="failed",
                       reason=f"{exc.__class__.__name__}: {exc}",
                       eval_failures=st["eval_failures"])
            return
        st["eval_done"] = True
        if st["status"] == q.TRAINED:
            st["status"] = q.DONE
        self.store.save(st)
        self.event(run=spec.run_id, phase="eval", state="done", S=rec.S,
                   R_intact=rec.R_intact)

    def cross_ckpt(self, arm: str, condition: str, seed: int) -> Path | None:
        """Checkpoint of one seed's model, None = untrained init."""
        spec = q.RunSpec(arm, seed, condition)
        ck = self.store.load(spec)["last_checkpoint"]
        return self.run_dir(spec) / ck if ck else None

    def cross_cell_ready(self, arm: str, condition: str) -> bool:
        """All seeds of the cell exist, are trained or failed, and evaluated."""
        for seed in q.SEEDS:
            spec = q.RunSpec(arm, seed, condition)
            if not self.store.path(spec).exists():
                return False
            st = self.store.load(spec)
            if st["status"] not in (q.TRAINED, q.TRAINING_FAILURE, q.DONE) \
                    or not st["eval_done"]:
                return False
        return True

    def cross_pass(self) -> bool:
        """Cross-seed cross-play probe for every ready (arm, condition) cell.

        KXKM only (x86 Procgen build). Resumable: a pair marked done is never
        repeated; a failed pair is retried up to MAX_EVAL_ATTEMPTS in total.
        Returns True when every ready cell is complete.
        """
        if self.o.host != "kxkm":
            raise GuardError("the cross-play probe runs only with --host kxkm "
                             "(x86 Procgen build)")
        complete = True
        out = self.dir / "cross"
        for arm, condition in q.cross_cells():
            if not self.cross_cell_ready(arm, condition):
                continue
            cs = self.cross_store.load(arm, condition)
            for pid, pair in cs["pairs"].items():
                s0, s1 = pair["seed_role0"], pair["seed_role1"]
                while q.cross_pair_pending(pair):
                    ck0 = self.cross_ckpt(arm, condition, s0)
                    ck1 = self.cross_ckpt(arm, condition, s1)
                    self.event(phase="cross", cell=cs["cell"], pair=pid, state="start")
                    try:
                        rec = self.cross_fn(ck0, ck1, arm, condition, s0, s1,
                                            env_factory=self.o.env_factory)
                        out.mkdir(parents=True, exist_ok=True)
                        (out / f"{cs['cell']}-s{s0}-s{s1}.json").write_text(rec.to_json())
                    except Exception as exc:  # noqa: BLE001
                        pair["failures"] += 1
                        self.cross_store.save(cs)
                        self.event(phase="cross", cell=cs["cell"], pair=pid,
                                   state="failed", failures=pair["failures"],
                                   reason=f"{exc.__class__.__name__}: {exc}")
                        continue
                    pair.update(done=True, R_cross=float(rec.R_cross),
                                untrained=ck0 is None or ck1 is None,
                                ckpt_role0=None if ck0 is None else str(ck0.name),
                                ckpt_role1=None if ck1 is None else str(ck1.name))
                    self.cross_store.save(cs)
                    self.event(phase="cross", cell=cs["cell"], pair=pid, state="done",
                               R_cross=pair["R_cross"])
            cs["done"] = all(p["done"] for p in cs["pairs"].values())
            self.cross_store.save(cs)
            complete = complete and cs["done"]
        return complete

    def archive(self, spec: q.RunSpec, st: dict) -> None:
        if not self.o.archive_to:
            if self.o.host == "kxkm":  # KXKM cannot reach electron-server:
                st["archive_via"] = "pull"  # pull_archive.sh runs there
                self.store.save(st)
            return
        rd = self.run_dir(spec)
        try:
            files = archive_files(rd, st["last_checkpoint"])
            self.archive_fn(rd, files, self.o.archive_to, spec.run_id)
        except Exception as exc:  # noqa: BLE001 - never blocks the campaign
            self.event(run=spec.run_id, phase="archive", state="failed",
                       reason=f"{exc.__class__.__name__}: {exc}")
            return
        st["archived"] = True
        self.store.save(st)
        self.event(run=spec.run_id, phase="archive", state="done")

    def find_last_checkpoint(self, spec: q.RunSpec) -> str | None:
        """Newest finite checkpoint of the latest attempt that wrote one,
        as a path relative to the run directory (current dir, then the
        ``.failed-<n>`` dirs from the most recent attempt back)."""
        rd = self.run_dir(spec)
        fails = sorted(rd.parent.glob(f"{spec.run_id}.failed-*"),
                       key=lambda d: int(d.name.rsplit("-", 1)[1]), reverse=True)
        for d in [rd, *fails]:
            cks = completed_ckpts(d)
            if cks:
                return cks[-1].name if d == rd else f"../{d.name}/{cks[-1].name}"
        return None

    def finalize_exhausted(self, spec: q.RunSpec, st: dict) -> None:
        """3 infra failures = training failure (amendment): keep the run,
        evaluate its last finite checkpoint, else the initialisation."""
        if not q.infra_exhausted(st):
            return
        st.update(
            status=q.TRAINING_FAILURE, infra_exhausted=True,
            last_checkpoint=self.find_last_checkpoint(spec),
            last_checkpoint_sha256=checkpoint_sha256(
                self.run_dir(spec), self.find_last_checkpoint(spec)),
            failure_reason=("still failing after 2 restarts: "
                            + str(st["failure_reason"])),
            finished_at=_now())
        self.store.save(st)
        self.event(run=spec.run_id, phase="train", state="exhausted_to_training_"
                   "failure", ckpt=st["last_checkpoint"])

    # -- the loop ---------------------------------------------------
    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    def run(self) -> int:
        queue = q.runs_for_host(self.o.host)
        if not self._pinned:
            reg = self.o.registration
            if reg is None and not self.o.trial:
                raise GuardError("a registered campaign cannot start without "
                                 "its registration context (provenance)")
            self.pin({} if reg is None else reg.pins())
        can_eval = self.o.host == "kxkm"
        for spec in q.build_queue():
            if self.stop_requested():
                self.event(phase="campaign", state="stopped_by_STOP_file")
                return EXIT_STOP
            if spec.host != self.o.host:
                # KXKM evaluates (never trains) macM1 runs once synced here
                if can_eval and self.store.path(spec).exists():
                    st = self.store.load(spec)
                    while q.needs_eval(st):
                        self.evaluate(spec, st)
                    if q.needs_archive(st):
                        self.archive(spec, st)
                continue
            st = self.store.recover_stale(spec)
            self.finalize_exhausted(spec, st)
            if q.needs_training(st):
                ok, free = self.gpu_ok()
                if not ok:
                    self.event(phase="campaign", state="gpu_guard_refused",
                               free_gb=free, min_free_gb=self.o.min_free_gb,
                               next_run=spec.run_id)
                    return EXIT_GPU
                while q.needs_training(st):  # infra restarts, from scratch
                    self.train(spec, st)
                    if st["status"] == q.INFRA_FAILED and q.needs_training(st):
                        ok, free = self.gpu_ok()
                        if not ok:
                            self.event(phase="campaign", state="gpu_guard_refused",
                                       free_gb=free, next_run=spec.run_id)
                            return EXIT_GPU
                self.finalize_exhausted(spec, st)
            while can_eval and q.needs_eval(st):
                self.evaluate(spec, st)
            if q.needs_archive(st):
                self.archive(spec, st)
        states = [self.store.load(s) for s in queue]
        awaiting = [s for s in q.runs_for_host("macm1")
                    if can_eval and not self.store.path(s).exists()]
        self.event(phase="campaign", state="finished", host=self.o.host,
                   summary=q.summarize(states), awaiting_sync=len(awaiting))
        problems = [s for s in states if can_eval
                    and s["status"] in (q.TRAINED, q.TRAINING_FAILURE)
                    and not s["eval_done"]]
        cross_ok = self.cross_pass() if can_eval else True
        return EXIT_PROBLEMS if problems or not cross_ok else EXIT_OK


CKPT_RE = re.compile(r"^ckpt_(\d+)\.safetensors$")


def completed_ckpts(d: Path) -> list[Path]:
    """Finished checkpoints only (never ``ckpt_<n>.tmp.safetensors``),
    ascending by step."""
    found = [(int(m.group(1)), f) for f in d.glob("ckpt_*.safetensors")
             if (m := CKPT_RE.match(f.name))]
    return [f for _, f in sorted(found)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(run_dir: Path, last_ckpt: str | None) -> Path | None:
    """The status-selected checkpoint as an existing file, resolving the
    ``../<run>.failed-N/ckpt_*.safetensors`` form of an exhausted run.
    None = untrained init; a recorded but missing file raises."""
    if not last_ckpt:
        return None
    p = (Path(run_dir) / last_ckpt).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"status-selected checkpoint {last_ckpt!r} not found at {p}")
    return p


def checkpoint_sha256(run_dir: Path, last_ckpt: str | None) -> str | None:
    try:
        p = resolve_checkpoint(run_dir, last_ckpt)
    except FileNotFoundError:
        return None
    return None if p is None else sha256_file(p)


def verify_checkpoint(run_dir: Path, last_ckpt: str | None, sha: str | None) -> None:
    """Raise unless the checkpoint on disk still has the recorded digest."""
    if not last_ckpt or not sha:
        return
    p = resolve_checkpoint(run_dir, last_ckpt)
    if p is not None and sha256_file(p) != sha:
        raise RuntimeError(f"checkpoint {last_ckpt} SHA-256 differs from the "
                           "recorded digest (corrupted copy?)")


def archive_name(run_dir: Path, f: Path) -> str:
    """Name of ``f`` in the archive: its basename inside the run directory;
    ``failed-<n>__<basename>`` for a file of a ``<run>.failed-<n>`` dir."""
    parent = f.resolve().parent
    if parent == Path(run_dir).resolve():
        return f.name
    return f"{parent.name.rsplit('.', 1)[-1]}__{f.name}"


def archive_files(run_dir: Path, last_ckpt: str | None) -> list[Path]:
    """run.json, eval.json, the newest finished checkpoint of the run
    directory and the STATUS-SELECTED checkpoint (which, for an exhausted
    run, lives in ``../<run>.failed-N/``). A recorded checkpoint that is
    missing raises instead of being silently dropped."""
    run_dir = Path(run_dir)
    files = [run_dir / "run.json", run_dir / "eval.json"]
    ckpts = completed_ckpts(run_dir)
    keep = {c.resolve() for c in ckpts[-1:]}
    sel = resolve_checkpoint(run_dir, last_ckpt)
    if sel is not None:
        keep.add(sel)
    files += sorted(keep)
    return [f for f in files if f.exists()]


def build_provenance(run_dir: Path, files: list[Path], run_id: str) -> dict:
    """Digests and provenance of what is archived for one run."""
    run_dir = Path(run_dir)
    camp = run_dir.parent.parent
    st_path = camp / "status" / f"{run_id}.json"
    st = json.loads(st_path.read_text()) if st_path.exists() else {}
    meta = {}
    if (camp / "campaign.json").exists():
        meta = json.loads((camp / "campaign.json").read_text())
    return {
        "run_id": run_id,
        "status": st.get("status"),
        "infra_exhausted": st.get("infra_exhausted", False),
        "status_last_checkpoint": st.get("last_checkpoint"),
        "recorded_checkpoint_sha256": st.get("last_checkpoint_sha256"),
        "freeze": {k: meta.get(k) for k in (
            "freeze_tag", "freeze_commit", "swhid", "manifest_sha256",
            "amendment_sha256")},
        "files": {archive_name(run_dir, f): {
            "sha256": sha256_file(f), "bytes": f.stat().st_size,
            "source": str(f.resolve().relative_to(run_dir.parent.resolve()))
            if f.resolve().is_relative_to(run_dir.parent.resolve()) else str(f)}
            for f in files},
    }


def rsync_archive(run_dir: Path, files: list[Path], target: str, run_id: str,
                  *, runner: Callable[..., Any] = subprocess.run) -> None:
    """Push one run to ``user@host:path/<run_id>/``.

    Files are staged under their archive names (a checkpoint of a
    ``.failed-N`` dir keeps its provenance in the name and in
    ``PROVENANCE.json``), sent with ``rsync -aL --checksum`` and their
    SHA-256 verified on the target; ``campaign.json`` and ``attempts.jsonl``
    go to ``<target>/_campaign/``.
    """
    import tempfile

    run_dir = Path(run_dir)
    camp = run_dir.parent.parent
    base = target.rstrip("/")
    prov = build_provenance(run_dir, files, run_id)
    rec = prov["recorded_checkpoint_sha256"]
    if rec and prov["status_last_checkpoint"]:
        name = archive_name(run_dir, resolve_checkpoint(
            run_dir, prov["status_last_checkpoint"]))
        if prov["files"][name]["sha256"] != rec:
            raise RuntimeError(f"checkpoint {name} SHA-256 differs from the "
                               "digest recorded at training time")
    ssh = ["-e", "ssh -o BatchMode=yes -o ControlPath=none"]
    with tempfile.TemporaryDirectory() as td:
        stage = Path(td)
        for f in files:
            (stage / archive_name(run_dir, f)).symlink_to(f.resolve())
        (stage / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
        r = runner(["rsync", "-aL", "--partial", "--checksum", *ssh,
                    f"{stage}/", f"{base}/{run_id}/"],
                   capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"rsync failed: {r.stderr.strip()}")
    host, _, path = base.partition(":")
    listing = "".join(f"{v['sha256']}  {k}\n" for k, v in prov["files"].items())
    v = runner(["ssh", "-o", "BatchMode=yes", "-o", "ControlPath=none", host,
                f"cd '{path}/{run_id}' && sha256sum -c -"],
               input=listing, capture_output=True, text=True, timeout=600)
    if v.returncode != 0:
        raise RuntimeError(f"remote SHA-256 verification failed: "
                           f"{(v.stdout + v.stderr).strip()}")
    extra = [camp / n for n in ("campaign.json", "attempts.jsonl")
             if (camp / n).exists()]
    if extra:
        r = runner(["rsync", "-a", *ssh, *[str(x) for x in extra], f"{base}/_campaign/"],
                   capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"rsync of campaign files failed: {r.stderr.strip()}")


DEFAULT_JUMP = "electron@100.78.191.52"  # electron-server


def rsync_stage(src: Path, target: str, jump: str) -> None:
    """rsync a staged directory (symlinks followed) to ``user@host:path``;
    never overwrites what already exists at the target."""
    e = f"ssh -o BatchMode=yes -o ControlPath=none -J {jump}"
    cmd = ["rsync", "-aL", "--partial", "--ignore-existing", "-e", e,
           f"{src}/", f"{target.rstrip('/')}/"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip()}")


def sync_runs(campaign_dir: Path, target: str, *, jump: str = DEFAULT_JUMP,
              rsync_fn: Callable[[Path, str, str], None] = rsync_stage) -> list[q.RunSpec]:
    """Copy trained macM1 runs to the KXKM campaign dir for evaluation.

    Per run, in two steps (so a status file never precedes its
    checkpoint): ``runs/<id>/`` (run.json + the last checkpoint, renamed to
    its basename), then ``status/<id>.json`` (marked ``synced_from``).
    Marks the local status ``synced`` only after both succeeded.
    """
    store = q.StatusStore(Path(campaign_dir) / "status")
    stage_root = Path(campaign_dir) / "sync-stage"
    done: list[q.RunSpec] = []
    for spec in q.runs_for_host("macm1"):
        if not store.path(spec).exists():
            continue
        st = store.load(spec)
        if st["status"] not in (q.TRAINED, q.TRAINING_FAILURE) or st.get("synced"):
            continue
        rd = Path(campaign_dir) / "runs" / spec.run_id
        st_runs = stage_root / f"{spec.run_id}.runs"
        st_stat = stage_root / f"{spec.run_id}.status"
        for d in (st_runs, st_stat):
            if d.exists():
                shutil.rmtree(d)
        out = st_runs / "runs" / spec.run_id
        out.mkdir(parents=True)
        (st_stat / "status").mkdir(parents=True)
        if (rd / "run.json").exists():
            (out / "run.json").symlink_to((rd / "run.json").resolve())
        ck = st["last_checkpoint"]
        remote = dict(st, synced_from="macm1", synced=False, eval_done=False,
                      eval_failures=0, archived=False)
        if ck:
            src = (rd / ck).resolve()
            (out / src.name).symlink_to(src)
            remote["last_checkpoint"] = src.name
        (st_stat / "status" / f"{spec.run_id}.json").write_text(
            json.dumps(remote, indent=2))
        try:  # checkpoint first, status file second
            rsync_fn(st_runs, target, jump)
            rsync_fn(st_stat, target, jump)
        except Exception as exc:  # noqa: BLE001 - retried at the next sync
            print(f"sync failed for {spec.run_id}: {exc.__class__.__name__}: {exc}",
                  file=sys.stderr)
            continue
        st["synced"] = True
        store.save(st)
        done.append(spec)
    return done


def sync_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="run_campaign sync",
        description="Copy trained macM1 runs to the KXKM campaign directory "
                    "(via the electron-server jump) so KXKM can evaluate them.")
    p.add_argument("--campaign-dir", type=Path, required=True)
    p.add_argument("--to", required=True,
                   help="user@kxkm:/path/to/KXKM/campaign-dir")
    p.add_argument("--jump", default=DEFAULT_JUMP)
    a = p.parse_args(argv)
    done = sync_runs(a.campaign_dir, a.to, jump=a.jump)
    print(f"synced {len(done)} run(s)")
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--campaign-dir", type=Path, required=True,
                   help="state + run outputs; must be OUTSIDE the git repo")
    p.add_argument("--host", choices=list(q.HOSTS),
                   help="run only this host's runs (required to train)")
    from .guards import add_guard_args

    add_guard_args(p)  # --manifest --manifest-sha256 --manifest-doi
    #                    --expect-commit --freeze-tag --swhid
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--gpu-wait-s", type=float, default=0.0)
    p.add_argument("--archive-to", help="user@host:path (rsync over ssh)")
    p.add_argument("--max-concurrent", type=int, default=1, choices=[1])
    p.add_argument("--throughput-json", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--total-pair-steps", type=int, default=q.TOTAL_PAIR_STEPS)
    p.add_argument("--non-registered-trial", action="store_true",
                   help="development trial: budget <= 1M pair-steps, fakes or "
                        "training levels only (never the real campaign)")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "sync":
        return sync_main(argv[1:])
    a = _parser().parse_args(argv)
    problems = guard_problems(
        repo=REPO, manifest=a.manifest, manifest_sha256=a.manifest_sha256,
        manifest_doi=a.manifest_doi, expect_commit=a.expect_commit,
        freeze_tag=a.freeze_tag, swhid=a.swhid)
    if not a.host and not a.dry_run:
        problems.append("--host {macm1,kxkm} is required")
    try:
        cdir = a.campaign_dir.resolve()
        if REPO in (cdir, *cdir.parents):
            problems.append("--campaign-dir is inside the git repo")
    except OSError as exc:
        problems.append(str(exc))
    if a.total_pair_steps != q.TOTAL_PAIR_STEPS and not a.non_registered_trial:
        problems.append("--total-pair-steps differs from the registered 50M")
    if a.non_registered_trial and a.total_pair_steps == q.TOTAL_PAIR_STEPS:
        problems.append("--non-registered-trial with the registered 50M budget")
    if a.non_registered_trial and a.total_pair_steps > DEV_MAX_PAIR_STEPS:
        problems.append(f"--non-registered-trial is limited to "
                        f"{DEV_MAX_PAIR_STEPS:,} pair-steps")
    if a.dry_run:
        return _dry_run(a, problems)
    if problems:
        print("REFUSED:\n- " + "\n- ".join(problems), file=sys.stderr)
        return EXIT_GUARD
    try:
        ctx = issue_context(
            repo=REPO, manifest=a.manifest, manifest_sha256=a.manifest_sha256,
            manifest_doi=a.manifest_doi, expect_commit=a.expect_commit,
            freeze_tag=a.freeze_tag, swhid=a.swhid)
    except GuardError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_GUARD
    opts = Options(campaign_dir=a.campaign_dir, host=a.host,
                   total_pair_steps=a.total_pair_steps,
                   min_free_gb=a.min_free_gb, gpu_wait_s=a.gpu_wait_s,
                   archive_to=a.archive_to, max_concurrent=a.max_concurrent,
                   trial=a.non_registered_trial, registration=ctx)
    camp = Campaign(opts)
    camp.dir.mkdir(parents=True, exist_ok=True)
    with open(camp.dir / "campaign.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("REFUSED: another campaign process holds the lock", file=sys.stderr)
            return EXIT_GUARD
        try:
            camp.pin(ctx.pins())
            return camp.run()
        except GuardError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT_GUARD


def _dry_run(a, problems: list[str]) -> int:
    queue = q.runs_for_host(a.host) if a.host else q.build_queue()
    store = q.StatusStore(a.campaign_dir / "status") if a.campaign_dir.exists() else None
    print(f"DRY RUN: nothing is trained. {len(queue)} runs, "
          f"{a.total_pair_steps:,} pair-steps each, max-concurrent 1"
          + (f", host {a.host}." if a.host else ", all hosts."))
    for i, s in enumerate(queue, 1):
        st = store.load(s)["status"] if store else q.PENDING
        print(f"{i:2d}. {s.run_id:28s} {s.host:6s} {st}")
    print("GUARDS: " + ("all pass (start would be allowed)" if not problems
                        else "WOULD REFUSE:"))
    for pr in problems:
        print(f"  - {pr}")
    try:
        free = gpu_free_gb()
        print("GPU guard: n/a (not Linux)" if free is None else
              f"GPU guard: {free:.1f} GiB free, need {a.min_free_gb} -> "
              + ("ok" if free >= a.min_free_gb else "WOULD REFUSE"))
    except GuardError as exc:
        print(f"GPU guard: WOULD REFUSE ({exc})")
    if a.throughput_json:
        e = estimate(json.loads(a.throughput_json.read_text()), a.total_pair_steps,
                     a.host)
        print(f"Estimated duration: {e['total_hours']:.0f} h "
              f"({e['total_days']:.1f} days), nocomm rate used for non-gtm arms.")
    else:
        print("Estimated duration: pass --throughput-json for an estimate.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
