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

``--registration n10c-bis`` (addendum 2026-10-25) runs the 48-run N10-C-bis
queue (``queue.build_queue_bis``, every run on KXKM-AI, tag
``n10c-bis-freeze-YYYYMMDD``) in a separate campaign directory; a
``gtm_warm`` run is trained with ``warm_start_from`` = the status-selected
checkpoint of its ``eng_gtmrx`` source (same seed and condition; the last
finite checkpoint of a failed source, or None = initialisation, flagged).
The existing STOP file pauses the runner between runs (N10-C evaluation
windows). The N10-C campaign (default) is unchanged.

Amendment 2026-10-25: a liveness abort restarts a run once from scratch,
a second abort is a training failure (``queue``). When the registered
preset uses a pre-trained trunk, KXKM-AI then trains and evaluates the
per-run-pretraining control (``queue.control_queue``) after all its
registered runs, each run with ``init_trunk`` = its own pinned trunk
(``guards.control_trunk_path``) and ``per_run_trunk=True``; campaign
completion requires them, cross-play and the estimands never use them.
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
from types import SimpleNamespace
from typing import Any

from ..harness.codes import HOST_SETTINGS
from ..harness.options import options_for
from . import guards as _guards
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
# KXKM finished its work but macM1 runs / cross-play cells still await `sync`
EXIT_AWAITING_SYNC = 6


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
             host: str | None = None,
             registration: str = q.DEFAULT_REGISTRATION) -> dict:
    """Duration estimate: gtm runs use the gtm rate, others the nocomm proxy
    (same convention as the manifest estimate); one 2x200 evaluation each."""
    res = throughput["results"]

    def hours(arm: str) -> float:
        r = res.get(arm) or res["nocomm"]
        return (total / r["pair_steps_per_s"]
                + r["eval_proxy"]["seconds_per_model_2x200"]) / 3600

    runs = (q.runs_for_host(host, registration) if host
            else q.build_queue(registration))
    per_run = {s.run_id: hours(s.arm) for s in runs}
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
    # which frozen queue / cells: "n10c" (default) or "n10c-bis"; must agree
    # with the context's registration when one is given
    registration_name: str = q.DEFAULT_REGISTRATION


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
        if opts.registration_name not in q.REGISTRATIONS:
            raise GuardError(f"unknown registration {opts.registration_name!r}")
        if (opts.registration is not None
                and opts.registration.registration != opts.registration_name):
            raise GuardError(
                f"registration context is {opts.registration.registration!r} but "
                f"the campaign is {opts.registration_name!r}")
        self.reg_name = opts.registration_name
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
        self.cross_summary: dict[str, str] = {}

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

    # -- queue of this registration ----------------------------------
    def queue_all(self) -> list[q.RunSpec]:
        return q.build_queue(self.reg_name)

    def queue_host(self) -> list[q.RunSpec]:
        return q.runs_for_host(self.o.host, self.reg_name)

    def cells(self) -> list[tuple[str, str]]:
        return q.cross_cells(self.reg_name)

    # -- one run ----------------------------------------------------
    def run_dir(self, spec: q.RunSpec) -> Path:
        return self.dir / "runs" / spec.run_id

    def warm_source(self, spec: q.RunSpec) -> tuple[Path | None, dict]:
        """Status-selected checkpoint of ``spec``'s warm source (N10-C-bis
        ``gtm_warm`` <- ``eng_gtmrx``, same seed and condition), digest
        verified; (None, info) when the source has no finite checkpoint
        (the warm run then starts from initialisation, flagged). A source
        that has not finished is a refusal, never a silent cold start."""
        src = spec.warm_source()
        sst = self.store.load(src)
        if sst["status"] not in (q.TRAINED, q.DONE, q.TRAINING_FAILURE):
            raise GuardError(f"warm source {src.run_id} of {spec.run_id} has not "
                             f"finished (status {sst['status']!r})")
        ck = sst["last_checkpoint"]
        rd = self.run_dir(src)
        verify_checkpoint(rd, ck, sst.get("last_checkpoint_sha256"))
        path = resolve_checkpoint(rd, ck)
        info = {"run_id": src.run_id, "status": sst["status"], "checkpoint": ck,
                "sha256": sst.get("last_checkpoint_sha256") if ck else None,
                "untrained": path is None}
        return path, info

    def train(self, spec: q.RunSpec, st: dict) -> None:
        attempt = len(st["attempts"]) + 1
        rd = self.run_dir(spec)
        self._settle_liveness_aside(spec, st)  # an interrupted transition
        if rd.exists():  # restart from scratch; keep the failed partial aside
            aside = rd.with_name(f"{spec.run_id}.failed-{st['infra_failures']}")
            shutil.move(str(rd), str(aside))
        st.update(status=q.RUNNING, started_at=_now())
        self.store.save(st)
        self.event(run=spec.run_id, attempt=attempt, phase="train", state="start")
        # registered per-host implementation settings (single source:
        # codes.HOST_SETTINGS); only a non-registered trial may override them
        settings = dict(HOST_SETTINGS[self.o.host])
        # registered training-options preset (amendment 2026-10-25) and, for
        # a pre-training preset, the seed's pinned trunk (hash checked by
        # the training boundary)
        opts = options_for(_guards.REGISTERED_TRAIN_OPTIONS)
        base: dict[str, Any] = {"options": opts}
        if spec.trunk == q.PER_RUN:  # per-run-pretraining control run
            if not opts.pretrained_trunk:
                raise GuardError(f"{spec.run_id}: the per-run-pretraining "
                                 "control needs a pre-trained-trunk preset")
            base["init_trunk"] = _guards.control_trunk_path(
                self.dir, spec.arm, spec.condition)
            base["per_run_trunk"] = True
        elif opts.pretrained_trunk:
            base["init_trunk"] = _guards.trunk_path(self.dir, spec.seed)
        kw = {**settings, **base, **(self.o.train_kwargs or {})}
        st["train_settings"] = {k: kw[k] for k in settings}
        st["train_options"] = kw["options"].preset
        if self.o.env_factory is not None:
            kw["env_factory"] = self.o.env_factory
        if spec.warm_from is not None:  # gtm_warm: resolved before training
            path, info = self.warm_source(spec)
            kw["warm_start_from"] = path
            st["warm_source"] = info
            self.store.save(st)
            self.event(run=spec.run_id, attempt=attempt, phase="train",
                       state="warm_source", warm_source=info)
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
        self._finish_attempt(spec, st, attempt, rec)

    def _finish_attempt(self, spec: q.RunSpec, st: dict, attempt: int,
                        rec: Any) -> None:
        """Record a finished training attempt (``rec``: the RunRecord, or
        its run.json when recovering an interrupted liveness abort)."""
        rd = self.run_dir(spec)
        n_live = st.get("liveness_restarts", 0)
        if getattr(rec, "liveness_abort", False) and n_live < q.MAX_LIVENESS_RESTARTS:
            self._liveness_restart(spec, st, attempt, rec)
            return
        st["last_checkpoint"] = rec.last_checkpoint
        st["last_checkpoint_sha256"] = checkpoint_sha256(rd, rec.last_checkpoint)
        if rec.last_checkpoint and st["last_checkpoint_sha256"] is None:
            q.fail_attempt(st, f"checkpoint {rec.last_checkpoint} missing after training")
            self.store.save(st)
            self.event(run=spec.run_id, attempt=attempt, phase="train",
                       state="infra_failed", reason=st["failure_reason"],
                       infra_failures=st["infra_failures"])
            return
        st["finished_at"] = _now()
        if rec.training_failure:
            reason = rec.failure_reason
            if getattr(rec, "liveness_abort", False):  # second abort: final
                st["liveness_abort"] = True
                reason = (f"liveness abort after {n_live} restart(s) from "
                          f"scratch: {reason}")
            st.update(status=q.TRAINING_FAILURE, failure_reason=reason)
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

    # -- liveness-abort restart (amendment 2026-10-25) -------------------
    def _liveness_restart(self, spec: q.RunSpec, st: dict, attempt: int,
                          rec: Any) -> None:
        """First liveness abort (outcome-blind: the trunk-liveness flag of
        the training record only): one restart from scratch, counted apart
        from infrastructure restarts. The transition is persisted BEFORE
        the aborted attempt is moved aside (idempotent, see
        ``_settle_liveness_aside``), so an interruption can never grant a
        second liveness restart."""
        n = st.get("liveness_restarts", 0) + 1
        aside = f"{spec.run_id}.liveness-abort-{n}"
        st["liveness_restarts"] = n
        st["attempts"].append({
            "n": attempt, "outcome": "liveness_abort", "at": _now(),
            "reason": rec.failure_reason, "kept_aside": aside,
            "wall_time_s": rec.wall_time_s,
            "pair_steps_done": rec.pair_steps_done})
        st.update(status=q.LIVENESS_RESTART, failure_reason=rec.failure_reason,
                  last_checkpoint=None, last_checkpoint_sha256=None,
                  liveness_aside=aside)
        self.store.save(st)
        self._settle_liveness_aside(spec, st)
        self.event(run=spec.run_id, attempt=attempt, phase="train",
                   state="liveness_abort_restart", reason=rec.failure_reason,
                   liveness_restarts=n, kept_aside=aside)

    def _settle_liveness_aside(self, spec: q.RunSpec, st: dict) -> None:
        """Move the aborted attempt's run directory to its recorded aside
        name if that has not happened yet (no-op otherwise)."""
        name = st.get("liveness_aside")
        if st.get("status") != q.LIVENESS_RESTART or not name:
            return
        rd, dst = self.run_dir(spec), self.run_dir(spec).with_name(name)
        if rd.exists() and not dst.exists():
            shutil.move(str(rd), str(dst))

    def recover_interrupted_abort(self, spec: q.RunSpec, st: dict) -> dict:
        """A ``running`` status whose run directory already holds the
        run.json of a liveness-aborted attempt was interrupted between the
        end of training and the status update: record that abort (restart
        or final training failure, as in ``train``) instead of counting an
        infrastructure failure."""
        f = self.run_dir(spec) / "run.json"
        if st["status"] != q.RUNNING or not f.exists():
            return st
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            return st
        if not d.get("liveness_abort"):
            return st
        rec = SimpleNamespace(**{k: d.get(k) for k in (
            "training_failure", "liveness_abort", "pair_steps_done",
            "last_checkpoint", "wall_time_s", "failure_reason")})
        self.event(run=spec.run_id, phase="train",
                   state="recovered_interrupted_liveness_abort")
        self._finish_attempt(spec, st, len(st["attempts"]) + 1, rec)
        return st

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

    def cross_verified_digest(self, arm: str, condition: str, seed: int) -> str | None:
        """SHA-256 of one seed's status-selected checkpoint, verified against
        the digest recorded at training time immediately before cross-play
        (None = untrained init). Raises on a missing file or a mismatch."""
        spec = q.RunSpec(arm, seed, condition)
        st = self.store.load(spec)
        rd = self.run_dir(spec)
        verify_checkpoint(rd, st["last_checkpoint"], st.get("last_checkpoint_sha256"))
        p = resolve_checkpoint(rd, st["last_checkpoint"])
        return None if p is None else sha256_file(p)

    def cross_cell_state(self, arm: str, condition: str) -> str:
        """ready | awaiting_sync (a seed's run is absent) | unresolved (a seed
        is present but cannot become ready: no terminal status, or its
        evaluation retries are exhausted)."""
        for seed in q.SEEDS:
            spec = q.RunSpec(arm, seed, condition)
            if not self.store.path(spec).exists():
                return "awaiting_sync"
        for seed in q.SEEDS:
            st = self.store.load(q.RunSpec(arm, seed, condition))
            if st["status"] in (q.TRAINED, q.TRAINING_FAILURE, q.DONE) \
                    and st["eval_done"]:
                continue
            if st["status"] in (q.TRAINED, q.TRAINING_FAILURE) and q.needs_eval(st):
                return "pending_eval"
            return "unresolved"
        return "ready"

    def cross_cell_ready(self, arm: str, condition: str) -> bool:
        """All seeds of the cell exist, are trained or failed, and evaluated."""
        return self.cross_cell_state(arm, condition) == "ready"

    def cross_pass(self) -> bool:
        """Cross-seed cross-play probe for every ready (arm, condition) cell.

        KXKM only (x86 Procgen build). Resumable: a pair marked done is never
        repeated; a failed pair is retried up to MAX_EVAL_ATTEMPTS in total.
        Returns True ONLY when all ten cells are complete; ``self.cross_summary``
        maps every cell to done | awaiting_sync | pending_eval | unresolved |
        pairs_failed, so incomplete cells are never mistaken for success.
        """
        if self.o.host != "kxkm":
            raise GuardError("the cross-play probe runs only with --host kxkm "
                             "(x86 Procgen build)")
        out = self.dir / "cross"
        self.cross_summary = {}
        for arm, condition in self.cells():
            cell = q.cross_cell_id(arm, condition)
            state = self.cross_cell_state(arm, condition)
            if state != "ready":
                self.cross_summary[cell] = state
                continue
            cs = self.cross_store.load(arm, condition)
            for pid, pair in cs["pairs"].items():
                s0, s1 = pair["seed_role0"], pair["seed_role1"]
                while q.cross_pair_pending(pair):
                    self.event(phase="cross", cell=cs["cell"], pair=pid, state="start")
                    try:
                        # both digests re-verified right before the pair runs
                        sha0 = self.cross_verified_digest(arm, condition, s0)
                        sha1 = self.cross_verified_digest(arm, condition, s1)
                        ck0 = self.cross_ckpt(arm, condition, s0)
                        ck1 = self.cross_ckpt(arm, condition, s1)
                        rec = self.cross_fn(ck0, ck1, arm, condition, s0, s1,
                                            env_factory=self.o.env_factory)
                        out.mkdir(parents=True, exist_ok=True)
                        body = json.loads(rec.to_json())
                        body.update(sha256_role0=sha0, sha256_role1=sha1)
                        (out / f"{cs['cell']}-s{s0}-s{s1}.json").write_text(
                            json.dumps(body, indent=2))
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
                                ckpt_role1=None if ck1 is None else str(ck1.name),
                                sha256_role0=sha0, sha256_role1=sha1)
                    self.cross_store.save(cs)
                    self.event(phase="cross", cell=cs["cell"], pair=pid, state="done",
                               R_cross=pair["R_cross"])
            cs["done"] = all(p["done"] for p in cs["pairs"].values())
            self.cross_store.save(cs)
            self.cross_summary[cell] = "done" if cs["done"] else "pairs_failed"
        return all(v == "done" for v in self.cross_summary.values()) \
            and len(self.cross_summary) == len(self.cells())

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

    def control_all(self) -> list[q.RunSpec]:
        """Per-run-pretraining control runs (only when the registered preset
        uses a pre-trained trunk; N10-C only)."""
        enabled = options_for(_guards.REGISTERED_TRAIN_OPTIONS).pretrained_trunk
        return q.control_queue(self.reg_name, enabled=enabled)

    def control_host(self) -> list[q.RunSpec]:
        return [s for s in self.control_all() if s.host == self.o.host]

    def _process(self, spec: q.RunSpec, can_eval: bool) -> int | None:
        """Train (with infrastructure and liveness restarts), evaluate and
        archive one run of this host; an exit code stops the campaign."""
        st = self.recover_interrupted_abort(spec, self.store.load(spec))
        st = self.store.recover_stale(spec) if st["status"] == q.RUNNING else st
        self.finalize_exhausted(spec, st)
        if q.needs_training(st):
            ok, free = self.gpu_ok()
            if not ok:
                self.event(phase="campaign", state="gpu_guard_refused",
                           free_gb=free, min_free_gb=self.o.min_free_gb,
                           next_run=spec.run_id)
                return EXIT_GPU
            # infra restarts and the one liveness restart, from scratch
            while q.needs_training(st):
                self.train(spec, st)
                if (st["status"] in (q.INFRA_FAILED, q.LIVENESS_RESTART)
                        and q.needs_training(st)):
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
        return None

    def run(self) -> int:
        queue = self.queue_host()
        if not self._pinned:
            reg = self.o.registration
            if reg is None and not self.o.trial:
                raise GuardError("a registered campaign cannot start without "
                                 "its registration context (provenance)")
            self.pin({} if reg is None else reg.pins())
        can_eval = self.o.host == "kxkm"
        for spec in self.queue_all():
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
            rc = self._process(spec, can_eval)
            if rc is not None:
                return rc
        # declared per-run-pretraining control (amendment 2026-10-25): after
        # every registered run of this host; never part of the estimands
        for spec in self.control_host():
            if self.stop_requested():
                self.event(phase="campaign", state="stopped_by_STOP_file")
                return EXIT_STOP
            rc = self._process(spec, can_eval)
            if rc is not None:
                return rc
        queue = queue + self.control_host()
        states = [self.store.load(s) for s in queue]
        if not can_eval:  # macM1: trains only; evaluation happens on KXKM
            self.event(phase="campaign", state="finished", host=self.o.host,
                       summary=q.summarize(states))
            return EXIT_OK
        # KXKM: completion needs ALL evaluations and ALL cross-play cells of
        # the registration (N10-C: 60 and 10; N10-C-bis: 48 and 8), plus the
        # per-run-pretraining control runs when there is one
        everything = self.queue_all() + self.control_all()
        awaiting, unresolved = [], []
        for s in everything:
            if not self.store.path(s).exists():
                awaiting.append(s.run_id)
                continue
            st = self.store.load(s)
            if not (st["eval_done"] and st["status"] in (q.TRAINED, q.TRAINING_FAILURE, q.DONE)):
                unresolved.append(s.run_id)
        cross_ok = self.cross_pass()
        cross_bad = {c: v for c, v in self.cross_summary.items()
                     if v in ("unresolved", "pairs_failed")}
        cross_wait = {c: v for c, v in self.cross_summary.items()
                      if v in ("awaiting_sync", "pending_eval")}
        complete = not (awaiting or unresolved or not cross_ok)
        self.event(phase="campaign", state="complete" if complete else "finished",
                   host=self.o.host, summary=q.summarize(states),
                   evaluations_done=len(everything) - len(awaiting) - len(unresolved),
                   evaluations_required=len(everything),
                   awaiting_sync=awaiting, unresolved=unresolved,
                   cross_cells_done=sum(v == "done" for v in self.cross_summary.values()),
                   cross_cells_required=len(self.cells()),
                   cross_pending=cross_wait, cross_failed=cross_bad)
        if unresolved or cross_bad:
            return EXIT_PROBLEMS
        return EXIT_OK if complete else EXIT_AWAITING_SYNC


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


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def verify_checkpoint(run_dir: Path, last_ckpt: str | None, sha: str | None) -> None:
    """Raise unless the selected checkpoint has its recorded SHA-256.

    A valid recorded digest is REQUIRED whenever ``last_ckpt`` is set; only
    the explicit untrained-initialisation case (no checkpoint) needs none."""
    if not last_ckpt:
        return
    if not isinstance(sha, str) or not SHA256_RE.match(sha):
        raise RuntimeError(
            f"checkpoint {last_ckpt} has no valid recorded SHA-256 digest "
            "(training-time evidence is required before evaluation / cross-play)")
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
        freeze_tag=a.freeze_tag, swhid=a.swhid, registration=a.registration)
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
            freeze_tag=a.freeze_tag, swhid=a.swhid, registration=a.registration)
    except GuardError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_GUARD
    opts = Options(campaign_dir=a.campaign_dir, host=a.host,
                   total_pair_steps=a.total_pair_steps,
                   min_free_gb=a.min_free_gb, gpu_wait_s=a.gpu_wait_s,
                   archive_to=a.archive_to, max_concurrent=a.max_concurrent,
                   trial=a.non_registered_trial, registration=ctx,
                   registration_name=a.registration)
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
    reg = getattr(a, "registration", q.DEFAULT_REGISTRATION)
    queue = q.runs_for_host(a.host, reg) if a.host else q.build_queue(reg)
    store = q.StatusStore(a.campaign_dir / "status") if a.campaign_dir.exists() else None
    print(f"DRY RUN ({reg}): nothing is trained. {len(queue)} runs, "
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
                     a.host, reg)
        print(f"Estimated duration: {e['total_hours']:.0f} h "
              f"({e['total_days']:.1f} days), nocomm rate used for non-gtm arms.")
    else:
        print("Estimated duration: pass --throughput-json for an estimate.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
