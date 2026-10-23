"""Sequential N10-C campaign runner (option D, 50 runs, one at a time).

    python -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign \
        --campaign-dir /ai/saisail/n10c-run/campaign \
        --manifest experiments/procgen_multiagent_gtm_comm/manifest/<file>.md \
        --manifest-sha256 <sha256> \
        --manifest-doi 10.17605/OSF.IO/XXXXX --expect-commit <40-hex> \
        [--archive-to user@host:/path] [--dry-run --throughput-json t.json]

Hard guards (campaign/guards.py, shared with the harness CLIs; all must
pass, else exit 2 before any training): real-looking OSF DOI, FILED
(not DRAFT) manifest under the manifest/ folder containing that DOI and
the expected commit, tracked in git, matching ``--manifest-sha256``; git
tree clean; HEAD == ``--expect-commit`` or a descendant whose every
intervening commit only touched docs/ or the manifest folder. The budget
and trial flag are pinned in campaign.json at first start. Before every
run on Linux: nvidia-smi free memory >= ``--min-free-gb``. Other
processes are never signalled or killed. After 3 infrastructure
failures a run becomes a training failure (amendment: evaluated at the
last finite checkpoint, else at initialisation; never replaced).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import queue as q
from .guards import (  # noqa: F401 - re-exported for callers and tests
    HEX40,
    REPO,
    GuardError,
    guard_problems,
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


def estimate(throughput: dict, total: int = q.TOTAL_PAIR_STEPS) -> dict:
    """Duration estimate: gtm runs use the gtm rate, others the nocomm proxy
    (same convention as the manifest estimate); one 2x200 evaluation each."""
    res = throughput["results"]

    def hours(arm: str) -> float:
        r = res.get(arm) or res["nocomm"]
        return (total / r["pair_steps_per_s"]
                + r["eval_proxy"]["seconds_per_model_2x200"]) / 3600

    per_run = {s.run_id: hours(s.arm) for s in q.build_queue()}
    tot = sum(per_run.values())
    return {"per_run_hours": per_run, "total_hours": tot, "total_days": tot / 24}


@dataclass
class Options:
    campaign_dir: Path
    total_pair_steps: int = q.TOTAL_PAIR_STEPS
    min_free_gb: float = 8.0
    gpu_wait_s: float = 0.0
    archive_to: str | None = None
    max_concurrent: int = 1
    trial: bool = False  # non-registered budget; never the real campaign
    train_kwargs: dict | None = None  # test / smoke overrides (n_pairs, ...)
    env_factory: Callable[[int], Any] | None = None


def _default_train(**kw):
    from experiments.procgen_multiagent_gtm_comm.harness.train import train_run
    return train_run(**kw)


def _default_eval(ckpt, arm, condition, seed, env_factory=None):
    from experiments.procgen_multiagent_gtm_comm.harness.evaluate import (
        HELD_OUT,
        evaluate_model,
    )
    return evaluate_model(ckpt, arm, condition, HELD_OUT, seed=seed,
                          env_factory=env_factory)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Campaign:
    def __init__(self, opts: Options, *, train_fn=None, eval_fn=None,
                 gpu_fn=None, archive_fn=None, log=print) -> None:
        self.o = opts
        self.dir = Path(opts.campaign_dir)
        self.store = q.StatusStore(self.dir / "status")
        self.train_fn = train_fn or _default_train
        self.eval_fn = eval_fn or _default_eval
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
                "trial": self.o.trial}
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "campaign.json"
        if p.exists():
            old = json.loads(p.read_text())
            diff = {k: (old.get(k), v) for k, v in pins.items() if old.get(k) != v}
            if diff:
                raise GuardError(
                    f"campaign pinned to other inputs (trial/registered "
                    f"cannot be mixed, nor budget changed): {diff}")
        else:
            if any((self.dir / "status").glob("*.json")):
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
        kw = dict(self.o.train_kwargs or {})
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
        rd = self.run_dir(spec)
        ckpt = rd / st["last_checkpoint"] if st["last_checkpoint"] else None
        self.event(run=spec.run_id, phase="eval", state="start",
                   ckpt=st["last_checkpoint"])
        try:  # held-out levels are touched only here, after training
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

    def archive(self, spec: q.RunSpec, st: dict) -> None:
        if not self.o.archive_to:
            return
        rd = self.run_dir(spec)
        files = archive_files(rd, st["last_checkpoint"])
        try:
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
        queue = q.build_queue()
        if not self._pinned:
            self.pin({})
        for spec in queue:
            if self.stop_requested():
                self.event(phase="campaign", state="stopped_by_STOP_file")
                return EXIT_STOP
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
            while q.needs_eval(st):
                self.evaluate(spec, st)
            if q.needs_archive(st):
                self.archive(spec, st)
        states = [self.store.load(s) for s in queue]
        self.event(phase="campaign", state="finished", summary=q.summarize(states))
        problems = [s for s in states if s["status"] in (q.TRAINED, q.TRAINING_FAILURE)
                    and not s["eval_done"]]
        return EXIT_PROBLEMS if problems else EXIT_OK


CKPT_RE = re.compile(r"^ckpt_(\d+)\.safetensors$")


def completed_ckpts(d: Path) -> list[Path]:
    """Finished checkpoints only (never ``ckpt_<n>.tmp.safetensors``),
    ascending by step."""
    found = [(int(m.group(1)), f) for f in d.glob("ckpt_*.safetensors")
             if (m := CKPT_RE.match(f.name))]
    return [f for _, f in sorted(found)]


def archive_files(run_dir: Path, last_ckpt: str | None) -> list[Path]:
    files = [run_dir / "run.json", run_dir / "eval.json"]
    ckpts = completed_ckpts(run_dir)
    keep = {c for c in ckpts[-1:]}
    if last_ckpt:
        keep.add(run_dir / last_ckpt)
    files += sorted(keep)
    return [f for f in files if f.exists()]


def rsync_archive(run_dir: Path, files: list[Path], target: str, run_id: str) -> None:
    dest = f"{target.rstrip('/')}/{run_id}/"
    cmd = ["rsync", "-a", "--partial", "-e", "ssh -o BatchMode=yes",
           *[str(f) for f in files], dest]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr.strip()}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--campaign-dir", type=Path, required=True,
                   help="state + run outputs; must be OUTSIDE the git repo")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--manifest-sha256")
    p.add_argument("--manifest-doi")
    p.add_argument("--expect-commit")
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--gpu-wait-s", type=float, default=0.0)
    p.add_argument("--archive-to", help="user@host:path (rsync over ssh)")
    p.add_argument("--max-concurrent", type=int, default=1, choices=[1])
    p.add_argument("--throughput-json", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--total-pair-steps", type=int, default=q.TOTAL_PAIR_STEPS)
    p.add_argument("--non-registered-trial", action="store_true",
                   help="allow a budget != 50M (never for the real campaign)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    problems = guard_problems(
        repo=REPO, manifest=a.manifest, manifest_sha256=a.manifest_sha256,
        manifest_doi=a.manifest_doi, expect_commit=a.expect_commit)
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
    if a.dry_run:
        return _dry_run(a, problems)
    if problems:
        print("REFUSED:\n- " + "\n- ".join(problems), file=sys.stderr)
        return EXIT_GUARD
    opts = Options(campaign_dir=a.campaign_dir, total_pair_steps=a.total_pair_steps,
                   min_free_gb=a.min_free_gb, gpu_wait_s=a.gpu_wait_s,
                   archive_to=a.archive_to, max_concurrent=a.max_concurrent,
                   trial=a.non_registered_trial)
    camp = Campaign(opts)
    camp.dir.mkdir(parents=True, exist_ok=True)
    with open(camp.dir / "campaign.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("REFUSED: another campaign process holds the lock", file=sys.stderr)
            return EXIT_GUARD
        try:
            camp.pin({"manifest_sha256": a.manifest_sha256.lower(),
                      "manifest_doi": normalize_doi(a.manifest_doi),
                      "commit": a.expect_commit.lower()})
            return camp.run()
        except GuardError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT_GUARD


def _dry_run(a, problems: list[str]) -> int:
    queue = q.build_queue()
    store = q.StatusStore(a.campaign_dir / "status") if a.campaign_dir.exists() else None
    print(f"DRY RUN: nothing is trained. {len(queue)} runs, "
          f"{a.total_pair_steps:,} pair-steps each, max-concurrent 1.")
    for i, s in enumerate(queue, 1):
        st = store.load(s)["status"] if store else q.PENDING
        print(f"{i:2d}. {s.run_id:28s} {st}")
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
        e = estimate(json.loads(a.throughput_json.read_text()), a.total_pair_steps)
        print(f"Estimated duration: {e['total_hours']:.0f} h "
              f"({e['total_days']:.1f} days), nocomm rate used for non-gtm arms.")
    else:
        print("Estimated duration: pass --throughput-json for an estimate.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
