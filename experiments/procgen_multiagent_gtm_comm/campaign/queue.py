"""Frozen N10-C run queue and per-run status files (option D, 60 runs).

Registered condition: 6 arms x 6 seeds = 36 runs. Closed condition:
gtm, dial, eng, nocomm x 6 seeds = 24 runs. Every run has a ``host`` (seed
block factor, ``harness.codes.SEED_HOST``): macM1 trains seeds 0, 17, 42;
KXKM trains 73, 101, 131. The order is a pure function
of the frozen lists below: seed-major (so a partial campaign always
holds complete seed blocks, plus a prefix of the next one) with the arm
order rotated by the seed index (so no arm is systematically last), and
the closed run of an arm directly after its registered run.

Status files: one JSON per run, ``<status_dir>/<run_id>.json``, written
atomically. Infrastructure failures restart the run from scratch, at
most ``MAX_INFRA_RESTARTS`` times (so at most 3 attempts); a training
failure (non-finite loss / parameter) is final and never retried.
Amendment 2026-10-25: a trunk-liveness abort restarts the run once from
scratch (status ``liveness_restart``, counter ``liveness_restarts``,
separate from ``infra_failures``; the aborted attempt is kept aside as
``<run_id>.liveness-abort-1`` and never evaluated); a second abort is a
training failure, evaluated at its last checkpoint and never replaced.
``control_queue`` is the declared per-run-pretraining control: the 10
runs of seed 101 (KXKM-AI) again, each with its own pre-trained trunk,
run ids ``prc-<condition>-<arm>-s101``, outside the frozen 60-run list.

N10-C-bis (addendum 2026-10-25, scope option bis-A): ``build_queue_bis``
is the frozen 48-run list of the second campaign (4 control arms x 2
conditions x 6 seeds, every run on KXKM-AI): seed-major, the arm units
rotated by the seed index, each arm's closed run directly after its
registered run, and ``eng_gtmrx``(s, c) always before ``gtm_warm``(s, c),
whose ``warm_from`` names that source run. The N10-C queue is untouched.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ..harness.codes import (  # noqa: F401 - re-exported
    HOSTS,
    PER_RUN_TRUNK_SEED,
    SEED_HOST,
    SEED_HOST_BIS,
    SEEDS,
)

ARMS_REGISTERED: tuple[str, ...] = ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
ARMS_CLOSED: tuple[str, ...] = ("gtm", "dial", "eng", "nocomm")
ARM_ORDER: tuple[str, ...] = ("gtm", "nocomm", "dial", "eng", "mlp", "atoc")
TOTAL_PAIR_STEPS = 50_000_000
MAX_INFRA_RESTARTS = 2
# Amendment 2026-10-25 (PI decision 12:07-12:16): a liveness abort restarts
# the run once from scratch (counted separately from infrastructure
# restarts); a second abort is a training failure, never replaced.
MAX_LIVENESS_RESTARTS = 1
MAX_EVAL_ATTEMPTS = 3
N_RUNS = 60

# -- N10-C-bis (addendum 2026-10-25) -----------------------------------------
REGISTRATIONS: tuple[str, ...] = ("n10c", "n10c-bis")
DEFAULT_REGISTRATION = "n10c"
ARMS_BIS: tuple[str, ...] = ("gtm_mis", "eng_gtmrx", "gtm_warm", "vq")
ARMS_BIS_REGISTERED = ARMS_BIS  # both conditions for every arm (bis-A)
ARMS_BIS_CLOSED = ARMS_BIS
# Order units: the (eng_gtmrx -> gtm_warm) dependency is one unit; the
# units rotate by seed index so no arm is systematically last.
ARM_UNITS_BIS: tuple[tuple[str, ...], ...] = (
    ("eng_gtmrx", "gtm_warm"), ("gtm_mis",), ("vq",))
WARM_SOURCE_ARM: dict[str, str] = {"gtm_warm": "eng_gtmrx"}
N_RUNS_BIS = 48

# -- per-run-pretraining control (amendment 2026-10-25, PI 19:12) ------------
PER_RUN = "per_run"
CONTROL_PREFIX = "prc-"
N_CONTROL_RUNS = 10  # every (arm, condition) run of seed PER_RUN_TRUNK_SEED

PENDING = "pending"
RUNNING = "running"
INFRA_FAILED = "infra_failed"
LIVENESS_RESTART = "liveness_restart"  # first liveness abort: retrain from scratch
TRAINED = "trained"  # training finished, evaluation still to do
TRAINING_FAILURE = "training_failure"
DONE = "done"


@dataclass(frozen=True)
class RunSpec:
    arm: str
    seed: int
    condition: str
    # N10-C-bis fields (None for every N10-C run, so N10-C specs compare and
    # serialise as before): the fixed host (KXKM for all bis runs) and the
    # run_id of the warm-start source (``gtm_warm`` only).
    host_fixed: str | None = None
    warm_from: str | None = None
    # amendment 2026-10-25: "per_run" = a run of the per-run-pretraining
    # control (own trunk; run_id prefixed "prc-"); None = registered run
    trunk: str | None = None

    @property
    def run_id(self) -> str:
        rid = f"{self.condition}-{self.arm}-s{self.seed}"
        return f"{CONTROL_PREFIX}{rid}" if self.trunk == PER_RUN else rid

    @property
    def host(self) -> str:
        return self.host_fixed or SEED_HOST[self.seed]

    def warm_source(self) -> RunSpec | None:
        """The run this one is warm-started from (same seed and condition)."""
        if self.warm_from is None:
            return None
        src = RunSpec(WARM_SOURCE_ARM[self.arm], self.seed, self.condition,
                      host_fixed=self.host_fixed)
        assert src.run_id == self.warm_from
        return src


def _check_registration(registration: str) -> None:
    if registration not in REGISTRATIONS:
        raise ValueError(f"unknown registration {registration!r}, expected one "
                         f"of {REGISTRATIONS}")


def build_queue(registration: str = DEFAULT_REGISTRATION) -> list[RunSpec]:
    """The frozen run list of a registration, deterministic order
    (N10-C: 60 runs; N10-C-bis: 48 runs, ``build_queue_bis``)."""
    _check_registration(registration)
    if registration == "n10c-bis":
        return build_queue_bis()
    out: list[RunSpec] = []
    for i, seed in enumerate(SEEDS):
        k = i % len(ARM_ORDER)
        for arm in ARM_ORDER[k:] + ARM_ORDER[:k]:
            out.append(RunSpec(arm, seed, "registered"))
            if arm in ARMS_CLOSED:
                out.append(RunSpec(arm, seed, "closed"))
    assert len(out) == N_RUNS and len({r.run_id for r in out}) == N_RUNS
    return out


def build_queue_bis() -> list[RunSpec]:
    """The frozen 48-run N10-C-bis list (scope option bis-A).

    Seed-major; per seed the arm units of ``ARM_UNITS_BIS`` rotated by the
    seed index; per arm the registered run then the closed run; so
    ``eng_gtmrx``(s, c) precedes ``gtm_warm``(s, c) for every (s, c).
    """
    out: list[RunSpec] = []
    for i, seed in enumerate(SEEDS):
        host = SEED_HOST_BIS[seed]
        k = i % len(ARM_UNITS_BIS)
        for unit in ARM_UNITS_BIS[k:] + ARM_UNITS_BIS[:k]:
            for arm in unit:
                for cond in ("registered", "closed"):
                    wf = None
                    if arm in WARM_SOURCE_ARM:
                        wf = RunSpec(WARM_SOURCE_ARM[arm], seed, cond,
                                     host_fixed=host).run_id
                    out.append(RunSpec(arm, seed, cond, host_fixed=host,
                                       warm_from=wf))
    ids = [r.run_id for r in out]
    assert len(out) == N_RUNS_BIS and len(set(ids)) == N_RUNS_BIS
    for r in out:  # the warm source always runs first
        if r.warm_from is not None:
            assert ids.index(r.warm_from) < ids.index(r.run_id)
    return out


def control_queue(registration: str = DEFAULT_REGISTRATION,
                  enabled: bool = True) -> list[RunSpec]:
    """The declared per-run-pretraining control (N10-C only): the 10 runs
    of seed ``PER_RUN_TRUNK_SEED`` (its host, KXKM-AI), in that seed's
    registered queue order, each with its OWN pre-trained trunk (run_id
    ``prc-<condition>-<arm>-s101``). Not part of the frozen 60-run list
    (``build_queue``), never an input of the confirmatory analysis.
    ``enabled`` is False when the registered preset has no pre-trained
    trunk (frozen configuration): then there is no control."""
    _check_registration(registration)
    if registration != DEFAULT_REGISTRATION or not enabled:
        return []
    out = [RunSpec(r.arm, r.seed, r.condition, trunk=PER_RUN)
           for r in build_queue(registration) if r.seed == PER_RUN_TRUNK_SEED]
    assert len(out) == N_CONTROL_RUNS
    assert not {r.run_id for r in out} & {r.run_id for r in build_queue()}
    return out


def control_partner(spec: RunSpec) -> RunSpec:
    """The shared-trunk registered run a control run is paired with."""
    if spec.trunk != PER_RUN:
        raise ValueError(f"{spec.run_id} is not a per-run-pretraining control run")
    return RunSpec(spec.arm, spec.seed, spec.condition)


def runs_for_host(host: str,
                  registration: str = DEFAULT_REGISTRATION) -> list[RunSpec]:
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}, expected one of {HOSTS}")
    return [r for r in build_queue(registration) if r.host == host]


def new_state(spec: RunSpec) -> dict:
    st = {
        "run_id": spec.run_id, "arm": spec.arm, "seed": spec.seed,
        "condition": spec.condition, "host": spec.host, "status": PENDING,
        "infra_failures": 0, "eval_failures": 0, "eval_done": False,
        "archived": False, "attempts": [], "last_checkpoint": None,
        "failure_reason": None, "started_at": None, "finished_at": None,
        "infra_exhausted": False, "synced": False,
        "liveness_restarts": 0,
    }
    if spec.trunk is not None:
        st["trunk"] = spec.trunk
    if spec.warm_from is not None:
        st["warm_from"] = spec.warm_from
    return st


class StatusStore:
    def __init__(self, status_dir: Path) -> None:
        self.dir = Path(status_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, spec: RunSpec) -> Path:
        return self.dir / f"{spec.run_id}.json"

    def load(self, spec: RunSpec) -> dict:
        p = self.path(spec)
        if not p.exists():
            return new_state(spec)
        return json.loads(p.read_text())

    def save(self, state: dict) -> None:
        p = self.dir / f"{state['run_id']}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, p)

    def recover_stale(self, spec: RunSpec) -> dict:
        """A ``running`` file found at start-up is an interrupted attempt.

        It is an infrastructure failure (restart from scratch, counted).
        """
        s = self.load(spec)
        if s["status"] == RUNNING:
            fail_attempt(s, "interrupted: found 'running' at resume")
            self.save(s)
        return s


def fail_attempt(state: dict, reason: str) -> None:
    state["infra_failures"] += 1
    state["status"] = INFRA_FAILED
    state["failure_reason"] = reason
    state["attempts"].append({
        "n": len(state["attempts"]) + 1, "outcome": "infra_failed",
        "reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def needs_training(state: dict) -> bool:
    return (state["status"] in (PENDING, RUNNING, INFRA_FAILED, LIVENESS_RESTART)
            and state["infra_failures"] <= MAX_INFRA_RESTARTS)


def needs_eval(state: dict) -> bool:
    return (state["status"] in (TRAINED, TRAINING_FAILURE)
            and not state["eval_done"]
            and state["eval_failures"] < MAX_EVAL_ATTEMPTS)


def infra_exhausted(state: dict) -> bool:
    return (state["status"] == INFRA_FAILED
            and state["infra_failures"] > MAX_INFRA_RESTARTS)


def needs_archive(state: dict) -> bool:
    return (state["eval_done"] and not state["archived"]
            and not state.get("archive_via"))


def summarize(states: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in states:
        key = s["status"]
        if s.get("infra_exhausted"):
            key = "training_failure(infra_exhausted)"
        out[key] = out.get(key, 0) + 1
    return out


# -- cross-seed cross-play probe (KXKM only, evaluation only) -------------
# One cell per (condition, arm). Its 6 pairs are the fixed pairing
# ``evaluate.cross_pairs`` (role 0 = seed s, role 1 = next seed, wrapping)
# and run only after all 6 seeds of the cell are trained AND evaluated.

def cross_cells(registration: str = DEFAULT_REGISTRATION) -> list[tuple[str, str]]:
    """(arm, condition) cells, in queue order of first appearance."""
    out: list[tuple[str, str]] = []
    for r in build_queue(registration):
        if (r.arm, r.condition) not in out:
            out.append((r.arm, r.condition))
    return out


def cross_pair_list() -> list[tuple[int, int]]:
    return [(s, SEEDS[(k + 1) % len(SEEDS)]) for k, s in enumerate(SEEDS)]


def cross_cell_id(arm: str, condition: str) -> str:
    return f"{condition}-{arm}"


def cross_pair_id(seed0: int, seed1: int) -> str:
    return f"{seed0}-{seed1}"


class CrossStore:
    """One JSON per cell: ``<dir>/<condition>-<arm>.json``, written atomically."""

    def __init__(self, status_dir: Path) -> None:
        self.dir = Path(status_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, arm: str, condition: str) -> Path:
        return self.dir / f"{cross_cell_id(arm, condition)}.json"

    def load(self, arm: str, condition: str) -> dict:
        p = self.path(arm, condition)
        if p.exists():
            return json.loads(p.read_text())
        return {"cell": cross_cell_id(arm, condition), "arm": arm,
                "condition": condition, "done": False,
                "pairs": {cross_pair_id(a, b): {
                    "seed_role0": a, "seed_role1": b, "done": False,
                    "failures": 0, "R_cross": None, "untrained": None}
                    for a, b in cross_pair_list()}}

    def save(self, state: dict) -> None:
        p = self.dir / f"{state['cell']}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, p)


def cross_pair_pending(pair: dict) -> bool:
    return not pair["done"] and pair["failures"] < MAX_EVAL_ATTEMPTS
