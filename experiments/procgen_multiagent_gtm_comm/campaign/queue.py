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
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ..harness.codes import HOSTS, SEED_HOST, SEEDS  # noqa: F401 - re-exported

ARMS_REGISTERED: tuple[str, ...] = ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
ARMS_CLOSED: tuple[str, ...] = ("gtm", "dial", "eng", "nocomm")
ARM_ORDER: tuple[str, ...] = ("gtm", "nocomm", "dial", "eng", "mlp", "atoc")
TOTAL_PAIR_STEPS = 50_000_000
MAX_INFRA_RESTARTS = 2
MAX_EVAL_ATTEMPTS = 3
N_RUNS = 60

PENDING = "pending"
RUNNING = "running"
INFRA_FAILED = "infra_failed"
TRAINED = "trained"  # training finished, evaluation still to do
TRAINING_FAILURE = "training_failure"
DONE = "done"


@dataclass(frozen=True)
class RunSpec:
    arm: str
    seed: int
    condition: str

    @property
    def run_id(self) -> str:
        return f"{self.condition}-{self.arm}-s{self.seed}"

    @property
    def host(self) -> str:
        return SEED_HOST[self.seed]


def build_queue() -> list[RunSpec]:
    """The frozen 60-run list, deterministic order."""
    out: list[RunSpec] = []
    for i, seed in enumerate(SEEDS):
        k = i % len(ARM_ORDER)
        for arm in ARM_ORDER[k:] + ARM_ORDER[:k]:
            out.append(RunSpec(arm, seed, "registered"))
            if arm in ARMS_CLOSED:
                out.append(RunSpec(arm, seed, "closed"))
    assert len(out) == N_RUNS and len({r.run_id for r in out}) == N_RUNS
    return out


def runs_for_host(host: str) -> list[RunSpec]:
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}, expected one of {HOSTS}")
    return [r for r in build_queue() if r.host == host]


def new_state(spec: RunSpec) -> dict:
    return {
        "run_id": spec.run_id, "arm": spec.arm, "seed": spec.seed,
        "condition": spec.condition, "host": spec.host, "status": PENDING,
        "infra_failures": 0, "eval_failures": 0, "eval_done": False,
        "archived": False, "attempts": [], "last_checkpoint": None,
        "failure_reason": None, "started_at": None, "finished_at": None,
        "infra_exhausted": False, "synced": False,
    }


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
    return (state["status"] in (PENDING, RUNNING, INFRA_FAILED)
            and state["infra_failures"] <= MAX_INFRA_RESTARTS)


def needs_eval(state: dict) -> bool:
    return (state["status"] in (TRAINED, TRAINING_FAILURE)
            and not state["eval_done"]
            and state["eval_failures"] < MAX_EVAL_ATTEMPTS)


def infra_exhausted(state: dict) -> bool:
    return (state["status"] == INFRA_FAILED
            and state["infra_failures"] > MAX_INFRA_RESTARTS)


def needs_archive(state: dict) -> bool:
    return state["eval_done"] and not state["archived"]


def summarize(states: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in states:
        key = s["status"]
        if s.get("infra_exhausted"):
            key = "training_failure(infra_exhausted)"
        out[key] = out.get(key, 0) + 1
    return out
