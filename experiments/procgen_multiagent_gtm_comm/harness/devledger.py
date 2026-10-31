"""Persistent development-run ledger (no mlx import).

One per-arm ledger, ``manifest/smoke_ledger.json``, accounts for EVERY
real-environment development run, whichever path launched it. Amendment,
Configuration manifest (with the changes of 2026-10-24): development may use
smoke runs, throughput / profiling runs with learning enabled, and throughput
measurements on a frozen random policy.

Kinds:

* ``smoke`` and ``dev_train``: runs whose policy learns; counted against the
  per-arm allowance (``allowance(arm)``: 3, gtm 4), each <= ``MAX_STEPS``.
* ``throughput_learning``: throughput / profiling runs with learning on,
  <= ``MAX_STEPS``, training levels only; a SEPARATE category (no per-arm
  count cap), counted per arm in the ledger; each records what was inspected.
* ``measurement``: throughput on a frozen random policy (learning rate 0).

A reservation is written BEFORE the run starts (a crashed run still counts),
has a persistent unique id and is CLAIMED exactly once, atomically, at the
training boundary; claimed or completed reservations are rejected.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
LEDGER = PKG / "manifest" / "smoke_ledger.json"
MAX_STEPS = 1_000_000
MAX_PER_ARM = 3  # default allowance per arm
# PI decision 2026-10-24 (amendment, development allowance): GTM-comm gets 4
# counted learning runs, because its gradient path changed after the earlier
# smoke runs and the third slot was consumed by a crash at 0 steps.
ARM_ALLOWANCE: dict[str, int] = {"gtm": 4}


def allowance(arm: str) -> int:
    """Counted development learning runs allowed for ``arm`` (single source)."""
    return ARM_ALLOWANCE.get(arm, MAX_PER_ARM)


COUNTED = ("smoke", "dev_train")
# Amendment 2026-10-25 (trunk-collapse fix): learnability-gate development
# runs, a separate category with its own cap, and single-agent trunk
# pre-training runs (arm "trunk", condition "single").
GATE_KINDS = ("gate_dev", "pretrain_dev")
# gate_dev, per arm: variants C, A, B (PI decision 2026-10-25 09:10: all
# three on KXKM-AI) + the slot spent by the 0-step crash of 09:05 = 4.
# gtm, PI decisions 2026-10-25 09:18 and 09:50: + the macM1 (Metal) C check
# (stopped by hand under memory pressure), + its rerun with the day model
# paused, + the B rerun after a crash caused by our own concurrent GPU test
# = 7. PI decisions 2026-10-25 12:07-12:10, before the results: + a second
# dev seed for C (both arms), + C with and without Adam bias correction on
# the PR head code (both arms) = gtm 10, eng 7. pretrain_dev, in total (arm
# "trunk"): + the second seed's pre-training (its CUDA attempt on KXKM-AI
# crashed at 12:07, then macM1) = 4, + one KXKM-AI validation of the
# micro-batched pre-training fix for that crash (PI 2026-10-25 12:16) = 5.
# PI decision 2026-10-25 13:33, before the results: deeper Adam bias-
# correction study (C vs C-bc): mechanism runs (gtm, 2 x 200k), 3 more dev
# seeds x 2 arms x 2 presets at 1M (+3 pre-trainings on KXKM-AI) and
# 2 long gtm runs (10M) = gtm 20, eng 13, pretrain_dev 8. gate_dev runs may
# then reach GATE_MAX_STEPS.
# PI decision 2026-10-26 22:02, before any result: every N10-C-bis arm
# must pass the learnability gate under fix C before the N10-C-bis tag;
# 3 gate_dev runs per new arm (one gate run + up to 2 infrastructure
# reruns), dev seed 9101, 1M pair-steps, the seed-9101 trunk.
# N10-C-bis necessity gate (addendum G6, G7 host annex 2026-10-31, before
# any gate run): eng and nocomm, condition closed, 10M pair-steps; one run
# per arm and seed, plus the one rerun per failed run of G2 (a crashed
# attempt still uses a slot), so 2k slots per arm for k seeds. eng had used
# 13 of 13. Values for the PROPOSED k = 4 seeds (9101-9104): eng 13 + 8 =
# 21, nocomm 8. TBD by PI (seed count); edit before merge.
GATE_DEV_ALLOWANCE = {"gtm": 20, "eng": 21, "nocomm": 8, "gtm_mis": 3,
                      "eng_gtmrx": 3, "gtm_warm": 3, "vq": 3}
GATE_ALLOWANCE = {"gate_dev": 4, "pretrain_dev": 8}
GATE_MAX_STEPS = {"gate_dev": 10_000_000}


def gate_allowance(kind: str, arm: str) -> int:
    if kind == "gate_dev":
        return GATE_DEV_ALLOWANCE.get(arm, GATE_ALLOWANCE["gate_dev"])
    return GATE_ALLOWANCE[kind]
PRETRAIN_ARM, PRETRAIN_CONDITION = "trunk", "single"
# Pre-freeze registered pre-training (PI decision 2026-10-28, option A of
# the freeze plan): the exact registered procedure of one registered seed
# (or one control unit of seed 101) before the tag. Not a development
# allowance: at most one completed trunk per (seed, unit), and an attempt
# that did not give a usable trunk gets ONE restart; then the PI decides.
PREFREEZE_KIND = "pretrain_registered"
PREFREEZE_MAX_ATTEMPTS = 2
KINDS = (*COUNTED, "throughput_learning", "measurement", *GATE_KINDS,
         PREFREEZE_KIND)
LEARNING_KINDS = (*COUNTED, "throughput_learning", "gate_dev")
SMOKE_SEED = 9001  # not a registered seed
_TOKEN = object()


class SmokeLimitError(RuntimeError):
    """The development allowance would be exceeded."""


@dataclass(frozen=True)
class Reservation:
    index: int
    arm: str
    condition: str
    steps: int
    kind: str
    path: Path
    id: str = ""
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise SmokeLimitError("a Reservation can only be issued by devledger.reserve")


def read(path: Path = LEDGER) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {"max_steps": MAX_STEPS, "max_per_arm": MAX_PER_ARM,
            "arm_allowance": ARM_ALLOWANCE, "runs": []}


def _write(path: Path, ledger: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    os.replace(tmp, path)


@contextmanager
def _locked(path: Path):
    """Exclusive inter-process (and inter-thread) lock for read-modify-write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def counts(path: Path = LEDGER) -> dict[str, int]:
    """Counted development training runs per arm (any outcome)."""
    out: dict[str, int] = {}
    for r in read(path)["runs"]:
        if r.get("kind", "smoke") in COUNTED:
            out[r["arm"]] = out.get(r["arm"], 0) + 1
    return out


def counts_by_kind(path: Path = LEDGER) -> dict[str, dict[str, int]]:
    """Runs per kind and arm (any outcome)."""
    out: dict[str, dict[str, int]] = {}
    for r in read(path)["runs"]:
        k = r.get("kind", "smoke")
        out.setdefault(k, {})
        out[k][r["arm"]] = out[k].get(r["arm"], 0) + 1
    return out


def remaining(arms, path: Path = LEDGER) -> dict[str, int]:
    c = counts(path)
    return {a: allowance(a) - c.get(a, 0) for a in arms}


def reserve(arm: str, condition: str, steps: int, *, kind: str = "smoke",
            path: Path = LEDGER, seed: int = SMOKE_SEED,
            inspected: list[str] | None = None,
            unit: list[str] | None = None) -> Reservation:
    """Account one development run before it starts; raise if not allowed."""
    unit = None if unit is None else list(unit)
    from .codes import ARM_CODES, CONDITION_CODES

    if kind not in KINDS:
        raise ValueError(f"unknown ledger kind {kind!r}")
    if kind in ("pretrain_dev", PREFREEZE_KIND):
        if (arm, condition) != (PRETRAIN_ARM, PRETRAIN_CONDITION):
            raise ValueError("pretrain_dev runs use arm 'trunk', condition "
                             "'single'")
    elif arm not in ARM_CODES or condition not in CONDITION_CODES:
        raise ValueError(f"unknown arm/condition {arm!r}/{condition!r}")
    if kind == "throughput_learning" and not inspected:
        raise ValueError("a throughput_learning run must record what was "
                         "inspected (throughput, losses, success rate, ...)")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    limit = GATE_MAX_STEPS.get(kind, MAX_STEPS)
    if steps > limit:
        raise SmokeLimitError(
            f"development runs are limited to {limit:,} pair-steps "
            f"(requested {steps:,}); larger runs only via the registered path")
    with _locked(path):
        ledger = read(path)
        if kind in COUNTED:
            used = sum(1 for r in ledger["runs"]
                       if r["arm"] == arm and r.get("kind", "smoke") in COUNTED)
            if used >= allowance(arm):
                raise SmokeLimitError(
                    f"arm {arm!r} already used {used} of {allowance(arm)} smoke runs")
        if kind in GATE_KINDS:
            used = sum(1 for r in ledger["runs"]
                       if r.get("kind") == kind and r["arm"] == arm)
            if used >= gate_allowance(kind, arm):
                raise SmokeLimitError(
                    f"{kind} for {arm!r}: already used {used} of "
                    f"{gate_allowance(kind, arm)}")
        if kind == PREFREEZE_KIND:
            same = [r for r in ledger["runs"] if r.get("kind") == kind
                    and r.get("seed") == seed and r.get("unit") == unit]
            if any(r.get("outcome") == "completed" for r in same):
                raise SmokeLimitError(
                    f"{kind} seed {seed} unit {unit}: a usable trunk exists")
            if any(r.get("outcome") == "started" for r in same):
                raise SmokeLimitError(
                    f"{kind} seed {seed} unit {unit}: an attempt is still "
                    "recorded as started (running, or killed: record its "
                    "outcome first)")
            if len(same) >= PREFREEZE_MAX_ATTEMPTS:
                raise SmokeLimitError(
                    f"{kind} seed {seed} unit {unit}: {len(same)} attempts "
                    "(one restart allowed); the PI decides")
        rid = uuid.uuid4().hex
        entry = {
            "id": rid, "claimed": False, "kind": kind, "arm": arm,
            "condition": condition, "seed": seed, "steps": steps,
            "levels": "training (start_level=0, num_levels=16)",
            "policy_updates": kind != "measurement",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "outcome": "started"}
        if inspected:
            entry["inspected"] = list(inspected)
        if unit is not None:
            entry["unit"] = list(unit)
        ledger["runs"].append(entry)
        _write(path, ledger)
        return Reservation(len(ledger["runs"]) - 1, arm, condition, steps, kind,
                           Path(path), rid, _TOKEN)


def finish(res: Reservation, outcome: str) -> None:
    with _locked(res.path):
        ledger = read(res.path)
        run = next(r for r in ledger["runs"] if r.get("id") == res.id)
        run["outcome"] = outcome
        _write(res.path, ledger)


def claim(res: Reservation | None, arm: str, condition: str, steps: int) -> Reservation:
    """Atomically claim ``res`` for exactly one run at the training boundary.

    Rejects forged objects, a reservation that does not cover this run, and
    any reservation already claimed or completed (also across processes)."""
    if not isinstance(res, Reservation) or res._token is not _TOKEN:
        raise SmokeLimitError("not a genuine development-ledger reservation")
    if (res.arm, res.condition, res.steps) != (arm, condition, steps):
        raise SmokeLimitError(
            f"reservation ({res.arm}, {res.condition}, {res.steps}) does not "
            f"cover this run ({arm}, {condition}, {steps})")
    with _locked(res.path):
        ledger = read(res.path)
        run = next((r for r in ledger["runs"] if r.get("id") == res.id), None)
        if run is None:
            raise SmokeLimitError("reservation not found in the ledger")
        if run.get("claimed") or run.get("outcome") != "started":
            raise SmokeLimitError(
                f"reservation {res.id} already claimed or completed")
        run["claimed"] = True
        run["claimed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write(res.path, ledger)
    return res
