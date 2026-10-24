"""Persistent development-run ledger (no mlx import).

One per-arm ledger, ``manifest/smoke_ledger.json``, accounts for EVERY
real-environment development run, whichever path launched it (smoke runs,
throughput measurements, a direct development ``train_run`` or the trainer
CLI). Amendment, Configuration manifest: development may use "smoke runs of
at most 1M steps (at most 3 per arm, all disclosed in the manifest with their
outcomes), throughput measurements on random or untrained policies".

Kinds:

* ``smoke`` and ``dev_train``: runs whose policy learns. At most
  ``allowance(arm)`` per arm (3; gtm 4), each at most ``MAX_STEPS`` pair-steps.
* ``measurement``: throughput measurements. The policy stays at its random
  initialisation (the optimiser runs with learning rate 0), so the amendment's
  "random or untrained policies" wording holds. Ledgered, not counted.
* ``legacy_measurement``: throughput / profiling runs executed BEFORE this
  rule with learning updates enabled (disclosed; not counted, not erased).

A reservation is written BEFORE the run starts: a crashed run still counts.
"""
from __future__ import annotations

import json
import time
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
KINDS = (*COUNTED, "measurement", "legacy_measurement")
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
    tmp.replace(path)


def counts(path: Path = LEDGER) -> dict[str, int]:
    """Counted development training runs per arm (any outcome)."""
    out: dict[str, int] = {}
    for r in read(path)["runs"]:
        if r.get("kind", "smoke") in COUNTED:
            out[r["arm"]] = out.get(r["arm"], 0) + 1
    return out


def remaining(arms, path: Path = LEDGER) -> dict[str, int]:
    c = counts(path)
    return {a: allowance(a) - c.get(a, 0) for a in arms}


def reserve(arm: str, condition: str, steps: int, *, kind: str = "smoke",
            path: Path = LEDGER, seed: int = SMOKE_SEED) -> Reservation:
    """Account one development run before it starts; raise if not allowed."""
    from .codes import ARM_CODES, CONDITION_CODES

    if arm not in ARM_CODES or condition not in CONDITION_CODES:
        raise ValueError(f"unknown arm/condition {arm!r}/{condition!r}")
    if kind not in ("smoke", "dev_train", "measurement"):
        raise ValueError(f"unknown ledger kind {kind!r}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if steps > MAX_STEPS:
        raise SmokeLimitError(
            f"development runs are limited to {MAX_STEPS:,} pair-steps "
            f"(requested {steps:,}); larger runs only via the registered path")
    ledger = read(path)
    if kind in COUNTED:
        used = sum(1 for r in ledger["runs"]
                   if r["arm"] == arm and r.get("kind", "smoke") in COUNTED)
        if used >= allowance(arm):
            raise SmokeLimitError(
                f"arm {arm!r} already used {used} of {allowance(arm)} smoke runs")
    ledger["runs"].append({
        "kind": kind, "arm": arm, "condition": condition, "seed": seed,
        "steps": steps, "levels": "training (start_level=0, num_levels=16)",
        "policy_updates": kind != "measurement",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "outcome": "started"})
    _write(path, ledger)
    return Reservation(len(ledger["runs"]) - 1, arm, condition, steps, kind,
                       Path(path), _TOKEN)


def finish(res: Reservation, outcome: str) -> None:
    ledger = read(res.path)
    ledger["runs"][res.index]["outcome"] = outcome
    _write(res.path, ledger)


def check(res: Reservation | None, arm: str, condition: str, steps: int) -> Reservation:
    """The reservation handed to a trainer must match the run it covers."""
    if not isinstance(res, Reservation) or res._token is not _TOKEN:
        raise SmokeLimitError("not a genuine development-ledger reservation")
    if (res.arm, res.condition, res.steps) != (arm, condition, steps):
        raise SmokeLimitError(
            f"reservation ({res.arm}, {res.condition}, {res.steps}) does not "
            f"cover this run ({arm}, {condition}, {steps})")
    return res
