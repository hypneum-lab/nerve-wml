"""N10-C-bis necessity gate A+ (G2 at k = 3): categorical decision; no mlx
or numpy import.

Plan ``docs/prereg/n10c-bis/three-seeds-plan-2026-11-06.md`` (sections 1.3,
6.5, 6.6) and ``docs/prereg/n10c-bis/RUNBOOK-aplus.md`` (sections 4, 5).
Development runs (ledger kind ``gate_dev``), condition ``closed``, preset
``fix-c-coin-pretrain``, 10,000,000 pair-steps, three cells per seed S on
one host and one trunk (the trunk of S):

* ``eng``        arm eng,    seed S;
* ``nocomm``     arm nocomm, seed S;
* ``nocomm_rep`` arm nocomm, seed S + 10 (same trunk, other random stream).

Run directories: ``closed-<seed>-<arm>-t<trunk seed>-a<attempt>`` holding
the trainer's ``run.json`` (attempt 1, or 2 for the one rerun of G2).

Two commands, neither prints a measured value:

``check-cell DIR ...`` prints ``OK`` or ``FAIL:<conditions>`` for one run
(used by the lanes to decide the one rerun): identity, completion, trunk,
host settings, liveness schedule, the last five logs and the per-run
learnability gate (no liveness flag after step 0, own-coin rate rising).

``decide ...`` prints ONE category and writes every value (team success
per run, D_s, D0_s, D-bar, eng control, chosen attempts, record SHA-256) to
a sealed JSON that must not exist yet (created read-only). Rules, in order:

1. learnability gate: a seed whose ``eng`` or ``nocomm`` has no usable
   attempt is not measurable -> ``NOT_MEASURABLE_LEARNING`` (G2: the PI
   decides);
2. ``eng`` control (final team success >= 0.05 and above its first log)
   fails on >= 2 seeds -> ``NOT_MEASURABLE_ENG_CONTROL``;
3. noise control: |D0_s| >= 0.10 (or no usable ``nocomm_rep``) on >= 2
   seeds -> ``INCONCLUSIVE_NOISE``;
4. D-bar = mean over seeds of (eng - nocomm) >= 0.20 -> ``PASS``, else
   ``FAIL``.

Team success of a run = mean of its last 5 logged success values (last 100
completed episodes at each log; G2, ``MS:337``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from itertools import pairwise
from pathlib import Path

from .gate_report import GATE_MIN_TEAM_SUCCESS, probe_flagged

STEPS = 10_000_000
LOG_EVERY = 200_000
LAST_N = 5
LAST_STEPS = [STEPS - LOG_EVERY * k for k in range(LAST_N - 1, -1, -1)]
WINDOW = 100
PROBE_EVERY = 100_000
N_PROBES = STEPS // PROBE_EVERY  # after step 0
PRESET = "fix-c-coin-pretrain"
CONDITION = "closed"
REP_OFFSET = 10  # nocomm_rep of seed S runs on seed S + 10 with trunk S
SEED_HOST = {9104: "kxkm", 9105: "macm1", 9106: "kxkm"}
SEEDS = tuple(sorted(SEED_HOST))
# init_trunk_sha256 of the Adam-study records (RUNBOOK-aplus.md section 2);
# the trunk of 9106 does not exist yet: --trunk-sha 9106=<hex>
TRUNK_SHA = {
    9104: "894688cd4a62b868ea1f18d7021be57a64acb52699a8742048c9e2508da4f16f",
    9105: "d4916633ae7203f5e6ba5353451abf32fd280fa70c1b2404d1fb5803e2538a7d",
}
# (micro_batch_pairs, clear_cache, n_workers), equal to codes.HOST_SETTINGS
# (asserted in tests; codes imports numpy)
HOST_SETTINGS = {"kxkm": (512, True, 8), "macm1": (None, False, 1)}
PASS_MARGIN = 0.20   # G2
NOISE_MAX = 0.10     # N10-C SESOI
ENG_MIN_SUCCESS = GATE_MIN_TEAM_SUCCESS  # 0.05 = 2 x uniform-random pair
MAX_FAILED_SEEDS = 1  # rules 2 and 3 trigger on >= 2 seeds of 3
EPS = 1e-9  # float tolerance on the thresholds (0.7 - 0.5 is not 0.2)
CATEGORIES = ("PASS", "FAIL", "NOT_MEASURABLE_LEARNING",
              "NOT_MEASURABLE_ENG_CONTROL", "INCONCLUSIVE_NOISE")
CELLS = (("eng", "eng", 0), ("nocomm", "nocomm", 0),
         ("nocomm_rep", "nocomm", REP_OFFSET))
_DIR = re.compile(r"^closed-(?P<seed>\d+)-(?P<arm>eng|nocomm)"
                  r"-t(?P<trunk>\d+)-a(?P<att>\d+)$")


class DecideError(RuntimeError):
    """The inputs do not allow a decision (message carries no value)."""


def _finite(x) -> bool:
    return (isinstance(x, int | float) and not isinstance(x, bool)
            and math.isfinite(x))


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def cell_dir(seed: int, arm: str, trunk: int, attempt: int) -> str:
    return f"closed-{seed}-{arm}-t{trunk}-a{attempt}"


def _curve(run: dict, key: str) -> list | None:
    c = run.get(key)
    if not isinstance(c, list) or not c:
        return None
    try:
        steps = [s for s, _ in c]
    except (TypeError, ValueError):
        return None
    if not all(_int(s) for s in steps) or any(
            b <= a for a, b in pairwise(steps)):
        return None
    return c


def _probes_cover(live) -> bool:
    if not isinstance(live, list) or not live:
        return False
    steps = [m.get("step") if isinstance(m, dict) else None for m in live]
    if steps[0] != 0 or not all(_int(s) for s in steps):
        return False
    late = steps[1:]
    return ([s // PROBE_EVERY for s in late] == list(range(1, N_PROBES + 1))
            and all(b > a for a, b in pairwise(late)) and late[-1] == STEPS)


def cell_problems(run: dict, *, arm: str, seed: int, trunk_sha: str,
                  host: str) -> list[str]:
    """Names of the failed conditions of one run (empty = usable)."""
    bad = []
    opts = run.get("train_options") or {}
    for name, ok in (
            ("arm", run.get("arm") == arm),
            ("seed", run.get("seed") == seed),
            ("condition", run.get("condition") == CONDITION),
            ("preset", opts.get("preset") == PRESET),
            ("using_stub", run.get("using_stub") is False),
            ("training_failure", run.get("training_failure") is False),
            ("liveness_abort", run.get("liveness_abort") is False),
            ("steps", run.get("total_pair_steps") == STEPS
             and run.get("pair_steps_done") == STEPS),
            ("trunk", run.get("init_trunk_sha256") == trunk_sha),
            ("liveness_mode", (opts.get("liveness"), opts.get("liveness_every"))
             == ("abort", PROBE_EVERY)),
            ("settings", (run.get("micro_batch_pairs"), run.get("clear_cache"),
                          run.get("n_workers")) == HOST_SETTINGS[host]),
            ("probes", _probes_cover(run.get("liveness")))):
        if not ok:
            bad.append(name)
    sc = _curve(run, "success_curve")
    last = sc[-LAST_N:] if sc else []
    if [p[0] for p in last] != LAST_STEPS:
        bad.append("last5_present")
    elif not all(_finite(p[1]) for p in last):
        bad.append("last5_finite")
    n = run.get("success_curve_n")
    if not (isinstance(n, list) and sc and len(n) == len(sc)
            and n[-LAST_N:] == [WINDOW] * LAST_N):
        bad.append("window_100")
    # per-run learnability gate (plan section 1.3, MS:290), at 10M
    live = run.get("liveness")
    if isinstance(live, list) and any(
            probe_flagged(m) for m in live[1:] if isinstance(m, dict)):
        bad.append("liveness_flagged")
    cc = _curve(run, "coin_rate_curve")
    if not (cc and len(cc) >= 2 and all(_finite(v) for _, v in cc)
            and cc[-1][1] > cc[0][1]):
        bad.append("coin_rate_rise")
    return bad


def team_success(run: dict) -> float:
    """Mean of the last 5 logged success values (G2)."""
    return sum(v for _, v in run["success_curve"][-LAST_N:]) / LAST_N


def eng_control_ok(run: dict) -> bool:
    """Positive control: final team success >= 0.05 and above its first
    logged value (plan section 1.3)."""
    sc = run["success_curve"]
    first, final = sc[0][1], sc[-1][1]
    return _finite(first) and final >= ENG_MIN_SUCCESS - EPS and final > first


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict | None:
    try:
        r = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return r if isinstance(r, dict) else None


def check_cell(d: Path, *, arm: str, seed: int, trunk_sha: str,
               host: str) -> str:
    """``OK`` or ``FAIL:<conditions>`` (never a value)."""
    f = Path(d) / "run.json"
    if not f.exists():
        return "FAIL:no_run_json"
    r = _load(f)
    if r is None:
        return "FAIL:malformed_run_json"
    bad = cell_problems(r, arm=arm, seed=seed, trunk_sha=trunk_sha, host=host)
    return "OK" if not bad else "FAIL:" + ",".join(bad)


def _index(roots: list[Path]) -> dict[str, Path]:
    """Cell directories of the A+ layout found under ``roots`` (one level);
    a name seen twice is refused (the two hosts' copies must not overlap)."""
    out: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise DecideError(f"runs root {root} is not a directory")
        for d in sorted(root.iterdir()):
            if d.is_dir() and _DIR.match(d.name):
                if d.name in out:
                    raise DecideError(f"cell directory {d.name} found twice")
                out[d.name] = d
    return out


def _pick(index: dict[str, Path], *, seed: int, arm: str, trunk: int,
          trunk_sha: str, host: str) -> dict:
    """The usable attempt of one cell: attempt 1 if usable, else attempt 2
    (the one rerun of G2). More than two attempts is refused."""
    extra = [n for n in index
             if (m := _DIR.match(n)) and (int(m["seed"]), m["arm"],
                                          int(m["trunk"])) == (seed, arm, trunk)
             and int(m["att"]) not in (1, 2)]
    if extra:
        raise DecideError(f"cell {seed}/{arm}/t{trunk}: attempts other than "
                          f"1 and 2 found: {sorted(extra)}")
    cell = {"seed": seed, "arm": arm, "trunk_seed": trunk, "attempts": {},
            "chosen": None, "anomalies": []}
    record = None
    for att in (1, 2):
        d = index.get(cell_dir(seed, arm, trunk, att))
        if d is None:
            continue
        f = d / "run.json"
        st = check_cell(d, arm=arm, seed=seed, trunk_sha=trunk_sha, host=host)
        cell["attempts"][att] = {"dir": str(d), "status": st,
                                 "sha256": _sha(f) if f.exists() else None}
        if st == "OK" and record is None:
            record, cell["chosen"] = _load(f), att
    if 1 not in cell["attempts"] and 2 in cell["attempts"]:
        cell["anomalies"].append("attempt 2 without attempt 1")
    if cell["chosen"] == 1 and 2 in cell["attempts"]:
        cell["anomalies"].append("attempt 2 exists although attempt 1 is usable")
    cell["record"] = record
    return cell


def decide(roots: list[Path], *, seeds=SEEDS,
           trunk_sha: dict[int, str] | None = None) -> tuple[str, dict]:
    """(category, sealed values) of the A+ gate."""
    shas = dict(TRUNK_SHA) | dict(trunk_sha or {})
    if len(seeds) != len(set(seeds)) or not seeds:
        raise DecideError("seeds must be distinct and non-empty")
    for s in seeds:
        if s not in SEED_HOST:
            raise DecideError(f"seed {s} is not an A+ seed {SEEDS}")
        if not re.fullmatch(r"[0-9a-f]{64}", shas.get(s) or ""):
            raise DecideError(f"no SHA-256 for the trunk of seed {s} "
                              f"(--trunk-sha {s}=<64 hex>)")
    index = _index(roots)
    per_seed, trace = {}, []
    for s in seeds:
        cells = {name: _pick(index, seed=s + off, arm=arm, trunk=s,
                             trunk_sha=shas[s], host=SEED_HOST[s])
                 for name, arm, off in CELLS}
        rec = {k: c.pop("record") for k, c in cells.items()}
        v = {"host": SEED_HOST[s], "trunk_sha256": shas[s], "cells": cells,
             "learned": rec["eng"] is not None and rec["nocomm"] is not None}
        for k, r in rec.items():
            v[f"success_{k}"] = None if r is None else team_success(r)
        v["eng_control_ok"] = (None if rec["eng"] is None
                               else eng_control_ok(rec["eng"]))
        if v["learned"]:
            v["D"] = v["success_eng"] - v["success_nocomm"]
        else:
            v["D"] = None
        if v["success_nocomm"] is not None and v["success_nocomm_rep"] is not None:
            v["D0"] = v["success_nocomm_rep"] - v["success_nocomm"]
        else:
            v["D0"] = None
        per_seed[s] = v
    k = len(seeds)
    lost = [s for s in seeds if not per_seed[s]["learned"]]
    eng_fail = [s for s in seeds if per_seed[s]["eng_control_ok"] is not True]
    noise_fail = [s for s in seeds if per_seed[s]["D0"] is None
                  or abs(per_seed[s]["D0"]) >= NOISE_MAX - EPS]
    ds = [per_seed[s]["D"] for s in seeds if per_seed[s]["D"] is not None]
    dbar = sum(ds) / len(ds) if len(ds) == k else None
    if lost:
        cat = "NOT_MEASURABLE_LEARNING"
        trace.append(f"rule 1: seeds {lost} have no usable eng or nocomm run")
    elif len(eng_fail) > MAX_FAILED_SEEDS:
        cat = "NOT_MEASURABLE_ENG_CONTROL"
        trace.append(f"rule 2: eng control failed on seeds {eng_fail}")
    elif len(noise_fail) > MAX_FAILED_SEEDS:
        cat = "INCONCLUSIVE_NOISE"
        trace.append(f"rule 3: noise control failed on seeds {noise_fail}")
    else:
        cat = "PASS" if dbar >= PASS_MARGIN - EPS else "FAIL"
        trace.append("rule 4: D-bar against 0.20")
    sealed = {
        "category": cat, "rule_trace": trace,
        "rules": {"pass_margin": PASS_MARGIN, "noise_max": NOISE_MAX,
                  "eng_min_success": ENG_MIN_SUCCESS,
                  "max_failed_seeds": MAX_FAILED_SEEDS, "eps": EPS,
                  "team_success": f"mean of the last {LAST_N} logs "
                                  f"(steps {LAST_STEPS}), window {WINDOW}"},
        "seeds": list(seeds), "rep_offset": REP_OFFSET,
        "lost_seeds": lost, "eng_control_failed": eng_fail,
        "noise_control_failed": noise_fail, "D_bar": dbar,
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "decide_script_sha256": _sha(Path(__file__)),
        "roots": [str(Path(r).resolve()) for r in roots],
    }
    return cat, sealed


def write_sealed(path: Path, sealed: dict) -> None:
    """Create ``path`` read-only; refuse to overwrite an earlier decision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError:
        raise DecideError(f"sealed file {path} exists: refuse") from None
    with os.fdopen(fd, "w") as f:
        json.dump(sealed, f, indent=2)
        f.write("\n")


def _trunk_arg(items: list[str]) -> dict[int, str]:
    out = {}
    for it in items or []:
        s, _, h = it.partition("=")
        if not s.isdigit() or not re.fullmatch(r"[0-9a-f]{64}", h):
            raise DecideError("--trunk-sha takes SEED=<64 lowercase hex>")
        out[int(s)] = h
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-cell", help="OK / FAIL:<conditions> for one run")
    c.add_argument("dir", type=Path)
    c.add_argument("--arm", required=True, choices=("eng", "nocomm"))
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--trunk-seed", type=int, required=True)
    c.add_argument("--host", required=True, choices=sorted(HOST_SETTINGS))
    c.add_argument("--trunk-sha", default=None,
                   help="SHA-256 of the trunk (default: known A+ trunks)")
    d = sub.add_parser("decide", help="print the category; seal the values")
    d.add_argument("--runs-root", type=Path, action="append", required=True,
                   help="directory holding closed-* cells (repeat per host)")
    d.add_argument("--sealed-out", type=Path, required=True,
                   help="new file for the sealed values (must not exist)")
    d.add_argument("--trunk-sha", action="append", default=[],
                   help="SEED=<sha256>, needed for 9106")
    d.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args(argv)
    try:
        if a.cmd == "check-cell":
            sha = a.trunk_sha or TRUNK_SHA.get(a.trunk_seed)
            if not sha or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise DecideError(f"no SHA-256 for the trunk of seed "
                                  f"{a.trunk_seed} (--trunk-sha)")
            st = check_cell(a.dir, arm=a.arm, seed=a.seed, trunk_sha=sha,
                            host=a.host)
            print(st)
            return 0 if st == "OK" else 1
        cat, sealed = decide(a.runs_root, seeds=tuple(a.seeds),
                             trunk_sha=_trunk_arg(a.trunk_sha))
        write_sealed(a.sealed_out, sealed)
        print(cat)
        return 0
    except DecideError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
