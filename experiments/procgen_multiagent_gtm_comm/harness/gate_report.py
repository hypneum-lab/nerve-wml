"""Learnability-gate report (amendment 2026-10-25), no mlx import.

Reads ``<gate_dir>/pretrain/pretrain.json`` and ``<gate_dir>/<preset>-<arm>
[suffix]/run.json`` and prints markdown tables: per run the liveness probes
over time, policy entropy (minimum vs mean of live agents), per-agent
own-coin rate, team success against the uniform-random baseline, wall time
and the gate verdict of section 4 of the amendment.

``table INDEX`` prints the gate-results table of the amendment (section
4.2) from the committed run records listed in a research-cell index
(``docs/superpowers/research/2026-10-25-n10c-gate/index.json``), after
checking each record's SHA-256 and identity against the index.

The verdict trusts no directory name: the record's arm, seed, condition and
preset must equal the expected ones, the liveness probes must cover the
schedule exactly (step 0, then one probe per 100,000-step interval up to
1M, no duplicate), the flags are recomputed from the probe metrics, and
the success and coin-rate curves must be finite, strictly increasing in
step and end at 1M.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from itertools import pairwise
from pathlib import Path

RANDOM_TEAM_SUCCESS_MAX = 0.025  # uniform-random pair, training levels 0-15
GATE_MIN_TEAM_SUCCESS = 2 * RANDOM_TEAM_SUCCESS_MAX
GATE_STEPS = 1_000_000
GATE_WINDOW = 100  # completed episodes in the final success measurement
GATE_PROBE_EVERY = 100_000
GATE_PROBES = GATE_STEPS // GATE_PROBE_EVERY  # after step 0 (11 with step 0)
GATE_PRESET = "fix-c-coin-pretrain"
GATE_ARMS = ("gtm", "eng")
GATE_SEED = 9101  # first development seed (not a registered seed)
GATE_CONDITION = "registered"
# Same thresholds as harness.train.LIVENESS_* (that module imports mlx;
# equality is asserted in tests/unit/n10c/test_trunk_fix.py).
LIVENESS_MIN_FRAC = 0.01
LIVENESS_STD_EPS = 1e-6
_DIR = re.compile(r"^(?P<preset>.+?)-(?P<arm>gtm|eng)(?P<suffix>-.+)?$")


def _finite(x) -> bool:
    return (isinstance(x, int | float) and not isinstance(x, bool)
            and math.isfinite(x))


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def probe_flagged(m: dict) -> bool:
    """The guard's flag recomputed from the probe metrics (missing or
    non-finite metrics count as flagged), or the recorded flag."""
    vals = [m.get(k) for k in ("block3_pos_frac", "live_h_frac",
                               "logit_std", "value_std")]
    if not all(_finite(v) for v in vals):
        return True
    b3, lh, ls, vs = vals
    return bool(m.get("flagged") or b3 < LIVENESS_MIN_FRAC
                or lh < LIVENESS_MIN_FRAC or ls <= LIVENESS_STD_EPS
                or vs <= LIVENESS_STD_EPS)


def _curve_ok(curve, name: str, why: list[str]) -> list[float] | None:
    """Finite values, strictly increasing integer steps, last step 1M."""
    if not isinstance(curve, list) or not curve:
        why.append(f"{name}: empty")
        return None
    try:
        steps = [s if isinstance(s, int) and not isinstance(s, bool)
                 else None for s, _ in curve]
        vals = [v for _, v in curve]
    except (TypeError, ValueError):
        why.append(f"{name}: malformed")
        return None
    if None in steps or any(b <= a for a, b in pairwise(steps)):
        why.append(f"{name}: steps not strictly increasing")
        return None
    if steps[-1] != GATE_STEPS:
        why.append(f"{name}: ends at {steps[-1]:,}, not {GATE_STEPS:,}")
        return None
    if not all(_finite(v) for v in vals):
        why.append(f"{name}: non-finite value")
        return None
    return vals


def verdict(run: dict, *, arm: str, preset: str = GATE_PRESET,
            seed: int = GATE_SEED,
            condition: str = GATE_CONDITION) -> tuple[bool, list[str]]:
    """Per-run gate criterion (amendment 2026-10-25, section 4) for the run
    expected to be (``arm``, ``preset``, ``seed``, ``condition``)."""
    why = []
    opts = run.get("train_options") or {}
    got = {"arm": run.get("arm"), "preset": opts.get("preset"),
           "seed": run.get("seed"), "condition": run.get("condition")}
    want = {"arm": arm, "preset": preset, "seed": seed, "condition": condition}
    for k in want:
        if got[k] != want[k]:
            why.append(f"{k} {got[k]!r} != expected {want[k]!r}")
    if (opts.get("liveness"), opts.get("liveness_every")) != (
            "abort", GATE_PROBE_EVERY):
        why.append("liveness guard not in abort mode every "
                   f"{GATE_PROBE_EVERY:,} steps")
    if run.get("training_failure") or run.get("liveness_abort"):
        why.append("training failure / liveness abort")
    if not (run.get("total_pair_steps") == run.get("pair_steps_done")
            == GATE_STEPS):
        why.append(f"not a complete {GATE_STEPS:,}-step run")
    live = run.get("liveness") or []
    steps = [m.get("step") for m in live]
    if not live or steps[0] != 0:
        why.append("no liveness probe at step 0")
    late = live[1:]
    ints = all(isinstance(s, int) and not isinstance(s, bool) for s in steps)
    buckets = [s // GATE_PROBE_EVERY for s in steps[1:]] if ints else None
    if (buckets != list(range(1, GATE_PROBES + 1))
            or any(b <= a for a, b in pairwise(steps[1:]))
            or steps[-1] != GATE_STEPS):
        why.append(f"liveness probes after step 0 do not cover the schedule "
                   f"(need exactly one per {GATE_PROBE_EVERY:,}-step interval, "
                   f"the last at {GATE_STEPS:,}, {GATE_PROBES} in all; got steps "
                   f"{steps[1:]})")
    if any(probe_flagged(m) for m in late):
        why.append("liveness flagged")
    sc = _curve_ok(run.get("success_curve"), "success curve", why)
    n = run.get("success_curve_n") or []
    if sc is not None:
        if not n or len(n) != len(run["success_curve"]) or n[-1] != GATE_WINDOW:
            why.append(f"no last-{GATE_WINDOW}-episode success at "
                       f"{GATE_STEPS:,}")
        elif sc[-1] < GATE_MIN_TEAM_SUCCESS:
            why.append(f"final team success {_f(sc[-1])} < "
                       f"{GATE_MIN_TEAM_SUCCESS}")
    cc = _curve_ok(run.get("coin_rate_curve"), "coin-rate curve", why)
    if cc is not None and (len(cc) < 2 or cc[-1] <= cc[0]):
        why.append("coin rate did not rise")
    return not why, why


def parse_run_dir(name: str) -> tuple[str, str, str] | None:
    """``<preset>-<arm>[suffix]`` -> (preset, arm, suffix), else None."""
    m = _DIR.match(name)
    return None if m is None else (m["preset"], m["arm"], m["suffix"] or "")


def combined_verdict(gate_dir: Path, preset: str = GATE_PRESET,
                     suffix: str = "", seed: int = GATE_SEED
                     ) -> tuple[bool, list[str]]:
    """The gate passes only if BOTH arms of ``preset`` pass on one host."""
    why = []
    for arm in GATE_ARMS:
        f = Path(gate_dir) / f"{preset}-{arm}{suffix}" / "run.json"
        if not f.exists():
            why.append(f"{preset}-{arm}{suffix}: missing")
            continue
        ok, w = verdict(json.loads(f.read_text()), arm=arm, preset=preset,
                        seed=seed)
        if not ok:
            why.append(f"{preset}-{arm}{suffix}: " + "; ".join(w))
    return not why, why


def report(gate_dir: Path, host: str = "", seed: int = GATE_SEED) -> str:
    out = []
    pj = gate_dir / "pretrain" / "pretrain.json"
    if pj.exists():
        p = json.loads(pj.read_text())
        cc = p["coin_rate_curve"]
        out.append("**Trunk pre-training** (single-agent CoinRun, levels 0-15, "
                   f"seed {p['seed']}): {p['steps_done']:,} steps, "
                   f"{p['wall_time_s'] / 60:.1f} min, usable={p['usable']}, "
                   f"trunk SHA-256 `{p['trunk_sha256']}`.")
        out.append("Coin rate (last 100 episodes): " + ", ".join(
            f"{s // 1000}k {_f(r, 2)}" for s, r in cc))
        out.append("Liveness (block-3 pos / live h): " + ", ".join(
            f"{m['step'] // 1000}k {m['block3_pos_frac']:.2f}/"
            f"{m['live_h_frac']:.2f}" for m in p["liveness"]))
        out.append("")
    runs = sorted(d for d in gate_dir.iterdir()
                  if d.is_dir() and (d / "run.json").exists())
    out.append("| Host | Run | Steps | Wall | Team success (final, max) vs random ≤0.025 "
               "| Own-coin rate first → final | Entropy mean (last) / min (all) "
               "| Liveness flagged | Gate |")
    out.append("|---|---|---:|---:|---|---|---|---|---|")
    detail = []
    for d in runs:
        r = json.loads((d / "run.json").read_text())
        sc = [x for _, x in r["success_curve"] if x is not None]
        cc = [x for _, x in r["coin_rate_curve"] if x is not None]
        ec = r["entropy_curve"]
        parsed = parse_run_dir(d.name)
        if parsed is None:
            ok, why = False, [f"run directory {d.name!r} is not <preset>-<arm>"]
        else:
            ok, why = verdict(r, arm=parsed[1], preset=parsed[0], seed=seed)
        out.append(
            f"| {host} | {d.name} | {r['pair_steps_done']:,} | "
            f"{r['wall_time_s'] / 60:.0f} min | "
            f"{_f(sc[-1] if sc else None)}, {_f(max(sc) if sc else None)} | "
            f"{_f(cc[0] if cc else None)} → {_f(cc[-1] if cc else None)} | "
            f"{_f(ec[-1][1] if ec else None, 2)} / "
            f"{_f(min(e[2] for e in ec) if ec else None, 2)} | "
            f"{'yes' if r['liveness_flagged'] else 'no'} | "
            f"{'PASS' if ok else 'FAIL: ' + '; '.join(why)} |")
        detail.append(f"- {d.name} liveness (step: block-3 pos / live h / "
                      "logit std / value std): " + "; ".join(
                          f"{m['step'] // 1000}k {m['block3_pos_frac']:.2f}/"
                          f"{m['live_h_frac']:.2f}/{m['logit_std']:.3f}/"
                          f"{m['value_std']:.3f}" for m in r["liveness"]))
        detail.append(f"- {d.name} team success every 40k: " + ", ".join(
            _f(x, 3) for _, x in r["success_curve"]))
    ok, why = combined_verdict(gate_dir, seed=seed)
    out.append("")
    out.append(f"**Combined gate ({GATE_PRESET}, both arms, {host or 'this host'}):"
               f" {'PASS' if ok else 'FAIL'}**" + ("" if ok else
                                                   " — " + "; ".join(why)))
    return "\n".join(out + [""] + detail)


# -- amendment tables from the committed research cells ----------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_index(index_path: Path) -> tuple[dict, list[str]]:
    """Read a gate-cell index; return it and the integrity problems found
    (SHA-256 mismatch, record identity differing from the index)."""
    index_path = Path(index_path)
    idx = json.loads(index_path.read_text())
    bad = []
    for e in (idx["runs"] + idx.get("pretrain", [])
              + idx.get("adam_study", [])):
        f = index_path.parent / e["path"]
        if not f.exists():
            bad.append(f"{e['path']}: missing")
            continue
        if sha256_file(f) != e["sha256"]:
            bad.append(f"{e['path']}: SHA-256 differs from the index")
            continue
        r = json.loads(f.read_text())
        if "arm" in e:
            got = (r.get("arm"), r.get("seed"),
                   (r.get("train_options") or {}).get("preset"),
                   r.get("condition"))
            want = (e["arm"], e["seed"], e["preset"], e["condition"])
            if got != want:
                bad.append(f"{e['path']}: record {got} != index {want}")
        elif r.get("seed") != e["seed"]:
            bad.append(f"{e['path']}: seed {r.get('seed')} != {e['seed']}")
    return idx, bad


_SHORT = {"fix-c-coin-pretrain": "**C**", "fix-c-coin-pretrain-bc": "C-bc",
          "fix-a-coin": "A", "fix-b-pretrain": "B"}


def table_rows(index_path: Path) -> list[str]:
    """Rows of the amendment's gate-results table (section 4.2)."""
    idx, bad = load_index(index_path)
    if bad:
        raise ValueError("gate-cell index check failed: " + "; ".join(bad))
    rows = []
    for e in idx["runs"]:
        r = json.loads((Path(index_path).parent / e["path"]).read_text())
        sc = [x for _, x in r["success_curve"]]
        cc = [x for _, x in r["coin_rate_curve"]]
        late = r["liveness"][1:]
        ok, _ = verdict(r, arm=e["arm"], preset=e["preset"], seed=e["seed"])
        c = e["preset"] == GATE_PRESET
        fin = f"{sc[-1]:.2f}"
        rows.append(
            f"| {e['row']} | {e['host']} | {e['seed']} | {e['arm']} | "
            f"{_SHORT.get(e['preset'], e['preset'])} | "
            f"{'**' + fin + '**' if c else fin} ({max(sc):.2f}) | "
            f"{cc[0]:.3f} → {cc[-1]:.3f} | "
            f"{min(m['block3_pos_frac'] for m in late):.3f} | "
            f"{min(m['live_h_frac'] for m in late):.2f} | "
            f"{'yes' if any(probe_flagged(m) for m in late) else 'no'} / "
            f"{'yes' if r.get('liveness_abort') else 'no'} | "
            f"{'PASS' if ok else 'FAIL'} | {r['wall_time_s']:,.0f} s |")
    return rows


TABLE_HEADER = (
    "| # | host | seed | arm | preset | team success at 1M (max) "
    "| coin rate first → 1M | min block-3 positive fraction "
    "| min live-h fraction | flagged / abort | criterion | wall time |\n"
    "|---|---|---|---|---|---|---|---:|---:|---|---|---:|")



# -- Adam bias-correction study, part A (amendment section 4.3) -------------
ADAM_PRESETS = ("fix-c-coin-pretrain", "fix-c-coin-pretrain-bc")
LAST_WINDOW = 200_000  # "last 200k": success-curve points after 800k


def _last_mean(run: dict) -> float:
    pts = [v for s, v in run["success_curve"] if s > GATE_STEPS - LAST_WINDOW]
    return sum(pts) / len(pts)


def _mean_block3(run: dict) -> float:
    late = run["liveness"][1:]
    return sum(m["block3_pos_frac"] for m in late) / len(late)


ADAM_HEADER = (
    "| seed | arm | C at 1M | C-bc at 1M | C last-200k mean "
    "| C-bc last-200k mean | C-bc − C | mean block-3 positive fraction "
    "C / C-bc | criterion C / C-bc |\n"
    "|---|---|---:|---:|---:|---:|---:|---|---|")


def adam_rows(index_path: Path) -> tuple[list[str], str]:
    """Rows of the part-A table (C vs C-bc per seed and arm, 1M runs) and a
    one-line summary, from the ``adam_study`` records of the index."""
    idx, bad = load_index(index_path)
    if bad:
        raise ValueError("gate-cell index check failed: " + "; ".join(bad))
    cells: dict[tuple[int, str], dict[str, dict]] = {}
    for e in idx.get("adam_study", []):
        if e.get("part") != "A" or e.get("kind") != "gate_dev":
            continue
        r = json.loads((Path(index_path).parent / e["path"]).read_text())
        cells.setdefault((e["seed"], e["arm"]), {})[e["preset"]] = r
    rows, diffs, b3_up = [], [], 0
    for (seed, arm) in sorted(cells):
        c, bc = (cells[(seed, arm)][p] for p in ADAM_PRESETS)
        lc, lb = _last_mean(c), _last_mean(bc)
        bc3, bb3 = _mean_block3(c), _mean_block3(bc)
        diffs.append(lb - lc)
        b3_up += bb3 > bc3
        ok = [verdict(r, arm=arm, preset=p, seed=seed)[0]
              for r, p in ((c, ADAM_PRESETS[0]), (bc, ADAM_PRESETS[1]))]
        rows.append(
            f"| {seed} | {arm} | {c['success_curve'][-1][1]:.2f} | "
            f"{bc['success_curve'][-1][1]:.2f} | {lc:.3f} | {lb:.3f} | "
            f"{lb - lc:+.3f} | {bc3:.3f} / {bb3:.3f} | "
            f"{'PASS' if ok[0] else 'FAIL'} / {'PASS' if ok[1] else 'FAIL'} |")
    n = len(diffs)
    summary = (f"C-bc >= C on the last-200k mean in {sum(d >= 0 for d in diffs)} "
               f"of {n} cells, mean difference {sum(diffs) / n:+.3f}; mean "
               f"block-3 positive fraction higher with bias correction in "
               f"{b3_up} of {n}.")
    return rows, summary

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("table", help="amendment gate table from a cell index")
    t.add_argument("index", type=Path)
    t2 = sub.add_parser("adam", help="Adam study part-A table from a cell index")
    t2.add_argument("index", type=Path)
    r = sub.add_parser("report", help="report on one host's gate directory")
    r.add_argument("gate_dir", type=Path)
    r.add_argument("--host", default="")
    r.add_argument("--seed", type=int, default=GATE_SEED)
    a = p.parse_args()
    if a.cmd == "table":
        print(TABLE_HEADER)
        print("\n".join(table_rows(a.index)))
    elif a.cmd == "adam":
        rows, summary = adam_rows(a.index)
        print(ADAM_HEADER)
        print("\n".join(rows))
        print()
        print(summary)
    elif a.cmd == "report":
        print(report(a.gate_dir, a.host, a.seed))
    else:
        p.error("choose a command: report, table or adam")


if __name__ == "__main__":
    main()
