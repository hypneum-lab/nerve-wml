"""Learnability-gate report (amendment 2026-10-25), no mlx import.

Reads ``<gate_dir>/pretrain/pretrain.json`` and ``<gate_dir>/<preset>-<arm>/
run.json`` and prints markdown tables: per run the liveness probes over
time, policy entropy (minimum vs mean of live agents), per-agent own-coin
rate, team success against the uniform-random baseline, wall time and the
gate verdict of section 4 of the amendment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RANDOM_TEAM_SUCCESS_MAX = 0.025  # uniform-random pair, training levels 0-15
GATE_MIN_TEAM_SUCCESS = 2 * RANDOM_TEAM_SUCCESS_MAX
GATE_STEPS = 1_000_000
GATE_WINDOW = 100  # completed episodes in the final success measurement
GATE_MIN_PROBES = 10  # liveness probes after step 0 (every 100k up to 1M)
GATE_PRESET = "fix-c-coin-pretrain"
GATE_ARMS = ("gtm", "eng")


def _finite(x) -> bool:
    return isinstance(x, int | float) and math.isfinite(x)


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def verdict(run: dict) -> tuple[bool, list[str]]:
    """Per-run gate criterion (amendment 2026-10-25, section 4)."""
    why = []
    if run.get("training_failure") or run.get("liveness_abort"):
        why.append("training failure / liveness abort")
    if not (run.get("total_pair_steps") == run.get("pair_steps_done")
            == GATE_STEPS):
        why.append(f"not a complete {GATE_STEPS:,}-step run")
    late = [m for m in run.get("liveness", []) if m.get("step", 0) > 0]
    if len(late) < GATE_MIN_PROBES:
        why.append(f"{len(late)} liveness probes after step 0 "
                   f"< {GATE_MIN_PROBES}")
    if any(m.get("flagged") for m in late):
        why.append("liveness flagged")
    sc = run.get("success_curve") or []
    n = run.get("success_curve_n") or []
    final = sc[-1] if sc else None
    if (final is None or final[0] != GATE_STEPS or not _finite(final[1])
            or not n or n[-1] != GATE_WINDOW):
        why.append(f"no finite last-{GATE_WINDOW}-episode success at "
                   f"{GATE_STEPS:,}")
    elif final[1] < GATE_MIN_TEAM_SUCCESS:
        why.append(f"final team success {_f(final[1])} < {GATE_MIN_TEAM_SUCCESS}")
    cc = [r for _, r in run.get("coin_rate_curve", []) if r is not None]
    if len(cc) < 2 or not all(_finite(x) for x in cc) or cc[-1] <= cc[0]:
        why.append("coin rate did not rise")
    return not why, why


def combined_verdict(gate_dir: Path, preset: str = GATE_PRESET,
                     suffix: str = "") -> tuple[bool, list[str]]:
    """The gate passes only if BOTH arms of ``preset`` pass on one host."""
    why = []
    for arm in GATE_ARMS:
        f = Path(gate_dir) / f"{preset}-{arm}{suffix}" / "run.json"
        if not f.exists():
            why.append(f"{preset}-{arm}{suffix}: missing")
            continue
        ok, w = verdict(json.loads(f.read_text()))
        if not ok:
            why.append(f"{preset}-{arm}{suffix}: " + "; ".join(w))
    return not why, why


def report(gate_dir: Path, host: str = "") -> str:
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
        ok, why = verdict(r)
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
    ok, why = combined_verdict(gate_dir)
    out.append("")
    out.append(f"**Combined gate ({GATE_PRESET}, both arms, {host or 'this host'}):"
               f" {'PASS' if ok else 'FAIL'}**" + ("" if ok else
                                                   " — " + "; ".join(why)))
    return "\n".join(out + [""] + detail)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("gate_dir", type=Path)
    p.add_argument("--host", default="")
    a = p.parse_args()
    print(report(a.gate_dir, a.host))


if __name__ == "__main__":
    main()
