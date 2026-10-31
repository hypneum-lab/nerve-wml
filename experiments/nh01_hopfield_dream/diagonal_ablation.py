"""NH-01 diagonal ablation: isolates the diagonal contribution to the capacity drop.

Pre-registration (binding, L0 draft — NOT frozen, not runnable until the PI
creates the annotated tag ``prereg-nh01-diagonal-20261031`` on the commit that
adds this file):
``docs/milestones/2026-10-31-nh01-diagonal-ablation.md``

Two arms on the same seeds and patterns (seeds 200-209, disjoint from
0-9 confirmatory and 100-109 correlated):
  - ``dream_offdiag``: J(t) with diagonal set to 0 before the dynamics
    (exact replica of the confirmatory dream arm; positive control).
  - ``dream_diag``:    J(t) with diagonal KEPT in the update fields
    (as in the original Fachechi-Agliari-Barra 2019 field equation).

This script imports the frozen functions from run.py and confirmatory.py
unchanged.  Do NOT modify run.py or confirmatory.py: they are frozen by tags.

T = 0 only; same t grid, load grid and cue noise as the confirmatory run.

Usage (pre-registered host only, after tag)::

    python -I -m py_compile experiments/nh01_hopfield_dream/diagonal_ablation.py
    python -m experiments.nh01_hopfield_dream.diagonal_ablation --probe
    python -m experiments.nh01_hopfield_dream.diagonal_ablation \\
        --out results/nh01_diagonal_ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.nh01_hopfield_dream import confirmatory as cf
from experiments.nh01_hopfield_dream import run as nh

# --------------------------------------------------------------------------- #
# Pre-registered constants
# --------------------------------------------------------------------------- #

N = cf.N_CONF                                    # 1 000
SEEDS: tuple[int, ...] = tuple(range(200, 210))  # new namespace, disjoint from 0-9 and 100-109
T_GRID: tuple[float, ...] = cf.T_GRID            # (0.0, 1.0, ..., 30.0, 100.0)
ALPHA_GRID: tuple[float, ...] = nh.ALPHA_GRID    # 24 loads
T_STAR_FIXED: float = 7.0                        # from confirmatory C2_unimodal.t_star; NOT re-picked
CONDITION: tuple[float, float] = (0.0, cf.FLIPS) # T = 0, f = 0.10 only
WALL_CAP_SECONDS = 8 * 3600
PROBE_SEED = 997                                  # timing only, output discarded; not pre-registered
PROBE_ALPHAS: tuple[float, ...] = (0.2, 0.5)

# Decision thresholds (TBD by PI — see pre-registration)
TOL_C2: float = 0.020             # C2' tolerance; drop ≤ this means diagonal is the cause
THRESH_NOT_CAUSE: float = 0.050   # drop ≥ this means diagonal is not the cause
CENSOR_MAX_SEEDS: int = 2         # maximum censored seeds allowed before C5 is inconclusive

# Confirmatory result used as the positive-control reference
# (from results/nh01_confirmatory/summary.json, verdict.C2_unimodal)
CONTROL_DROP_MEAN: float = 0.0876          # D = alpha_c(t*=7) - alpha_c(t=100), seeds 0-9
CONTROL_DROP_CI: tuple[float, float] = (0.0823, 0.0923)  # 95% bootstrap CI

# --------------------------------------------------------------------------- #
# Cell evaluation (T = 0 only, two dream arm variants)
# --------------------------------------------------------------------------- #


def evaluate_task(args: tuple) -> list[dict]:
    """All cells for one seed: dream_offdiag and dream_diag at T = 0.

    ``args`` is ``(N, seed)`` or ``(N, seed, cfg)`` where cfg may override
    ``alpha_grid``, ``t_grid``, ``max_probes`` (for tests and timing probes).
    """
    n, seed = args[0], args[1]
    cfg = args[2] if len(args) > 2 else {}
    alpha_grid = tuple(cfg.get("alpha_grid", ALPHA_GRID))
    t_grid = tuple(cfg.get("t_grid", T_GRID))
    max_probes = int(cfg.get("max_probes", nh.MAX_PROBES))

    t_start = time.time()
    records: list[dict] = []
    p_max = max(1, int(round(alpha_grid[-1] * n)))
    # Same pattern-generation stream as confirmatory (stream id 1), new seeds 200-209
    xi_all = np.where(nh.stream(seed, n, 1).random((p_max, n)) < 0.5, -1.0, 1.0)

    for ai, alpha in enumerate(alpha_grid):
        p = max(1, int(round(alpha * n)))
        xi = xi_all[:p]
        j_full = nh.hebbian_kernel(xi)
        # One eigen-decomposition per (seed, alpha), reused for all t (efficiency)
        w, v = np.linalg.eigh(j_full)
        w = np.clip(w, 0.0, None)

        # Build arms: at each t, one offdiag variant and one diag-kept variant.
        # At t = 0: J(0) = J (Hebbian). dream_offdiag uses offdiag(J); dream_diag uses J.
        arms_list: list[tuple[str, float, np.ndarray]] = []
        for t in t_grid:
            if t == 0.0:
                jt_full = np.array(j_full, dtype=np.float64, copy=True)
            else:
                wt = w * (1.0 + t) / (1.0 + t * w)
                jt_full = (v * wt) @ v.T      # dreaming kernel, diagonal included
            jt_off = nh.offdiag(jt_full)       # diagonal zeroed
            arms_list.append(("dream_offdiag", t, jt_off))
            arms_list.append(("dream_diag", t, jt_full))

        probe_idx = np.arange(min(p, max_probes))
        temp, flips = CONDITION
        cues = nh.make_cues(xi, probe_idx, flips, nh.stream(seed, n, 2, ai, 0))
        m0 = float(np.mean(np.einsum("bi,bi->b", cues, xi[probe_idx]) / n))

        for arm, t, j_arm in arms_list:
            # Identical rng across arms (same stream id as confirmatory ci=0)
            rng_dyn = nh.stream(seed, n, 3, ai, 0)
            s, sweeps, conv = nh.run_zero_temperature(j_arm, cues, rng_dyn)
            m_all = s @ xi.T / n
            m_cued = m_all[np.arange(probe_idx.size), probe_idx]
            m_cued_eff = np.where(conv, m_cued, -np.inf)
            rec: dict = {
                "N": n, "seed": seed, "alpha": alpha, "P": p,
                "arm": arm, "t": t,
                "temp": temp, "flips": flips,
                "n_probes": int(probe_idx.size), "cue_overlap": m0,
            }
            rec.update(nh.classify(m_cued_eff, m_all, probe_idx, nh.RETRIEVED_THRESH[0.0]))
            rec["nonconverged_frac"] = float((~conv).mean())
            rec["sweeps"] = int(sweeps)
            rec["mean_overlap"] = float(np.mean(m_cued))
            records.append(rec)

    for r in records:
        r["task_seconds"] = round(time.time() - t_start, 3)
    return records


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def _capacity_by_arm_seed(
    records: list[dict],
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    t_grid: tuple[float, ...] = T_GRID,
) -> dict[str, dict[float, dict[int, tuple[float, str]]]]:
    """Return {arm: {t: {seed: (alpha_c, flag)}}}."""
    alphas = np.array(alpha_grid)
    curves: dict[str, dict[float, dict[int, np.ndarray]]] = {}
    for rec in records:
        arm = rec["arm"]
        t = float(rec["t"])
        s = int(rec["seed"])
        arm_t = curves.setdefault(arm, {}).setdefault(t, {})
        curve = arm_t.setdefault(s, np.full(len(alpha_grid), np.nan))
        curve[list(alpha_grid).index(rec["alpha"])] = rec["retrieved_frac"]
    caps: dict[str, dict[float, dict[int, tuple[float, str]]]] = {}
    for arm, by_t in curves.items():
        caps[arm] = {}
        for t, by_seed in by_t.items():
            caps[arm][t] = {
                s: nh.capacity_from_curve(alphas, r) for s, r in by_seed.items()
            }
    return caps


def analyse(
    records: list[dict],
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    t_grid: tuple[float, ...] = T_GRID,
    t_star_fixed: float = T_STAR_FIXED,
) -> dict:
    """Primary and secondary analyses (pre-registered).

    Censoring guard: if more than CENSOR_MAX_SEEDS seeds are right-censored
    (flag == 'censored') at t_star_fixed or t_grid[-1] in either arm, the
    corresponding ``censoring_guard`` entry sets ``exceeds_limit = True``.
    ``verdict()`` reports C5 = 'inconclusive_censoring' in that case.
    """
    caps = _capacity_by_arm_seed(records, alpha_grid, t_grid)
    seeds = sorted(caps["dream_offdiag"][t_grid[0]].keys())

    def drop(arm: str) -> np.ndarray:
        """Per-seed drop D = alpha_c(t_star_fixed) - alpha_c(t_grid[-1])."""
        at_star = np.array([caps[arm][t_star_fixed][s][0] for s in seeds])
        at_end = np.array([caps[arm][t_grid[-1]][s][0] for s in seeds])
        return at_star - at_end

    d_off = drop("dream_offdiag")
    d_diag = drop("dream_diag")
    delta_d = d_off - d_diag  # ΔD = D_offdiag - D_diag

    mean_off, lo_off, hi_off = nh.bootstrap_mean_ci(d_off)
    mean_diag, lo_diag, hi_diag = nh.bootstrap_mean_ci(d_diag)
    mean_dd, lo_dd, hi_dd = nh.bootstrap_mean_ci(delta_d)

    # Positive control: D_offdiag CI must overlap with the confirmatory CI
    control_ci_overlap = lo_off <= CONTROL_DROP_CI[1] and hi_off >= CONTROL_DROP_CI[0]

    # Censoring guard: count right-censored seeds at t* and t_end for each arm
    t_end = t_grid[-1]
    censoring_guard: dict = {}
    for arm in ("dream_offdiag", "dream_diag"):
        censoring_guard[arm] = {}
        for t_key in (t_star_fixed, t_end):
            flags = [caps[arm][t_key][s][1] for s in seeds]
            n_censored = sum(1 for f in flags if f == "censored")
            censoring_guard[arm][f"{t_key:g}"] = {
                "n_censored": n_censored,
                "exceeds_limit": n_censored > CENSOR_MAX_SEEDS,
            }

    result: dict = {
        "seeds": seeds,
        "t_star_fixed": t_star_fixed,
        "D_offdiag": {
            "per_seed": d_off.tolist(),
            "mean_ci": [mean_off, lo_off, hi_off],
        },
        "D_diag": {
            "per_seed": d_diag.tolist(),
            "mean_ci": [mean_diag, lo_diag, hi_diag],
        },
        "delta_D": {
            "per_seed": delta_d.tolist(),
            "mean_ci": [mean_dd, lo_dd, hi_dd],
        },
        "positive_control": {
            "reference_mean": CONTROL_DROP_MEAN,
            "reference_ci": list(CONTROL_DROP_CI),
            "D_offdiag_mean_ci": [mean_off, lo_off, hi_off],
            "ci_overlap": bool(control_ci_overlap),
        },
        "censoring_guard": censoring_guard,
        # Descriptive: full alpha_c(t) curves per arm
        "curves": {
            arm: {
                f"{t:g}": {
                    "alpha_c_per_seed": [caps[arm][t][s][0] for s in seeds],
                    "flags": [caps[arm][t][s][1] for s in seeds],
                    "median": float(np.median([caps[arm][t][s][0] for s in seeds])),
                }
                for t in t_grid
            }
            for arm in ("dream_offdiag", "dream_diag")
        },
    }
    return result


def verdict(summary: dict) -> dict:
    """Apply the pre-registered decision rule to an analyse() result.

    Returns {"C5": <label>, "C6": <label or None>} where:
      C5 ∈ {positive_control_failed, inconclusive_censoring,
             diagonal_is_cause, diagonal_is_not_cause, inconclusive}
      C6 ∈ {dD_positive, dD_zero, dD_negative, None}

    The positive-control gate is applied first; if it fails, C6 is None
    and no diagonal numbers are interpreted.  The censoring guard is applied
    second; if it triggers, C6 is also None.
    """
    # Gate 1: positive control
    if not summary["positive_control"]["ci_overlap"]:
        return {"C5": "positive_control_failed", "C6": None}

    # Gate 2: censoring guard
    cg = summary["censoring_guard"]
    for arm in ("dream_offdiag", "dream_diag"):
        for entry in cg[arm].values():
            if entry["exceeds_limit"]:
                return {"C5": "inconclusive_censoring", "C6": None}

    # C5: primary diagonal ablation
    _mean_diag, lo_diag, hi_diag = summary["D_diag"]["mean_ci"]
    mean_diag = _mean_diag
    _mean_dd, lo_dd, hi_dd = summary["delta_D"]["mean_ci"]

    if mean_diag <= TOL_C2 and hi_diag < THRESH_NOT_CAUSE and lo_dd > 0.0:
        c5 = "diagonal_is_cause"
    elif mean_diag >= THRESH_NOT_CAUSE and lo_diag > TOL_C2:
        c5 = "diagonal_is_not_cause"
    else:
        c5 = "inconclusive"

    # C6: secondary ΔD direction
    if lo_dd > 0.0:
        c6: str | None = "dD_positive"
    elif hi_dd < 0.0:
        c6 = "dD_negative"
    else:
        c6 = "dD_zero"

    return {"C5": c5, "C6": c6}


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _sha256file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/nh01_diagonal_ablation"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="Timing probe only; output discarded.")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args(argv)
    out = args.out

    if args.probe:
        probe_cfg = {
            "alpha_grid": PROBE_ALPHAS,
            "t_grid": T_GRID,
            "max_probes": nh.MAX_PROBES,
        }
        t0 = time.time()
        recs = evaluate_task((N, PROBE_SEED, probe_cfg))
        elapsed = time.time() - t0
        n_cells = len(recs)
        rate = elapsed / n_cells if n_cells else float("nan")
        n_arms_total = 2 * len(T_GRID)
        total_cells = n_arms_total * len(ALPHA_GRID) * len(SEEDS)
        projected = total_cells * rate / (args.workers * 3600)
        probe_result = {
            "probe_cells": n_cells,
            "probe_seconds": round(elapsed, 1),
            "seconds_per_cell": round(rate, 3),
            "total_cells_estimated": total_cells,
            "projected_total_hours": round(projected, 2),
            "workers_assumed": args.workers,
        }
        print(json.dumps(probe_result, indent=2))
        return

    out.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        meta: dict = {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpus": os.cpu_count(),
            "workers": args.workers,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seeds": list(SEEDS),
            "N": N,
            "t_grid": list(T_GRID),
            "alpha_grid": list(ALPHA_GRID),
        }
        t_wall = time.time()
        tasks = [(N, s) for s in SEEDS]
        all_records: list[dict] = []
        with Pool(args.workers) as pool:
            for recs in pool.imap_unordered(evaluate_task, tasks):
                all_records.extend(recs)
                if time.time() - t_wall > WALL_CAP_SECONDS:
                    print("Wall-time cap reached; stopping.", file=sys.stderr)
                    break
        meta["wall_seconds"] = round(time.time() - t_wall, 1)
        meta["n_cells"] = len(all_records)
        _write_jsonl(out / "cells.jsonl", all_records)
        with open(out / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    else:
        all_records = []
        with open(out / "cells.jsonl") as f:
            for line in f:
                all_records.append(json.loads(line))

    summary = analyse(all_records)
    summary["verdict"] = verdict(summary)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # SHA256SUMS for cells.jsonl, run_meta.json and summary.json
    sha_lines = []
    for name in ("cells.jsonl", "run_meta.json", "summary.json"):
        p = out / name
        if p.exists():
            sha_lines.append(f"{_sha256file(p)}  {name}\n")
    with open(out / "SHA256SUMS", "w") as f:
        f.writelines(sha_lines)


if __name__ == "__main__":
    main()
