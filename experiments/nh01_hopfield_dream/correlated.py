"""NH-01 follow-up: dreaming kernel vs matched shrink on correlated patterns.

Pre-registration (binding, L1):
``docs/milestones/2026-10-27-nh01-correlated-patterns.md``
(tag ``prereg-nh01-correlated-20261027``). Same kernels, dynamics,
estimators, t grid, temperatures and per-level decision rule as the N = 1000
confirmatory run (``confirmatory.py``); what is new is the pattern generator
(one random ancestor, each bit kept with probability (1+m)/2), three
correlation levels, an extended low-load grid and the cross-level
criterion C4.

Usage (pre-registered host only)::

    python -m experiments.nh01_hopfield_dream.correlated --probe
    python -m experiments.nh01_hopfield_dream.correlated --workers 2 --out results/nh01_correlated
    python -m experiments.nh01_hopfield_dream.correlated --analyze-only --out ...
"""

from __future__ import annotations

import argparse
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

N = cf.N_CONF
SEEDS: tuple[int, ...] = tuple(range(100, 110))  # new seeds, disjoint from 0-9
M_LEVELS: tuple[float, ...] = (0.0, 0.15, 0.30)  # ancestor overlap; pairwise q = m**2
M_LOW, M_HIGH = M_LEVELS[0], M_LEVELS[-1]
ALPHA_GRID: tuple[float, ...] = (0.01, 0.02) + nh.ALPHA_GRID
MAX_BELOW_GRID = 2  # more Hebbian seeds below the grid than this => C1 inconclusive
WALL_CAP_SECONDS = 12 * 3600
PROBE_SEED = 998  # timing only, output discarded
PROBE_ALPHAS: tuple[float, ...] = (0.2, 0.5)


def cfg_for(m: float) -> dict:
    return {
        "alpha_grid": ALPHA_GRID,
        "t_grid": cf.T_GRID,
        "conditions": cf.CONDITIONS,
        "max_probes": nh.MAX_PROBES,
        "spectral": False,
        "ancestor_m": m,
    }


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def analyze_level(records: list[dict]) -> dict:
    caps, spur = nh._collect(records, cf.SPURIOUS_ALPHAS, ALPHA_GRID)
    conds: dict = {}
    for cond, by_arm in caps.items():
        temp, flips, n = cond
        conds[f"T{temp:g}_f{flips:g}_N{n}"] = nh._condition_entry(
            cond, by_arm, spur.get(cond, {}), cf.T_GRID, cf.T_PRIMARY, cf.SPURIOUS_ALPHAS
        )
    v = cf.verdict(conds)
    e0 = conds[f"T0_f{cf.FLIPS:g}_N{N}"]
    n_below = sum(1 for f in e0["hebb"]["flags"] if f == "below_grid")
    v["hebb_below_grid_seeds"] = n_below
    if n_below > MAX_BELOW_GRID and v["C1_gain"]["label"] != "refuted":
        v["C1_gain"]["label"] = "inconclusive"
        if v["overall"] == "confirme":
            v["overall"] = "non_concluant"
    return {"conditions": conds, "verdict": v}


def c4_trend(levels: dict) -> dict:
    """Paired-by-seed difference of the T = 0, t = 10 ratio between m_high and m = 0."""
    key = f"T0_f{cf.FLIPS:g}_N{N}"
    tk = f"{cf.T_PRIMARY:g}"
    lo = np.array(levels[f"{M_LOW:g}"]["conditions"][key]["dream"][tk]["ratio_per_seed"])
    hi = np.array(levels[f"{M_HIGH:g}"]["conditions"][key]["dream"][tk]["ratio_per_seed"])
    d_mean, d_lo, d_hi = nh.bootstrap_mean_ci(hi - lo)
    if d_lo > 0:
        label = "gain_increases"
    elif d_hi < 0:
        label = "gain_decreases"
    else:
        label = "inconclusive"
    means = {
        f"{m:g}": levels[f"{m:g}"]["conditions"][key]["dream"][tk]["ratio_mean_ci"]
        for m in M_LEVELS
    }
    return {
        "label": label,
        "diff_high_minus_zero_mean_ci": [d_mean, d_lo, d_hi],
        "ratio_by_level": means,
        "prediction_met": label == "gain_increases",
    }


def analyze(records: list[dict]) -> dict:
    levels = {}
    for m in M_LEVELS:
        sub = [r for r in records if r.get("ancestor_m") == m]
        levels[f"{m:g}"] = analyze_level(sub)
    return {"levels": levels, "C4_trend": c4_trend(levels)}


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def make_figure(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.array(cf.T_GRID[1:])
    x = np.log10(ts)
    colors = dict(zip([f"{m:g}" for m in M_LEVELS], ("#1b9e77", "#7570b3", "#d95f02"), strict=True))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for temp, ax_i in ((0.0, 0), (cf.T_CONTROL, 1)):
        ax = axes[ax_i]
        for mk, lv in summary["levels"].items():
            e = lv["conditions"][f"T{temp:g}_f{cf.FLIPS:g}_N{N}"]
            d = np.array([e["dream"][f"{t:g}"]["ratio_mean_ci"] for t in ts])
            s = np.array([e["shrink"][f"{t:g}"]["ratio_mean_ci"] for t in ts])
            ax.errorbar(x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]],
                        color=colors[mk], marker="o", label=f"dream m={mk}")
            ax.plot(x, s[:, 0], color=colors[mk], marker="s", ls="--", label=f"shrink m={mk}")
        ax.axhline(3.0, color="grey", lw=0.8, ls=":")
        ax.axhline(1.1, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("log10 dreaming time t")
        ax.set_ylabel("alpha_c ratio to Hebb (mean, 95% CI)")
        ax.set_title(f"({'ab'[ax_i]}) Ratios, N={N}, T={temp:g}")
        ax.legend(fontsize=7)
    ax = axes[2]
    for mk, lv in summary["levels"].items():
        e = lv["conditions"][f"T0_f{cf.FLIPS:g}_N{N}"]
        h = e["hebb"]["mean_ci"][0]
        d = [h] + [e["dream"][f"{t:g}"]["mean_ci"][0] for t in ts]
        ax.plot(np.log10(np.array([0.3] + list(ts))), d, color=colors[mk], marker="o",
                label=f"m={mk} (t*={lv['verdict']['C2_unimodal']['t_star']:g})")
    ax.set_xlabel("log10 t (leftmost point = Hebb)")
    ax.set_ylabel("alpha_c (mean over seeds)")
    ax.set_title("(c) Capacity vs t, T=0")
    ax.legend(fontsize=8)
    c4 = summary["C4_trend"]
    fig.suptitle(
        "NH-01 correlated patterns, N=1000 — "
        + ", ".join(f"m={k}: {v['verdict']['overall']}" for k, v in summary["levels"].items())
        + f" ; C4 {c4['label']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/nh01_correlated"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--probe", action="store_true", help="timing probe, seed 998, discarded")
    args = ap.parse_args(argv)

    if args.probe:
        t0 = time.time()
        recs = nh.evaluate_task((N, PROBE_SEED, dict(cfg_for(M_HIGH), alpha_grid=PROBE_ALPHAS)))
        wall = time.time() - t0
        per_task = len(ALPHA_GRID) * (1 + 2 * (len(cf.T_GRID) - 1)) * len(cf.CONDITIONS)
        total = wall / len(recs) * per_task * len(SEEDS) * len(M_LEVELS)
        print(json.dumps({
            "probe_cells": len(recs), "probe_seconds": round(wall, 1),
            "seconds_per_cell": round(wall / len(recs), 2),
            "cells_total": per_task * len(SEEDS) * len(M_LEVELS),
            "projected_total_hours_2_workers": round(total / 2 / 3600, 2),
        }, indent=2))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"
    meta_path = args.out / "run_meta.json"
    if not args.analyze_only:
        tasks = [(N, s, cfg_for(m)) for m in M_LEVELS for s in SEEDS]
        t0 = time.time()
        n_done = 0
        with cells_path.open("w") as f, Pool(processes=max(1, args.workers)) as pool:
            for recs in pool.imap_unordered(nh.evaluate_task, tasks):
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                n_done += 1
                print(f"[{time.time() - t0:8.1f}s] {n_done}/{len(tasks)} tasks", flush=True)
                if time.time() - t0 > WALL_CAP_SECONDS:
                    print("wall-time cap reached: stopping (pre-registered rule)", flush=True)
                    pool.terminate()
                    break
        meta = {
            "host": platform.node(), "platform": platform.platform(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cpus": os.cpu_count(), "workers": args.workers,
            "wall_seconds": round(time.time() - t0, 1), "git_commit": nh.git_commit(),
            "tasks_done": n_done, "tasks_total": len(tasks),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))

    records = [json.loads(line) for line in cells_path.read_text().splitlines() if line.strip()]
    summary = analyze(records)
    summary["run_meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figure(summary, args.out / "figure.png")
    print(json.dumps({k: v["verdict"]["overall"] for k, v in summary["levels"].items()}))
    print(json.dumps(summary["C4_trend"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
