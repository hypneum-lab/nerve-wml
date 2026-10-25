"""NH-01 confirmatory run at N = 1000 (amended pre-registration).

Pre-registration (binding, L1 amendment):
``docs/milestones/2026-10-25-nh01-confirmatory-n1000.md``
(tag ``prereg-nh01-confirmatory-20261025``). Kernels, dynamics, estimators and
bootstrap are those of the pilot (``run.py``); only the grid, the decision
rule (C2' unimodal with a finite optimum t*, C3' decided at T > 0) and the
outputs differ. Every constant below is the pre-registered value.

Usage (pre-registered host only)::

    python -m experiments.nh01_hopfield_dream.confirmatory --probe      # timing, seed 999
    python -m experiments.nh01_hopfield_dream.confirmatory --out results/nh01_confirmatory
    python -m experiments.nh01_hopfield_dream.confirmatory --analyze-only --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.nh01_hopfield_dream import run as nh

# --------------------------------------------------------------------------- #
# Pre-registered constants (amendment)
# --------------------------------------------------------------------------- #

N_CONF = 1000
SEEDS_CONF: tuple[int, ...] = tuple(range(10))
ALPHA_GRID: tuple[float, ...] = nh.ALPHA_GRID
T_GRID: tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 100.0)
FLIPS = 0.10
T_CONTROL = 0.2  # Glauber arm, decisive for C3'
CONDITIONS: tuple[tuple[float, float], ...] = ((0.0, FLIPS), (T_CONTROL, FLIPS))
T_PRIMARY = 10.0  # C1 and C3' are read at the catalogue's t = 10
SPURIOUS_ALPHAS: tuple[float, ...] = (0.14, 0.20)
UNIMODAL_TOL = 0.02  # C2' tolerance on the median curve
T_STAR_PRED: tuple[float, float] = (5.0, 20.0)  # P2 prediction for t*
WALL_CAP_SECONDS = 8 * 3600
PROBE_SEED = 999  # not a pre-registered seed; timing only, output discarded
PROBE_ALPHAS: tuple[float, ...] = (0.2, 0.5)

CFG: dict = {
    "alpha_grid": ALPHA_GRID,
    "t_grid": T_GRID,
    "conditions": CONDITIONS,
    "max_probes": nh.MAX_PROBES,
    "spectral": False,  # spectral c omitted at N = 1000 (descriptive only in the pilot)
}


# --------------------------------------------------------------------------- #
# Amended decision rule
# --------------------------------------------------------------------------- #


def unimodality(values: list[float], tol: float) -> tuple[int, bool]:
    """Index of the (first) maximum and whether the curve is unimodal within tol.

    Non-decreasing up to the maximum and non-increasing after it, each step
    allowed to go the wrong way by at most ``tol``.
    """
    k_star = int(np.argmax(values))
    rising = all(values[i + 1] >= values[i] - tol for i in range(k_star))
    falling = all(values[i + 1] <= values[i] + tol for i in range(k_star, len(values) - 1))
    return k_star, rising and falling


def per_seed_t_star(entry: dict, t_grid: tuple[float, ...]) -> list[float]:
    """Per-seed argmax of alpha_c(dream, t) over the grid (t = 0 is Hebb)."""
    hebb = entry["hebb"]["alpha_c_per_seed"]
    cols = [hebb] + [entry["dream"][f"{t:g}"]["alpha_c_per_seed"] for t in t_grid[1:]]
    mat = np.array(cols).T  # seeds x t
    return [float(t_grid[int(k)]) for k in np.argmax(mat, axis=1)]


def verdict(conds: dict) -> dict:
    e0 = conds[f"T0_f{FLIPS:g}_N{N_CONF}"]
    e2 = conds[f"T{T_CONTROL:g}_f{FLIPS:g}_N{N_CONF}"]
    tk = f"{T_PRIMARY:g}"

    # C1 (unchanged from the pilot): gain at t = 10, T = 0
    r_mean, r_lo, r_hi = e0["dream"][tk]["ratio_mean_ci"]
    d_mean, d_lo, d_hi = e0["dream"][tk]["diff_mean_ci"]
    if r_lo >= 3.0:
        c1 = "confirmed"
    elif d_hi < 0.1:
        c1 = "refuted"
    else:
        c1 = "inconclusive"

    # C2' (amended): unimodal median curve with an interior optimum t*
    medians = [float(np.median(e0["hebb"]["alpha_c_per_seed"]))]
    medians += [e0["dream"][f"{t:g}"]["median"] for t in T_GRID[1:]]
    k_star, unimodal = unimodality(medians, UNIMODAL_TOL)
    t_star = float(T_GRID[k_star])
    interior = 0 < k_star < len(T_GRID) - 1
    if not unimodal:
        c2 = "violated"
    elif not interior:
        c2 = "inconclusive"
    else:
        c2 = "holds"
    seeds_t_star = per_seed_t_star(e0, T_GRID)
    hebb0 = np.array(e0["hebb"]["alpha_c_per_seed"])
    at_star = hebb0 if k_star == 0 else np.array(e0["dream"][f"{t_star:g}"]["alpha_c_per_seed"])
    at_end = np.array(e0["dream"][f"{T_GRID[-1]:g}"]["alpha_c_per_seed"])
    fin_mean, fin_lo, fin_hi = nh.bootstrap_mean_ci(at_star - at_end)
    finite_optimum = bool(fin_lo > 0.0)

    # C3' (amended): matched shrink at T_CONTROL, t = 10
    s_mean, s_lo, s_hi = e2["shrink"][tk]["ratio_mean_ci"]
    if s_hi < 1.1:
        c3 = "holds"
    elif s_lo >= 2.0:
        c3 = "refutes"
    else:
        c3 = "inconclusive"
    s0 = np.array(e0["shrink"][tk]["ratio_per_seed"], dtype=np.float64)
    exact_one_t0 = bool(np.all(np.isclose(s0[np.isfinite(s0)], 1.0)))

    if not exact_one_t0:
        overall = "void_implementation_failure"
    elif c1 == "confirmed" and c2 == "holds" and c3 == "holds":
        overall = "confirme"
    elif c1 == "refuted" or c2 == "violated" or c3 == "refutes":
        overall = "refute"
    else:
        overall = "non_concluant"

    p4 = {f"{a:g}": e0["spurious"][f"{a:g}"]["relative_reduction_dream"] for a in SPURIOUS_ALPHAS}
    p4_ok = all(math.isfinite(v) and v >= 0.5 for v in p4.values())
    return {
        "cells": {
            "C1_C2": f"T0_f{FLIPS:g}_N{N_CONF}",
            "C3": f"T{T_CONTROL:g}_f{FLIPS:g}_N{N_CONF}",
        },
        "t_primary": T_PRIMARY,
        "C1_gain": {
            "label": c1,
            "ratio_mean_ci": [r_mean, r_lo, r_hi],
            "diff_mean_ci": [d_mean, d_lo, d_hi],
        },
        "C2_unimodal": {
            "label": c2,
            "t_star": t_star,
            "t_star_in_predicted_range": bool(T_STAR_PRED[0] <= t_star <= T_STAR_PRED[1]),
            "per_seed_t_star": seeds_t_star,
            "median_alpha_c_by_t": dict(zip([f"{t:g}" for t in T_GRID], medians, strict=True)),
            "finite_optimum": finite_optimum,
            "drop_star_to_end_mean_ci": [fin_mean, fin_lo, fin_hi],
            "tolerance": UNIMODAL_TOL,
        },
        "C3_shrink_T_control": {
            "label": c3,
            "temperature": T_CONTROL,
            "ratio_mean_ci": [s_mean, s_lo, s_hi],
            "exact_one_at_T0": exact_one_t0,
        },
        "P4_spurious": {"label": "holds" if p4_ok else "fails", "relative_reduction_by_alpha": p4},
        "descriptive": {
            "ratio_T0_at_t_star": (
                e0["dream"][f"{t_star:g}"]["ratio_mean_ci"] if k_star > 0 else [1.0, 1.0, 1.0]
            ),
            "ratio_T_control_at_t_primary": e2["dream"][tk]["ratio_mean_ci"],
            "ratio_T_control_best_t": max(
                ((e2["dream"][f"{t:g}"]["ratio_mean_ci"][0], t) for t in T_GRID[1:]),
                key=lambda x: x[0],
            ),
        },
        "overall": overall,
    }


def analyze(records: list[dict]) -> dict:
    caps, spur = nh._collect(records, SPURIOUS_ALPHAS, ALPHA_GRID)
    out: dict = {"conditions": {}}
    for cond, by_arm in caps.items():
        temp, flips, n = cond
        out["conditions"][f"T{temp:g}_f{flips:g}_N{n}"] = nh._condition_entry(
            cond, by_arm, spur.get(cond, {}), T_GRID, T_PRIMARY, SPURIOUS_ALPHAS
        )
    out["verdict"] = verdict(out["conditions"])
    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def _mean_curve(records: list[dict], temp: float, arm: str, t: float, field: str) -> np.ndarray:
    curve = np.zeros(len(ALPHA_GRID))
    seeds = {r["seed"] for r in records}
    for r in records:
        if r["temp"] == temp and r["arm"] == arm and r["t"] == t:
            curve[ALPHA_GRID.index(r["alpha"])] += r[field] / len(seeds)
    return curve


def make_figure(records: list[dict], summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = np.array(ALPHA_GRID)
    ts = np.array(T_GRID[1:])
    x = np.log10(ts)
    conds = summary["conditions"]
    v = summary["verdict"]
    t_star = v["C2_unimodal"]["t_star"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) alpha_c vs t at both temperatures
    ax = axes[0, 0]
    for temp, ls in ((0.0, "-"), (T_CONTROL, "--")):
        e = conds[f"T{temp:g}_f{FLIPS:g}_N{N_CONF}"]
        d = np.array([e["dream"][f"{t:g}"]["mean_ci"] for t in ts])
        s = np.array([e["shrink"][f"{t:g}"]["mean_ci"] for t in ts])
        ax.errorbar(
            x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]],
            color="#1b9e77", ls=ls, marker="o", label=f"dream T={temp:g}",
        )
        ax.plot(x, s[:, 0], color="#d95f02", ls=ls, marker="s", label=f"shrink c(t) T={temp:g}")
        ax.axhline(e["hebb"]["mean_ci"][0], color="k", ls=ls, lw=0.8)
    ax.axvline(np.log10(t_star), color="grey", lw=0.8, ls=":")
    ax.set_xlabel("log10 dreaming time t")
    ax.set_ylabel("alpha_c (mean, 95% bootstrap CI over seeds)")
    ax.set_title(f"(a) Capacity vs t, N={N_CONF} (black: Hebb ; dotted: t*={t_star:g})")
    ax.legend(fontsize=7)

    # (b) retrieval curves at T = 0
    ax = axes[0, 1]
    sel = [
        ("hebb", 0.0, "Hebb", "k", "-"),
        ("dream", T_PRIMARY, f"dream t={T_PRIMARY:g}", "#1b9e77", "-"),
        ("dream", t_star, f"dream t*={t_star:g}", "#7570b3", "-"),
        ("shrink", T_PRIMARY, f"shrink c({T_PRIMARY:g})", "#d95f02", "--"),
    ]
    for arm, t, label, color, ls in sel:
        if label.startswith("dream t*") and t_star == T_PRIMARY:
            continue
        ax.plot(alphas, _mean_curve(records, 0.0, arm, t, "retrieved_frac"),
                marker="o", ms=3, color=color, ls=ls, label=label)
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_xlabel("load alpha = P/N")
    ax.set_ylabel("retrieved fraction (mean over seeds)")
    ax.set_title(f"(b) Retrieval curves, N={N_CONF}, T=0")
    ax.legend(fontsize=8)

    # (c) spurious fraction at T = 0
    ax = axes[1, 0]
    for arm, t, label, color, ls in sel[:2] + sel[3:]:
        ax.plot(alphas, _mean_curve(records, 0.0, arm, t, "spurious_frac"),
                marker="o", ms=3, color=color, ls=ls, label=label)
    ax.set_xlabel("load alpha = P/N")
    ax.set_ylabel("spurious final states (fraction)")
    ax.set_title(f"(c) Spurious attractors, N={N_CONF}, T=0")
    ax.legend(fontsize=8)

    # (d) ratios vs t at both temperatures
    ax = axes[1, 1]
    for temp, ls in ((0.0, "-"), (T_CONTROL, "--")):
        e = conds[f"T{temp:g}_f{FLIPS:g}_N{N_CONF}"]
        d = np.array([e["dream"][f"{t:g}"]["ratio_mean_ci"] for t in ts])
        s = np.array([e["shrink"][f"{t:g}"]["ratio_mean_ci"] for t in ts])
        ax.errorbar(
            x, d[:, 0], yerr=[d[:, 0] - d[:, 1], d[:, 2] - d[:, 0]],
            color="#1b9e77", ls=ls, marker="o", label=f"dream / Hebb, T={temp:g}",
        )
        ax.errorbar(
            x, s[:, 0], yerr=[s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]],
            color="#d95f02", ls=ls, marker="s", label=f"shrink / Hebb, T={temp:g}",
        )
    ax.axhline(3.0, color="grey", lw=0.8, ls=":")
    ax.axhline(1.1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("log10 dreaming time t")
    ax.set_ylabel("alpha_c ratio to Hebb (mean, 95% CI)")
    ax.set_title(f"(d) Ratios, N={N_CONF}: T=0 (solid) vs T={T_CONTROL:g} (dashed)")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"NH-01 confirmatory N={N_CONF} — {v['overall']} (C1 {v['C1_gain']['label']}, "
        f"C2' {v['C2_unimodal']['label']} t*={t_star:g}, "
        f"C3' {v['C3_shrink_T_control']['label']} at T={T_CONTROL:g})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/nh01_confirmatory"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--probe", action="store_true", help="timing probe, seed 999, output discarded")
    args = ap.parse_args(argv)

    if args.probe:
        cfg = dict(CFG, alpha_grid=PROBE_ALPHAS)
        t0 = time.time()
        recs = nh.evaluate_task((N_CONF, PROBE_SEED, cfg))
        wall = time.time() - t0
        n_arms = 1 + 2 * (len(T_GRID) - 1)
        full_cells = len(ALPHA_GRID) * n_arms * len(CONDITIONS)
        print(json.dumps({
            "probe_cells": len(recs), "probe_seconds": round(wall, 1),
            "seconds_per_cell": round(wall / len(recs), 2),
            "projected_task_seconds": round(wall / len(recs) * full_cells),
            "projected_total_hours_4_workers": round(
                wall / len(recs) * full_cells * len(SEEDS_CONF) / 4 / 3600, 2
            ),
        }, indent=2))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    cells_path = args.out / "cells.jsonl"
    meta_path = args.out / "run_meta.json"

    if not args.analyze_only:
        tasks = [(N_CONF, s, CFG) for s in SEEDS_CONF]
        t0 = time.time()
        records: list[dict] = []
        with Pool(processes=max(1, args.workers)) as pool:
            for recs in pool.imap_unordered(nh.evaluate_task, tasks):
                records.extend(recs)
                done = {(r["N"], r["seed"]) for r in records}
                print(
                    f"[{time.time() - t0:7.1f}s] {len(done)}/{len(tasks)} tasks, "
                    f"{len(records)} cells",
                    flush=True,
                )
                if time.time() - t0 > WALL_CAP_SECONDS:
                    print("wall-time cap reached: stopping (pre-registered rule)", flush=True)
                    pool.terminate()
                    break
        wall = time.time() - t0
        with cells_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        meta = {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cpus": os.cpu_count(),
            "workers": args.workers,
            "wall_seconds": round(wall, 1),
            "git_commit": nh.git_commit(),
            "n_cells": len(records),
            "tasks_done": sorted({(r["N"], r["seed"]) for r in records}),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))

    records = [json.loads(line) for line in cells_path.read_text().splitlines() if line.strip()]
    summary = analyze(records)
    summary["run_meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figure(records, summary, args.out / "figure.png")
    print(json.dumps(summary["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
