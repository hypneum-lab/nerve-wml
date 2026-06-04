"""Renf 11: seed-window robustness for the synchrony-replacement ordering.

Renf 7 used seeds 0-49. Are seeds 0-49 a lucky window? Re-run with
window A = range(0,50), B = range(50,100), C = range(1000,1050).

Output JSON shape::

    {
      "windows": {"A": [0..50], "B": [50..100], "C": [1000..1050]},
      "steps": S,
      "per_window": {
        "A": {arm: {metric: {mean, std, values}}}, ...
      },
      "cross_window_mannwhitneyu": {
        arm: {metric: {"A_vs_B": p, "A_vs_C": p, "B_vs_C": p}}
      },
      "wall_clock_s": float,
    }
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.synchrony_replacement_eval import _run_one_seed_kw

_ARMS = ["gtm", "simple_gating", "akorn_best", "null"]
_METRICS = ["spectral_entropy", "participation_ratio",
            "effective_rank", "top3_dispersion"]


def _aggregate(results):
    agg = {arm: {m: {"values": []} for m in _METRICS} for arm in _ARMS}
    for _, row in results:
        for arm in _ARMS:
            for m in _METRICS:
                agg[arm][m]["values"].append(row[arm][m])
    for arm in _ARMS:
        for m in _METRICS:
            v = np.asarray(agg[arm][m]["values"])
            agg[arm][m]["mean"] = float(v.mean())
            agg[arm][m]["std"]  = float(v.std())
    return agg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    windows = {
        "A": list(range(0, 50)),
        "B": list(range(50, 100)),
        "C": list(range(1000, 1050)),
    }

    t0 = time.perf_counter()
    all_tasks = []
    for wname, seeds in windows.items():
        for s in seeds:
            all_tasks.append((wname, s, args.steps, args.device))

    workers = args.workers or min(len(all_tasks), mp.cpu_count())
    print(f"running {len(all_tasks)} jobs (3 windows x 50 seeds x 4 arms) on {workers}")
    pool_args = [(s, steps, dev) for _, s, steps, dev in all_tasks]

    if workers > 1:
        with mp.Pool(workers) as pool:
            raw = pool.map(_run_one_seed_kw, pool_args)
    else:
        raw = [_run_one_seed_kw(a) for a in pool_args]

    # bucket by window
    per_window = {}
    idx = 0
    for wname, seeds in windows.items():
        chunk = raw[idx: idx + len(seeds)]
        idx += len(seeds)
        per_window[wname] = _aggregate(chunk)

    # cross-window Mann-Whitney per arm per metric
    cross = {arm: {m: {} for m in _METRICS} for arm in _ARMS}
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for arm in _ARMS:
        for m in _METRICS:
            for a, b in pairs:
                va = np.asarray(per_window[a][arm][m]["values"])
                vb = np.asarray(per_window[b][arm][m]["values"])
                try:
                    r = stats.mannwhitneyu(va, vb, alternative="two-sided")
                    pv = float(r.pvalue)
                except ValueError:
                    pv = 1.0
                cross[arm][m][f"{a}_vs_{b}"] = pv

    out = {
        "windows": windows,
        "steps": args.steps,
        "per_window": per_window,
        "cross_window_mannwhitneyu": cross,
        "wall_clock_s": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
