"""Renf 8: transducer benchmark at relative_rep best HP, 50 seeds.

PR #18's v3 ran defaults (n_anchors=32). v2 ran n_anchors=64 at 10 seeds
and found learned == relative_rep. This run uses 50 seeds at n_anchors=64
to confirm the tie at high power.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.transducer_baselines_pilot import run_transducer_benchmark

_OUT = Path("docs/superpowers/research/2026-05-20-transducer-anchors64-50s.json")


def _one(seed: int) -> dict[str, float]:
    res = run_transducer_benchmark(steps=2000, seed=seed, n_anchors=64)
    return {
        "learned":      float(res["learned"]["mi_bits"]),
        "relative_rep": float(res["relative_rep"]["mi_bits"]),
        "procrustes":   float(res["procrustes"]["mi_bits"]),
        "vec2vec":      float(res["vec2vec"]["mi_bits"]),
        "null":         float(res["null"]["mi_bits"]),
    }


def main() -> None:
    seeds = list(range(50))
    t0 = time.perf_counter()
    workers = min(len(seeds), mp.cpu_count())
    print(f"transducer n_anchors=64, 50 seeds on {workers} workers")
    with mp.Pool(workers) as pool:
        rows = pool.map(_one, seeds)
    wall = time.perf_counter() - t0
    learned = np.asarray([r["learned"] for r in rows])
    relrep = np.asarray([r["relative_rep"] for r in rows])
    procru = np.asarray([r["procrustes"] for r in rows])
    v2v = np.asarray([r["vec2vec"] for r in rows])
    null = np.asarray([r["null"] for r in rows])

    def paired(a, b):
        if np.allclose(a, b):
            return {"p_value": 1.0, "median_diff": 0.0, "cohens_dz": 0.0}
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        diff = a - b
        sd = float(np.std(diff, ddof=1))
        return {
            "p_value": float(res.pvalue),
            "median_diff": float(np.median(diff)),
            "cohens_dz": float(diff.mean() / sd) if sd > 0 else 0.0,
        }

    out = {
        "seeds": 50, "n_anchors": 64, "wall_clock_s": wall,
        "rows": rows,
        "summary": {
            arm: {"mean": float(v.mean()), "std": float(v.std()),
                  "min": float(v.min()), "max": float(v.max())}
            for arm, v in [
                ("learned", learned), ("relative_rep", relrep),
                ("procrustes", procru), ("vec2vec", v2v), ("null", null),
            ]
        },
        "paired_vs_learned": {
            "relative_rep": paired(learned, relrep),
            "procrustes":   paired(learned, procru),
            "vec2vec":      paired(learned, v2v),
            "null":         paired(learned, null),
        },
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {_OUT} ({wall:.1f}s)")


if __name__ == "__main__":
    main()
