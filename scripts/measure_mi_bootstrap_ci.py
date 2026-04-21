"""Bootstrap confidence interval on MLP<->LIF MI from saved codes.

Companion to measure_mi_null_model.py. Loads the same NPZ, reports
median + IQR + 95% CI via non-parametric bootstrap.

GROSMAC-SAFE: pure numpy, loads pre-computed NPZ.

Usage:
    uv run python scripts/measure_mi_bootstrap_ci.py \\
        --codes tests/golden/codes_mlp_lif.npz \\
        --resamples 1000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nerve_wml.methodology.bootstrap_ci_mi import bootstrap_ci_mi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("papers/paper1/figures/mi_bootstrap_ci.json"),
    )
    args = parser.parse_args()

    if not args.codes.exists():
        raise FileNotFoundError(
            f"{args.codes} not found. Produce it first via "
            "scripts/save_codes_for_checks.py on Tower/kxkm-ai."
        )

    data = np.load(args.codes)
    mlp_codes = data["mlp_codes"]
    lif_codes = data["lif_codes"]

    per_seed = []
    for seed_idx, s in enumerate(args.seeds):
        result = bootstrap_ci_mi(
            mlp_codes[seed_idx].astype(np.int64),
            lif_codes[seed_idx].astype(np.int64),
            n_resamples=args.resamples,
            seed=s,
        )
        per_seed.append({
            "seed":         s,
            "mi_point":     result.mi_point,
            "mi_median":    result.mi_median,
            "mi_p25":       result.mi_p25,
            "mi_p75":       result.mi_p75,
            "mi_ci95_low":  result.mi_ci95_low,
            "mi_ci95_high": result.mi_ci95_high,
            "n_resamples":  result.n_resamples,
        })

    summary = {
        "mi_point_mean":  float(np.mean([r["mi_point"] for r in per_seed])),
        "mi_ci95_low_min": float(min(r["mi_ci95_low"] for r in per_seed)),
        "mi_ci95_high_max": float(max(r["mi_ci95_high"] for r in per_seed)),
        "iqr_width_mean":  float(
            np.mean([r["mi_p75"] - r["mi_p25"] for r in per_seed])
        ),
        "n_seeds":    len(args.seeds),
        "n_resamples": args.resamples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_seed": per_seed, "summary": summary}, indent=2)
    )

    print(
        f"Bootstrap CI -- {len(args.seeds)} seeds x "
        f"{args.resamples} resamples"
    )
    print()
    header = (
        f"{'seed':>6}{'point':>10}{'median':>10}"
        f"{'p25':>10}{'p75':>10}{'ci95_low':>12}{'ci95_high':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in per_seed:
        print(
            f"{r['seed']:>6}"
            f"{r['mi_point']:>10.4f}{r['mi_median']:>10.4f}"
            f"{r['mi_p25']:>10.4f}{r['mi_p75']:>10.4f}"
            f"{r['mi_ci95_low']:>12.4f}{r['mi_ci95_high']:>12.4f}"
        )
    print()
    print(
        f"Mean MI point: {summary['mi_point_mean']:.4f}, "
        f"95% CI spans [{summary['mi_ci95_low_min']:.4f}, "
        f"{summary['mi_ci95_high_max']:.4f}], "
        f"mean IQR width {summary['iqr_width_mean']:.4f}"
    )
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
