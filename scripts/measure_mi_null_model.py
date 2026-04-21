"""Null-model MI significance check on pre-computed substrate codes.

Loads an NPZ artefact containing emitted codes from two substrates
across N seeds and runs the permutation significance test from
``nerve_wml.methodology.mi_null_model``. Reports per-seed z-scores
and p-values plus an aggregated verdict.

GROSMAC-SAFE: this script is pure numpy and loads pre-computed
arrays. No training, no GPU. The heavy lift (training MLP/LIF and
saving their emitted codes) must be done separately on Tower or
kxkm-ai via scripts/save_codes_for_checks.py (Phase A generation
step, to be written day 1 PM on remote host).

Expected NPZ schema (produced by save_codes_for_checks.py):

    codes_mlp_lif.npz
      * mlp_codes : int64[n_seeds, n_samples]
      * lif_codes : int64[n_seeds, n_samples]
      * config    : object dict with task, substrate params, etc.

Usage:

    uv run python scripts/measure_mi_null_model.py \\
        --codes tests/golden/codes_mlp_lif.npz \\
        --shuffles 1000 \\
        --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nerve_wml.methodology.mi_null_model import null_model_mi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codes",
        type=Path,
        required=True,
        help="NPZ with keys mlp_codes, lif_codes; shape [n_seeds, n_samples].",
    )
    parser.add_argument("--shuffles", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("papers/paper1/figures/mi_null_model.json"),
    )
    args = parser.parse_args()

    if not args.codes.exists():
        raise FileNotFoundError(
            f"{args.codes} not found.\n"
            f"Generate it first on Tower:\n"
            f"  ssh clems@192.168.0.120 "
            f"'cd ~/nerve-wml && uv run python scripts/save_codes_for_checks.py'\n"
            f"then scp back the resulting NPZ to tests/golden/."
        )

    data = np.load(args.codes, allow_pickle=True)
    mlp_codes = data["mlp_codes"]
    lif_codes = data["lif_codes"]

    if mlp_codes.shape != lif_codes.shape:
        raise ValueError(
            f"code arrays disagree: mlp {mlp_codes.shape} vs lif {lif_codes.shape}"
        )
    if len(args.seeds) > mlp_codes.shape[0]:
        raise ValueError(
            f"requested {len(args.seeds)} seeds but NPZ only has {mlp_codes.shape[0]}"
        )

    per_seed = []
    for seed_idx, s in enumerate(args.seeds):
        result = null_model_mi(
            mlp_codes[seed_idx].astype(np.int64),
            lif_codes[seed_idx].astype(np.int64),
            n_shuffles=args.shuffles,
            seed=s,
        )
        per_seed.append({
            "seed":         s,
            "mi_observed":  result.mi_observed,
            "mi_null_mean": result.mi_null_mean,
            "mi_null_std":  result.mi_null_std,
            "z_score":      result.z_score,
            "p_value":      result.p_value,
            "n_shuffles":   result.n_shuffles,
        })

    observed_mean = float(np.mean([r["mi_observed"] for r in per_seed]))
    z_mean = float(np.mean([r["z_score"] for r in per_seed]))
    p_max = float(max(r["p_value"] for r in per_seed))

    summary = {
        "mi_observed_mean": observed_mean,
        "z_score_mean":     z_mean,
        "p_value_max":      p_max,
        "significant":      p_max < 0.01,
        "n_seeds":          len(args.seeds),
        "n_shuffles":       args.shuffles,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_seed": per_seed, "summary": summary}, indent=2)
    )

    print(
        f"Null-model MI check -- {len(args.seeds)} seeds x "
        f"{args.shuffles} shuffles"
    )
    print()
    for r in per_seed:
        print(
            f"  seed {r['seed']}: "
            f"MI_obs={r['mi_observed']:.4f}, "
            f"null={r['mi_null_mean']:.4f}+/-{r['mi_null_std']:.4f}, "
            f"z={r['z_score']:.2f}, p={r['p_value']:.4f}"
        )
    print()
    print(f"Mean MI observed: {observed_mean:.4f}")
    print(f"Mean z-score:     {z_mean:.2f}")
    print(f"Max p-value:      {p_max:.4f}")
    print(
        f"Verdict: {'SIGNIFICANT (null rejected)' if p_max < 0.01 else 'NOT SIGNIFICANT'}"
    )
    print()
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
