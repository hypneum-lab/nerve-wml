"""Q1 Task 14 — GTM ablation runner.

3 variants × 5 seeds × HardFlowProxyTask N=2.
Reuses train_one() from runner.py for metric parity.

Output: experiments/benchmark_multiplexer_vs_baselines/ablation_results.json
Wallclock estimate: ~75% of T13 (3 instead of 4 archs).

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 14.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from experiments.benchmark_multiplexer_vs_baselines.architectures.gtm_ablation_bridges import (
    GTMGammaOnlyBridge,
    GTMNoPlasticityBridge,
    GTMThetaOnlyBridge,
)
from experiments.benchmark_multiplexer_vs_baselines.runner import (
    SEEDS,
    train_one,
)

ABLATIONS: dict[str, type] = {
    "gtm_gamma_only": GTMGammaOnlyBridge,
    "gtm_theta_only": GTMThetaOnlyBridge,
    "gtm_no_plasticity": GTMNoPlasticityBridge,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-classes", type=int, default=12)
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--ablations", nargs="+", default=list(ABLATIONS.keys())
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments/benchmark_multiplexer_vs_baselines/"
            "ablation_results.json"
        ),
    )
    args = parser.parse_args()

    cfg = {
        "steps": args.steps,
        "batch": args.batch,
        "dim": args.dim,
        "n_classes": args.n_classes,
        "code_dim": args.code_dim,
        "seeds": args.seeds,
        "ablations": args.ablations,
        "out": str(args.out),
    }

    all_results: list[dict] = []
    for abl_name in args.ablations:
        if abl_name not in ABLATIONS:
            raise SystemExit(f"unknown ablation: {abl_name}")
        for seed in args.seeds:
            print(f"=== {abl_name} seed={seed} ===", flush=True)
            metrics = train_one(
                ABLATIONS[abl_name],
                seed,
                dim=args.dim,
                n_classes=args.n_classes,
                code_dim=args.code_dim,
                steps=args.steps,
                batch=args.batch,
            )
            all_results.append(asdict(metrics))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {"config": cfg, "results": all_results},
                    indent=2,
                    default=str,
                )
            )
            print(
                f"  rtf={metrics.round_trip_fidelity:.4f} "
                f"mi_h={metrics.mi_h:.4f} "
                f"bw_eff={metrics.bandwidth_efficiency:.4f} "
                f"time={metrics.train_time_s:.1f}s",
                flush=True,
            )

    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
