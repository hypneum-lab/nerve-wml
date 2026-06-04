"""Q1 benchmark runner — 4 archs × 5 seeds × HardFlowProxyTask N=2.

Trains each (arch, seed) on a round-trip MSE objective on
HardFlowProxyTask, then evaluates three metrics on a held-out
batch=1024:

    round_trip_fidelity = 1 - MSE(x, decode(encode(x))) / Var(x)
    mi_h                = mean per-dim mutual information
                          between binned code and label y
    bandwidth_efficiency = effective rank @ 95% PCA variance
                           divided by code_dim

Results are written to results.json as a list of run dicts; the
file is rewritten after each (arch, seed) so a kill mid-sweep
keeps a partial record.

Wallclock estimate: 8 cores Linux CPU, ~1.5-3h for the full
20-run sweep. Designed to be launched on electron-server.

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 13.
Pre-reg: docs/milestones/q1-multiplexer-benchmark-2026-05-10.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import mutual_info_score

from experiments.benchmark_multiplexer_vs_baselines.architectures.cross_attention import (
    CrossAttentionBridge,
)
from experiments.benchmark_multiplexer_vs_baselines.architectures.gtm import (
    GTMBridge,
)
from experiments.benchmark_multiplexer_vs_baselines.architectures.mlp_bridge import (
    MLPBridge,
)
from experiments.benchmark_multiplexer_vs_baselines.architectures.recursive_link import (
    RecursiveLinkBridge,
)
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask

SEEDS = [0, 17, 42, 73, 101]
ARCHITECTURES: dict[str, type] = {
    "gtm": GTMBridge,
    "recursive_link": RecursiveLinkBridge,
    "mlp_bridge": MLPBridge,
    "cross_attention": CrossAttentionBridge,
}


@dataclass
class RunMetrics:
    arch: str
    seed: int
    train_loss_final: float
    round_trip_fidelity: float
    mi_h: float
    bandwidth_efficiency: float
    n_params: int
    train_time_s: float


def _build_model(
    arch_cls: type, seed: int, dim: int, code_dim: int
) -> torch.nn.Module:
    return arch_cls(dim=dim, code_dim=code_dim, seed=seed)


def _compute_metrics(
    model: torch.nn.Module, task: HardFlowProxyTask, batch: int = 1024
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        x_eval, y_eval = task.sample(batch=batch)
        code_eval = model.encode(x_eval)
        x_hat_eval = model.decode(code_eval)

        mse = F.mse_loss(x_hat_eval, x_eval).item()
        var = x_eval.var().item()
        rtf = max(0.0, 1.0 - mse / var)

        code_np = code_eval.cpu().numpy()
        y_np = y_eval.cpu().numpy()

        per_dim_mi = []
        for d in range(code_np.shape[1]):
            edges = np.percentile(code_np[:, d], np.linspace(5, 95, 15))
            binned = np.digitize(code_np[:, d], edges)
            per_dim_mi.append(mutual_info_score(binned, y_np))
        mi_h = float(np.mean(per_dim_mi))

        cov = np.cov(code_np.T)
        eigs = np.linalg.eigvalsh(cov)[::-1]
        eig_sum = eigs.sum()
        if eig_sum <= 0:
            bw_eff = 0.0
        else:
            cumvar = np.cumsum(eigs) / eig_sum
            eff_rank = int(np.searchsorted(cumvar, 0.95) + 1)
            bw_eff = eff_rank / code_np.shape[1]

    return rtf, mi_h, bw_eff


def train_one(
    arch_cls: type,
    seed: int,
    *,
    dim: int,
    n_classes: int,
    steps: int,
    batch: int,
    code_dim: int,
) -> RunMetrics:
    torch.manual_seed(seed)
    np.random.seed(seed)
    task = HardFlowProxyTask(dim=dim, n_classes=n_classes, seed=seed)
    model = _build_model(arch_cls, seed, dim, code_dim)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    last_loss = float("nan")
    for step in range(steps):
        x, _ = task.sample(batch=batch)
        x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Advance plasticity clock for any GTM-derived submodule so the
        # cosine decay schedule actually progresses across training. A
        # no-op for non-GTM archs.
        gtm = getattr(model, "gtm", None)
        if gtm is not None and hasattr(gtm, "step"):
            gtm.step()
        last_loss = float(loss.item())
        if step % 100 == 0:
            print(
                f"  [{arch_cls.__name__} seed={seed} step={step}] "
                f"loss={last_loss:.4f}",
                flush=True,
            )
    train_time_s = time.perf_counter() - t0

    rtf, mi_h, bw_eff = _compute_metrics(model, task, batch=1024)
    n_params = sum(p.numel() for p in model.parameters())

    return RunMetrics(
        arch=arch_cls.__name__,
        seed=seed,
        train_loss_final=last_loss,
        round_trip_fidelity=rtf,
        mi_h=mi_h,
        bandwidth_efficiency=bw_eff,
        n_params=n_params,
        train_time_s=train_time_s,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-classes", type=int, default=12)
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--archs", nargs="+", default=list(ARCHITECTURES.keys())
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "experiments/benchmark_multiplexer_vs_baselines/results.json"
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
        "archs": args.archs,
        "out": str(args.out),
    }

    all_results: list[dict] = []
    for arch_name in args.archs:
        if arch_name not in ARCHITECTURES:
            raise SystemExit(f"unknown arch: {arch_name}")
        for seed in args.seeds:
            print(f"=== {arch_name} seed={seed} ===", flush=True)
            metrics = train_one(
                ARCHITECTURES[arch_name],
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
