"""Q1++ benchmark runner — FlowProxyTask 4-class (canonical task).

Mirror of runner.py, but uses FlowProxyTask instead of HardFlowProxyTask.
Tests cross-task generalization of GTM benchmark verdict (Q1 was on
HardFlowProxyTask XOR-on-noise hard task ; this is the easier
linearly-separable canonical task).

Plan: HYPNEUM-PLANS/2026-05-11-niveau9-scaling-experiments.md (Q1++).
Pre-reg: docs/milestones/q1plusplus-flowproxytask-4class-2026-05-11.md.
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
from experiments.benchmark_multiplexer_vs_baselines.architectures.gtm import GTMBridge
from experiments.benchmark_multiplexer_vs_baselines.architectures.mlp_bridge import (
    MLPBridge,
)
from experiments.benchmark_multiplexer_vs_baselines.architectures.recursive_link import (
    RecursiveLinkBridge,
)
from track_w.tasks.flow_proxy import FlowProxyTask

SEEDS = [0, 17, 42, 73, 101]
ARCHITECTURES = {
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


def train_one(arch_cls: type, seed: int, dim: int, n_classes: int,
              steps: int, batch: int, code_dim: int) -> RunMetrics:
    torch.manual_seed(seed)
    np.random.seed(seed)
    task = FlowProxyTask(dim=dim, n_classes=n_classes, seed=seed)
    model = arch_cls(dim=dim, code_dim=code_dim, seed=seed)
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
        last_loss = loss.item()
        if step % 100 == 0:
            print(f"  [{arch_cls.__name__} seed={seed} step={step}] "
                  f"loss={last_loss:.4f}", flush=True)
    train_time_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        x_eval, y_eval = task.sample(batch=1024)
        code_eval = model.encode(x_eval)
        x_hat_eval = model.decode(code_eval)
        mse = F.mse_loss(x_hat_eval, x_eval).item()
        var = x_eval.var().item()
        rtf = max(0.0, 1.0 - mse / var)
        code_np = code_eval.numpy()
        y_np = y_eval.numpy()
        per_dim_mi = []
        for d in range(code_np.shape[1]):
            binned = np.digitize(
                code_np[:, d],
                np.percentile(code_np[:, d], np.linspace(5, 95, 15)),
            )
            per_dim_mi.append(mutual_info_score(binned, y_np))
        mi_h = float(np.mean(per_dim_mi))
        cov = np.cov(code_np.T)
        eigs = np.linalg.eigvalsh(cov)[::-1]
        cumvar = np.cumsum(eigs) / eigs.sum()
        eff_rank = int(np.searchsorted(cumvar, 0.95) + 1)
        bw_eff = eff_rank / code_np.shape[1]

    n_params = sum(p.numel() for p in model.parameters())
    return RunMetrics(
        arch=arch_cls.__name__, seed=seed,
        train_loss_final=last_loss,
        round_trip_fidelity=rtf, mi_h=mi_h,
        bandwidth_efficiency=bw_eff,
        n_params=n_params, train_time_s=train_time_s,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-classes", type=int, default=4)  # Q1++ canonical
    parser.add_argument("--code-dim", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--archs", nargs="+",
                        default=list(ARCHITECTURES.keys()))
    parser.add_argument(
        "--out", type=Path,
        default=Path("experiments/benchmark_multiplexer_vs_baselines/results_q1plusplus.json"),
    )
    args = parser.parse_args()

    all_results: list[dict] = []
    for arch_name in args.archs:
        for seed in args.seeds:
            print(f"=== {arch_name} seed={seed} ===", flush=True)
            metrics = train_one(
                ARCHITECTURES[arch_name], seed,
                dim=args.dim, n_classes=args.n_classes, code_dim=args.code_dim,
                steps=args.steps, batch=args.batch,
            )
            all_results.append(asdict(metrics))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({
                "config": vars(args) | {"out": str(args.out)},
                "task": "FlowProxyTask",
                "results": all_results,
            }, indent=2, default=str))
            print(f"  rtf={metrics.round_trip_fidelity:.4f} "
                  f"mi_h={metrics.mi_h:.4f} "
                  f"bw_eff={metrics.bandwidth_efficiency:.4f} "
                  f"time={metrics.train_time_s:.1f}s", flush=True)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
