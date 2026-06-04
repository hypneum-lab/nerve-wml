"""Renf 7: test 4 synchrony alternatives on the 4 GTM-ablation arms.

Runs each arm `seeds` times, captures the trained carrier, computes
4 candidate metrics. Reports per-arm mean +/- std and paired Wilcoxon
tests pairwise across arms to identify which metric is MONOTONE
(saine > intermediate > degenerate) and gives the cleanest ordering.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam

from scripts.synchrony_alternatives import (
    effective_rank,
    participation_ratio,
    spectral_entropy,
    top3_dispersion,
)
from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer
from track_p.transducer_baselines import (
    KuramotoMultiplexer,
    SimpleGatingMultiplexer,
)

_METRICS = {
    "spectral_entropy":   spectral_entropy,
    "participation_ratio": participation_ratio,
    "effective_rank":     effective_rank,
    "top3_dispersion":    top3_dispersion,
}


def _train_and_carrier_gtm(
    codes: torch.Tensor, steps: int, seed: int, device: str = "cpu"
) -> torch.Tensor:
    cfg = GammaThetaConfig(symbols_per_theta=codes.shape[-1])
    g = GammaThetaMultiplexer(cfg=cfg, seed=seed).to(device)
    opt = Adam(g.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        c = g.forward(codes)
        soft = g.demodulate(c, hard=False, tau=1.0)
        loss = F.cross_entropy(
            torch.log(soft + 1e-9).reshape(-1, soft.shape[-1]),
            codes.reshape(-1),
        )
        loss.backward()
        opt.step()
        g.step()
    with torch.no_grad():
        return g.forward(codes).detach().cpu()


def _train_and_carrier_simple(codes: torch.Tensor, steps: int, device: str = "cpu") -> torch.Tensor:
    m = SimpleGatingMultiplexer(alphabet_size=64, n_symbols=codes.shape[-1]).to(device)
    opt = Adam(m.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        c = m.forward(codes)
        logits = m.demodulate_logits(c)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        return m.forward(codes).detach().cpu()


def _train_and_carrier_akorn_best(
    codes: torch.Tensor, steps: int, device: str = "cpu"
) -> torch.Tensor:
    """AKOrN with the Renf 1 winning config (n_osc=64, n_steps=32, lr=0.05)."""
    m = KuramotoMultiplexer(
        alphabet_size=64, n_symbols=codes.shape[-1],
        n_oscillators=64, n_steps=32,
    ).to(device)
    opt = Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        c = m.forward(codes)
        logits = m.demodulate_logits(c)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        return m.forward(codes).detach().cpu()


def _train_and_carrier_null(
    codes: torch.Tensor, steps: int, seed: int, device: str = "cpu"
) -> torch.Tensor:
    """GTM trained on shuffled codes vs original codes -> degenerate."""
    g_rng = torch.Generator()
    g_rng.manual_seed(seed + 9973)
    perm = torch.randperm(codes.shape[0], generator=g_rng).to(device)
    shuffled = codes[perm]
    cfg = GammaThetaConfig(symbols_per_theta=codes.shape[-1])
    g = GammaThetaMultiplexer(cfg=cfg, seed=seed).to(device)
    opt = Adam(g.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        c = g.forward(shuffled)
        soft = g.demodulate(c, hard=False, tau=1.0)
        loss = F.cross_entropy(
            torch.log(soft + 1e-9).reshape(-1, soft.shape[-1]),
            codes.reshape(-1),
        )
        loss.backward()
        opt.step()
        g.step()
    with torch.no_grad():
        return g.forward(shuffled).detach().cpu()


def run_one_seed(*, seed: int, steps: int, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    codes_cpu = torch.randint(0, 64, (128, 7))
    codes = codes_cpu.to(device)
    carriers = {
        "gtm":              _train_and_carrier_gtm(codes, steps, seed, device),
        "simple_gating":    _train_and_carrier_simple(codes, steps, device),
        "akorn_best":       _train_and_carrier_akorn_best(codes, steps, device),
        "null":             _train_and_carrier_null(codes, steps, seed, device),
    }
    out: dict[str, dict[str, float]] = {}
    for arm, carrier in carriers.items():
        out[arm] = {name: fn(carrier) for name, fn in _METRICS.items()}
    return out


def _run_one_seed_kw(args: tuple) -> tuple[int, dict]:
    s, steps, device = args
    return s, run_one_seed(seed=s, steps=steps, device=device)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    # MPS/CUDA cannot share device contexts across forked workers reliably.
    # Force workers=1 (single-process) when running on GPU.
    if args.device != "cpu" and args.workers != 1:
        print(f"[warn] forcing workers=1 for device={args.device}")
        args.workers = 1

    t0 = time.perf_counter()
    seeds = list(range(args.seeds))
    workers = args.workers or min(len(seeds), mp.cpu_count())
    print(
        f"running {len(seeds)} seeds x 4 arms x 4 metrics on "
        f"{workers} workers, device={args.device}"
    )
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(
                _run_one_seed_kw, [(s, args.steps, args.device) for s in seeds]
            )
    else:
        results = [_run_one_seed_kw((s, args.steps, args.device)) for s in seeds]

    arms = ["gtm", "simple_gating", "akorn_best", "null"]
    metrics = list(_METRICS)
    agg: dict[str, dict[str, dict]] = {
        arm: {m: {"values": []} for m in metrics} for arm in arms
    }
    for _, row in results:
        for arm in arms:
            for m in metrics:
                agg[arm][m]["values"].append(row[arm][m])
    for arm in arms:
        for m in metrics:
            v = np.asarray(agg[arm][m]["values"])
            agg[arm][m]["mean"] = float(v.mean())
            agg[arm][m]["std"] = float(v.std())

    paired: dict[str, dict[str, dict[str, float]]] = {}
    for m in metrics:
        paired[m] = {}
        for arm_a in arms:
            for arm_b in arms:
                if arm_a == arm_b:
                    continue
                va = np.asarray(agg[arm_a][m]["values"])
                vb = np.asarray(agg[arm_b][m]["values"])
                if np.allclose(va, vb):
                    paired[m][f"{arm_a}_vs_{arm_b}"] = {"p_value": 1.0, "median_diff": 0.0}
                    continue
                res = stats.wilcoxon(va, vb, alternative="two-sided", zero_method="wilcox")
                paired[m][f"{arm_a}_vs_{arm_b}"] = {
                    "p_value": float(res.pvalue),
                    "median_diff": float(np.median(va - vb)),
                }

    monotone_score: dict[str, dict] = {}
    for m in metrics:
        ordered = sorted(arms, key=lambda a: agg[a][m]["mean"])
        means = [agg[a][m]["mean"] for a in ordered]
        gaps = [means[i + 1] - means[i] for i in range(len(means) - 1)]
        monotone_score[m] = {
            "ordering_low_to_high": ordered,
            "min_gap": min(gaps),
            "max_gap": max(gaps),
            "smallest_pvalue_in_chain": min(
                paired[m][f"{ordered[i]}_vs_{ordered[i+1]}"]["p_value"]
                for i in range(len(ordered) - 1)
            ),
        }

    out = {
        "seeds": args.seeds, "steps": args.steps,
        "aggregated": agg, "paired": paired,
        "monotone_score": monotone_score,
        "wall_clock_s": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({len(seeds)} seeds, {out['wall_clock_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
