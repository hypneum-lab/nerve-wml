"""Renf 12: AKOrN top-cell (n_osc=64, n_steps=32, lr=0.05) at 50 seeds.

Renf 1 reported synchrony 0.4542 ± 0.1624 for the top cell on 5 seeds.
The std=0.16 is large; at 50 seeds, the CI should tighten by ~sqrt(10).
This script re-runs that single cell at 50 seeds and reports the new
mean and bootstrap CI to check the original estimate.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam

from nerve_wml.methodology.mi_estimators import mi_miller_madow_discrete
from nerve_wml.methodology.paired_tests import bootstrap_ci
from scripts.gtm_ablation_pilot import _synchrony_index
from track_p.transducer_baselines import KuramotoMultiplexer

_BITS = float(np.log2(np.e))


def _run_one(args):
    seed, train_steps = args
    torch.manual_seed(seed)
    codes = torch.randint(0, 64, (128, 7))
    m = KuramotoMultiplexer(
        alphabet_size=64, n_symbols=codes.shape[-1],
        n_oscillators=64, n_steps=32,
    )
    opt = Adam(m.parameters(), lr=0.05)
    for _ in range(train_steps):
        opt.zero_grad()
        carrier = m.forward(codes)
        logits = m.demodulate_logits(carrier)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        carrier = m.forward(codes)
        pred = m.demodulate(carrier)
        acc = float((pred == codes).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            codes.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return seed, acc, mi, sync


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t0 = time.perf_counter()
    tasks = [(s, args.train_steps) for s in range(args.seeds)]
    workers = args.workers or min(len(tasks), mp.cpu_count())
    print(f"running {len(tasks)} seeds on {workers} workers")
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_run_one, tasks)
    else:
        results = [_run_one(t) for t in tasks]

    accs  = [r[1] for r in results]
    mis   = [r[2] for r in results]
    syncs = [r[3] for r in results]

    def _summary(vals):
        v = np.asarray(vals)
        ci = bootstrap_ci(list(vals), n_resamples=2000, seed=0)
        return {
            "values": list(vals),
            "mean": float(v.mean()),
            "std":  float(v.std()),
            "ci95_low":  ci["ci95_low"],
            "ci95_high": ci["ci95_high"],
        }

    out = {
        "cell": "n64_s32_lr0.05",
        "renf1_reference": {
            "n_seeds": 5,
            "synchrony_mean": 0.4542,
            "synchrony_std":  0.1624,
        },
        "n_seeds": args.seeds,
        "train_steps": args.train_steps,
        "accuracy": _summary(accs),
        "mi_bits":  _summary(mis),
        "synchrony_index": _summary(syncs),
        "wall_clock_s": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['wall_clock_s']:.1f}s")
    s = out["synchrony_index"]
    print(f"sync at 50s: {s['mean']:.4f} ± {s['std']:.4f}")
    print(f"  CI95: [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
