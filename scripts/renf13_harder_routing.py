"""Renf 13: harder routing task — does the GTM ablation separate on
accuracy/MI when the task gets bigger?

Renf 7 used alphabet=64, K=7; all arms saturated to accuracy=1.0 /
MI=2.22 bits. This script probes a harder regime to see whether
the arms separate on MI/accuracy too, not just on synchrony-spectrum.

Constraints discovered:
 * GTMConfig enforces the Lisman-Idiart capacity bound (K <= 9). We
   cap K at 9 (not 14).
 * Alphabet_size is a free parameter; we push to 128.

Arms: gtm, simple_gating, akorn_best, null — same training paths as
synchrony_replacement_eval.py but parametrised on (alphabet_size, K).
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
from scripts.synchrony_alternatives import spectral_entropy
from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer
from track_p.transducer_baselines import (
    KuramotoMultiplexer,
    SimpleGatingMultiplexer,
)

_BITS = float(np.log2(np.e))


def _train_gtm(codes, steps, seed, alphabet_size):
    cfg = GammaThetaConfig(
        symbols_per_theta=codes.shape[-1],
        alphabet_size=alphabet_size,
    )
    g = GammaThetaMultiplexer(cfg=cfg, seed=seed)
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
        c = g.forward(codes)
        pred = g.demodulate(c, hard=True)
    return c.detach().cpu(), pred.detach().cpu()


def _train_simple(codes, steps, alphabet_size):
    m = SimpleGatingMultiplexer(alphabet_size=alphabet_size, n_symbols=codes.shape[-1])
    opt = Adam(m.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        c = m.forward(codes)
        logits = m.demodulate_logits(c)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.reshape(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        c = m.forward(codes)
        pred = m.demodulate(c)
    return c.detach().cpu(), pred.detach().cpu()


def _train_akorn(codes, steps, alphabet_size):
    m = KuramotoMultiplexer(
        alphabet_size=alphabet_size, n_symbols=codes.shape[-1],
        n_oscillators=64, n_steps=32,
    )
    opt = Adam(m.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        c = m.forward(codes)
        logits = m.demodulate_logits(c)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.reshape(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        c = m.forward(codes)
        pred = m.demodulate(c)
    return c.detach().cpu(), pred.detach().cpu()


def _train_null(codes, steps, seed, alphabet_size):
    rng = torch.Generator()
    rng.manual_seed(seed + 9973)
    perm = torch.randperm(codes.shape[0], generator=rng)
    shuffled = codes[perm]
    cfg = GammaThetaConfig(
        symbols_per_theta=codes.shape[-1],
        alphabet_size=alphabet_size,
    )
    g = GammaThetaMultiplexer(cfg=cfg, seed=seed)
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
        c = g.forward(shuffled)
        pred = g.demodulate(c, hard=True)
    return c.detach().cpu(), pred.detach().cpu()


def _score(carrier, pred, codes_cpu):
    acc = float((pred == codes_cpu).float().mean())
    mi = mi_miller_madow_discrete(
        pred.reshape(-1).numpy().astype(np.int64),
        codes_cpu.reshape(-1).numpy().astype(np.int64),
    ) * _BITS
    se = spectral_entropy(carrier)
    return {"accuracy": acc, "mi_bits": mi, "spectral_entropy": se}


def _run_one(args):
    seed, alphabet_size, K, steps = args  # noqa: N806 (K matches Lisman-Idiart symbol)
    torch.manual_seed(seed)
    codes = torch.randint(0, alphabet_size, (128, K))

    out = {}
    c, p = _train_gtm(codes, steps, seed, alphabet_size)
    out["gtm"] = _score(c, p, codes)
    c, p = _train_simple(codes, steps, alphabet_size)
    out["simple_gating"] = _score(c, p, codes)
    c, p = _train_akorn(codes, steps, alphabet_size)
    out["akorn_best"] = _score(c, p, codes)
    c, p = _train_null(codes, steps, seed, alphabet_size)
    out["null"] = _score(c, p, codes)
    return seed, out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--alphabet_size", type=int, default=128)
    p.add_argument("--K", type=int, default=9,
                   help="symbols per theta. GTM caps at 9 (Lisman-Idiart).")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t0 = time.perf_counter()
    tasks = [(s, args.alphabet_size, args.K, args.steps) for s in range(args.seeds)]
    workers = args.workers or min(len(tasks), mp.cpu_count())
    print(f"running {len(tasks)} seeds (alphabet={args.alphabet_size}, K={args.K}) "
          f"on {workers} workers")
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_run_one, tasks)
    else:
        results = [_run_one(t) for t in tasks]

    arms = ["gtm", "simple_gating", "akorn_best", "null"]
    metrics = ["accuracy", "mi_bits", "spectral_entropy"]
    agg = {arm: {m: {"values": []} for m in metrics} for arm in arms}
    for _, row in results:
        for arm in arms:
            for m in metrics:
                agg[arm][m]["values"].append(row[arm][m])
    for arm in arms:
        for m in metrics:
            v = np.asarray(agg[arm][m]["values"])
            agg[arm][m]["mean"] = float(v.mean())
            agg[arm][m]["std"]  = float(v.std())

    # mi_chance: log2(alphabet_size) is the theoretical max per symbol
    mi_max_per_symbol = float(np.log2(args.alphabet_size))

    out = {
        "config": {
            "alphabet_size": args.alphabet_size,
            "K": args.K,
            "seeds": args.seeds,
            "steps": args.steps,
            "mi_max_per_symbol_bits": mi_max_per_symbol,
        },
        "aggregated": agg,
        "wall_clock_s": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['wall_clock_s']:.1f}s")
    for arm in arms:
        print(f"  {arm:14s} acc={agg[arm]['accuracy']['mean']:.3f} "
              f"mi={agg[arm]['mi_bits']['mean']:.3f} "
              f"se={agg[arm]['spectral_entropy']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
