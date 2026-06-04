"""macm1 scientific eval: wide-variety alignment metrics over scale & noise.

Runs entirely on MPS via torch. 50 seeds x N values x noise levels.
Output: JSON of every cell + skipped log.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _device(name: str) -> str:
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("[abort] mps not available on this host")
    return name


def hsic_debiased_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    n = x.shape[0]
    kx = x @ x.T
    ky = y @ y.T
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    kx = kx * (1 - eye)
    ky = ky * (1 - eye)
    kx_row = kx.sum(dim=1)
    ky_row = ky.sum(dim=1)
    kx_sum = kx_row.sum()
    ky_sum = ky_row.sum()
    term_trace = (kx * ky).sum()
    term_outer = (kx_sum * ky_sum) / ((n - 1) * (n - 2))
    term_cross = 2.0 * (kx_row @ ky_row) / (n - 2)
    return float(((term_trace + term_outer - term_cross) / (n * (n - 3))).item())


def cknna_torch(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    n = x.shape[0]
    sq_x = (x * x).sum(dim=-1)
    sq_y = (y * y).sum(dim=-1)
    dist_x = sq_x[:, None] + sq_x[None, :] - 2 * (x @ x.T)
    dist_y = sq_y[:, None] + sq_y[None, :] - 2 * (y @ y.T)
    # Mask diagonal with +inf so a point is never its own nearest neighbour.
    # NOTE: do NOT use `eye * inf` — that produces NaN off-diagonal (0*inf=nan).
    dist_x = dist_x.clone()
    dist_y = dist_y.clone()
    idx = torch.arange(n, device=x.device)
    dist_x[idx, idx] = float("inf")
    dist_y[idx, idx] = float("inf")
    _, nn_x = torch.topk(dist_x, k=k, largest=False, dim=-1)
    _, nn_y = torch.topk(dist_y, k=k, largest=False, dim=-1)
    mx = torch.zeros(n, n, dtype=torch.bool, device=x.device).scatter_(1, nn_x, True)
    my = torch.zeros(n, n, dtype=torch.bool, device=y.device).scatter_(1, nn_y, True)
    return float((((mx & my).sum(dim=-1).float()) / k).mean().item())


def linear_cka_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    hxy = hsic_debiased_torch(x, y)
    hxx = hsic_debiased_torch(x, x)
    hyy = hsic_debiased_torch(y, y)
    denom = (hxx * hyy) ** 0.5
    return hxy / denom if denom > 1e-12 else 0.0


def procrustes_r2_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """1 - ||X R - Y||^2 / ||Y||^2 with R from orthogonal Procrustes."""
    # SVD can be slow/unstable on MPS for some shapes; fallback to CPU if needed.
    m = y.T @ x
    try:
        u, _, vh = torch.linalg.svd(m)
    except RuntimeError:
        u, _, vh = torch.linalg.svd(m.cpu())
        u = u.to(x.device)
        vh = vh.to(x.device)
    r = u @ vh
    pred = x @ r
    num = ((pred - y) ** 2).sum()
    den = (y ** 2).sum()
    return float((1.0 - num / den).item()) if float(den.item()) > 1e-12 else 0.0


def mutual_knn_corr_torch(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    n = x.shape[0]
    sq_x = (x * x).sum(dim=-1)
    sq_y = (y * y).sum(dim=-1)
    dist_x = sq_x[:, None] + sq_x[None, :] - 2 * (x @ x.T)
    dist_y = sq_y[:, None] + sq_y[None, :] - 2 * (y @ y.T)
    dist_x = dist_x.clone()
    dist_y = dist_y.clone()
    idx = torch.arange(n, device=x.device)
    dist_x[idx, idx] = float("inf")
    dist_y[idx, idx] = float("inf")
    rank_x = dist_x.argsort(dim=-1)[:, :k]
    rank_y = dist_y.argsort(dim=-1)[:, :k]
    matches = (rank_x == rank_y).float().mean()
    return float(matches.item())


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def run_cell(*, n: int, sigma: float, seed: int, device: str) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_cpu = torch.randn(n, 32, generator=g)
    eps = torch.randn(n, 32, generator=g)
    y_cpu = x_cpu + sigma * eps
    perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed + 100003))
    y_null_cpu = y_cpu[perm]
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    y_null = y_null_cpu.to(device)
    _sync(device)
    t0 = time.perf_counter()
    real = {
        "hsic":               hsic_debiased_torch(x, y),
        "cknna_5":            cknna_torch(x, y, 5),
        "cknna_10":           cknna_torch(x, y, 10),
        "cknna_20":           cknna_torch(x, y, 20),
        "cknna_50":           cknna_torch(x, y, 50),
        "linear_cka":         linear_cka_torch(x, y),
        "procrustes_r2":      procrustes_r2_torch(x, y),
        "mutual_knn_corr_10": mutual_knn_corr_torch(x, y, 10),
    }
    null = {
        "hsic":          hsic_debiased_torch(x, y_null),
        "cknna_10":      cknna_torch(x, y_null, 10),
        "procrustes_r2": procrustes_r2_torch(x, y_null),
    }
    _sync(device)
    wall = time.perf_counter() - t0
    del x, y, y_null
    if device == "mps":
        torch.mps.empty_cache()
    return {"n": n, "sigma": sigma, "seed": seed, "wall_s": wall, "real": real, "null": null}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ns", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.001, 0.01, 0.05, 0.2, 1.0])
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--budget-s", type=float, default=5400.0,
                   help="Soft wall-clock budget. After this elapses, stop accepting larger N.")
    args = p.parse_args()

    device = _device(args.device)
    print(f"device={device}, torch={torch.__version__}, "
          f"mps_avail={torch.backends.mps.is_available()}", flush=True)

    cells: list[dict] = []
    skipped: list[dict] = []
    t_start = time.perf_counter()

    max_n_allowed = max(args.ns)
    budget_hit = False
    for n in sorted(args.ns):
        if n > max_n_allowed:
            skipped.append({"n": n, "sigma": None, "seed": None,
                            "error": "skipped: above max_n_allowed cap"})
            continue
        for sigma in args.sigmas:
            broke = False
            for seed in range(args.seeds):
                if time.perf_counter() - t_start > args.budget_s:
                    print(f"[budget] hit at n={n}, sigma={sigma}, seed={seed}; capping",
                          flush=True)
                    budget_hit = True
                    max_n_allowed = n
                    broke = True
                    break
                try:
                    row = run_cell(n=n, sigma=sigma, seed=seed, device=device)
                    cells.append(row)
                except RuntimeError as exc:
                    skipped.append({
                        "n": n, "sigma": sigma, "seed": seed,
                        "error": str(exc)[:200],
                    })
                    print(f"[skip] n={n} sigma={sigma} seed={seed}: {exc!r}",
                          flush=True)
                    if "MPS" in str(exc) or "out of memory" in str(exc).lower():
                        max_n_allowed = min(max_n_allowed, n - 1)
                        broke = True
                        break
            if broke:
                break
        elapsed = time.perf_counter() - t_start
        print(f"[progress] n={n} elapsed={elapsed:.1f}s "
              f"cells={len(cells)} skipped={len(skipped)}", flush=True)
        if budget_hit:
            break

    out = {
        "device": device,
        "torch_version": torch.__version__,
        "ns": args.ns,
        "sigmas": args.sigmas,
        "seeds": args.seeds,
        "cells": cells,
        "skipped": skipped,
        "total_wall_s": time.perf_counter() - t_start,
        "budget_hit": budget_hit,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({len(cells)} cells, {len(skipped)} skipped, "
          f"{out['total_wall_s']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
