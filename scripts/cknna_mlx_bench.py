"""Renf 9: CKNNA bench on Apple MLX (GPU on Apple Silicon).

Complements Renf 5/5b (CPU/MPS/CUDA) with the 4th backend: Apple's MLX
framework. MLX uses unified memory and is supposed to be the lightest
Apple-Silicon-native backend.

NOTE: MLX has no `scatter`; we build the k-NN mask via broadcast
equality (`nn[:, :, None] == cols[None, None, :]`).any(dim=1).

Run with the mlx-stack venv: /Users/electron/mlx-stack/.venv/bin/python
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def cknna_mlx(x: mx.array, y: mx.array, k: int) -> float:
    """Mutual k-NN alignment in MLX."""
    n = x.shape[0]
    sq_x = (x * x).sum(axis=-1)
    sq_y = (y * y).sum(axis=-1)
    dist_x = sq_x[:, None] + sq_x[None, :] - 2 * (x @ x.T)
    dist_y = sq_y[:, None] + sq_y[None, :] - 2 * (y @ y.T)
    # mask self-pairs to +inf
    inf_diag = mx.eye(n, dtype=x.dtype) * 1e10
    dist_x = dist_x + inf_diag
    dist_y = dist_y + inf_diag
    # k smallest indices per row (argpartition equivalent of topk-smallest)
    nn_x = mx.argpartition(dist_x, kth=k, axis=-1)[:, :k]  # [N, k]
    nn_y = mx.argpartition(dist_y, kth=k, axis=-1)[:, :k]
    # build masks via broadcast (no scatter in MLX)
    cols = mx.arange(n)
    mask_x = (nn_x[:, :, None] == cols[None, None, :]).any(axis=1)
    mask_y = (nn_y[:, :, None] == cols[None, None, :]).any(axis=1)
    intersection = (mask_x & mask_y).sum(axis=-1).astype(mx.float32)
    score = (intersection / k).mean()
    mx.eval(score)
    return float(score.item())


def bench(sizes: list[int]) -> list[dict]:
    rows = []
    mx.random.seed(0)
    for n in sizes:
        x = mx.random.normal((n, 32))
        eps = mx.random.normal((n, 32))
        y = x + 0.05 * eps
        # warm-up
        _ = cknna_mlx(x, y, 10)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            score = cknna_mlx(x, y, 10)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        wall = sum(ts[1:-1]) / max(len(ts) - 2, 1)
        rows.append({
            "n": n,
            "wall_s_trimmed_mean": wall,
            "wall_s_min": ts[0],
            "wall_s_max": ts[-1],
            "score": float(score),
        })
        print(f"  n={n:>6}  wall={wall*1e3:.2f} ms  score={score:.4f}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[256, 1024, 4096, 8192, 16384])
    args = p.parse_args()
    print(f"backend=MLX  device={mx.default_device()}")
    rows = bench(args.sizes)
    out = {
        "backend": "MLX",
        "device": str(mx.default_device()),
        "mlx_version": mx.__version__,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
