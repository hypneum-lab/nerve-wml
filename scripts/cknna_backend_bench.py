"""Renf 5: backend benchmark (CPU / MPS / CUDA) for the two ops in nerve-wml
that could plausibly benefit from GPU acceleration.

Usage:
    uv run python -m scripts.cknna_backend_bench --device cpu --out cpu.json
    uv run python -m scripts.cknna_backend_bench --device mps --out mps.json
    uv run python -m scripts.cknna_backend_bench --device cuda --out cuda.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    raise ValueError(f"unknown device {device!r}")


def cknna_torch(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """Pure-torch CKNNA on the device ``x`` lives on."""
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"row mismatch: x {n}, y {y.shape[0]}")
    sq_x = (x * x).sum(dim=-1)
    sq_y = (y * y).sum(dim=-1)
    dist_x = sq_x[:, None] + sq_x[None, :] - 2 * (x @ x.T)
    dist_y = sq_y[:, None] + sq_y[None, :] - 2 * (y @ y.T)
    # Mask self by setting diagonal to +inf.
    eye_inf = torch.eye(n, device=x.device, dtype=x.dtype) * float("inf")
    dist_x = dist_x + eye_inf
    dist_y = dist_y + eye_inf
    _, nn_x = torch.topk(dist_x, k=k, largest=False, dim=-1)
    _, nn_y = torch.topk(dist_y, k=k, largest=False, dim=-1)
    mask_x = torch.zeros(n, n, dtype=torch.bool, device=x.device)
    mask_y = torch.zeros(n, n, dtype=torch.bool, device=y.device)
    mask_x.scatter_(1, nn_x, True)
    mask_y.scatter_(1, nn_y, True)
    intersection = (mask_x & mask_y).sum(dim=-1).float()
    return intersection.mean() / k


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def bench_cknna(device: str, sizes: list[int], k: int = 10) -> list[dict]:
    rows = []
    # MPS generator does not support seeded sampling in some torch builds; use CPU gen + .to.
    g = torch.Generator()
    g.manual_seed(0)
    for n in sizes:
        try:
            x_cpu = torch.randn(n, 32, generator=g)
            y_cpu = x_cpu + 0.05 * torch.randn(n, 32, generator=g)
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            # Warm-up.
            _ = cknna_torch(x, y, k)
            _sync(device)
            ts: list[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                score = cknna_torch(x, y, k)
                _sync(device)
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
            print(f"  cknna n={n:6d}  wall={wall*1000:8.2f} ms  score={float(score):.3f}")
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "n": n,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"  cknna n={n:6d}  ERROR {type(exc).__name__}: {exc}")
    return rows


def bench_transducer_train(device: str, steps: int = 500) -> dict:
    """Tiny Transducer training (64x64 + 1024 codes). Reports total wall."""
    try:
        from torch.nn import functional as F  # noqa: N812
        from torch.optim import Adam

        from track_p.transducer import Transducer, TransducerGating

        g = torch.Generator()
        g.manual_seed(0)
        src_codes = torch.randint(0, 64, (1024,), generator=g)
        perm = torch.randperm(64, generator=g)
        dst_codes = perm[src_codes]
        src_codes = src_codes.to(device)
        dst_codes = dst_codes.to(device)
        t = Transducer(
            alphabet_size=64, gating=TransducerGating.GUMBEL_SOFTMAX,
        ).to(device)
        opt = Adam(t.parameters(), lr=0.05)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad()
            soft = t.forward(src_codes, hard=False, tau=1.0)
            loss = F.cross_entropy(torch.log(soft + 1e-9), dst_codes)
            loss.backward()
            opt.step()
        _sync(device)
        wall = time.perf_counter() - t0
        print(f"  transducer steps={steps}  wall={wall:.3f} s")
        return {"steps": steps, "wall_s": wall, "final_loss": float(loss.detach().cpu())}
    except Exception as exc:  # noqa: BLE001
        print(f"  transducer ERROR {type(exc).__name__}: {exc}")
        return {"steps": steps, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--sizes", type=int, nargs="+",
        default=[256, 1024, 4096, 8192, 16384],
    )
    args = p.parse_args()
    if not _device_available(args.device):
        print(f"device {args.device} not available; aborting")
        return 2
    print(f"benching device={args.device} on sizes={args.sizes}")
    rows = bench_cknna(args.device, args.sizes)
    train = bench_transducer_train(args.device)
    out = {
        "device": args.device,
        "torch_version": torch.__version__,
        "cknna": rows,
        "transducer_train": train,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
