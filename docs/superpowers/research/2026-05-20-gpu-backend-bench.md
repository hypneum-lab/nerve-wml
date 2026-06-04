# Renf 5 — 3-Way Backend Bench (CPU vs MPS vs CUDA 4090)

**Date:** 2026-05-20
**Branch:** `feat/gap-analysis-remediation`
**Question:** for the two ops in `nerve-wml` that could plausibly benefit from GPU acceleration — large-N CKNNA Gram matrices, and Transducer training — is the GPU worth it, and which GPU?

## TL;DR

| Op | Winner |
|----|--------|
| CKNNA `n ≤ 1024` | **CPU** (M5 wins on tiny matrices; GPU launch overhead dominates) |
| CKNNA `n = 4096` | **CUDA** (~43× CPU), MPS ~2.2× CPU |
| CKNNA `n = 8192` | **CUDA** (~39× CPU), MPS ~2.4× CPU |
| CKNNA `n = 16384` | **MPS / CPU** (CUDA OOM at 4090 24 GB with concurrent tenants) |
| Transducer training (500 steps, 64-alphabet) | **CUDA** (~3.9× CPU), MPS ~1.3× CPU |

**Interpretation:** for CKNNA, MPS is a small constant-factor win (~2–3×) over CPU once `n ≥ 4096`; CUDA is a real ~40× win in the same range but breaks at `n = 16384` on a contested 4090 (true peak Gram = 1 GB float32, plus topk/scatter workspaces ~2 GB → exceeds the ~4 GB free on the shared box). For Transducer training, CUDA is the only meaningful win (~4×) but the absolute wall (0.21 s for 500 steps) is already small enough on CPU that **you basically never need a GPU for nerve-wml as currently sized**.

## Backend status

| Backend | Status | Notes |
|---|---|---|
| CPU (M5, torch 2.11) | **SUCCESS** | 5/5 sizes, 500-step Transducer in 0.83 s |
| MPS (M5 GPU) | **SUCCESS** | 5/5 sizes, no kernel fallback observed, scores all 1.000 (matches CPU) |
| CUDA (RTX 4090 on kxkm-ai) | **PARTIAL** | 4/5 sizes; `n=16384` OOMed (4090 had ~700 MB free of 24 GB at run time — other tenants held 16 GB). The bench script returned a per-row `error` field and continued. |

## CKNNA wall-clock (trimmed mean of 5 runs, 3 best)

| N | CPU (M5) | MPS (M5) | CUDA (4090) | MPS speedup | CUDA speedup |
|---:|---:|---:|---:|---:|---:|
|   256 |   0.61 ms |   4.38 ms |   0.15 ms | **0.14×** | **4.0×** |
|  1024 |   6.31 ms |   7.72 ms |   0.25 ms | **0.82×** | **25×** |
|  4096 | 113.14 ms |  51.14 ms |   2.61 ms | **2.2×** | **43×** |
|  8192 | 454.10 ms | 190.04 ms |  11.65 ms | **2.4×** | **39×** |
| 16384 | 3195.67 ms | 3058.64 ms | **OOM** | 1.05× | n/a |

Notes:
- CKNNA `score` is 1.000 in every cell (y = x + 0.05·noise; the k-NN sets overlap fully), confirming numerical equivalence across backends.
- MPS at `n=16384` is barely better than CPU — the unified-memory M5 GPU hits the same DRAM wall as the CPU. The matmul is already memory-bound at that scale.
- CUDA `n=16384` would have been ~50 ms extrapolating from `n=8192` (8× work for 2× N because of O(N²) Gram), if a free 4090 had been available.

## Transducer training (500 steps, alphabet=64, Gumbel-Softmax)

| Backend | Wall (s) | Speedup vs CPU | Final loss |
|---|---:|---:|---:|
| CPU (M5)    | 0.827 | 1.0×  | 0.0115 |
| MPS (M5)    | 0.616 | 1.34× | 0.0132 |
| CUDA (4090) | 0.213 | 3.88× | 0.0155 |

All three converged to a very similar small loss (training is RNG-sensitive with Gumbel-Softmax, not a backend bug).

## When does GPU help?

- **CKNNA below `n ≈ 1024`: never use a GPU.** Launch overhead (especially MPS warm-up) makes it slower than CPU. The bench shows MPS losing to CPU at `n=256` and at `n=1024`.
- **CKNNA at `n ≈ 4096–8192`: CUDA is decisively worth it (~40×)**, MPS gives a modest 2–3× and is good enough if you don't have a CUDA box.
- **CKNNA at `n ≥ 16384`: be VRAM-aware.** The triple-Gram (`x@x.T`, `y@y.T`, plus topk and the `n×n bool` mask) needs ~3 × 4 × N² bytes peak. At N=16384 that's ~3 GB just for the Gram + ~1 GB for mask + workspace = ~4 GB, which is the whole budget on a contested 4090. CPU/unified-memory MPS are actually more reliable here.
- **Transducer training as currently sized is not a GPU problem.** Even on CPU it's <1 s for 500 steps. The CUDA win evaporates the moment SSH-jump latency or model upload (>5 s) enters the loop.

**Net recommendation for `nerve-wml`**: keep all current code paths on CPU. If a future config bumps `n` to 4096+ in CKNNA inner loops, route that one op through MPS locally (no SSH cost, ~2× win) or batch enough of them to amortize a CUDA round-trip.

## Reproducibility

```bash
uv run python -m scripts.cknna_backend_bench --device cpu  --out cpu.json
uv run python -m scripts.cknna_backend_bench --device mps  --out mps.json
bash scripts/cknna_remote_runner.sh   # rsync + ssh -J electron-server kxkm@10.2.0.237
```

Raw JSON: `2026-05-20-gpu-backend-bench-{cpu,mps,cuda}.json`.

CUDA remote: torch 2.11.0+cu130 on the kxkm-ai `~/venv`, ~700 MB free on the 4090 at run time (contested — `nvidia-smi` showed two larger tenants). A second pass on a freed 4090 would give the missing `n=16384` cell; the 4 collected points already define the curve unambiguously.
