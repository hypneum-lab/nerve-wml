# macm1 CKNNA backend bench (CPU + MPS)

Renf 5b: the same backend bench as Renf 5 (`scripts/cknna_backend_bench.py`)
but executed on the remote `macm1` (M1, 32 GB RAM, MPS-capable). Complements
the Renf 5 grosmac (M5 16 GB) numbers — comparing two M-series chips with
different unified-memory budgets.

## Configuration

- Host: `macm1` (Tailscale `100.112.121.126`).
- Python: 3.12.13 (uv-managed venv at `/tmp/bench-venv`; system `python3` is
  3.9.6 without torch).
- torch: 2.12.0, `torch.backends.mps.is_available() == True`.
- N sizes: 256, 1024, 4096, 8192.
- 5 repeats per cell, trimmed mean (drop best+worst).

## Results

| N    | CPU wall (ms) | MPS wall (ms) | speedup (CPU / MPS) |
|------|---------------|---------------|---------------------|
| 256  | 1.06          | 3.92          | 0.27× (MPS slower)  |
| 1024 | 6.49          | 5.83          | 1.11×               |
| 4096 | 133.38        | 30.93         | 4.31×               |
| 8192 | 549.10        | 93.37         | 5.88×               |

## Interpretation

MPS beats CPU on macm1 starting at N=1024 (marginal, 1.1×) and the gap widens
to ~5.9× at N=8192. The crossover is therefore between N=256 (CPU wins by
~3.7×, dispatch overhead dominates) and N=1024. For the CKNNA workload sizes
typical in nerve-wml (N≥1024), MPS is the clear win on M1; below that, the
extra dispatch/copy overhead makes CPU faster. Cross-reference with the
Renf 5 grosmac (M5) report when available — both M-series chips should show
the same crossover shape, with absolute numbers reflecting the M5's newer
GPU and the M1's larger 32 GB unified-memory budget at the high end.

## Caveats

- Single host run, no multi-seed (the bench measures raw op throughput, not
  scientific output). Trimmed-mean over 5 reps is the only smoothing.
- macm1 had no system torch; a uv venv (Python 3.12 + torch 2.12.0) was
  provisioned at `/tmp/bench-venv` specifically for this run. Numbers may
  differ from the worktree's torch version (whatever Renf 5 uses on grosmac).
- macm1 is a shared inference host (`mlx_lm.server` on :8502/:8503 plus
  LoRA adapters); the GPU was not idle-isolated. Some MPS measurement noise
  is expected, though the 4-8× CPU/MPS gap is well above any noise floor.
