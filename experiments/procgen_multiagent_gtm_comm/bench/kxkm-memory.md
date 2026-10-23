# KXKM-AI memory record (N10-C, 2026-10-23)

Host: KXKM-AI, NVIDIA GeForce RTX 4090 (24564 MiB), driver 595.84,
MLX 0.31.1 CUDA, Python 3.10 (`/ai/saisail/n10c-bench/venv310`). GPU
services stopped (PI-approved); one foreign 652 MiB process present.

## Out-of-memory evidence (no accumulation)

- Harness default minibatch (1024 pair-steps = 2048 agent samples),
  `harness/throughput.py` -> `train_run`: `RuntimeError:
  cudaMallocAsync(&data, size, stream) failed: out of memory` at
  `train.py` `mx.eval(loss, norm)` (traceback kept on the host in
  `/ai/saisail/n10c-bench/tp.out`). Peak before failure reported by the
  earlier session as 23.08 GB of 24.5 GB.
- Unmodified `bench_ppo_mlx.py` (minibatch 2048 samples) ran with a
  ~24.0 GB peak, and only after VRAM had been freed (earlier session
  report; the bench logs `b1.out`/`b2.out` hold timings only, not peaks).
  The 23.08 / ~24.0 GB figures are therefore reported values, not raw
  logs in this repo.

## Gradient accumulation (Task 11), measured this session

Free VRAM before start: 22.7 GB. gtm, seed 9001, registered,
64 pairs x 128 horizon, real procgen, training levels only,
64 updates (524,288 pair-steps) per configuration, one run each;
pair-steps/s includes compile warm-up. Raw `run.json` files in
`kxkm-memory/`. Registered minibatch (1024 pair-steps) in every row.

| micro_batch_pairs | clear_cache | pair-steps/s | mx peak (GB) | nvidia-smi peak (MiB) |
|---|---|---|---|---|
| 256 | off | 1159.7 | 5.21 | 22014 |
| 256 | on  | 1298.2 | 4.86 | 15410 |
| 512 | off | 1266.4 | 7.06 | 13700 |
| 512 | on  | 1286.2 | 6.73 | 14360 |

The mx peak was not reset at run start in these runs (reset added
afterwards; each run was its own process, so the value is per run).
nvidia-smi peaks include the allocator cache and are noisy.
