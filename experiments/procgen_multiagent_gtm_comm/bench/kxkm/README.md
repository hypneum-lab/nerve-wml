# KXKM-AI memory measurements (N10-C Task 11), 2026-10-23

Host KXKM-AI, RTX 4090 (24564 MiB), driver 595.84, MLX 0.31.1 CUDA,
Python 3.10. Only `gpu-worker` and `whisper-dex` were stopped for the
campaign; one foreign 652 MiB process (pid 712018) stayed. Free VRAM
before each run: see `*.before.txt` (22.7 / 22.6 GB, above the 20 GB
gate). Command (cwd = repo root, `PYTHONPATH=.`):

    python -m experiments.procgen_multiagent_gtm_comm.harness.train \
      --arm gtm --seed 9001 --condition registered --steps N --out DIR \
      --log-every N --ckpt-every N [--micro-batch-pairs 512 --clear-cache]

`nvidia-smi --query-gpu=memory.used --format=csv -lms 200` ran alongside
(`*.smi.csv`). Raw files: `*.smi.csv`, `*.before.txt`, `*.out`,
`*.run.json` (run record, incl. `peak_memory_bytes` = mx.get_peak_memory
of the process). Real procgen, training levels only, 64 pairs x 128.

| run | updates | micro | clear_cache | mx peak (GB) | nvidia-smi peak (MiB) | pair-steps/s (incl. warm-up) | outcome |
|---|---|---|---|---|---|---|---|
| noaccum_2upd | 2 | none (1024 pairs) | off | 11.998 | 23252 | 1172.7 | completed, no OOM |
| micro512_clear_8upd | 8 | 512 | on | 6.681 | 13788 | 1294.5 | completed |

Reading. With the GPU otherwise free, the registered minibatch did NOT
run out of memory in 2 updates: the process' allocator cache grew to
23.3 GB of 24.6 GB (nvidia-smi), i.e. almost no headroom, of which
12.0 GB is the tracked peak. An earlier OOM
(`cudaMallocAsync ... out of memory` at `train.py` `mx.eval(loss, norm)`,
traceback in `/ai/saisail/n10c-bench/tp.out` on the host, not in this
repo) occurred under `throughput.py` while other GPU services were
running; its peak value is not traceable and is therefore not cited. The
case for accumulation is the margin (6.7 GB tracked / 13.8 GB reserved
vs 12.0 GB / 23.3 GB), not a reproduced OOM at 2 updates. A longer
no-accumulation run was not made.
