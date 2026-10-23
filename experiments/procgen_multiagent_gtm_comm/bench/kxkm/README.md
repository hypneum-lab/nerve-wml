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

## Same-seed repeat control on CUDA (added)

`repeat_control.py` (2 runs per config, separate processes, real procgen,
8 updates = 65536 pair-steps, gtm and mlp, no accumulation and micro 512)
and `grad_repeat.py` (fixed minibatch gradient repeats). Outputs under
`repeat/` (`*.info.json`, `*.params.npz`, `analysis.txt`,
`grad_localisation.txt`).

Result: init, forward and the FIRST rollout (obs/actions/logp/value/
returns hashes) are bit-identical across repeats in all 4 configs. Params
after update 1 (24 Adam steps) and after update 8 are NOT identical in
any of them (every tensor differs; max |d| 3.9e-2 to 5.2e-2 after update
1, 1.1e-1 to 1.9e-1 after update 8); the update-8 rollout then differs
too, so trajectories diverge. Localisation on a fixed minibatch: forward
bit-identical; the only non-identical gradients are conv weight/bias in
`trunk.blocks.*` (all 22 conv tensors), max relative diff 1.4e-5 to
4.5e-5 (CUDA/cuDNN; ~1e-7 on Metal), same with and without
accumulation, mlp and gtm alike. The head, fc and GTM tensors are
bit-identical. Divergence beyond the single-step Adam bound comes from
the 24 sequential steps amplifying the conv-gradient noise.

## n_workers end-to-end measurement (added 2026-10-23, after PR #54)

Same command as above with `--micro-batch-pairs 512 --clear-cache` and
`--n-workers N`, gtm registered, seed 9001, 8 updates (65,536
pair-steps), real procgen, training levels only, run one at a time in
the order 1, 8, 1, 8. Free VRAM before each run 22.6-22.7 GB (the same
foreign 652 MiB process only; nothing was stopped or killed). Raw files
in `nworkers/` (`*.run.json`, `*.before.txt`, `*.out`).

| n_workers | pair-steps/s (incl. warm-up) | wall s | mx peak GB |
|---|---|---|---|
| 1 (a, b) | 1,286.8 / 1,290.0 | 50.9 / 50.8 | 6.68 / 7.04 |
| 8 (a, b) | 1,391.1 / 1,390.2 | 47.1 / 47.1 | 6.69 / 6.67 |

Reading. `n_workers = 8` is about 8 % faster end to end than 1 in both
pairs, with the same memory. n_workers is an implementation setting
(not registered); the unit tests check that the first rollout is
identical across worker counts. The manifest records `n_workers = 8`
for KXKM-AI and 1 for macM1 (no macM1 measurement of workers exists in
the repository).
