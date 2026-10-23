# N10-C compute evidence (macM1, 2026-10-23)

Evidence behind the N10-C amendment's compute section (option D on
macM1 with MLX). Hardware: Apple M1 Max, 32 GB, macOS 27.0.1.

## Procgen 0.10.7 on arm64

No arm64 wheel exists on PyPI, so Procgen was built from source
(tag 0.10.7, commit 5e1dbf3) with Qt 5.15.19 (Homebrew `qt@5`) and
Python 3.10 (`requirements-procgen-py310.lock`). Build-only changes:

- the Xcode 26.5 SDK is forced (`SDKROOT`, `DEVELOPER_DIR`): the
  Command Line Tools macOS 27 SDK fails to link with Xcode's `ld`;
- `PROCGEN_CMAKE_PREFIX_PATH=$(brew --prefix qt@5)/lib/cmake` and
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` (CMake 4.4);
- `-Werror` removed (`procgen-0.10.7-arm64-build.patch`): clang 21
  flags two unused variables in `chaser.cpp` and `starpilot.cpp`.

Game logic is unchanged. Equivalence with x86-64 builds was not
checked.

CoinRun, random policy: 30,118 steps/s (1 env), 116,614 steps/s
(64 envs).

## PPO network cost (MLX 0.31.1)

`bench_ppo_mlx.py` (`requirements-mlx.lock`), IMPALA CNN, 64 envs x
256 steps per iteration, 3 epochs, minibatch 2048, `mx.compile`:

| Run | Inference (64 obs) | Update (16,384 samples) | Throughput | Per run (NN only) |
|---|---:|---:|---:|---:|
| 1 | 3.36 ms | 8.20 s | 1,807 samples/s | 15.4 h |
| 2 | 3.39 ms | 8.29 s | 1,789 samples/s | 15.5 h |

Without `mx.compile` the update took 99.4 s (163 samples/s). A run is
2 agents x 50M steps. Env stepping, the GTM message path and
evaluation are not included; the stage-2 manifest requires an
end-to-end measurement before any run.
