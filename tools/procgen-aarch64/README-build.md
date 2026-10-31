# Procgen 0.10.7 on Linux aarch64 (CPython 3.12)

Target: DGX Station GB300 (Grace ARM, Ubuntu 24.04, 64 KiB pages) for the
N10-C-bis gate (nerve-wml PR #99, annex G7). Procgen 0.10.7 ships no
aarch64 wheel and no sdist on PyPI, so it is built from source.

## Pinned inputs

| Item | Value |
|---|---|
| Source | `https://github.com/openai/procgen`, tag `0.10.7` |
| Commit | `5e1dbf341d291eff40d1f9e0c0a0d5003643aebf` |
| `git archive` (tar, uncompressed) sha256 | `881b1f4a8bb7326446be3c8ed787069ff89143868da0301f8051a16499960794` |
| Patches | `patches/0001..0003` (sha256 in `patches.sha256`) |
| Python deps | `requirements-aarch64-cp312.lock` (hash-pinned, resolved with `uv pip compile --python-platform aarch64-unknown-linux-gnu --python-version 3.12`), sha256 `b49aba5ae95cb2ce825a7118bcbcea7eb68f9db627b1abae1b3aef98064ebea1` |
| apt (noble) | build-essential 12.10ubuntu1, g++ 4:13.2.0-7ubuntu1, cmake 3.28.3-1build7, ninja-build 1.11.1-2, pkg-config 1.8.1-2build1, git 1:2.43.0-1ubuntu7.4, qtbase5-dev 5.15.13+dfsg-1ubuntu1, libgl-dev 1.7.0-1build1, python3.12-dev and python3.12-venv 3.12.3-1ubuntu0.17 |

The script sha256 is the git blob of `build_procgen_aarch64.sh` at the
PR head (see the PR); it is not repeated here to avoid a stale value.

## Patch series

Applied with `git apply` on the pinned commit (no fork):

1. `setup.py`: drop `filelock<4.0.0` (PI decision; the lock resolves
   filelock 4.1.0). `builder.py` only uses `filelock.FileLock(path)` as a
   context manager.
2. `CMakeLists.txt`: `-march=ivybridge` / `-march=native` only on x86
   processors. This flag (the "at least AVX" requirement) is the only
   x86-specific code: grep finds no SSE/AVX intrinsics (`_mm*`,
   `*mmintrin.h`) in `procgen/src`, so **sse2neon is not needed and is not
   vendored**.
3. `CMakeLists.txt`: on non-x86 Linux add `-fsigned-char
   -ffp-contract=off`. Plain `char` is unsigned on aarch64 Linux and GCC
   fuses multiply-add into FMA there; the ivybridge x86 build has neither
   behaviour. This keeps level generation and physics as close as possible
   to the x86/macOS builds. It is a precaution, see risks.

## Use

```
bash tools/procgen-aarch64/build_procgen_aarch64.sh
# env: WORK (default ~/procgen-aarch64-build), JOBS (4), SKIP_APT=1, APT_LOOSE=1
```

The script is idempotent: it hard-resets the clone to the pinned commit,
checks the patch sha256, re-applies the patches and rebuilds under
`nice -n 19`. It then installs the wheel into a clean venv outside the
clone, runs a CoinRun smoke test, and prints the wheel path and sha256.
Report that sha256 and the PR head in PR #99 / annex G7.

## Tested / not tested

Tested (electron-server, x86_64):
- `bash -n` on the script;
- the 3 patches apply cleanly on the pinned commit;
- CMake flag logic evaluated for `x86_64` (`-march=ivybridge`) and
  `aarch64` (`-fsigned-char -ffp-contract=off`);
- the hash-pinned lock resolves for aarch64 / cp312, filelock 4.1.0
  included.

NOT tested: the compile, the wheel and the smoke test. No arm64 host was
usable: KXKM, macM1 and macM3 run campaigns. On Studio, the Docker CLI
29.5.3 and Docker.app exist but user `ailiance` gets "permission denied"
on `/var/run/docker.sock` (engine owned by another account); nothing was
installed or escalated. Emulation on x86 ES was ruled out. The build must
be run on the GB300 (or any arm64 Ubuntu 24.04). The smoke test prints an
observation hash that is not yet a reference value.

## Remaining risks

- First real build may hit GCC 13 diagnostics or Qt5 lookup issues
  (`PROCGEN_CMAKE_PREFIX_PATH` is set to `/usr/lib/aarch64-linux-gnu/cmake`).
- Procgen levels and frames are not guaranteed bit-identical to
  x86/macOS (libm, Qt rasterizer); compare a few seeds before pooling
  results with macM1/KXKM runs.
- Pinned apt versions can vanish from the arm64 archive: use
  `APT_LOOSE=1` and record `dpkg -l`.
- gym 0.26.2 and gym3 0.3.3 with numpy 1.26.4 on Python 3.12 are untested
  together here; gym 0.26.2 is sdist-only (built by pip, hash-pinned).
- filelock 4.1.0 is outside the range declared by upstream.

## Build test on arm64 (2026-10-31, GrosMac, colima)

Tested in a colima Linux arm64 VM (2 CPU, 4 GB) on GrosMac, container
`ubuntu@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852`
(`ubuntu:24.04`, index digest, arm64 variant), kit at PR head 8ba0d8a,
`JOBS=2`, pinned apt versions installed without `APT_LOOSE`.

- Build and wheel OK in under 2 minutes, CoinRun smoke test OK.
- Wheel: `procgen-0.10.7+5e1dbf3-cp312-cp312-linux_aarch64.whl`,
  sha256 `e977b3f2340dbc41da04182332a985aeaebadd71676d61f0ba34d0ad13c4ea21`.
- Smoke obs-hash (coinrun, 2 envs, 50 zero-action steps, seed 0):
  `7c082778387f1eafd230ed3104323d2169e5ca99bb9f7a280d2bbbb9a93198af`
  (arm64 result only; not compared to x86).
- Not verified: the VM page size is 4 KiB (`getconf PAGESIZE` = 4096), the
  GB300 uses 64 KiB pages. Rebuild/run on the GB300 to confirm. Also not
  compared: frames vs x86/macOS builds. The colima VM image download
  was not recorded (image was cached or pulled by colima itself).
- The wheel hash may differ on the GB300 (timestamps, toolchain build).
