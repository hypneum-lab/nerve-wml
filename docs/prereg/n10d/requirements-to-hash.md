# N10-D dependency candidates - to be hashed and HITL-reviewed

Nothing is installed. Versions are the latest on PyPI/GitHub on 2026-10-31 that are
at least two weeks old (supply-chain policy: pin, mirror in the `ailiance` org, diff
review before merge, `pip --require-hashes`). Each line: **to be hashed and
HITL-reviewed**; wheel availability unchecked.

**Host change (v2, 2026-11-08):** all runs moved from KXKM-AI (CUDA) to macM1
(Apple M1, CPU/MPS). Consequences:
- `jax-cuda12-plugin` is **not needed** on macM1; replace with `jax-metal` or
  the CPU-only wheel (JAX MPS / Metal backend availability must be confirmed at
  the pilot, TBD-11; see protocol.md v2 section 10).
- `torch` wheel must be `aarch64` (macOS arm64), not CUDA. The 2.14.1 candidate
  is listed for EGG; verify that `torch.mps` is available in this version.
- No CUDA-specific packages (`jax-cuda12-plugin`, `torch` CUDA extras) are needed.

| package | candidate pin | released | note |
|---|---|---|---|
| jaxmarl | 0.2.0 | 2026-09-10 | JAX MARL envs |
| jax / jaxlib | 0.11.2 | 2026-09-17 | 0.12.0.1 (10-21) too recent |
| jax-metal | TBD | — | macM1 MPS backend; version must equal jax; replaces jax-cuda12-plugin |
| jax-cuda12-plugin | 0.11.2 | 2026-09-17 | CUDA only — NOT needed on macM1; kept for reference |
| flax | 0.12.10 | 2026-09-24 | 0.12.11 (10-26) and 0.12.12 (10-27) too recent |
| optax | 0.2.8 | 2026-03-20 | |
| chex | 0.1.92 | 2026-06-12 | |
| gymnax | 1.0.0 | 2026-08-16 | 1.0.1 (10-18) too recent |
| numpy | 2.5.4 | 2026-10-10 | check jax 0.11.2 compatibility |
| torch (EGG only) | 2.14.1 | 2026-09-30 | aarch64 macOS arm64 wheel; verify torch.mps availability |
| EGG | commit 54eb11c6b4ba3d3171811378f7ad7a117fa93a97 | 2025-05-06 | no PyPI release; PI decision 2026-10-31: fork into `ailiance` org at this commit, with code review |

Both backends are in scope (PI decision 2026-10-31). Grouped HITL review at lock
time. Transitive closure and SHA-256 of every wheel: still to produce.

**New protocol files added in v2 (2026-11-08) — to include in the freeze manifest:**

| file | role |
|---|---|
| `docs/prereg/n10d/protocol.md` | v2 protocol (this PR) |
| `docs/prereg/n10d/null-v2.md` | null pseudocode and TOST specification (this PR) |
| `docs/prereg/n10d/nh96-null.md` | NH-96 null (superseded banner added; kept for PC3) |
