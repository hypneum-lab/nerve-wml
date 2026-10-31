# N10-D dependency candidates - to be hashed and HITL-reviewed

Nothing is installed. Versions are the latest on PyPI/GitHub on 2026-10-31 that are
at least two weeks old (supply-chain policy: pin, mirror in the `ailiance` org, diff
review before merge, `pip --require-hashes`). Each line: **to be hashed and
HITL-reviewed**; aarch64/CUDA wheel availability unchecked.

| package | candidate pin | released | note |
|---|---|---|---|
| jaxmarl | 0.2.0 | 2026-09-10 | JAX MARL envs |
| jax / jaxlib | 0.11.2 | 2026-09-17 | 0.12.0.1 (10-21) too recent |
| jax-cuda12-plugin | 0.11.2 | 2026-09-17 | must equal jax |
| flax | TBD | - | latest 0.12.12 (2026-10-27) too recent; resolve an older pin |
| optax | 0.2.8 | 2026-03-20 | |
| chex | 0.1.92 | 2026-06-12 | |
| gymnax | 1.0.0 | 2026-08-16 | 1.0.1 (10-18) too recent |
| numpy | 2.5.4 | 2026-10-10 | check jax 0.11.2 compatibility |
| torch (EGG only) | 2.14.1 | 2026-09-30 | aarch64 CUDA wheel to verify |
| EGG | commit 54eb11c6b4ba3d3171811378f7ad7a117fa93a97 | 2025-05-06 | no PyPI release; vendor/fork into `ailiance` org |

Transitive closure, SHA-256 of every wheel, and the fork/mirror step: TBD-8.
