"""Pool factory for N=16 / N=32 scaling experiments.

Builds a heterogeneous WML pool (mix of MlpWML and LifWML) with deterministic
per-WML seeds derived from a single pool_seed. No global torch RNG mutation.
"""
from __future__ import annotations

import math

from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML


def k_for_n(n_wmls: int) -> int:
    """Fan-out per WML: max(2, ceil(log2(N))).

    At N=4 → k=2. At N=16 → k=4. At N=32 → k=5. Keeps the graph
    connected with high probability while remaining sparse.
    """
    if n_wmls <= 1:
        raise ValueError(f"n_wmls must be > 1, got {n_wmls}")
    return max(2, math.ceil(math.log2(n_wmls)))


def build_pool(n_wmls: int, mlp_frac: float = 0.5, *, seed: int = 0) -> list:
    """Build a heterogeneous pool of WMLs.

    Odd ids → MlpWML, even ids → LifWML when mlp_frac == 0.5, so the two
    substrates are interleaved. For mlp_frac ≠ 0.5, the first ceil(N*frac)
    ids are MlpWML and the rest are LifWML.

    Each WML is seeded from `seed * 1000 + wml_id` so the pool is fully
    deterministic from a single seed. Local torch.Generator only — no
    global RNG mutation.
    """
    if not 0.0 <= mlp_frac <= 1.0:
        raise ValueError(f"mlp_frac must be in [0, 1], got {mlp_frac}")

    n_mlp = n_wmls // 2 if mlp_frac == 0.5 else math.ceil(n_wmls * mlp_frac)
    pool: list = []
    for i in range(n_wmls):
        wml_seed = seed * 1000 + i
        if mlp_frac == 0.5:
            is_mlp = (i % 2 == 0)
        else:
            is_mlp = i < n_mlp
        if is_mlp:
            pool.append(MlpWML(id=i, d_hidden=16, seed=wml_seed))
        else:
            pool.append(LifWML(id=i, n_neurons=16, seed=wml_seed))
    return pool
