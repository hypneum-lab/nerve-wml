import torch

from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.pool_factory import build_pool, k_for_n


def test_k_for_n_formula():
    assert k_for_n(2) == 2  # floor at 2
    assert k_for_n(4) == 2  # log2(4)=2
    assert k_for_n(8) == 3  # log2(8)=3
    assert k_for_n(16) == 4 # log2(16)=4
    assert k_for_n(32) == 5 # log2(32)=5


def test_build_pool_returns_correct_count_and_mix():
    """Half MLP / half LIF at mlp_frac=0.5 — interleaved by parity."""
    pool = build_pool(n_wmls=8, mlp_frac=0.5, seed=0)
    assert len(pool) == 8
    types = [type(w).__name__ for w in pool]
    # Deterministic interleaving: id 0 MLP, 1 LIF, 2 MLP, 3 LIF, ...
    assert types[0] == "MlpWML"
    assert types[1] == "LifWML"
    assert types[2] == "MlpWML"
    # Exactly half of each.
    assert sum(1 for w in pool if isinstance(w, MlpWML)) == 4
    assert sum(1 for w in pool if isinstance(w, LifWML)) == 4


def test_build_pool_ids_are_stable():
    pool = build_pool(n_wmls=8, mlp_frac=0.5, seed=0)
    assert [w.id for w in pool] == list(range(8))


def test_build_pool_seeds_are_derived_deterministically():
    """Same pool_seed → same per-WML params across two calls."""
    p1 = build_pool(n_wmls=4, mlp_frac=0.5, seed=42)
    p2 = build_pool(n_wmls=4, mlp_frac=0.5, seed=42)
    for a, b in zip(p1, p2, strict=True):
        # Compare first parameter tensor element-wise.
        pa = next(iter(a.parameters()))
        pb = next(iter(b.parameters()))
        assert torch.equal(pa, pb)


def test_build_pool_global_rng_is_untouched():
    """Building a pool must not pollute the global torch RNG."""
    torch.manual_seed(7)
    expected = torch.rand(1).item()

    torch.manual_seed(7)
    _ = build_pool(n_wmls=8, mlp_frac=0.5, seed=99)
    observed = torch.rand(1).item()

    assert expected == observed
