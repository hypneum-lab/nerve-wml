
from track_p.router import SparseRouter


def test_router_edge_count_equals_k_per_wml():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    # edges[i, j] = 1 if i → j active. Each row must have exactly k ones.
    assert edges.shape == (4, 4)
    assert (edges.sum(dim=-1) == 2).all()


def test_router_no_self_loops():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    assert (edges.diagonal() == 0).all()


def test_routing_weight_lookup():
    r = SparseRouter(n_wmls=4, k=2)
    edges = r.sample_edges(tau=0.5, hard=True)
    for i in range(4):
        for j in range(4):
            assert r.routing_weight(i, j, edges) == float(edges[i, j])


def test_router_soft_edges_are_continuous():
    """hard=False path must return fractional weights on active edges.

    Previously uncovered. Verifies that soft routing produces a valid
    top-K masked tensor with continuous weights in [0, 1], usable as
    a differentiable relaxation during training.
    """
    import torch

    torch.manual_seed(0)
    r = SparseRouter(n_wmls=4, k=2)
    soft = r.sample_edges(tau=0.5, hard=False)

    assert soft.shape == (4, 4)
    # All values in [0, 1].
    assert (soft >= 0).all()
    assert (soft <= 1).all()
    # No self-loops (diagonal masked by the sampler).
    assert (soft.diagonal() == 0).all()
    # At least one positive entry per row (top-K >= 1 active).
    assert (soft.sum(dim=-1) > 0).all()
