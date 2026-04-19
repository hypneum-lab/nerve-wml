
import torch

from interpret.clustering import cluster_codes_by_activation


def test_clustering_returns_one_label_per_code():
    centroids = torch.randn(64, 32)
    labels = cluster_codes_by_activation(centroids, n_clusters=8, seed=0)
    assert labels.shape == (64,)
    assert labels.dtype == torch.long
    assert labels.min().item() >= 0
    assert labels.max().item() < 8


def test_clustering_is_deterministic():
    centroids = torch.randn(64, 32)
    a = cluster_codes_by_activation(centroids, n_clusters=8, seed=0)
    b = cluster_codes_by_activation(centroids, n_clusters=8, seed=0)
    assert torch.equal(a, b)


def test_clustering_entropy_non_degenerate():
    """On well-separated data, assignment entropy > 2 bits (≥ 4 effective clusters)."""
    import torch

    # 8 Gaussian blobs.
    torch.manual_seed(0)
    gen = torch.Generator()
    gen.manual_seed(0)
    centres = torch.randn(8, 32, generator=gen) * 5
    cluster_ids = torch.randint(0, 8, (64,), generator=gen)
    centroids = centres[cluster_ids] + torch.randn(64, 32, generator=gen) * 0.1

    labels = cluster_codes_by_activation(centroids, n_clusters=8, seed=0)
    counts = torch.bincount(labels, minlength=8).float()
    p = counts / counts.sum()
    entropy_bits = -(p * (p + 1e-12).log2()).sum().item()
    assert entropy_bits > 2.0, f"entropy {entropy_bits:.2f} bits < 2"


def test_clustering_seed_is_local():
    """Calling the clusterer must NOT mutate the global torch RNG."""
    centroids = torch.randn(64, 32)
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = cluster_codes_by_activation(centroids, n_clusters=8, seed=99)
    observed = torch.rand(1).item()
    assert expected == observed
