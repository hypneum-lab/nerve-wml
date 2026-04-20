"""MNIST real-data validation of Claims A and B (v1.2 track 1)."""
import os
import pytest

pytest.importorskip("torchvision")

if os.environ.get("NERVE_WML_SKIP_MNIST") == "1":
    pytest.skip("MNIST tests skipped via env flag", allow_module_level=True)

from scripts.run_mnist_pilots import (  # noqa: E402
    run_mnist_mi,
    run_mnist_polymorphism,
)


@pytest.mark.slow
def test_mnist_polymorphism_gap_bounded():
    """MNIST scaling law: median gap < 8 % at N=16 (a looser bound
    than the synthetic 5 %, appropriate because MNIST has real
    structure that amplifies substrate differences)."""
    r = run_mnist_polymorphism(n_wmls=16, steps=300, seeds=[0, 1, 2])
    assert r["median_gap"] < 0.08, (
        f"MNIST median gap {r['median_gap']:.3f} > 8 %"
    )
    assert r["mean_acc_mlp"] > 0.60
    assert r["mean_acc_lif"] > 0.55


@pytest.mark.slow
def test_mnist_mi_shared_code():
    """Claim B on MNIST: MI/H > 0.50 across 3 seeds."""
    results = run_mnist_mi(seeds=[0, 1, 2], steps=300, batch=1024)
    ratios = [r["mi_over_h"] for r in results]
    assert sum(ratios) / len(ratios) > 0.50
