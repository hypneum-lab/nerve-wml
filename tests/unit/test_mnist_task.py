import os
import pytest
import torch

pytest.importorskip("torchvision")
from track_w.tasks.mnist import MNISTTask


@pytest.fixture(autouse=True)
def _mnist_cache(tmp_path, monkeypatch):
    """Redirect torchvision's MNIST cache to a per-test tmp path on first run,
    or point at user's existing cache via MNIST_ROOT env."""
    root = os.environ.get("MNIST_ROOT", str(tmp_path / "mnist"))
    monkeypatch.setenv("MNIST_ROOT", root)


def test_mnist_task_sample_is_seed_stable():
    task_a = MNISTTask(batch_classes=10, seed=42)
    task_b = MNISTTask(batch_classes=10, seed=42)
    xa, ya = task_a.sample(batch=32)
    xb, yb = task_b.sample(batch=32)
    assert torch.equal(xa, xb)
    assert torch.equal(ya, yb)


def test_mnist_task_shape_and_range():
    task = MNISTTask(batch_classes=10, seed=0)
    x, y = task.sample(batch=16)
    assert x.shape == (16, 784)
    assert x.min() >= 0.0
    assert x.max() <= 1.0
    assert y.shape == (16,)
    assert y.min() >= 0 and y.max() <= 9
    assert y.dtype == torch.long
