"""MNISTTask — real-data validation for Claims A + B.

Flattens 28×28 MNIST into 784-dim vectors in [0, 1]. Seed-stable
sampling via a local torch.Generator. Cache location governed by
the MNIST_ROOT env var (default ~/.cache/nerve-wml/mnist).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


class MNISTTask:
    """Flattened-MNIST task. sample(batch) returns (x [B, 784], y [B])."""

    def __init__(
        self,
        batch_classes: int = 10,
        *,
        seed: int = 0,
        train: bool = True,
    ) -> None:
        if batch_classes != 10:
            raise ValueError("MNISTTask currently uses the full 10 classes")
        from torchvision import datasets, transforms  # lazy import

        root = os.environ.get(
            "MNIST_ROOT",
            str(Path.home() / ".cache" / "nerve-wml" / "mnist"),
        )
        Path(root).mkdir(parents=True, exist_ok=True)
        tf = transforms.Compose([transforms.ToTensor()])
        self._dataset = datasets.MNIST(
            root=root, train=train, download=True, transform=tf,
        )
        self.n_classes = 10
        self.dim = 784
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self._dataset)
        idx = torch.randint(0, n, (batch,), generator=self.generator)
        xs, ys = [], []
        for i in idx.tolist():
            x, y = self._dataset[i]
            xs.append(x.view(-1))
            ys.append(y)
        x_batch = torch.stack(xs, dim=0)
        y_batch = torch.tensor(ys, dtype=torch.long)
        return x_batch, y_batch
