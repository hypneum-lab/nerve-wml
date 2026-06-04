"""RehearsalBuffer — replay-based continual learning helper.

Encapsulates the batch-mixing logic from run_w4_rehearsal so that
run_w4_compare can use the same scaffold for all three methods
(none / rehearsal / ewc).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@dataclass
class RehearsalBuffer:
    """Fixed-replay buffer: mixes old-task samples into each Task-1 step.

    Parameters
    ----------
    rehearsal_frac : float
        Fraction of each mini-batch filled with Task-0 samples (default 0.3,
        matching run_w4_rehearsal).
    total_batch : int
        Total mini-batch size before the mix (default 64).

    Usage
    -----
    buf = RehearsalBuffer()
    buf.store(task0)          # capture the Task-0 sampler
    for _ in range(steps):
        loss = buf.mixed_loss(wml, task1, n_classes=12)
        opt.zero_grad(); loss.backward(); opt.step()
    """

    rehearsal_frac: float = 0.3
    total_batch: int = 64
    _task0: object | None = field(default=None, init=False, repr=False)

    def store(self, task0: object) -> None:
        """Register the Task-0 sampler for replay."""
        self._task0 = task0

    def mixed_loss(
        self,
        wml: object,
        task1: object,
        n_classes: int,
    ) -> Tensor:
        """Return weighted cross-entropy over a mixed Task0/Task1 mini-batch.

        Weights proportional to batch sizes (mirrors run_w4_rehearsal exactly):
            loss = (loss_new * n_new + loss_old * n_old) / total_batch
        """
        n_old = int(self.total_batch * self.rehearsal_frac)
        n_new = self.total_batch - n_old

        def _loss(task, n):
            x, y = task.sample(batch=n)
            logits = wml.emit_head_pi(wml.core(x))[:, :n_classes]
            return F.cross_entropy(logits, y)

        loss_new = _loss(task1, n_new)
        loss_old = _loss(self._task0, n_old)
        return (loss_new * n_new + loss_old * n_old) / self.total_batch
