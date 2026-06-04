"""HardSplitTask — two sequential sub-tasks over a shared 12-class head.

Both sub-tasks use HardFlowProxyTask with n_classes=12 but distinct seeds,
so the class centroids and XOR-gating hyperplanes differ. A vanilla
shared-head learner trained Task0 → Task1 with no mitigation forgets ≥ 50 %
of Task0 accuracy (verified by test_w4_hard_split_baseline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .hard_flow_proxy import HardFlowProxyTask


@dataclass
class HardSplitTask:
    """Sequential pair of HardFlowProxyTask over a shared 12-class label space.

    Attributes
    ----------
    seed : int
        Base seed. subtasks[0] uses seed, subtasks[1] uses seed + 1.
    dim : int
        Input feature dimension (matches MlpWML default d_hidden=16).
    n_classes : int
        Number of shared output classes (12; same head for both sub-tasks).
    subtasks : list[HardFlowProxyTask]
        [subtasks[0], subtasks[1]] — train sequentially.
    """

    seed: int = 0
    dim: int = 16
    n_classes: int = 12
    subtasks: list = field(init=False)

    def __post_init__(self) -> None:
        self.subtasks = [
            HardFlowProxyTask(
                dim=self.dim, n_classes=self.n_classes, seed=self.seed
            ),
            HardFlowProxyTask(
                dim=self.dim, n_classes=self.n_classes, seed=self.seed + 1
            ),
        ]
