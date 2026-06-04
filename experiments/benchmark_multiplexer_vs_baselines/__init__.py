"""Q1 — GTM benchmark : 4 architectures × HardFlowProxyTask N=2 × 5 seeds.

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md (Task 12).
Pre-registration: docs/milestones/q1-multiplexer-benchmark-2026-05-10.md.

This package houses :
- architectures/  — 4 BridgeArchitecture wrappers
- runner.py       — 5-seed sweep harness (Task 13)
- analyse.py      — Welch's t-test + Bonferroni analysis (Task 15)
"""
from collections.abc import Iterator
from typing import Protocol

import torch.nn as nn
from torch import Tensor


class BridgeArchitecture(Protocol):
    """Common interface for the 4 sender→receiver bridge architectures.

    encode  : [B, dim] -> [B, code_dim]
    decode  : [B, code_dim] -> [B, dim]
    """

    def encode(self, x: Tensor) -> Tensor: ...

    def decode(self, code: Tensor) -> Tensor: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...


__all__ = ["BridgeArchitecture"]
