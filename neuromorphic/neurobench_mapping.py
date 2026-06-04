"""NeuroBench mapping — make nerve-wml results externally comparable.

NeuroBench (Yik et al., "NeuroBench: a framework for benchmarking
neuromorphic computing algorithms and systems", Nature
Communications, 2025) is an MLPerf-style, open, community harness
for neuromorphic algorithms. It scores a model on a task with both
a *correctness* metric (e.g. accuracy) and *complexity* metrics
(synaptic operations, connection sparsity, parameter footprint).

nerve-wml runs cross-substrate validations whose primary outcome is
a streaming-classification accuracy over a tick sequence. This
module maps one such ValidationOutcome onto NeuroBench's
"streaming classification" task so a nerve-wml substrate
(MlpWML / LifWML / TransformerWML / BioWML) can be quoted on the
same axes as published NeuroBench entries.

Deliberately minimal: one task mapping. See scripts/neurobench_map.py
for the CLI that emits a NeuroBench-shaped row.
"""
from __future__ import annotations

from dataclasses import dataclass

# The single NeuroBench task this module targets.
NEUROBENCH_TASK = "streaming_classification"
# Identifies which harness produced the row, for provenance.
HARNESS_TAG = "neurobench-v1 (Yik et al., Nat. Commun. 2025)"


@dataclass(frozen=True)
class ValidationOutcome:
    """A nerve-wml cross-substrate validation result.

    substrate:    "MlpWML" | "LifWML" | "TransformerWML" | "BioWML".
    n_correct:    correctly classified streaming items.
    n_total:      total streaming items (> 0).
    total_synops: synaptic operations consumed over the run.
    param_count:  trainable parameter footprint of the substrate.
    latency_ms:   mean per-tick latency (per-roundtrip for BioWML).
    """

    substrate: str
    n_correct: int
    n_total: int
    total_synops: int
    param_count: int
    latency_ms: float


@dataclass(frozen=True)
class NeuroBenchResult:
    """A NeuroBench-shaped result row for one substrate."""

    task: str
    substrate: str
    accuracy: float
    synaptic_ops: int
    connection_sparsity: float | None
    footprint_params: int
    latency_ms: float
    harness: str

    def as_row(self) -> dict:
        """Flat dict suitable for CSV / JSON / a NeuroBench table."""
        return {
            "task": self.task,
            "substrate": self.substrate,
            "accuracy": self.accuracy,
            "synaptic_ops": self.synaptic_ops,
            "connection_sparsity": self.connection_sparsity,
            "footprint_params": self.footprint_params,
            "latency_ms": self.latency_ms,
            "harness": self.harness,
        }


def map_to_neurobench(outcome: ValidationOutcome) -> NeuroBenchResult:
    """Map a nerve-wml ValidationOutcome to a NeuroBench result.

    Raises ValueError if n_total <= 0.
    """
    if outcome.n_total <= 0:
        raise ValueError("n_total must be > 0 to compute accuracy")
    accuracy = outcome.n_correct / outcome.n_total
    # NeuroBench connection-sparsity convention: fraction of the
    # dense connectivity that carried no synaptic op. We have no
    # per-edge trace here, so derive a conservative proxy from the
    # synop / param ratio, clamped to [0, 1]. A substrate that uses
    # far fewer synops than it has params is "sparse".
    if outcome.param_count > 0:
        density = outcome.total_synops / outcome.param_count
        sparsity = max(0.0, min(1.0, 1.0 - density))
    else:
        sparsity = None
    return NeuroBenchResult(
        task=NEUROBENCH_TASK,
        substrate=outcome.substrate,
        accuracy=accuracy,
        synaptic_ops=outcome.total_synops,
        connection_sparsity=sparsity,
        footprint_params=outcome.param_count,
        latency_ms=outcome.latency_ms,
        harness=HARNESS_TAG,
    )
