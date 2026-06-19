"""Methodology toolkit for MI and representation alignment robustness checks.

This package provides primitives used by both nerve-wml and bouba_sens
(and any downstream consumer) to validate MI-based scientific claims:

* `null_model_mi`    -- permutation test for MI significance
                        (rejects chance-level claims).
* `bootstrap_ci_mi`  -- confidence interval via resampling
                        (quantifies MI uncertainty).
* `mi_estimators`    -- multi-estimator MI (argmax one-hot,
                        Kraskov kNN, MINE) for cross-estimator
                        robustness.
* `accuracy_edge_bootstrap_ci` -- bootstrap 95 % CI on per-seed
                        signed accuracy edges (LIF vs MLP or any
                        scalar difference list).
* `same_substrate_null_result` -- summary of an MLP-vs-MLP init-
                        variance null, providing the band against
                        which the real LIF edge is compared.

All functions are pure-python/numpy and are designed to run on
commodity CPU (light-client laptop compatible). Heavy workloads
(generating input codes from trained substrates) live under
scripts/ and should run on Tower / kxkm-ai, not locally.

See bouba_sens paper section 6.3 for the methodological rationale
behind the three checks. This module is the canonical home for the
primitives; bouba_sens imports via
``from nerve_wml.methodology import null_model_mi``.
"""
from nerve_wml.methodology.accuracy_edge_null import (
    AccuracyEdgeCiResult,
    SameSubstrateNullResult,
    accuracy_edge_bootstrap_ci,
    same_substrate_null_result,
)
from nerve_wml.methodology.bootstrap_ci_mi import (
    BootstrapCiResult,
    bootstrap_ci_mi,
)
from nerve_wml.methodology.mi_estimators import (
    entropy_discrete,
    mi_kraskov_ksg_continuous,
    mi_miller_madow_discrete,
    mi_plugin_discrete,
)
from nerve_wml.methodology.mi_mine_estimator import mi_mine
from nerve_wml.methodology.mi_null_model import (
    NullModelResult,
    mi_argmax_onehot,
    null_model_mi,
)

__all__ = [
    "AccuracyEdgeCiResult",
    "BootstrapCiResult",
    "NullModelResult",
    "SameSubstrateNullResult",
    "accuracy_edge_bootstrap_ci",
    "bootstrap_ci_mi",
    "entropy_discrete",
    "mi_argmax_onehot",
    "mi_kraskov_ksg_continuous",
    "mi_miller_madow_discrete",
    "mi_mine",
    "mi_plugin_discrete",
    "null_model_mi",
    "same_substrate_null_result",
]
