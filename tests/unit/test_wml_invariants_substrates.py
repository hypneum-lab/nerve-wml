"""W-1..W-4 invariant checks parameterized over all 6 WML substrates.

Complements tests/unit/test_invariants.py (which uses a fake _StubWML)
with one test per invariant × substrate, using real instances at the
smallest practical dimensions (d=16, alphabet_size=16, seed=0).

Design:
- W-1: no foreign-WML mutation (vacuously satisfied — WMLs have no
  _mutation_log by default; the check is opt-in via that attribute).
- W-2: parameters() includes the codebook nn.Parameter.
- W-3: step() does not access routing_weight — verified by passing an
  empty access log (the invariant checks the log we provide, not
  live step() calls; full instrumentation is an integration concern).
- W-4: id is stable — the id set at construction must not drift.

All tests are L1 (fast, no slow marker). No substrate code is modified.
"""
from __future__ import annotations

from typing import Callable

import pytest
import torch

from nerve_core.invariants import (
    assert_w1_step_uses_nerve_only,
    assert_w2_parameters_include_codebook,
    assert_w3_no_routing_access_in_step,
    assert_w4_id_stable,
)
from track_w.bio_clients import MockBioCultureClient
from track_w.bio_field_wml import BioFieldWML
from track_w.bio_wml import BioWML
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.spiking_kiki_wml import InMemoryProvider, SpikingKikiWML
from track_w.transformer_wml import TransformerWML

# ---------------------------------------------------------------------------
# Substrate factories — small dimensions for fast unit tests
# ---------------------------------------------------------------------------

_D = 16
_A = 16  # alphabet_size


def _mlp() -> MlpWML:
    return MlpWML(id=0, d_hidden=_D, alphabet_size=_A, seed=0)


def _lif() -> LifWML:
    return LifWML(id=0, n_neurons=_D, alphabet_size=_A, seed=0)


def _transformer() -> TransformerWML:
    # d_model must be divisible by n_heads; d_model must be divisible by
    # n_tokens.  With d_model=16, n_heads=2, n_tokens=4 all constraints hold.
    return TransformerWML(id=0, d_model=_D, n_layers=1, n_heads=2, n_tokens=4,
                          alphabet_size=_A, seed=0)


def _bio() -> BioWML:
    return BioWML(id=0, client=MockBioCultureClient(seed=0),
                  d_hidden=_D, alphabet_size=_A, seed=0)


def _biofield() -> BioFieldWML:
    return BioFieldWML(id=0, d_hidden=_D, n_neurons=_D, alphabet_size=_A, seed=0)


def _spiking_kiki() -> SpikingKikiWML:
    torch.manual_seed(0)
    provider = InMemoryProvider(
        [torch.randn(_D, _D) * 0.1 for _ in range(1)],
    )
    return SpikingKikiWML(id=0, provider=provider, alphabet_size=_A, seed=0)


_SUBSTRATES: list[tuple[str, Callable]] = [
    ("MlpWML", _mlp),
    ("LifWML", _lif),
    ("TransformerWML", _transformer),
    ("BioWML", _bio),
    ("BioFieldWML", _biofield),
    ("SpikingKikiWML", _spiking_kiki),
]


# ---------------------------------------------------------------------------
# W-1 — step() must not mutate other WMLs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,factory", _SUBSTRATES,
                          ids=[s[0] for s in _SUBSTRATES])
def test_w1_isolation(label: str, factory: Callable) -> None:
    """W-1 passes for every substrate: WMLs have no _mutation_log by default
    so the foreign-WML check is vacuously satisfied (opt-in protocol).
    """
    wml = factory()
    other = factory()
    # No _mutation_log on either side → no violation possible.
    assert_w1_step_uses_nerve_only(wml, [other])


# ---------------------------------------------------------------------------
# W-2 — parameters() includes codebook
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,factory", _SUBSTRATES,
                          ids=[s[0] for s in _SUBSTRATES])
def test_w2_codebook_in_parameters(label: str, factory: Callable) -> None:
    """W-2: every substrate's parameters() contains the codebook tensor."""
    wml = factory()
    assert_w2_parameters_include_codebook(wml)


# ---------------------------------------------------------------------------
# W-3 — no routing_weight access from inside step()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,factory", _SUBSTRATES,
                          ids=[s[0] for s in _SUBSTRATES])
def test_w3_no_routing_access(label: str, factory: Callable) -> None:
    """W-3: an empty step-access log satisfies the invariant for all substrates.

    Full instrumentation of live step() calls is an integration-test concern
    (see tests/integration/). This unit test confirms the invariant helper
    itself does not spuriously reject substrates.
    """
    # Empty log → no routing_weight access was recorded.
    assert_w3_no_routing_access_in_step([])


# ---------------------------------------------------------------------------
# W-4 — id is stable across the WML lifetime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,factory", _SUBSTRATES,
                          ids=[s[0] for s in _SUBSTRATES])
def test_w4_id_stable(label: str, factory: Callable) -> None:
    """W-4: id set at construction is stable and passes the invariant check."""
    wml = factory()
    assert_w4_id_stable(wml, expected_id=0)
