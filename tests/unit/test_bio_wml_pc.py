"""Unit tests for BioFieldWML Phase 2 — Palacios SNN-PC VMP.

Reference: Palacios et al. 2024 (arXiv:2409.05386), variational message
passing in spiking predictive-coding networks.  See
docs/superpowers/plans/2026-05-19-bio-substrate-wml.md §"Ref B-3".

Invariants verified
-------------------
  N-1: sleep calls remain silent AND leave belief state untouched.
  N-3: role == ERROR ↔ phase == THETA  /  role == PREDICTION ↔ γ.
       Every emitted Neuroletter carries one of these two pairs —
       no other role is ever emitted by BioFieldWML.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from nerve_core.neuroletter import Phase, Role
from track_w.bio_field_wml import BioFieldWML
from track_w.mock_nerve import MockNerve

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Frame:
    spikes: np.ndarray
    latency_ms: float = 0.0


class _ConstantBioClient:
    """Bio client that always returns the same per-channel spike vector.

    The vector is supplied as a numpy array of length ``n_read_channels``;
    each call to ``roundtrip`` returns shape ``(1, n_read_channels, 1)``
    so the BioFieldWML ``_measure_spikes`` sum-over-bins reduction yields
    exactly the supplied vector.
    """

    def __init__(self, per_channel: np.ndarray) -> None:
        self._per_channel = per_channel.astype(np.float32)
        self.n_stim_channels = 8
        self.n_read_channels = int(per_channel.shape[0])
        self.n_bins = 1

    def encode_stimulus(self, codes):  # pragma: no cover - unused
        return None

    def decode_activity(self, frame):  # pragma: no cover - unused
        return [0]

    def roundtrip(self, codes):
        spikes = self._per_channel.reshape(1, -1, 1).copy()
        return _Frame(spikes=spikes)

    def close(self) -> None:
        return None


def _make_nerve(n_wmls: int = 2) -> MockNerve:
    nerve = MockNerve(n_wmls=n_wmls, k=n_wmls - 1, seed=0)
    return nerve


def _drain_all(nerve: MockNerve, wml_id: int) -> list:
    """Drain γ-pending then θ-pending neuroletters.

    MockNerve enforces γ-priority: θ letters are only delivered when γ
    is inactive.  To inspect both populations in tests we listen twice,
    flipping the gate between calls.
    """
    nerve.set_phase_active(gamma=True, theta=False)
    gamma = nerve.listen(wml_id=wml_id)
    nerve.set_phase_active(gamma=False, theta=True)
    theta = nerve.listen(wml_id=wml_id)
    # Restore default (γ on, θ off).
    nerve.set_phase_active(gamma=True, theta=False)
    return gamma + theta


# ---------------------------------------------------------------------------
# Test 1 — N-3 role partition (PREDICTION ∪ ERROR, nothing else)
# ---------------------------------------------------------------------------


def test_neuroletter_role_partition() -> None:
    """Every emitted neuroletter carries Role.PREDICTION or Role.ERROR.

    Drives a deterministic observation that produces a mix of low-error
    (γ/PREDICTION) and high-error (θ/ERROR) neurons, and asserts both
    populations appear AND no other role/phase pair is emitted.
    """
    n_neurons = 8
    # Half observations large (will trigger ERROR), half ~0 (PREDICTION).
    per_channel = np.zeros(n_neurons, dtype=np.float32)
    per_channel[:4] = 10.0  # huge residual vs prior μ=0 → ERROR
    # Tails stay 0.0 → residual 0, posterior σ collapses → PREDICTION.

    nerve = _make_nerve(n_wmls=2)
    wml = BioFieldWML(
        id=0,
        sleep_every=10,
        n_neurons=n_neurons,
        error_threshold=1.0,
        sigma_confident=2.0,  # generous so tail neurons get PREDICTION
        bio_client=_ConstantBioClient(per_channel),
        seed=0,
    )

    # First wake call: tails have μ=0, σ=1 → σ_post ≈ 0.707 < 2.0 → γ.
    wml.step(nerve, t=0.0)
    delivered = _drain_all(nerve, wml_id=1)
    assert len(delivered) == n_neurons, (
        f"Expected {n_neurons} neuroletters (one per neuron), "
        f"got {len(delivered)}."
    )

    roles = {nl.role for nl in delivered}
    assert roles == {Role.PREDICTION, Role.ERROR}, (
        f"Expected both PREDICTION and ERROR populations, got {roles}."
    )

    # No role outside the partition (N-3).
    for nl in delivered:
        assert nl.role in (Role.PREDICTION, Role.ERROR)
        if nl.role is Role.ERROR:
            assert nl.phase is Phase.THETA
        else:
            assert nl.phase is Phase.GAMMA


# ---------------------------------------------------------------------------
# Test 2 — repeated stable observations shrink posterior σ
# ---------------------------------------------------------------------------


def test_belief_update_convergence() -> None:
    """Posterior σ monotonically decreases under repeated stable obs."""
    n_neurons = 4
    per_channel = np.full(n_neurons, 2.0, dtype=np.float32)

    nerve = _make_nerve(n_wmls=2)
    wml = BioFieldWML(
        id=0,
        sleep_every=100,  # no sleep within the run
        n_neurons=n_neurons,
        bio_client=_ConstantBioClient(per_channel),
        seed=0,
    )

    sigma_trace: list[torch.Tensor] = []
    for tick in range(5):
        wml.step(nerve, t=float(tick))
        sigma_trace.append(wml._sigma.clone())
        _ = _drain_all(nerve, wml_id=1)  # drain

    # Strict monotone decrease (closed-form precision sums).
    for k in range(1, len(sigma_trace)):
        assert torch.all(sigma_trace[k] < sigma_trace[k - 1]), (
            f"σ did not decrease at step {k}: "
            f"prev={sigma_trace[k-1]}, curr={sigma_trace[k]}"
        )
    # And final σ must be strictly below the prior σ=1.
    assert torch.all(sigma_trace[-1] < 1.0)


# ---------------------------------------------------------------------------
# Test 3 — large residual triggers Role.ERROR (θ)
# ---------------------------------------------------------------------------


def test_error_threshold_triggering() -> None:
    """An observation far above prior μ → all neuroletters are ERROR/θ."""
    n_neurons = 4
    per_channel = np.full(n_neurons, 50.0, dtype=np.float32)  # huge

    nerve = _make_nerve(n_wmls=2)
    wml = BioFieldWML(
        id=0,
        sleep_every=10,
        n_neurons=n_neurons,
        error_threshold=1.0,
        sigma_confident=0.5,
        bio_client=_ConstantBioClient(per_channel),
        seed=0,
    )

    wml.step(nerve, t=0.0)
    delivered = _drain_all(nerve, wml_id=1)
    assert len(delivered) == n_neurons
    for nl in delivered:
        assert nl.role is Role.ERROR, (
            f"large residual must yield Role.ERROR, got {nl.role}"
        )
        assert nl.phase is Phase.THETA


# ---------------------------------------------------------------------------
# Test 4 — sleep call leaves belief state unchanged (N-1 strengthened)
# ---------------------------------------------------------------------------


def test_sleep_suppresses_belief_update() -> None:
    """Sleep calls neither emit nor mutate (μ, σ)."""
    n_neurons = 4
    per_channel = np.full(n_neurons, 5.0, dtype=np.float32)

    nerve = _make_nerve(n_wmls=2)
    wml = BioFieldWML(
        id=0,
        sleep_every=2,  # call #2 will be sleep
        n_neurons=n_neurons,
        bio_client=_ConstantBioClient(per_channel),
        seed=0,
    )

    # Call 1 — wake: μ, σ update.
    wml.step(nerve, t=0.0)
    _ = _drain_all(nerve, wml_id=1)
    mu_after_wake    = wml._mu.clone()
    sigma_after_wake = wml._sigma.clone()
    assert not torch.equal(mu_after_wake, torch.zeros(n_neurons)), (
        "wake call should have updated μ from the zero prior."
    )

    # Call 2 — sleep: must be silent AND state unchanged.
    wml.step(nerve, t=1.0)
    received = _drain_all(nerve, wml_id=1)
    assert received == [], "sleep call must emit nothing (N-1)."
    assert torch.equal(wml._mu, mu_after_wake), (
        "sleep call must not mutate μ."
    )
    assert torch.equal(wml._sigma, sigma_after_wake), (
        "sleep call must not mutate σ."
    )


# ---------------------------------------------------------------------------
# Test 5 — N-3 conformance: no neuroletter has a role outside the partition
# ---------------------------------------------------------------------------


def test_conformance_n3_invariant() -> None:
    """Over many wake steps and varied stimuli, every emission obeys N-3."""
    n_neurons = 6
    rng = np.random.default_rng(0)
    nerve = _make_nerve(n_wmls=2)

    # Vary the stimulus across steps to exercise both regimes.
    stimuli = [
        np.zeros(n_neurons, dtype=np.float32),
        np.full(n_neurons, 0.1, dtype=np.float32),
        np.full(n_neurons, 8.0, dtype=np.float32),
        rng.uniform(0.0, 5.0, size=n_neurons).astype(np.float32),
    ]

    for s in stimuli:
        wml = BioFieldWML(
            id=0,
            sleep_every=100,
            n_neurons=n_neurons,
            error_threshold=1.0,
            sigma_confident=0.5,
            bio_client=_ConstantBioClient(s),
            seed=0,
        )
        for tick in range(3):
            wml.step(nerve, t=float(tick))
            delivered = _drain_all(nerve, wml_id=1)
            for nl in delivered:
                assert nl.role in (Role.PREDICTION, Role.ERROR), (
                    f"N-3 violated: role={nl.role} not in PREDICTION/ERROR."
                )
                if nl.role is Role.ERROR:
                    assert nl.phase is Phase.THETA, (
                        f"N-3 violated: ERROR with phase={nl.phase}."
                    )
                else:
                    assert nl.phase is Phase.GAMMA, (
                        f"N-3 violated: PREDICTION with phase={nl.phase}."
                    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_n_neurons_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_neurons must be >= 1"):
        BioFieldWML(id=0, n_neurons=0, seed=0)


def test_error_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="error_threshold must be > 0"):
        BioFieldWML(id=0, error_threshold=0.0, seed=0)
