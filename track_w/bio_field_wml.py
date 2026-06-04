"""BioFieldWML — Biological-field-inspired WML substrate.

Phase 1 (Bellitto, 2024 — Wake-Sleep Continual Learning):
    Internal wake/sleep scheduler. Sleep calls are silent (N-1).

Phase 2 (Palacios et al. 2024 — arXiv:2409.05386):
    Variational message passing (VMP) for spiking predictive coding.
    Each neuron maintains a belief (μ, σ) over expected spike counts
    from a ``BioCultureClient``.  On wake calls, a closed-form Gaussian
    VMP update fuses prior beliefs with observed spike counts and
    emits role-tagged Neuroletters:

      - low posterior σ (confident prediction)     → Role.PREDICTION / γ
      - high σ OR large residual > error_threshold → Role.ERROR      / θ

    The N-3 invariant (role == ERROR ↔ phase == THETA) is preserved
    by construction.

See docs/superpowers/plans/2026-05-19-bio-substrate-wml.md
§"Ref B-3" (Palacios SNN-PC) for the design input.

Wake/sleep convention (documented for TDD tests)
-------------------------------------------------
After ``_call_count`` is incremented at the start of ``step()``:
  - sleep  if ``_call_count % _sleep_every == 0``
  - wake   otherwise

With ``sleep_every=4`` and calls numbered 1-8:
  1→wake, 2→wake, 3→wake, 4→sleep, 5→wake, 6→wake, 7→wake, 8→sleep.

Phase scope deferral
--------------------
  Phase 3 (deferred): Tomé STDP triplet + heterosynaptic + inhibitory.
  Phase 4 (deferred): Pignatelli IE plasticity.
  Phase 5 (deferred): Tucker–Friston E/I gain on sleep calls.
"""
from __future__ import annotations

__all__ = ["BioFieldWML"]

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve


class BioFieldWML(nn.Module):
    """Wake/sleep-scheduled WML substrate with Palacios SNN-PC VMP.

    Parameters
    ----------
    id:
        Stable WML identifier.  Required by W-4.
    sleep_every:
        Number of ``step()`` calls per wake/sleep cycle.  A call is
        classified as a sleep call when ``_call_count % sleep_every == 0``
        (after increment).  Default: 4.  Must be ``>= 1``.
    alphabet_size:
        Size of the local codebook vocabulary.  Default: 64.
    d_hidden:
        Dimensionality of the local codebook embeddings.  Default: 16.
    n_neurons:
        Number of belief slots (per-neuron μ/σ).  Default: 32.  Must be
        ``>= 1``.  When a ``bio_client`` is attached, it should expose at
        least ``n_neurons`` read channels (extras are ignored, missing
        channels are zero-padded).
    error_threshold:
        Threshold on |observed − μ_prior| above which a neuron emits a
        Role.ERROR neuroletter (θ phase) regardless of posterior σ.
        Default: 1.0.  Must be ``> 0``.
    sigma_confident:
        Posterior σ below which the neuron is considered "confident" and
        emits a Role.PREDICTION neuroletter (γ phase).  Default: 0.5.
        Must be ``> 0``.
    bio_client:
        Optional ``BioCultureClient`` (track_w.bio_clients) that supplies
        observed spike counts.  When ``None``, Phase 2 falls back to the
        Phase 1 placeholder emission (single γ PREDICTION per edge) so
        existing Phase 1 tests keep passing.
    seed:
        Optional seed for the local ``torch.Generator`` used to
        initialise the codebook.  Does NOT mutate the global RNG (W-4).
    """

    def __init__(
        self,
        id:               int,
        sleep_every:      int   = 4,
        alphabet_size:    int   = 64,
        d_hidden:         int   = 16,
        n_neurons:        int   = 32,
        error_threshold:  float = 1.0,
        sigma_confident:  float = 0.5,
        *,
        bio_client:       Any | None = None,
        seed:             int | None = None,
    ) -> None:
        super().__init__()
        if sleep_every < 1:
            raise ValueError(f"sleep_every must be >= 1, got {sleep_every}")
        if n_neurons < 1:
            raise ValueError(f"n_neurons must be >= 1, got {n_neurons}")
        if error_threshold <= 0:
            raise ValueError(
                f"error_threshold must be > 0, got {error_threshold}"
            )
        if sigma_confident <= 0:
            raise ValueError(
                f"sigma_confident must be > 0, got {sigma_confident}"
            )

        self.id              = id
        self.alphabet_size   = alphabet_size
        self.n_neurons       = n_neurons
        self.error_threshold = error_threshold
        self.sigma_confident = sigma_confident
        self._sleep_every    = sleep_every
        self._call_count     = 0
        self._bio_client     = bio_client

        # Palacios SNN-PC belief state.
        # μ: per-neuron expected spike count (rate prediction).
        # σ: per-neuron rate uncertainty (standard deviation).
        # Initialised to vague prior (μ=0, σ=1) — uninformative.
        self._mu    = torch.zeros(n_neurons, dtype=torch.float32)
        self._sigma = torch.ones(n_neurons, dtype=torch.float32)

        # Local codebook — N-5: each WML owns its vocabulary.
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        saved_rng = torch.get_rng_state()
        try:
            self.codebook = nn.Parameter(
                torch.randn(alphabet_size, d_hidden, generator=gen) * 0.1
            )
        finally:
            # Restore global RNG so construction does not mutate it.
            torch.set_rng_state(saved_rng)

    # ------------------------------------------------------------------
    # Palacios SNN-PC — variational message passing
    # ------------------------------------------------------------------

    def _measure_spikes(self, t: float) -> Tensor:
        """Read observed spike counts per neuron from the bio client.

        Returns a 1-D ``Tensor`` of shape ``(n_neurons,)``.  When no
        ``bio_client`` is attached, returns the prior μ unchanged (no
        observation → no update).

        The bio client API surface is ``roundtrip(codes) -> ActivityFrame``
        with ``spikes`` shape ``[k, n_read, n_bins]`` or ``[n_read, n_bins]``.
        We collapse over time bins (sum) and, if a per-code axis is
        present, average over it.  Channels beyond ``n_neurons`` are
        dropped; missing channels are zero-padded.
        """
        if self._bio_client is None:
            return self._mu.clone()  # no observation → keep prior

        frame = self._bio_client.roundtrip([0])  # placeholder stim
        spikes = np.asarray(frame.spikes, dtype=np.float32)
        if spikes.ndim == 3:
            per_channel = spikes.sum(axis=-1).mean(axis=0)
        elif spikes.ndim == 2:
            per_channel = spikes.sum(axis=-1)
        else:
            per_channel = spikes.astype(np.float32)

        observed = torch.zeros(self.n_neurons, dtype=torch.float32)
        nc = min(self.n_neurons, int(per_channel.shape[0]))
        observed[:nc] = torch.from_numpy(
            np.asarray(per_channel[:nc], dtype=np.float32)
        )
        return observed

    def _vmp_update(
        self,
        observed: Tensor,
        t: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Closed-form Gaussian variational message-passing update.

        Approximates the Poisson-rate likelihood (Palacios 2024) with a
        Laplace-approximated Gaussian where σ²_obs = max(observed, 1) so
        the update has a closed form:

            μ_post  = (σ²_obs · μ_prior + σ²_prior · obs) /
                      (σ²_obs + σ²_prior)
            σ²_post = (σ²_prior · σ²_obs) /
                      (σ²_obs + σ²_prior)

        The prediction error is ``observed − μ_prior`` (Poisson rate
        residual, Palacios eq. 5).

        Mutates ``self._mu`` and ``self._sigma`` in-place and returns
        the new (μ, σ) plus the residual error (all detached, no grad).
        """
        del t  # currently unused; reserved for time-varying priors.

        mu_prior  = self._mu
        var_prior = self._sigma ** 2
        # Laplace approximation of Poisson likelihood variance.
        var_obs   = torch.clamp(observed, min=1.0)

        denom      = var_obs + var_prior
        mu_post    = (var_obs * mu_prior + var_prior * observed) / denom
        var_post   = (var_prior * var_obs) / denom
        sigma_post = torch.sqrt(var_post)

        error = observed - mu_prior

        self._mu    = mu_post.detach()
        self._sigma = sigma_post.detach()
        return self._mu, self._sigma, error.detach()

    # ------------------------------------------------------------------
    # WML Protocol surface
    # ------------------------------------------------------------------

    def step(self, nerve: Nerve, t: float) -> None:
        """One simulation tick.

        1. Increment ``_call_count``.
        2. Classify phase: SLEEP if ``_call_count % _sleep_every == 0``.
        3. SLEEP: silent — belief state (_mu, _sigma) unchanged (N-1).
        4. WAKE:
           a. Read observed spike counts via ``_measure_spikes(t)``.
           b. Run ``_vmp_update`` → posterior (μ, σ), residual error.
           c. For each neuron i, decide role from posterior + residual:
               - |error_i| > error_threshold        → Role.ERROR / θ
               - sigma_post_i < sigma_confident     → Role.PREDICTION / γ
               - otherwise (high uncertainty)       → Role.ERROR / θ
           d. Emit one Neuroletter per neuron per routed edge.

        N-3 (role == ERROR ↔ phase == THETA) is preserved by construction.
        W-1: communicates only via ``nerve.send()``.
        DR-3: emission shape is unchanged from Phase 1 — single-tick
        Neuroletter contract.

        Backward compatibility: when ``bio_client is None`` the routine
        falls back to the Phase 1 placeholder (single γ PREDICTION per
        edge with ``code=0``).  This keeps the Phase 1 test-suite green.
        """
        from nerve_core.neuroletter import Neuroletter, Phase, Role

        # Step 1 — advance call counter.
        self._call_count += 1

        # Step 2 — classify phase.
        is_sleep = (self._call_count % self._sleep_every == 0)

        if is_sleep:
            # SLEEP: silent — no neuroletters, no belief update (N-1).
            return

        # WAKE — Phase 1 fallback (no bio client): preserve legacy behaviour.
        if self._bio_client is None:
            for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
                if dst == self.id:
                    continue
                if nerve.routing_weight(self.id, dst) == 1.0:
                    nerve.send(Neuroletter(
                        code=0,
                        role=Role.PREDICTION,
                        phase=Phase.GAMMA,
                        src=self.id,
                        dst=dst,
                        timestamp=t,
                    ))
            return

        # WAKE — Phase 2 Palacios SNN-PC VMP path.
        observed = self._measure_spikes(t)
        mu_post, sigma_post, error = self._vmp_update(observed, t)

        abs_err   = torch.abs(error)
        confident = sigma_post < self.sigma_confident
        big_error = abs_err > self.error_threshold

        for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
            if dst == self.id:
                continue
            if nerve.routing_weight(self.id, dst) != 1.0:
                continue

            for i in range(self.n_neurons):
                if bool(big_error[i]) or not bool(confident[i]):
                    role, phase = Role.ERROR, Phase.THETA
                else:
                    role, phase = Role.PREDICTION, Phase.GAMMA
                code = int(round(float(mu_post[i]))) % self.alphabet_size
                nerve.send(Neuroletter(
                    code=code,
                    role=role,
                    phase=phase,
                    src=self.id,
                    dst=dst,
                    timestamp=t,
                ))

    def parameters(self, recurse: bool = True) -> Iterable[Tensor]:  # type: ignore[override]
        """W-2: parameters() includes codebook and all internal weights."""
        return super().parameters(recurse=recurse)
