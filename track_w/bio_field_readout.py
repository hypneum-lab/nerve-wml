"""BioFieldWMLReadout — task-evaluation adapter for the BioFieldWML substrate.

Why this exists
---------------
``track_w.bio_field_wml.BioFieldWML`` implements the Lee et al. 2024 SNN-PC
variational-message-passing (VMP) belief dynamics and exposes only
``step(nerve, t)`` — it emits role-tagged Neuroletters but has **no**
``core`` / ``emit_head_pi`` surface, so it cannot slot into the supervised
``run_w2_hard`` / ``train_wml_on_task`` harness the way ``MlpWML`` and
``LifWML`` do.  As a result the substrate had emission-scheduling unit
tests but no task/accuracy result.

This adapter adds a thin, **additive** readout so BioFieldWML can be
measured on ``HardFlowProxyTask`` exactly like MLP/LIF, WITHOUT touching
the substrate's internal dynamics:

  - ``core(x)`` maps a batch ``[B, input_dim]`` through the SNN-PC VMP
    belief update and exposes the resulting per-neuron belief state as a
    fixed-width feature vector — the **belief-state probe**.  The probe is
    the concatenation ``[mu_post, sigma_post, |error|]`` over the
    ``n_neurons`` belief slots, giving feature width ``3 * n_neurons``.
  - ``emit_head_pi`` is a trainable ``nn.Linear(3*n_neurons, alphabet_size)``
    readout over that feature — the ONLY trained parameters, symmetric to
    how ``MlpWML.emit_head_pi`` reads the MLP ``core`` output and
    ``LifWML.emit_head_pi`` reads the spike pattern.
  - ``codebook`` is exposed (a ``3*n_neurons``-wide local codebook) so the
    VQ commitment loss in ``train_wml_on_task`` has a target, matching the
    MLP path.

Design decision: probe the belief STATE
---------------------------------------
We probe the substrate's *internal representation* (the posterior belief
``(mu, sigma)`` and the prediction-error residual ``|obs - mu_prior|``),
mirroring how MLP/LIF expose a learned head over their ``core`` output.
The belief dynamics are the closed-form Gaussian VMP of
``BioFieldWML._vmp_update`` (Laplace-approximated Poisson likelihood),
reproduced here as a pure, batched, input-differentiable function so the
*input projection* and the *readout head* train end-to-end while the VMP
update rule itself is fixed (no learned parameters inside the belief
update).  The belief is reset to the vague prior (mu=0, sigma=1) for every
input row, so ``core(x)`` is a deterministic function of ``x`` (no hidden
cross-batch state) — required for a clean supervised probe.

This is intentionally a *probe*, not a re-implementation: if the SNN-PC
belief features carry no task signal, the readout will sit at chance, and
that is a valid (null) outcome to report.
"""
from __future__ import annotations

__all__ = ["BioFieldWMLReadout"]

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from track_w.bio_field_wml import BioFieldWML


class BioFieldWMLReadout(nn.Module):
    """Belief-state readout adapter over a :class:`BioFieldWML` substrate.

    Exposes the ``core`` / ``codebook`` / ``emit_head_pi`` surface the
    Track-W supervised harness expects, by probing the per-neuron SNN-PC
    belief posterior produced by the wrapped substrate's VMP dynamics.

    Parameters
    ----------
    id:
        Stable WML identifier.
    n_neurons:
        Number of belief slots in the wrapped substrate.  Feature width
        is ``3 * n_neurons`` (mu, sigma, |error| per neuron).
    alphabet_size:
        Readout head output width (codebook vocabulary).  Default 64.
    input_dim:
        Task input dimensionality.  A trainable ``nn.Linear(input_dim,
        n_neurons)`` projects the input to a per-neuron "observed spike
        count" that drives the VMP update.  Default 16.
    error_threshold / sigma_confident / sleep_every:
        Forwarded to the wrapped :class:`BioFieldWML` for parity with the
        emission path; they do not affect the belief posterior used by the
        probe (only role tagging in ``step``).
    seed:
        Local-generator seed for the codebook / projection / head init.
        Does NOT mutate the global RNG (W-4).
    """

    feature_width: int

    def __init__(
        self,
        id: int,
        n_neurons: int = 16,
        alphabet_size: int = 64,
        input_dim: int = 16,
        *,
        error_threshold: float = 1.0,
        sigma_confident: float = 0.5,
        sleep_every: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.id = id
        self.n_neurons = n_neurons
        self.alphabet_size = alphabet_size
        self.input_dim = input_dim
        self.feature_width = 3 * n_neurons

        # Wrapped substrate — its dynamics (VMP update rule, emission
        # scheduling) are reused unchanged. We instantiate it so the
        # adapter is genuinely backed by the BioFieldWML substrate and so
        # downstream code can still drive its step()/emission path.
        self.substrate = BioFieldWML(
            id=id,
            sleep_every=sleep_every,
            alphabet_size=alphabet_size,
            n_neurons=n_neurons,
            error_threshold=error_threshold,
            sigma_confident=sigma_confident,
            seed=seed,
        )

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        saved_rng = torch.get_rng_state()
        try:
            # Local codebook over the belief-feature space (N-5).
            self.codebook = nn.Parameter(
                torch.randn(alphabet_size, self.feature_width, generator=gen)
                * 0.1
            )

            # Trainable input projection: input -> per-neuron observed
            # spike-count drive for the VMP update.
            self.input_proj = nn.Linear(input_dim, n_neurons)
            with torch.no_grad():
                self.input_proj.weight.data = (
                    torch.randn(n_neurons, input_dim, generator=gen) * 0.1
                )
                self.input_proj.bias.data.zero_()

            # Trainable π readout over the fixed-width belief feature.
            self.emit_head_pi = nn.Linear(self.feature_width, alphabet_size)
            with torch.no_grad():
                self.emit_head_pi.weight.data = (
                    torch.randn(alphabet_size, self.feature_width, generator=gen)
                    * 0.1
                )
                self.emit_head_pi.bias.data.zero_()
        finally:
            torch.set_rng_state(saved_rng)

    # ------------------------------------------------------------------
    # Belief-state probe — batched, pure VMP update (dynamics unchanged)
    # ------------------------------------------------------------------

    def _vmp_belief_features(self, observed: Tensor) -> Tensor:
        """Batched closed-form Gaussian VMP belief features.

        Reproduces ``BioFieldWML._vmp_update`` exactly (Laplace-
        approximated Poisson likelihood) but vectorised over a batch and
        starting each row from the vague prior (mu=0, sigma=1) so the
        feature is a deterministic, input-differentiable function of
        ``observed``.  Returns ``[B, 3*n_neurons]`` =
        ``concat(mu_post, sigma_post, |error|)``.

        No learned parameters live inside this update — the belief rule is
        fixed; only ``input_proj`` (upstream) and ``emit_head_pi``
        (downstream) are trained.
        """
        # Vague prior, identical to BioFieldWML.__init__.
        mu_prior = torch.zeros_like(observed)
        var_prior = torch.ones_like(observed)  # sigma_prior = 1 -> var = 1

        # Laplace approximation of the Poisson likelihood variance.
        var_obs = torch.clamp(observed, min=1.0)

        denom = var_obs + var_prior
        mu_post = (var_obs * mu_prior + var_prior * observed) / denom
        var_post = (var_prior * var_obs) / denom
        sigma_post = torch.sqrt(var_post)

        error = observed - mu_prior  # = observed (prior mu is 0)
        return torch.cat([mu_post, sigma_post, error.abs()], dim=-1)

    def core(self, x: Tensor) -> Tensor:
        """Map task input ``[B, input_dim]`` to belief features ``[B, 3*n]``.

        Mirrors ``MlpWML.core`` / ``LifWML`` core surface: a fixed-width
        representation the trained ``emit_head_pi`` reads.  Here the
        representation is the SNN-PC posterior belief over per-neuron
        spike counts driven by a learned projection of the input.
        """
        observed = self.input_proj(x)
        return self._vmp_belief_features(observed)

    def parameters(  # type: ignore[override]
        self, recurse: bool = True
    ) -> Iterable[Tensor]:
        """Trainable parameters: input_proj, emit_head_pi, codebook.

        The wrapped substrate's ``codebook`` parameter is excluded — it is
        not part of the readout training (the VMP belief update has no
        learned weights), keeping the trained capacity matched to the MLP
        baseline (a learned input projection + a learned linear head).
        """
        yield from self.input_proj.parameters(recurse=recurse)
        yield from self.emit_head_pi.parameters(recurse=recurse)
        yield self.codebook
