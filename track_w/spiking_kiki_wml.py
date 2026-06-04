"""SpikingKikiWML — a WML whose core is one transformer block of the
``clemsail/spikingkiki-35b-a3b-v4`` rate-coded LIF conversion.

The full 70 GB spikingkiki conversion stores per-module ``.npz`` weights
for 40 transformer blocks (31 070 files total). This substrate exposes
ONE block's worth of LIF rate-coding through ``nerve_core.protocols.WML``
so the nerve-wml harness can drive it side-by-side with the other
substrates (MlpWML, LifWML, TransformerWML, BioFieldWML).

Weights are produced by an injected ``SpikingKikiWeightsProvider`` so
tests stay tiny and real deployment can lazily mmap the ``.npz`` bundle.

This first cut implements the WML protocol with a simplified LIF
integration: at each ``step()`` we project inbound nerve traffic into the
block's hidden space, integrate ``T`` LIF micro-ticks, decode via the
codebook + emit head, and send a Neuroletter when confidence exceeds
``threshold_eps``. Full spike-attention and MoE expert routing — the
parts of the spikingkiki forward path that benefit most from the per-
module weight bundle — land in a follow-up.

LIF constants match the spikingkiki HF card: ``T=128``, ``threshold=0.0625``,
``tau=1.0``.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve

_DEFAULT_T:         int   = 128
_DEFAULT_THRESHOLD: float = 0.0625
_DEFAULT_TAU:       float = 1.0


@runtime_checkable
class SpikingKikiWeightsProvider(Protocol):
    """Provides one transformer block's worth of spikingkiki weights.

    The protocol intentionally hides the storage medium (in-memory dict,
    ``np.memmap`` over a ``.npz``, remote HF stream, …) so tests inject
    tiny fixtures while production wires the real 70 GB bundle.
    """

    n_blocks: int
    hidden_dim: int

    def get_block_projection(self, block_idx: int) -> Tensor:
        """Return a ``(hidden_dim, hidden_dim)`` linear weight for the block.

        For the protocol-level WML cluster we collapse the full block's
        attention + MoE machinery into one effective projection. Concrete
        providers can do this collapse however they like (mean of expert
        gates, principal component, identity passthrough).
        """
        ...


class InMemoryProvider:
    """In-memory weights provider for tests and quick experiments."""

    def __init__(self, projections: list[Tensor]) -> None:
        if not projections:
            raise ValueError("InMemoryProvider needs at least one projection")
        hidden = projections[0].shape[-1]
        for i, p in enumerate(projections):
            if p.shape != (hidden, hidden):
                raise ValueError(
                    f"projection {i} has shape {tuple(p.shape)}, expected ({hidden},{hidden})",
                )
        self._projections = projections
        self.hidden_dim = hidden
        self.n_blocks = len(projections)

    def get_block_projection(self, block_idx: int) -> Tensor:
        if not 0 <= block_idx < self.n_blocks:
            raise IndexError(
                f"block_idx {block_idx} outside [0, {self.n_blocks})",
            )
        return self._projections[block_idx]


@runtime_checkable
class SpikingKikiBlockProvider(SpikingKikiWeightsProvider, Protocol):
    """Extended provider that surfaces the full per-module weight layout.

    Extends ``SpikingKikiWeightsProvider`` with optional access to the
    Q/K/V attention projections, attention output, MoE router, per-expert
    gate/up/down projections, and RMSNorm scales. This is the surface
    needed for the spike-native attention + MoE routing follow-up (PR #5).

    ``get_block_projection`` from the v1 Protocol is preserved so existing
    ``InMemoryProvider``, ``MmapNpzProvider``, and all existing unit tests
    remain green — this Protocol only adds optional methods. Providers
    that do NOT implement ``SpikingKikiBlockProvider`` continue to work
    with ``SpikingKikiWML.step()`` via the v1 collapsed-projection path.

    NOTE: The spike-native attention formula and MoE routing strategy
    (spike-route vs rate-route) are open research questions that require
    auditing ``convert_spikingkiki_35b.py`` from the upstream repo
    (55 KB, not included in nerve-wml). The forward implementation using
    this Protocol is deferred pending that audit. Only the Protocol
    surface and a synthetic test fixture are shipped in this PR.

    Naming conventions (subject to audit confirmation):
      W_qkv  — stacked Q/K/V projections, shape (3*n_heads*head_dim, hidden_dim)
      W_out  — attention output projection, shape (hidden_dim, n_heads*head_dim)
      W_router — MoE router logit weights, shape (n_experts, hidden_dim)
      W_gate, W_up, W_down — expert gate/up/down projections per expert
      gamma_in, gamma_post — RMSNorm scale vectors, shape (hidden_dim,) each
    """

    n_experts: int
    n_heads: int

    def get_qkv(self, block_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (W_Q, W_K, W_V) weight matrices for the block.

        Shape of each: ``(n_heads * head_dim, hidden_dim)`` where
        ``head_dim = hidden_dim // n_heads``.
        """
        ...

    def get_attn_out(self, block_idx: int) -> Tensor:
        """Return the attention output projection ``W_out``.

        Shape: ``(hidden_dim, n_heads * head_dim)``.
        """
        ...

    def get_router(self, block_idx: int) -> Tensor:
        """Return the MoE router logit weight matrix.

        Shape: ``(n_experts, hidden_dim)``.
        """
        ...

    def get_expert(
        self, block_idx: int, expert_idx: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (W_gate, W_up, W_down) for one MoE expert.

        Shapes:
          W_gate: ``(intermediate_dim, hidden_dim)``
          W_up:   ``(intermediate_dim, hidden_dim)``
          W_down: ``(hidden_dim, intermediate_dim)``
        """
        ...

    def get_norms(self, block_idx: int) -> tuple[Tensor, Tensor]:
        """Return (gamma_in, gamma_post) RMSNorm scale vectors.

        Each has shape ``(hidden_dim,)``.
        """
        ...


class InMemoryBlockProvider:
    """Synthetic in-memory provider satisfying SpikingKikiBlockProvider.

    Used by tests for the spike-attention + MoE Protocol surface (PR #5).
    Weights are random tensors of the correct shapes. The v1
    ``get_block_projection`` is computed as a mean of expert gate rows.

    Parameters
    ----------
    hidden_dim:
        Hidden dimension of the transformer block.
    n_blocks:
        Number of blocks in the provider.
    n_experts:
        Number of MoE experts per block.
    n_heads:
        Number of attention heads per block.
    seed:
        Optional random seed for reproducible fixtures.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        n_blocks: int = 1,
        n_experts: int = 2,
        n_heads: int = 2,
        seed: int | None = None,
    ) -> None:
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.n_experts = n_experts
        self.n_heads = n_heads

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        head_dim = hidden_dim // n_heads
        intermediate = hidden_dim * 2  # standard 2× MLP ratio

        # Pre-generate all weights so get_* calls are deterministic.
        self._qkv: list[tuple[Tensor, Tensor, Tensor]] = []
        self._out: list[Tensor] = []
        self._router: list[Tensor] = []
        self._experts: list[list[tuple[Tensor, Tensor, Tensor]]] = []
        self._norms: list[tuple[Tensor, Tensor]] = []

        for _ in range(n_blocks):
            self._qkv.append((
                torch.randn(n_heads * head_dim, hidden_dim, generator=gen) * 0.1,
                torch.randn(n_heads * head_dim, hidden_dim, generator=gen) * 0.1,
                torch.randn(n_heads * head_dim, hidden_dim, generator=gen) * 0.1,
            ))
            self._out.append(
                torch.randn(hidden_dim, n_heads * head_dim, generator=gen) * 0.1
            )
            self._router.append(
                torch.randn(n_experts, hidden_dim, generator=gen) * 0.1
            )
            block_experts = []
            for _ in range(n_experts):
                block_experts.append((
                    torch.randn(intermediate, hidden_dim, generator=gen) * 0.1,
                    torch.randn(intermediate, hidden_dim, generator=gen) * 0.1,
                    torch.randn(hidden_dim, intermediate, generator=gen) * 0.1,
                ))
            self._experts.append(block_experts)
            self._norms.append((
                torch.ones(hidden_dim),   # gamma_in (unit init)
                torch.ones(hidden_dim),   # gamma_post (unit init)
            ))

    def _check_block(self, block_idx: int) -> None:
        if not 0 <= block_idx < self.n_blocks:
            raise IndexError(
                f"block_idx {block_idx} outside [0, {self.n_blocks})"
            )

    def _check_expert(self, expert_idx: int) -> None:
        if not 0 <= expert_idx < self.n_experts:
            raise IndexError(
                f"expert_idx {expert_idx} outside [0, {self.n_experts})"
            )

    def get_block_projection(self, block_idx: int) -> Tensor:
        """v1 fallback: mean of expert gate rows as collapsed projection."""
        self._check_block(block_idx)
        gates = torch.stack(
            [self._experts[block_idx][e][0] for e in range(self.n_experts)]
        )  # [n_experts, intermediate, hidden]
        # Collapse to (hidden, hidden): mean over experts, mean over intermediate.
        return gates.mean(dim=0).mean(dim=0, keepdim=True).expand(
            self.hidden_dim, self.hidden_dim
        )

    def get_qkv(self, block_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        self._check_block(block_idx)
        return self._qkv[block_idx]

    def get_attn_out(self, block_idx: int) -> Tensor:
        self._check_block(block_idx)
        return self._out[block_idx]

    def get_router(self, block_idx: int) -> Tensor:
        self._check_block(block_idx)
        return self._router[block_idx]

    def get_expert(
        self, block_idx: int, expert_idx: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._check_block(block_idx)
        self._check_expert(expert_idx)
        return self._experts[block_idx][expert_idx]

    def get_norms(self, block_idx: int) -> tuple[Tensor, Tensor]:
        self._check_block(block_idx)
        return self._norms[block_idx]


class SpikingKikiWML(nn.Module):
    """One transformer block of spikingkiki as a ``WML`` substrate.

    The substrate is *frozen* on the block projection (rate-coded LIF
    conversion is a static representation of an upstream ANN). Only the
    codebook and the readout heads are learned, mirroring the other
    track_w substrates.
    """

    v_mem: Tensor

    def __init__(
        self,
        id: int,
        provider: SpikingKikiWeightsProvider,
        block_idx: int = 0,
        alphabet_size: int = 64,
        n_micro_ticks: int = _DEFAULT_T,
        threshold: float = _DEFAULT_THRESHOLD,
        tau: float = _DEFAULT_TAU,
        threshold_eps: float = 0.30,
        input_dim: int | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(provider, SpikingKikiWeightsProvider):
            raise TypeError(
                "provider must conform to SpikingKikiWeightsProvider",
            )
        if n_micro_ticks <= 0:
            raise ValueError(f"n_micro_ticks must be positive, got {n_micro_ticks}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if not 0.0 < tau:
            raise ValueError(f"tau must be positive, got {tau}")

        self.id = id
        self.alphabet_size = alphabet_size
        self.block_idx = block_idx
        self.n_micro_ticks = n_micro_ticks
        self.threshold = threshold
        self.tau = tau
        self.threshold_eps = threshold_eps
        self.hidden_dim = provider.hidden_dim
        self.input_dim = input_dim if input_dim is not None else self.hidden_dim

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Frozen block projection from the provider — the substrate-specific
        # "wiring" inherited from the spikingkiki conversion.
        block_w = provider.get_block_projection(block_idx).detach().clone()
        self.register_buffer("block_projection", block_w)

        # Learnable WML surface — same shape contract as LifWML.
        codebook_init = (
            torch.rand(alphabet_size, self.hidden_dim, generator=gen) > 0.7
        ).float()
        self.codebook = nn.Parameter(codebook_init)
        # Python 3.14 runtime_checkable workaround (mirrors BioWML / LifWML).
        self.__dict__["codebook"] = self._parameters["codebook"]

        self.register_buffer("v_mem", torch.zeros(self.hidden_dim))

        saved_rng = torch.get_rng_state()
        try:
            self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
            with torch.no_grad():
                self.input_proj.weight.data = (
                    torch.randn(self.hidden_dim, self.input_dim, generator=gen) * 0.1
                )
                self.input_proj.bias.data.zero_()

            self.emit_head_pi = nn.Linear(self.hidden_dim, alphabet_size)
            with torch.no_grad():
                self.emit_head_pi.weight.data = (
                    torch.randn(alphabet_size, self.hidden_dim, generator=gen) * 0.1
                )
                self.emit_head_pi.bias.data.zero_()
        finally:
            torch.set_rng_state(saved_rng)

    def reset_state(self) -> None:
        self.v_mem.zero_()

    def _lif_integrate(self, drive: Tensor) -> Tensor:
        """Rate-coded LIF: integrate ``n_micro_ticks`` steps, return spike count.

        Drive is held constant across the micro-ticks (the upstream nerve
        operates on a slower clock); the rate code is the sum of binary
        spikes over the window.
        """
        spike_count = torch.zeros_like(drive)
        for _ in range(self.n_micro_ticks):
            self.v_mem = self.v_mem + (1.0 / self.tau) * (-self.v_mem + drive)
            spikes = (self.v_mem > self.threshold).float()
            spike_count = spike_count + spikes
            self.v_mem = self.v_mem * (1.0 - spikes)
        return spike_count

    def step(self, nerve: Nerve, t: float, dt: float = 1e-3) -> None:
        del dt  # spikingkiki micro-tick cadence is internal (n_micro_ticks)
        from nerve_core.neuroletter import Neuroletter, Phase, Role
        from track_w._decode import embed_inbound

        inbound = nerve.listen(self.id)
        pooled = embed_inbound(inbound, self.codebook)            # (hidden,)
        i_in = self.input_proj(pooled)                            # (hidden,)
        drive = i_in @ self.block_projection                      # (hidden,)
        spike_count = self._lif_integrate(drive)                  # (hidden,)

        if spike_count.sum().item() == 0:
            return  # N-1: silent when no spikes — same convention as LifWML

        # π readout — pick the most active code; threshold guards against
        # weak / noisy emissions.
        logits = self.emit_head_pi(spike_count)
        best = int(logits.argmax().item())
        conf = float(torch.softmax(logits, dim=-1)[best].item())
        if conf < self.threshold_eps:
            return
        n_wmls = getattr(nerve, "n_wmls", 0)
        for dst in range(n_wmls):
            if dst == self.id:
                continue
            if nerve.routing_weight(self.id, dst) == 1.0:
                nerve.send(
                    Neuroletter(
                        code=best,
                        role=Role.PREDICTION,
                        phase=Phase.GAMMA,
                        src=self.id,
                        dst=dst,
                        timestamp=t,
                    ),
                )
