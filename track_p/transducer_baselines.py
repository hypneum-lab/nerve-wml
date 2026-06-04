"""Comparison baselines for the learned :class:`track_p.transducer.Transducer`.

Closes gap 1 of the validation suite. The learned Transducer is a free
[64x64] logits matrix; on its own it has no point of comparison. This module
provides three alternative src->dst code maps, each consuming the two WML
codebooks (`MlpWML.codebook` etc.) and exposing a `forward(src_code) -> dst_code`
that returns a `[B]` long tensor of dst indices, so all four can be plugged
into the same code-translation benchmark (`scripts/transducer_baselines_pilot.py`).

- ProcrustesTransducer  -- orthogonal Procrustes map (Maystre et al. 2025,
  arXiv:2510.13406). Closed-form SVD solution, supervised by code index.
- RelativeRepTransducer -- anchor-based cosine encoding, zero-shot, no fit
  (Moschella et al., ICLR 2023, arXiv:2209.15430).
- Vec2VecTransducer     -- unsupervised GAN + cycle-consistency translation,
  no paired data (Jha et al. 2025, arXiv:2505.12540).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class ProcrustesTransducer(nn.Module):
    """Orthogonal Procrustes src->dst code map.

    Fits the orthogonal matrix `R` minimising `||src @ R - dst||_F` over the
    paired codebook rows (pairing = shared code index), via the SVD of
    `dst^T @ src`. At inference, a src code is mapped by projecting its src
    embedding through `R` and taking the nearest dst codebook row.

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors. D must match between the two.
    """

    rotation: Tensor
    _dst_codebook: Tensor
    _src_codebook: Tensor

    def __init__(self, src_codebook: Tensor, dst_codebook: Tensor) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        self.alphabet_size = src_codebook.shape[0]
        src = src_codebook.detach().to(torch.float64)
        dst = dst_codebook.detach().to(torch.float64)
        # Orthogonal Procrustes: R = U V^T from SVD of dst^T @ src.
        u, _, vh = torch.linalg.svd(dst.T @ src)
        rotation = (u @ vh).to(torch.float32)
        self.register_buffer("rotation", rotation)
        self.register_buffer("_dst_codebook", dst_codebook.detach().clone())
        self.register_buffer("_src_codebook", src_codebook.detach().clone())

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        src_emb = self._src_lookup(src_code)  # [B, D]
        projected = src_emb @ self.rotation  # [B, D]
        dist = torch.cdist(projected, self._dst_codebook)  # [B, alphabet]
        return dist.argmin(dim=-1)

    def _src_lookup(self, src_code: Tensor) -> Tensor:
        return self._src_codebook[src_code]


class RelativeRepTransducer(nn.Module):
    """Anchor-based relative-representation src->dst map (zero-shot).

    Moschella et al. (ICLR 2023): encode each codebook row by its vector of
    cosine similarities to a fixed set of `n_anchors` anchor rows. Because the
    anchors are shared by code index across src and dst, the relative encoding
    is invariant to any rotation/reflection between the two latent spaces, so
    no fitting is needed. A src code maps to the dst code whose relative
    encoding is closest (cosine).

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors, same shape.
    n_anchors
        Number of anchor codes (sampled without replacement from the alphabet).
    seed
        RNG seed for anchor selection.
    """

    src_rel: Tensor
    dst_rel: Tensor

    def __init__(
        self,
        src_codebook: Tensor,
        dst_codebook: Tensor,
        *,
        n_anchors: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        alphabet = src_codebook.shape[0]
        if not 1 <= n_anchors <= alphabet:
            raise ValueError(f"n_anchors must be in [1, {alphabet}]")
        gen = torch.Generator()
        gen.manual_seed(seed)
        anchors = torch.randperm(alphabet, generator=gen)[:n_anchors]
        src = src_codebook.detach()
        dst = dst_codebook.detach()
        self.register_buffer("src_rel", self._relative(src, anchors))
        self.register_buffer("dst_rel", self._relative(dst, anchors))

    @staticmethod
    def _relative(codebook: Tensor, anchors: Tensor) -> Tensor:
        """`[alphabet, n_anchors]` cosine-similarity encoding."""
        normed = F.normalize(codebook, dim=-1)
        anchor_vecs = normed[anchors]  # [n_anchors, D]
        return normed @ anchor_vecs.T  # [alphabet, n_anchors]

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        query = F.normalize(self.src_rel[src_code], dim=-1)  # [B, n_anchors]
        keys = F.normalize(self.dst_rel, dim=-1)  # [alphabet, n_anchors]
        sim = query @ keys.T  # [B, alphabet]
        return sim.argmax(dim=-1)


class Vec2VecTransducer(nn.Module):
    """Unsupervised src->dst code map (vec2vec-style GAN + cycle-consistency).

    Jha et al. 2025 (arXiv:2505.12540): translate between two latent spaces
    with NO paired supervision, using an adversarial loss (discriminator must
    not tell translated-src from real-dst) plus a cycle-consistency loss
    (`G_dst->src(G_src->dst(x)) ~= x`). Here the two "spaces" are the src and
    dst WML codebooks treated as unpaired point clouds.

    `fit` trains the two generators and one discriminator; `forward` maps a
    src code by translating its embedding and taking the nearest dst row.

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors, same shape.
    hidden
        Width of the generator / discriminator MLPs.
    lambda_cycle
        Weight of the cycle-consistency term.
    seed
        RNG seed for parameter init.
    """

    _src_codebook: Tensor
    _dst_codebook: Tensor
    alphabet_size: int
    lambda_cycle: float
    g_src2dst: nn.Sequential
    g_dst2src: nn.Sequential
    discriminator: nn.Sequential



    def __init__(
        self,
        src_codebook: Tensor,
        dst_codebook: Tensor,
        *,
        hidden: int = 64,
        lambda_cycle: float = 10.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        torch.manual_seed(seed)
        self.alphabet_size, dim = src_codebook.shape
        self.lambda_cycle = float(lambda_cycle)
        self.register_buffer("_src_codebook", src_codebook.detach().clone())
        self.register_buffer("_dst_codebook", dst_codebook.detach().clone())

        def _mlp(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d_out),
            )

        self.g_src2dst = _mlp(dim, dim)
        self.g_dst2src = _mlp(dim, dim)
        self.discriminator = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def fit(self, *, steps: int = 2000, lr: float = 1e-3) -> list[float]:
        """Adversarial + cycle training. Returns the per-step cycle loss."""
        gen_params = list(self.g_src2dst.parameters()) + list(
            self.g_dst2src.parameters()
        )
        opt_g = torch.optim.Adam(gen_params, lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        src = self._src_codebook
        dst = self._dst_codebook
        ones = torch.ones(self.alphabet_size, 1)
        zeros = torch.zeros(self.alphabet_size, 1)
        cycle_history: list[float] = []
        for _ in range(steps):
            # --- discriminator step ---
            opt_d.zero_grad()
            fake = self.g_src2dst(src).detach()
            loss_d = bce(self.discriminator(dst), ones) + bce(
                self.discriminator(fake), zeros
            )
            loss_d.backward()
            opt_d.step()
            # --- generator step: fool D + cycle-consistency ---
            opt_g.zero_grad()
            fake_dst = self.g_src2dst(src)
            loss_adv = bce(self.discriminator(fake_dst), ones)
            cyc_src = F.mse_loss(self.g_dst2src(fake_dst), src)
            cyc_dst = F.mse_loss(
                self.g_src2dst(self.g_dst2src(dst)), dst
            )
            loss_cycle = cyc_src + cyc_dst
            (loss_adv + self.lambda_cycle * loss_cycle).backward()
            opt_g.step()
            cycle_history.append(float(loss_cycle.detach()))
        return cycle_history

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        with torch.no_grad():
            src_emb = self._src_codebook[src_code]  # [B, D]
            translated = self.g_src2dst(src_emb)  # [B, D]
            dist = torch.cdist(translated, self._dst_codebook)
        return dist.argmin(dim=-1)


class SimpleGatingMultiplexer(nn.Module):
    """Plain learned-gating control for the GTM ablation.

    Closes gap 2 (ablation arm): a minimal alternative to
    `track_p.multiplexer.GammaThetaMultiplexer` with NO theta/gamma band
    multiplexing. Each of the `n_symbols` code slots is embedded and summed
    into a flat carrier through a single learned linear gate; demodulation is
    a learned linear read-out. Same `forward(codes)->carrier` /
    `demodulate(carrier)->codes` contract as GTM, so the ablation script can
    swap them.

    Parameters
    ----------
    alphabet_size
        Code alphabet size.
    n_symbols
        Code slots per carrier (GTM's `symbols_per_theta`).
    carrier_dim
        Width of the flat carrier vector.
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 64,
        n_symbols: int = 7,
        carrier_dim: int = 64,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        self.n_symbols = n_symbols
        self.carrier_dim = carrier_dim
        self.embed = nn.Embedding(alphabet_size, carrier_dim)
        # One learned scalar gate weight per symbol slot.
        self.gate = nn.Linear(n_symbols, n_symbols, bias=False)
        self.readout = nn.Linear(carrier_dim, n_symbols * alphabet_size)

    def forward(self, codes: Tensor) -> Tensor:
        """Encode `[B, n_symbols]` long codes to a `[B, carrier_dim]` carrier."""
        if codes.shape[-1] != self.n_symbols:
            raise ValueError(
                f"expected {self.n_symbols} symbols, got {codes.shape[-1]}"
            )
        emb = self.embed(codes)  # [B, n_symbols, carrier_dim]
        slot_id = torch.eye(self.n_symbols, device=codes.device)
        gates = self.gate(slot_id).diagonal()  # [n_symbols]
        return (emb * gates[None, :, None]).sum(dim=1)  # [B, carrier_dim]

    def demodulate(self, carrier: Tensor) -> Tensor:
        """Recover `[B, n_symbols]` long codes from a `[B, carrier_dim]` carrier."""
        logits = self.readout(carrier)  # [B, n_symbols * alphabet]
        logits = logits.view(-1, self.n_symbols, self.alphabet_size)
        return logits.argmax(dim=-1)

    def demodulate_logits(self, carrier: Tensor) -> Tensor:
        """`[B, n_symbols, alphabet_size]` logits — for training the read-out."""
        logits = self.readout(carrier)
        return logits.view(-1, self.n_symbols, self.alphabet_size)


class KuramotoMultiplexer(nn.Module):
    """Minimal AKOrN-style coupled-oscillator multiplexer.

    Inspired by Miyato et al. (ICLR 2025, arXiv:2410.13821) Artificial
    Kuramoto Oscillatory Neurons. Each of `n_oscillators` units has a
    learned natural frequency and is coupled to every other through a
    learned [N, N] coupling matrix. A stimulus code is written as a
    phase-bias injection across a subset of oscillators (selected
    deterministically by code bits, same trick as the simple gating
    control). After `n_steps` of Euler integration of the Kuramoto
    update, the terminal phase pattern is read out linearly to logits.

    Provides the same `forward(codes) -> carrier` /
    `demodulate(carrier) -> codes` contract as `SimpleGatingMultiplexer`
    so the gtm ablation script can drop it in as a third arm.

    Parameters
    ----------
    alphabet_size : int
        Code alphabet size.
    n_symbols : int
        Code slots per carrier.
    n_oscillators : int
        Oscillator pool size (default 32; smaller = faster, less expressive).
    n_steps : int
        Default Euler-step count per forward (default 8).
    dt : float
        Integration step size (default 0.1).
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 64,
        n_symbols: int = 7,
        n_oscillators: int = 32,
        n_steps: int = 8,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        self.n_symbols = n_symbols
        self.n_oscillators = n_oscillators
        self.n_steps = n_steps
        self.dt = dt
        # Learned natural frequencies, one per oscillator.
        self.natural_freqs = nn.Parameter(
            torch.randn(n_oscillators) * 0.5
        )
        # Learned coupling matrix [N, N]. Symmetric initialisation
        # for stability; not enforced symmetric afterwards.
        init_coupling = torch.randn(n_oscillators, n_oscillators) * 0.1
        self.coupling = nn.Parameter(
            0.5 * (init_coupling + init_coupling.T)
        )
        # Phase-bias injection: each code maps to a small fixed pattern
        # of phase offsets across the oscillator pool, derived from the
        # code bits — matches the SimpleGating idiom.
        self.code_bias = nn.Parameter(
            torch.randn(alphabet_size, n_oscillators) * 0.1
        )
        # Linear read-out: phase pattern -> per-symbol logits.
        self.readout = nn.Linear(
            n_oscillators, n_symbols * alphabet_size,
        )

    def forward(
        self, codes: Tensor, *, n_steps: int | None = None,
    ) -> Tensor:
        """Encode `[B, n_symbols]` long codes to a `[B, n_oscillators]` carrier."""
        if codes.shape[-1] != self.n_symbols:
            raise ValueError(
                f"expected {self.n_symbols} symbols, got {codes.shape[-1]}"
            )
        steps = n_steps if n_steps is not None else self.n_steps
        # Initial phase = mean of injected biases across symbol slots.
        biases = self.code_bias[codes]  # [B, n_symbols, n_oscillators]
        phase = biases.mean(dim=1)      # [B, n_oscillators]
        # Discretised Kuramoto update for `steps` Euler steps.
        for _ in range(steps):
            # dphi_i/dt = omega_i + sum_j K_ij * sin(phi_j - phi_i)
            diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)
            # diff[b, i, j] = phi_i - phi_j; we want phi_j - phi_i for the formula.
            # Use -diff so the sign is right.
            interaction = (self.coupling * torch.sin(-diff)).sum(dim=-1)
            phase = phase + self.dt * (self.natural_freqs + interaction)
        return phase  # [B, n_oscillators]

    def demodulate(self, carrier: Tensor) -> Tensor:
        """Recover `[B, n_symbols]` long codes from a `[B, n_oscillators]` carrier."""
        logits = self.readout(carrier)
        logits = logits.view(-1, self.n_symbols, self.alphabet_size)
        return logits.argmax(dim=-1)

    def demodulate_logits(self, carrier: Tensor) -> Tensor:
        """`[B, n_symbols, alphabet_size]` logits — for training the read-out."""
        logits = self.readout(carrier)
        return logits.view(-1, self.n_symbols, self.alphabet_size)
