"""ProtocolNerve — opt-in wire of Transducer and GammaThetaMultiplexer into SimNerve.

Adds two flags to SimNerve:

- ``use_transducer``: build one :class:`Transducer` per active edge (mirroring
  :class:`bridge.sim_nerve_adapter.SimNerveAdapter`). On ``send()``, the src code
  is rewritten to the dst codebook via the per-edge transducer before queuing.
  Only the ``code`` field changes; ``role``, ``phase``, ``src``, ``dst``, and
  ``timestamp`` are preserved, so N-3 continues to hold.

- ``use_gtm``: a single shared :class:`GammaThetaMultiplexer` is instantiated.
  After phase-gating in ``listen()``, deliverable codes are batched (up to
  ``symbols_per_theta`` at a time) and passed through the GTM round-trip
  ``demodulate(forward(codes))`` — the real PAC channel, which is lossy with an
  untrained constellation.  Only the ``code`` field is affected; all other
  Neuroletter metadata survives intact.

**Default (both False) is byte-identical to SimNerve** — existing results
reproduce without any change.

Implementation notes
--------------------
- Transducers are stored in a :class:`torch.nn.ModuleDict` (keyed ``"src_dst"``)
  so they appear in ``parameters()`` alongside the router logits.
- The GTM is stored as a sub-module with a fixed seed for determinism.
- ``symbols_per_theta`` from the GTM config caps the batch size; if more letters
  are deliverable in one listen call than ``K``, they are chunked and processed
  sequentially — each chunk traverses one full PAC round-trip.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from nerve_core.invariants import assert_n3_role_phase_consistent
from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.multiplexer import GammaThetaMultiplexer
from track_p.sim_nerve import SimNerve
from track_p.transducer import Transducer


class ProtocolNerve(SimNerve):
    """SimNerve extended with per-edge transducers and the GTM PAC channel.

    Parameters
    ----------
    n_wmls, k, seed, strict_n3, priority_rule
        Forwarded verbatim to :class:`SimNerve`.
    use_transducer
        When ``True``, build one :class:`Transducer` per active edge.
        ``send()`` rewrites ``letter.code`` src→dst via the transducer before
        queuing. Default ``False`` — identical to base SimNerve.
    use_gtm
        When ``True``, ``listen()`` passes deliverable codes through one
        :class:`GammaThetaMultiplexer` round-trip (``demodulate(forward(codes))``)
        before returning them.  The constellation is randomly initialised (seeded),
        so fidelity is low until trained — that loss IS the scientific signal.
        Default ``False`` — identical to base SimNerve.
    gtm_seed
        RNG seed for the GTM constellation initialisation.  Separate from the
        topology ``seed`` so the two axes are independently reproducible.
    """

    def __init__(
        self,
        n_wmls: int,
        k: int,
        *,
        seed: int | None = 0,
        strict_n3: bool = True,
        priority_rule: bool = True,
        use_transducer: bool = False,
        use_gtm: bool = False,
        gtm_seed: int | None = 42,
    ) -> None:
        super().__init__(
            n_wmls=n_wmls,
            k=k,
            seed=seed,
            strict_n3=strict_n3,
            priority_rule=priority_rule,
        )
        self.use_transducer = use_transducer
        self.use_gtm = use_gtm

        # Per-edge transducers (N-5: each edge maps one codebook to another).
        self._transducers: nn.ModuleDict = nn.ModuleDict()
        if use_transducer:
            for src in range(n_wmls):
                for dst in range(n_wmls):
                    if self._edges[src, dst].item() == 1.0:
                        self._transducers[f"{src}_{dst}"] = Transducer(
                            alphabet_size=self.ALPHABET_SIZE
                        )

        # Shared GTM instance (one per ProtocolNerve, seeded for determinism).
        self._gtm: GammaThetaMultiplexer | None = None
        if use_gtm:
            self._gtm = GammaThetaMultiplexer(seed=gtm_seed)

    # ------------------------------------------------------------------
    # send: optionally apply per-edge transducer before queueing
    # ------------------------------------------------------------------

    def send(self, letter: Neuroletter) -> None:  # type: ignore[override]
        """Queue a Neuroletter, optionally rewriting code via per-edge transducer.

        N-3 is still asserted on the original letter (role ⟺ phase).
        The transducer only rewrites ``code``; ``role``, ``phase``, ``src``,
        ``dst``, and ``timestamp`` pass through unchanged, so N-3 holds for the
        queued letter as well.
        """
        assert_n3_role_phase_consistent(letter, strict=self._strict_n3)
        if self._edges[letter.src, letter.dst].item() == 0:
            return

        if self.use_transducer:
            key = f"{letter.src}_{letter.dst}"
            if key in self._transducers:
                src_t = torch.tensor([letter.code], dtype=torch.long)
                dst_code = int(self._transducers[key].forward(src_t, hard=True).item())
                letter = Neuroletter(
                    code=dst_code,
                    role=letter.role,
                    phase=letter.phase,
                    src=letter.src,
                    dst=letter.dst,
                    timestamp=letter.timestamp,
                )

        self._queues[letter.dst].append(letter)

    # ------------------------------------------------------------------
    # listen: optionally pass deliverable codes through GTM round-trip
    # ------------------------------------------------------------------

    def listen(
        self,
        wml_id: int,
        role: Role | None = None,
        phase: Phase | None = None,
    ) -> list[Neuroletter]:
        """Phase-gated delivery, optionally routing codes through the GTM PAC channel.

        The GTM round-trip (``demodulate(forward(codes))``) is the real γ/θ
        channel: codes travel as a PAC carrier and are recovered by nearest-
        constellation distance.  With an untrained constellation the fidelity
        is low; that loss is the scientific signal measured by the smoke test.
        Only ``code`` traverses the carrier; ``role``, ``phase``, ``src``,
        ``dst``, and ``timestamp`` are kept as metadata.
        """
        delivered = super().listen(wml_id=wml_id, role=role, phase=phase)

        if not self.use_gtm or self._gtm is None or not delivered:
            return delivered

        # Chunk delivered letters into batches of ≤ symbols_per_theta (K).
        k_cap: int = self._gtm.cfg.symbols_per_theta
        result: list[Neuroletter] = []

        for chunk_start in range(0, len(delivered), k_cap):
            chunk = delivered[chunk_start : chunk_start + k_cap]
            codes_t = torch.tensor(
                [letter.code for letter in chunk], dtype=torch.long
            ).unsqueeze(0)  # [1, K]

            with torch.no_grad():
                carrier = self._gtm.forward(codes_t)          # [1, T]
                recovered = self._gtm.demodulate(carrier, hard=True)  # [1, K]

            for i, letter in enumerate(chunk):
                new_code = int(recovered[0, i].item())
                result.append(
                    Neuroletter(
                        code=new_code,
                        role=letter.role,
                        phase=letter.phase,
                        src=letter.src,
                        dst=letter.dst,
                        timestamp=letter.timestamp,
                    )
                )

        return result

    # ------------------------------------------------------------------
    # parameters: include transducers + GTM constellation
    # ------------------------------------------------------------------

    def parameters(self) -> Iterable[Tensor]:  # type: ignore[override]
        """Yield all learnable tensors: router logits + transducers + GTM.

        Follows the pattern in :class:`SimNerve` (yields individual tensors
        rather than relying on ``nn.Module.parameters()`` so the API stays
        compatible with both Module and non-Module consumers).
        """
        yield self.router.logits
        for p in self._transducers.parameters():
            yield p
        if self._gtm is not None:
            for p in self._gtm.parameters():
                yield p
