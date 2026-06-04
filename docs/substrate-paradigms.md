# WML substrate paradigms — Hypneum Lab map

> **Status**: design note. Lays out the four (eventually five) WML
> implementations that conform to `nerve_core.protocols.WML` and clarifies
> how they relate to each other and to neighbouring repos
> (`micro-kiki`, `dream-of-kiki`, `baby-brain`).

## The contract

All substrates are stateful `nn.Module`s exposing the same protocol:

```python
@runtime_checkable
class WML(Protocol):
    id:       int
    codebook: Tensor
    def step(self, nerve: Nerve, t: float) -> None: ...
    def parameters(self) -> Iterable[Tensor]: ...
```

`step()` does three things in order: listen on the nerve for inbound
Neuroletters, compute internally with whatever substrate the
implementation chooses, then emit outbound Neuroletters. The codebook is
the only required learnable surface; everything else is private to the
substrate.

## The four paradigms

| Paradigm | Implementation | Forward dynamics | Learning rule | Provenance |
|---|---|---|---|---|
| **In-silico MLP baseline** | `track_w/mlp_wml.py::MlpWML` | Dense linear + nonlinearity | Standard backprop on `codebook` + readout heads | Track W foundation |
| **In-silico LIF (rate-coded)** | `track_w/lif_wml.py::LifWML` | Leaky integrate-and-fire over `T` micro-ticks, rate decode | Surrogate-gradient backprop | Track W foundation |
| **In-silico Transformer** | `track_w/transformer_wml.py::TransformerWML` | Multi-head attention over codebook history | Standard backprop | Track W foundation |
| **BioField PC-VMP** | `track_w/bio_field_wml.py::BioFieldWML` | Spiking field with `μ/σ` belief + `PRED γ` / `ERROR θ` Neuroletter roles | Predictive coding via variational message passing (Palacios SNN-PC, arXiv:2409.05386) | nerve-wml Phase 2 (PR #24, 2026-05-20) |
| **Wet biology** | `track_w/bio_wml.py::BioWML` | Remote biological neural culture via `BioCultureClient` (stim/read-back) | Codebook + emit heads via backprop on decoded spike codes | Track W bio-substrate plan, mock in CI |

## Cross-repo bridges

Three Hypneum Lab repos own SNN-adjacent surfaces. Their roles in the
substrate landscape:

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  micro-kiki  ──→ provides the ANN backbone (Qwen3.6-35B)   │
│      │            and the 35 SOTA LoRA adapters.           │
│      │                                                     │
│      ▼            convert_spikingkiki_35b.py               │
│  spikingkiki ──→ rate-coded LIF conversion (Phase C/D/E).  │
│  35b-a3b-v4       Stored module-wise as 31 070 .npz on HF. │
│      │                                                     │
│      ▼            (planned: SpikingKikiWML substrate)      │
│  nerve-wml   ──→ exposes spikingkiki as one more WML       │
│  track_w/         beside MlpWML / LifWML / BioFieldWML.    │
│                                                            │
│  baby-brain  ──→ ontogenetic harness; consumes any         │
│  adapters/        WML substrate through NerveAdapter ABC.  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Why two SNN paradigms (LIF rate-code vs PC-VMP) coexist

- **Rate-coded LIF** answers *"what if we ran the existing ANN backbone
  as a spiking network?"* — useful for neuromorphic deployment and energy
  budgeting; preserves the upstream micro-kiki competence.
- **Predictive Coding VMP** answers *"what if learning itself were
  spike-native and biologically plausible?"* — useful for theory work on
  active inference and for Track-W's contribution to the larger Hypneum
  Lab research programme.

They are complementary, not redundant. The bridge programme below makes
this explicit.

## Bridge programme (sequenced)

1. **Doc** — this file. Anchors the paradigm vocabulary.
2. **`SpikingKikiWML`** in `nerve-wml/track_w/` — exposes the
   `clemsail/spikingkiki-35b-a3b-v4` weight bundle as a `WML` substrate,
   with lazy mmap of the per-module `.npz` files and `T=128` LIF
   integration. Forward path only at first; no in-place learning.
3. **`SpikingKikiNerveAdapter`** in `baby-brain/baby_brain/adapters/` —
   plugs a `SpikingKikiWML` instance into the ontogenetic harness via
   the existing `NerveAdapter` ABC. Maturity gating: substrate becomes
   active from `INFANT` onward (rate-coding `T=128` is too costly for
   `NEONATE` tick rates).
4. **Learning hand-off** — couple `SpikingKikiWML`'s spike trains to
   `BioFieldWML`'s `μ/σ` belief tracker so that spikingkiki provides the
   **forward dynamics** and BioFieldWML provides the **biologically
   plausible learning rule**. Optional follow-up.

## Out of scope (for now)

- Wet-bio coupling with spikingkiki — `BioWML`'s `BioCultureClient` is
  an entirely different surface and not part of this bridge.
- Real on-device neuromorphic deployment (Loihi, Akida) — flagged as
  future work in the spikingkiki HF card.
- Re-converting micro-kiki to PC-VMP — out of scope; PC-VMP starts from
  a small in-silico substrate by design.
