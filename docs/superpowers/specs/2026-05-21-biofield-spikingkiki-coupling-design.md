# BioFieldWML × SpikingKikiWML coupling — design note

**Status:** design only, no implementation.
**Author:** session 2026-05-21.
**Prereqs:** PRs #37/#38/#39 merged. Reads `track_w/bio_clients.py` (328 L),
`track_w/bio_field_wml.py` (309 L), `track_w/spiking_kiki_wml.py` (226 L).

## Core insight

BioFieldWML is **not a learning rule** that can be plugged into
SpikingKikiWML's optimizer. It is itself a WML substrate that
consumes a `BioCultureClient`: each wake call samples spike activity
from the client and runs a closed-form Gaussian VMP update on its
per-neuron (μ, σ) belief.

The elegant coupling is therefore **SpikingKikiWML ⊑ BioCultureClient**:
wrap a `SpikingKikiWML` in an adapter that satisfies the
`BioCultureClient` Protocol. BioFieldWML then consumes spikingkiki
spike trains exactly as it consumes Mock / CL1 / FinalSpark cultures.
The Palacios SNN-PC VMP update becomes a Predictive-Coding learning
signal **on the silicon culture**, with no change to BioFieldWML
itself.

This makes spikingkiki the first **simulated culture** that satisfies
the same wire contract as wet biology.

## The BioCultureClient Protocol (recap from bio_clients.py)

```python
@runtime_checkable
class BioCultureClient(Protocol):
    n_stim_channels: int
    n_read_channels: int
    n_bins: int

    def encode_stimulus(self, codes: list[int]) -> StimulusFrame: ...
    def decode_activity(self, frame: ActivityFrame) -> list[int]: ...
    def roundtrip(self, codes: list[int]) -> ActivityFrame: ...
    def close(self) -> None: ...
```

`StimulusFrame.channels` is `float32 [n_codes, n_stim_channels]` in
`[0, 1]`. `ActivityFrame.spikes` is `float32 [n_read_channels, n_bins]`
spike counts.

## Proposed adapter

New module: `track_w/spiking_kiki_bio_client.py`.

```python
@dataclass(frozen=True)
class SpikingKikiBioConfig:
    n_stim_channels: int = 8       # match Mock / CL1 default
    n_read_channels: int = 32      # match Mock / CL1 default
    n_bins: int = 16               # number of spike-count windows
    micro_ticks_per_bin: int = 8   # T = n_bins * micro_ticks_per_bin
    block_idx: int = 0
    alphabet_size: int = ALPHABET_SIZE


class SpikingKikiBioClient:
    """BioCultureClient that drives a SpikingKikiWML as the 'culture'.

    encode_stimulus: codes -> low-rank channel pattern that drives the
        SpikingKikiWML input_proj on the next roundtrip.
    roundtrip: run SpikingKikiWML.step() for T = n_bins * micro_ticks
        ticks; capture per-neuron spike events into n_read_channels
        groups (e.g. hidden_dim // n_read_channels neurons per channel)
        and bin them into n_bins time bins.
    decode_activity: argmax over the emit_head_pi readout on the last
        bin -> alphabet code.
    """

    n_stim_channels: int
    n_read_channels: int
    n_bins: int

    def __init__(
        self,
        substrate: SpikingKikiWML,
        config: SpikingKikiBioConfig | None = None,
    ) -> None: ...
```

## Spike → channel-bin reduction

SpikingKikiWML internally tracks `v_mem ∈ R^hidden` and fires when
`v_mem >= threshold`. To deliver `ActivityFrame.spikes` shaped
`[n_read_channels, n_bins]`, we need a fixed bucketing:

- **Spatial:** partition the `hidden_dim` neurons into `n_read_channels`
  contiguous groups (default 32 groups). Each group's per-bin spike
  count = sum of fires in the group during the bin window.
- **Temporal:** divide the `n_micro_ticks` (default 128) into `n_bins`
  windows of `micro_ticks_per_bin` (default 8) ticks.

Result: `spikes[ch, t] = sum_{n in group(ch)} sum_{tick in window(t)}
fired(n, tick)`. Float32 cast.

## Stimulus injection

`encode_stimulus` returns a `StimulusFrame.channels[i, ch]` matrix.
The simplest spikingkiki-side mapping: extend the input pathway with
an injection vector

    inject = StimulusEncoder.project(stim.channels)  # [hidden_dim]

added to `input_proj(embed_inbound(neuroletter))` for the first
`micro_ticks_per_bin` ticks of the next roundtrip. This keeps the
existing `step(nerve, t)` signature intact — the bio client owns the
stimulus state.

Open: per-code or per-roundtrip stimulus? Mock and CL1 deliver
per-code (`[k, n_stim_channels]`), spikingkiki naturally per-roundtrip.
For v1, average across the `k` codes; revisit if BioFieldWML's VMP loss
benefits from per-code resolution.

## Tying it to BioFieldWML

No change to `bio_field_wml.py`. Wiring in a test:

```python
sk = SpikingKikiWML(id=0, provider=InMemoryProvider([w]), ...)
client = SpikingKikiBioClient(substrate=sk)
bio = BioFieldWML(id=1, bio_client=client, ...)

# Now bio.step(nerve, t) drives sk via roundtrip(), reads spikes,
# runs the Palacios VMP belief update, emits PRED/γ or ERR/θ
# Neuroletters as it does with any other BioCultureClient.
```

The Palacios VMP update is **the learning signal** on spikingkiki:
high posterior σ on a neuron group = ERROR / θ Neuroletter, which
downstream learners can use to update spikingkiki's `codebook` or
`input_proj` (out of scope for this design — it's the next step).

## Test plan

1. **Protocol conformance** — `SpikingKikiBioClient` is a
   `BioCultureClient` at runtime.
2. **Shape contract** — `roundtrip([c0, c1, c2])` returns
   `ActivityFrame.spikes.shape == (n_codes, n_read_channels, n_bins)`
   for k=3 codes (or `(n_read_channels, n_bins)` if we choose k-collapse;
   align with Mock semantics).
3. **Determinism** — same seed → same spikes; different seed → different.
4. **Encode/decode roundtrip** — fidelity > 25 % (chance = 1/64 ≈ 1.6 %)
   over 100 random codes; tighter targets after stimulus tuning.
5. **Cross-substrate integration** — `BioFieldWML(bio_client=SpikingKiki
   BioClient(SpikingKikiWML(...)))` runs N wake-sleep cycles and emits
   the expected mix of PREDICTION/γ and ERROR/θ Neuroletters. Reuse
   `tests/integration/track_w/test_bio_cross_substrate.py` patterns.
6. **Comparability** — at low spikingkiki capacity, the VMP belief
   stats (mean μ, mean σ) should track the MockBioCultureClient regime;
   at high capacity they should diverge (silicon culture is more
   structured than a noisy mock).

## Open questions for the implementing session

- Per-code vs per-roundtrip stimulus shape (see above).
- Spatial bucketing: contiguous groups vs hash-partitioned neurons.
  Contiguous is simpler and tied to the .npz layout; hash-partition
  avoids structural bias if blocks have non-uniform firing.
- `n_micro_ticks` budget: defaulting to 128 matches the HF card but is
  slow for cross-substrate integration tests. Provide a `fast`
  shortcut (T=16) for unit / CI use.
- Does decode_activity need an `emit_head_pi` head on every bin, or
  only on the final bin? Final-only is cheaper; per-bin gives BioFieldWML
  finer-grained spike rasters.

## Out of scope

- Using BioFieldWML's VMP residual to **update** SpikingKikiWML's
  weights. That's the "BioField as learning rule" follow-up, separate
  from this Protocol-bridging PR.
- Multi-block SpikingKikiWML stacks (single block sufficient until #5
  unlocks the full forward).
- Real wet-bio comparison (CL1 / FinalSpark) — requires API key.

## Relation to #5

Independent. If #5 ships first (spike-attention + MoE full forward),
this adapter benefits — spikingkiki produces richer spike statistics
and BioFieldWML's PC update gets more signal. But this PR can ship
on the v1 collapsed projection: BioFieldWML doesn't care whether the
silicon culture is realistic, only that it satisfies the wire contract.
