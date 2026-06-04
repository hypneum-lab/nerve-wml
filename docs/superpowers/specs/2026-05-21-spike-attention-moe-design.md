# Spike-attention + MoE routing for SpikingKikiWML — design note

**Status:** design only, no implementation. Carries the "real per-module
forward" follow-up from PR #38 (SpikingKikiWML scaffold).
**Author:** session 2026-05-21.
**Prereqs:** PRs #37 (paradigm map), #38 (substrate scaffold), #39
(MmapNpzProvider) all merged on master at `0be3fca`.

## Problem

The current `SpikingKikiWML.step()` collapses a transformer block to a
single `(hidden_dim, hidden_dim)` projection — the Q slice of the
linear-attention QKV stack, surfaced by `SpikingKikiWeightsProvider.
get_block_projection()`. This is enough to exercise the protocol but
discards ~774 of the ~775 module files per block in the on-disk
`clemsail/spikingkiki-35b-a3b-v4` bundle. The real spikingkiki forward
needs:

1. **Spike-native attention** over Q/K/V projections, accumulating into
   a Gaussian belief over output tokens rather than a softmax.
2. **MoE expert routing** — each block has N MLP experts; the router
   selects top-k experts per token in the spike domain.

These are research-grade design choices, not a refactor. They merit a
dedicated PR with its own bench evaluation against the micro-kiki ANN
reference forward.

## Module inventory per block

Naming pattern in the .npz bundle:

- `model_layers_<i>_linear_attn_in_proj_qkv.npz` — stacked W_qkv (already
  surfaced by `MmapNpzProvider`).
- `model_layers_<i>_linear_attn_out_proj.npz` — attention output projection.
- `model_layers_<i>_mlp_experts_<j>_gate_proj.npz` — expert j gate (e.g.
  j ∈ [0, 64)).
- `model_layers_<i>_mlp_experts_<j>_up_proj.npz` — expert j up.
- `model_layers_<i>_mlp_experts_<j>_down_proj.npz` — expert j down.
- `model_layers_<i>_router_weights.npz` — MoE router logits weight.
- `model_layers_<i>_norm_*.npz` — RMSNorm scales (input + post-attn).

Audit needs `convert_spikingkiki_35b.py` (55 KB script in the upstream
repo) to confirm exact naming on the v4 mirror and whether the bundle
also ships LIF state buffers (`v_mem`, `refractory_count`).

## Proposed interface (in `track_w/spiking_kiki_wml.py`)

Extend `SpikingKikiWeightsProvider` Protocol with optional surface for
the full block layout. Keep `get_block_projection` as the v1 minimal
contract so the existing in-memory test fixtures and MmapNpzProvider
stay green:

```python
@runtime_checkable
class SpikingKikiBlockProvider(SpikingKikiWeightsProvider, Protocol):
    n_experts: int
    n_heads: int

    def get_qkv(self, block_idx: int) -> tuple[Tensor, Tensor, Tensor]: ...
    def get_attn_out(self, block_idx: int) -> Tensor: ...
    def get_router(self, block_idx: int) -> Tensor: ...
    def get_expert(self, block_idx: int, expert_idx: int) -> tuple[Tensor, Tensor, Tensor]: ...
    def get_norms(self, block_idx: int) -> tuple[Tensor, Tensor]: ...
```

`SpikingKikiWML.step()` would branch: if `provider` satisfies
`SpikingKikiBlockProvider`, run the full forward; else fall back to the
v1 collapsed projection.

## Spike-native attention sketch

Per-tick t over T = `n_micro_ticks`:

1. **Input spikes** s_in ∈ {0, 1}^h driven by `embed_inbound(neuroletter)`
   through `input_proj`.
2. Compute Q, K, V via mmapped matmuls. To stay spike-native, accumulate
   into a leaky integrator on each head dimension rather than a softmax.
3. **Attention via belief accumulation** (Palacios SNN-PC compatible):
   each output position holds a (μ, σ) belief; spikes drive the residual.
   On the final micro-tick, emit the top-1 token from
   `argmax_z μ_attn(z)`.

Open: does the spikingkiki upstream use linear attention (state-space
recursion) or rate-coded softmax? `convert_spikingkiki_35b.py` audit
required before locking the formula.

## MoE routing sketch

Two paths to evaluate empirically:

- **Spike-route:** route per-token by accumulating router logits into a
  LIF, top-k experts = first k to fire. Cheaper, but tail experts may
  never fire at low T.
- **Rate-route:** compute router logits in the rate domain on the last
  micro-tick, top-k argmax. Loses the spike contract but matches the
  upstream's per-token routing.

Decision criterion: parity test — output cosine similarity vs the
micro-kiki ANN reference forward on a frozen 100-prompt eval set.
Target ≥ 0.85 cos sim for the same expert set, ≥ 0.95 with router on.

## Test plan

1. **Protocol conformance** — `SpikingKikiBlockProvider` runtime-checkable;
   `MmapNpzProvider` extended to satisfy it.
2. **Unit, small bundle** — synthetic 2-expert / 1-head block built by
   the existing `_write_bundle` test helper, drives a single step.
3. **Numerical parity** — against a torch.nn reference module loading
   the same .npz weights with float32 rate-mode (no LIF). Cos sim ≥ 0.85.
4. **Bench** — 100 prompts × 3 experts-per-token (top-3), measure cos
   sim vs ANN reference and tokens/sec on M3 Ultra / KXKM 4090.

## Carve-out

This PR should NOT change:

- The v1 `SpikingKikiWML.step()` codepath for the scaffold collapsed
  projection (used by the existing 12 unit tests).
- The `SpikingKikiNerveAdapter` API in baby-brain.
- The `MmapNpzProvider` v1 surface (`hidden_dim`, `n_blocks`,
  `block_indices`, `get_block_projection`).

## Open questions for the implementing session

- Audit `convert_spikingkiki_35b.py` first: confirm naming, head count,
  expert count, normalisation kind (RMSNorm vs LayerNorm), residual
  connection placement, KV cache shape.
- Decide spike-route vs rate-route for the MoE router empirically.
- Decide whether the spike-attention output is the final token argmax or
  a Neuroletter-shaped `(role, code, weight)` triple — if the latter,
  this PR also unlocks the BioFieldWML coupling design (#6).

## Out of scope (defer further)

- Training the spikingkiki weights — this is inference-only.
- LoRA hot-swap on spikingkiki experts.
- Quantisation (the bundle is float32 .npz; int8 is its own PR).
- Multi-block forward — start with single-block, extend to stacked once
  parity is achieved.
