# Neuromorphic deployment guide

Plan 6 ships the **export layer** for `LifWML` → INT8 artefact → pure-numpy mock runner. Two vendor stubs exist: `akida_stub` (BrainChip Akida; wiring possible once the SDK is installed) and `loihi_stub` (Intel Loihi 2; **historical interface only** — the `lava-nc` SDK is archived upstream, see below, and Loihi 2 is not a deployment target). No vendor SDK is a runtime dep of `nerve-wml` itself; both stubs raise `NotImplementedError`.

## Pipeline overview

```
LifWML (torch)
    │
    ▼  quantize_lif_wml (INT8 symmetric per-tensor)
    │
    ├─► artefact.json + weights.npz     (serializable, bit-stable)
    │       │
    │       ├─► MockNeuromorphicRunner  (pure numpy — CI path)
    │       │       │
    │       │       └─► compare_software_vs_neuromorphic (delta check)
    │       │
    │       ├─► LoihiCompiler.compile() (always raises — lava-nc archived, no Loihi 2 path)
    │       │
    │       └─► AkidaCompiler.compile() (→ akida SDK, not installed by default)
```

## Install the vendor SDKs

### Loihi 2 (Intel `lava-nc`) — not a supported target

**Status (2026-10-25, phase 0 requalification, issue #64).** Intel's Lava
repositories (`lava-nc/lava`, `lava-nc/lava-dl`) are archived on GitHub
(`archived: true` via the GitHub API, checked 2026-10-25; last push
2026-05-13). nerve-wml never compiled anything for Loihi 2, has no
Loihi 2 hardware or Intel NRC access, and does not promise a Loihi 2
deployment path. `neuromorphic/loihi_stub.py` is kept for import
compatibility only; `LoihiCompiler.compile` always raises.

The recipe this guide used to give is retained below for the record. It
was never executed and is **not** a supported integration. Any future
neuromorphic target (SpiNNaker2, a NIR-based export, …) is a new
decision for the PI, not an extension of this stub.

<details>
<summary>Historical recipe (unsupported; archived SDK)</summary>

```bash
pip install lava-nc   # archived upstream — not maintained
```

Illustrative only, written against the pre-archive `lava-nc` API:

```python
from lava.magma.compiler.compiler import Compiler
from lava.proc.lif.process import LIF

def compile(artefact: dict) -> "Executable":
    # Build a LIF process graph from the artefact.
    graph = LIF(
        shape=(artefact["n_neurons"],),
        vth=artefact["v_thr"],
        du=1.0 / artefact["tau_mem"],
        weights=artefact["input_proj_int8"] * artefact["input_proj_scale"],
    )
    return Compiler().compile(graph, board="Oheo Gulch")
```

</details>

### Akida (BrainChip)

```bash
pip install akida
```

Akida consumes a slightly different model format; you'll typically wrap the artefact into a `akida.Model` subclass and call `.compile(input_shape=...)`.

## Artefact format (schema v0)

`artefact.json` holds scalars:

```json
{
  "codebook_scale":    0.0078125,
  "input_proj_scale":  0.0015625,
  "v_thr":             1.0,
  "tau_mem":           0.02,
  "n_neurons":         16,
  "alphabet_size":     64,
  "bits":              8
}
```

`weights.npz` holds arrays:

- `codebook_int8`      — shape `[alphabet_size, n_neurons]`, dtype `int8`
- `input_proj_int8`    — shape `[n_neurons, n_neurons]`, dtype `int8`
- `input_proj_bias`    — shape `[n_neurons]`,             dtype `float32`

Dequantization: `w_fp32 = w_int8.astype(np.float32) * scale`. Maximum quantization error is bounded by the scale.

## Round-trip verification

The `compare_software_vs_neuromorphic(wml, inputs, artefact)` helper returns:

```python
{
  "pytorch_codes":      np.ndarray[B],  # codes from torch LIF
  "neuromorphic_codes": np.ndarray[B],  # codes from MockNeuromorphicRunner
  "agreement":          float,          # fraction with identical code
  "delta":              float,          # 1 - agreement
}
```

Gate `gate-neuro-passed` (see `scripts/neuro_pilot.py`) asserts:

- Round-trip save/load is bit-stable.
- Rate encoder output rate matches input within ±10 %.
- PyTorch↔mock delta < 25 % on untrained LifWML (headroom for trained gains).

## When hardware is procured

1. Replace the `NotImplementedError` body in `akida_stub.py` (the Loihi 2 stub has no supported SDK behind it — see above).
2. Extend `scripts/neuro_pilot.py` with a `run_akida_gate()` that exercises the real compiler.
3. Compare hardware accuracy vs `MockNeuromorphicRunner` — the mock is the repeatable CI reference; hardware deviations flag real-chip-specific quantization or timing effects.

## Follow-up

- Trained LifWML checkpoints: the honest delta baseline (~19 %) uses an untrained LifWML. Plan 4a's `run_w2_true_lif` produces trained LIFs; extending `gate-neuro` to consume a trained checkpoint would tighten the delta to the 1-5 % range expected on hardware.
- Rate vs temporal coding: `rate_encode` and `temporal_encode` are both shipped. Hardware deployment benchmarks should compare which coding gives tighter pytorch↔chip agreement on the target task.
