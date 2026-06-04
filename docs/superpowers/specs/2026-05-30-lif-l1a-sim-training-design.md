# LifWML L1a — Full-LIF sim-training design

**Date**: 2026-05-30
**Status**: design (awaiting review)
**Chantier**: L1a (WML Full-LIF neuromorphique, sim-first)
**Scope**: nerve-wml training only — no NIR export, no hardware, no baby-brain coupling.

---

## Context

The current `LifWML` (`track_w/lif_wml.py:20`) is a toy substrate with
`n_neurons=16–100`, a `codebook` of size 64, and a fixed cosine-match decoder.
It participates in the W2 polymorphism gate but has never been benchmarked as a
standalone learner on a non-trivial task: in every W2 variant it trains *alongside*
an MLP reference and is evaluated for gap, not absolute learning ability.

The surrogate-gradient infrastructure is complete: `track_w/_surrogate.py:16`
implements a `_SpikeFn` with fast-sigmoid backward
(`α / (π · (1 + (α·(v − v_thr))²))`). The end-to-end `input_encoder →
input_proj → spike → emit_head_pi` pipeline is already exercised in
`scripts/track_w_pilot.py:479–488` (`run_w2_hard`). The pilot, however, never
reports loss curves or multi-epoch convergence; it returns a single accuracy
number against an MLP baseline.

**L1a asks a strictly scoped question**: can a LifWML that is *scaled up* learn a
non-trivial classification task from scratch, in simulation, using surrogate
gradients — and can we prove it by showing monotone loss descent?

Compute target: **macM1** (ssh macm1, 16 GB, CPU-only PyTorch). No CUDA,
no MLX, no GPUs. All training invocations in L1a run locally on macM1
(or via `ssh macm1 uv run python ...`). This is a deliberate constraint:
L1b (NIR/INT8 export) and L1c (baby-brain coupling) are out of scope.

---

## Goals / non-goals

**Goals**
- Scale `LifWML` to a capacity that can learn `HardFlowProxyTask` (12-class
  XOR-on-noise, ~8 % chance) through surrogate-gradient optimisation alone.
- Design a self-contained training loop (no `MockNerve` required, task-fed
  directly) and a gate that proves learning without cherry-picking.
- Emit a JSON artefact traceable by `factcheck_audit.py`.

**Non-goals (explicitly out of scope for L1a)**
- NIR export or INT8 quantisation (L1b).
- Hardware deployment on neuromorphic silicon (L1b / Phase 2).
- Integration with baby-brain or any embodied/multimodal input (L1c).
- Recurrent membrane state across samples (temporal BPTT) — L1a uses
  single-sample T-tick unrolling, no inter-sample state persistence.
- Multi-neuron populations communicating via `SimNerve` (GATE M scope).
- Spike-timing coding; rate coding over T ticks is sufficient for L1a.

---

## Architecture

### 3.1 Scaled LifWML

The current toy defaults are `n_neurons=16`, `alphabet_size=64`,
`input_dim=n_neurons`. L1a uses:

| Hyperparameter | Toy (current) | L1a target |
|---|---|---|
| `n_neurons` | 16 | **1024** |
| `alphabet_size` | 64 | 64 (unchanged — codebook is not the bottleneck) |
| `input_dim` | 16 (= task dim) | **64** (see §3.2) |
| `threshold_eps` | 0.30 | 0.30 (unchanged) |
| `tau_mem` | 20 ms | 20 ms (unchanged) |
| `v_thr` | 1.0 | 1.0 (unchanged) |

Rationale: `n_neurons=1024` is the L1a scaling target. A sweep over
`{128, 256, 512, 1024}` is run to measure the capacity curve (acc vs n_neurons)
and confirm that the gain is monotone. `input_dim=64` decouples task-input
dimension from neuron count more aggressively than the prior 32-dim value
to give the encoder more expressivity at large n_neurons.

### 3.1b Scaling sweep

An explicit capacity experiment runs training for each of:

```
n_neurons ∈ {128, 256, 512, 1024}
```

For each size: same hyperparameters (T=8, Adam lr=3e-3, N_STEPS=2000, batch=64,
seed=0), same HardFlowProxyTask. The sweep measures:

- `acc_trained` — final accuracy after N_STEPS
- `acc_untrained` — accuracy before any training (random weights)
- `loss_start` — loss at step 1 (after first Adam step)
- `loss_end` — loss at step N_STEPS

Results are written to
`docs/superpowers/research/2026-05-30-lif-l1a-scaling.json` as an array keyed
by n_neurons. The curve answers: "is there a monotone capacity benefit to
scaling n_neurons on HardFlowProxyTask?"

`emit_head_pi` (`track_w/lif_wml.py:73`) maps `n_neurons → alphabet_size`.
For L1a the task has 12 classes; we read `logits = emit_head_pi(spikes)[:, :12]`
exactly as `run_w2_hard` does at line 487.

**No structural changes to `LifWML`**. Scaling is achieved purely via constructor
arguments. This satisfies YAGNI and preserves invariants W-1..W-4.

### 3.2 Input encoding

Task: `HardFlowProxyTask(dim=16, n_classes=12)` (`track_w/tasks/hard_flow_proxy.py:25`).
Chosen because:
- 12 classes, ~8 % chance accuracy — non-trivial for a spiking substrate.
- XOR-on-noise nonlinearity requires nonlinear representation beyond a linear probe.
- Already used in `run_w2_hard` and pool-scale pilots; dataset generation is
  reproducible and dependency-free.
- **Not** raw MNIST pixels: avoids extra preprocessing and macM1 download on
  first run.

Input pipeline:
```
x ∈ ℝ^{16}  →  nn.Linear(16, 64)  →  LifWML.input_proj(·)  →  spikes ∈ {0,1}^{N}
                input_encoder            nn.Linear(64, N)
```
where N = n_neurons (128, 256, 512, or 1024 in the sweep).

`input_encoder` is a separate `nn.Linear(16, 64)` trained end-to-end alongside
LifWML parameters (same pattern as `run_w2_hard`, `track_w_pilot.py:253`).
The intermediate 64-dim projection decouples task-input dimension from neuron
count and is the only new component. `input_dim=64` is passed to `LifWML`
so `input_proj` is `nn.Linear(64, N)`.

### 3.3 Spike generation (T-tick unrolling)

For each mini-batch sample we unroll T membrane integration steps:

```
for t in 0..T-1:
    i_in   = input_proj(input_encoder(x))          # ℝ^{256}
    v_mem  = v_mem + dt/tau * (-v_mem + i_in)      # LIF: track_w/lif_wml.py:95
    spikes = spike_with_surrogate(v_mem, v_thr)    # _surrogate.py:32
    v_mem  = v_mem * (1 - spikes)                  # hard reset
```

**T = 8 ticks** is the L1a default. Rationale: enough for the membrane to
integrate and fire at least once for most inputs at `n_neurons=256`, while
keeping per-step cost linear in `n_neurons`. At macM1 CPU with batch=64,
8 ticks × 256 neurons is within the few-millisecond-per-step regime.

Spike accumulation across ticks: `spikes_sum = Σ_t spikes_t ∈ ℝ^{256}`.
Classifier input: `logits = emit_head_pi(spikes_sum)[:, :12]`.

The membrane is **reset to zero between samples** (`reset_state()`, line 82).
No inter-sample recurrence — simplest possible semantics for L1a.

### 3.4 Training loop

```python
# Pseudocode — final script: scripts/lif_l1a_train.py
# N = n_neurons (1024 for the primary run; {128,256,512,1024} for the sweep)
optimizer = Adam(list(lif.parameters()) + list(input_encoder.parameters()), lr=3e-3)

for step in range(N_STEPS):          # N_STEPS = 2000 (L1a default)
    x, y = task.sample(batch=64)
    lif.reset_state()

    v_mem = torch.zeros(batch, N)
    spikes_acc = torch.zeros(batch, N)   # batch × n_neurons
    for t in range(T):                  # T = 8
        i_in = lif.input_proj(input_encoder(x))
        v_mem = v_mem + dt / tau * (-v_mem + i_in)
        spikes = spike_with_surrogate(v_mem, v_thr=lif.v_thr)
        v_mem = v_mem * (1 - spikes)
        spikes_acc = spikes_acc + spikes

    logits = lif.emit_head_pi(spikes_acc)[:, :12]
    loss = F.cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()          # surrogate backward flows through spike_with_surrogate
    optimizer.step()
```

Optimizer: `Adam`, `lr=3e-3`. Rationale: `run_w2_hard` uses `lr=1e-2`; SNN
training is more sensitive to learning rate — 3e-3 is a conservative starting
point. If the loss does not descend in the first 200 steps, the implementer
should try `lr=1e-2` and document the result.

`N_STEPS=2000` at `batch=64` corresponds to ~128 k samples, sufficient for
`HardFlowProxyTask` to be covered many times over (the task is synthetic and
infinite). This matches the step budget used for pool-scale experiments.

Logging: every 100 steps emit `{"step": s, "loss": float, "acc": float}` to
`docs/superpowers/research/2026-05-30-lif-l1a-train.jsonl`. The
`factcheck_audit.py` harness requires a JSON cell for every numerical claim.

---

## Data flow (end-to-end)

```
HardFlowProxyTask.sample(batch=64)
        │
        │ x ∈ ℝ^{64×16}, y ∈ ℤ^{64}
        ▼
nn.Linear(16, 64)  [input_encoder]
        │
        │ ℝ^{64×64}
        ▼
LifWML.input_proj  [nn.Linear(64, N)]      ← lif_wml.py:66  (N = n_neurons)
        │
        │ ℝ^{64×N}
        ▼
T=8 × spike_with_surrogate(v_thr=1.0)     ← _surrogate.py:32
        │
        │ spikes_acc ∈ ℝ^{64×N}  (sum of binary spike tensors, grad flows)
        ▼
LifWML.emit_head_pi  [nn.Linear(N, 64)]   ← lif_wml.py:73
        │
        │ logits[:, :12]  ∈ ℝ^{64×12}
        ▼
F.cross_entropy(logits, y)  → scalar loss
        │
        ▼
Adam.step()  [∂loss/∂params via fast-sigmoid surrogate]
```

No MockNerve, no Neuroletter, no routing. The nerve protocol is deliberately
bypassed: L1a is a *substrate* proof-of-learning, not a protocol experiment.

---

## Testing and gate (L1a gate)

### 5.1 Correctness tests (L1 — unit)

- `LifWML(n_neurons=256, input_dim=32)` constructs without error.
- `forward` (T-tick loop) produces gradients on all parameters (no grad
  detachment regression).
- `spike_with_surrogate` backward is non-zero at `v ≈ v_thr` (already tested
  implicitly by existing suite).

### 5.2 Learning gate (measured, honest)

**Gate condition**: a single LifWML trained for 2000 steps on
`HardFlowProxyTask(dim=16, n_classes=12, seed=0)` achieves **all** of:

1. Final accuracy ≥ 20 % (baseline chance = 8.3 %).
2. Loss at step 2000 < loss at step 100 (monotone descent confirmed over the
   training window, not just final).
3. Untrained LifWML (same architecture, step=0) accuracy ≤ 10 % on the same
   eval batch — proves the gain is learned, not structural.

The gate is intentionally loose (20 % vs 8 % chance): SNN convergence on a
XOR task is non-trivial and over-specifying the target creates a risk of gate
gaming. If accuracy plateaus below 20 %, the implementer must report it
honestly and open an issue — the gate fails, the science is not hidden.

**Evidence format**: emit to
`docs/superpowers/research/2026-05-30-lif-l1a-gate.json`:
```json
{
  "task": "HardFlowProxyTask",
  "n_classes": 12,
  "n_neurons": 1024,
  "input_dim": 64,
  "T_ticks": 8,
  "n_steps": 2000,
  "seed": 0,
  "acc_untrained": <float>,
  "loss_step100": <float>,
  "loss_step2000": <float>,
  "acc_final": <float>,
  "gate_passed": <bool>
}
```

The scaling sweep emits `docs/superpowers/research/2026-05-30-lif-l1a-scaling.json`
with per-n_neurons entries (see §3.1b).

Multi-seed robustness (n=3, seeds 0/1/2) is **recommended** but not required
for gate passage. If only seed=0 is run, the claim must be qualified as
single-seed in any paper text.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Surrogate gradient vanishing / loss stuck | High | Tune `alpha` in `_SpikeFn` (default 2.0, try 4.0–8.0); reduce `lr` to 1e-3 if spiky |
| Dead neuron collapse (`spikes_acc ≡ 0` for all inputs) | Medium | Log mean spike rate per tick; if < 1 % saturate `v_thr` to 0.5 |
| T-tick unrolling too slow on macM1 CPU | Low | T=8 × batch=64 × n_neurons=256 is ~3 M FLOPs/step; 2000 steps ≈ minutes not hours |
| Scaling breaks W-1..W-4 invariants | Low | Only constructor arguments change; no structural modifications to `LifWML`; run existing test suite before committing results |
| Loss spiky (oscillating) due to surrogate sharpness | Medium | Use gradient clipping (`clip_grad_norm_`, max_norm=1.0) as first-line fix |
| macM1 unavailable (SSH down) | Low | Fall back to grosmac CPU for development; macM1 is the stated compute target for runs that will be cited |

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Task | `HardFlowProxyTask(dim=16, n_classes=12)` | Non-trivial (XOR + 12 classes), already in codebase, no download required |
| n_neurons target | 1024 | 64× toy size; fits macM1 16 GB; sweep {128,256,512,1024} measures capacity curve |
| input_dim | 64 | Decouples task input from neuron count; more expressivity than 32-dim |
| Scaling sweep | {128, 256, 512, 1024} | Acc vs n_neurons capacity curve; answers whether scaling helps |
| T ticks | 8 | Enough for integrate-and-fire at scale; macM1 CPU-safe |
| Optimizer | Adam, lr=3e-3 | Conservative vs existing lr=1e-2; SNN more sensitive |
| N_STEPS | 2000 | Sufficient coverage of synthetic task; matches pool-scale pilots |
| Gate threshold | acc ≥ 20 % + loss descent | Honest lower bound; 20 % = 2.4× chance; avoids over-specifying |
| Compute target | macM1 (CPU PyTorch) | Stated constraint; no MLX/CUDA in L1a |
| No MockNerve | Direct task→LifWML | L1a is substrate proof, not protocol experiment |
| No structural change to LifWML | Constructor args only | Preserves W-1..W-4; smallest viable change |

---

## Out of scope (explicit)

- **L1b**: NIR graph export, INT8 quantisation, hardware-agnostic execution via
  `nir` / `snntorch` bridges.
- **L1c**: integration with baby-brain, multimodal input, embodied simulation
  loop.
- **Phase 2**: deployment on neuromorphic hardware (Intel Loihi, SpiNNaker,
  BrainScaleS).
- Any WML substrate other than `LifWML` (MLP, Transformer, BioField comparisons
  belong to Track-W polymorphism gates already passed).
- Temporal BPTT across samples (recurrent state between mini-batches).
- Multi-population nerve communication.
