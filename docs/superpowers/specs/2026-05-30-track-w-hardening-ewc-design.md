# Track-W hardening — EWC vs rehearsal on a hard continual task

**Date**: 2026-05-30
**Status**: design (awaiting review)
**Chantier**: #2 (durcir gate-W)
**Scope**: nerve-wml only — no baby-brain coupling.

## Context

gate-W is already passed (tag `gate-w-passed`): MlpWML/LifWML polymorphie
gap plateaus at ~2–3 % for N≥32, and W4 continual-learning forgetting drops
from 100 % → 0 % via **rehearsal**. Two limitations weaken the scientific
claim:

1. **Rehearsal is the only continual-learning lever in the codebase.** There
   is no EWC / Fisher-based method. A single-method result cannot support a
   comparative claim.
2. **The W4 task is trivial.** `SplitMnistLikeTask` uses two *disjoint*
   class sets ({0,1} then {2,3}) with separate output heads — forgetting is
   avoided *by construction*, not by the algorithm (spec §13.1 already flags
   this as a known shortcut).

This work adds a **harder continual task with a shared output head** and a
**second mitigation (EWC)**, then compares `none` / `rehearsal` / `ewc`
honestly. If EWC underperforms on the hard task, we report it as-is — no
gate is tuned to pass by construction (spec §13 "honest measurements"
principle).

## Goals / non-goals

**Goals**
- A continual task that produces catastrophic forgetting *without* mitigation
  on a **shared full-class head** (not disjoint heads).
- An EWC implementation (diagonal Fisher + quadratic penalty) usable by the
  existing MlpWML training loop.
- A reproducible, multi-seed comparison `none` vs `rehearsal` vs `ewc`.
- Honest reporting tied to the factcheck audit trail.

**Non-goals (YAGNI)**
- No online/streaming EWC, no synaptic intelligence, no GEM/A-GEM.
- No new substrates; MlpWML is the test vehicle (LifWML reuse is a follow-up).
- No change to the existing `gate-w-passed` criteria or to Track-P.

## Architecture & components

New isolated sub-package `track_w/continual/` keeps continual-learning
mechanisms in one place; pilots stay in `scripts/track_w_pilot.py`.

### 1. `track_w/tasks/hard_split.py` — `HardSplitTask`
- Two sequential sub-tasks over a **shared class space** (same output head,
  same lr for both), built from the existing `HardFlowProxyTask` geometry
  (XOR-on-noise, overlapping centroids).
- Contract: a vanilla shared-head learner trained Task 0 → Task 1 with **no
  mitigation** forgets ≥ 50 % of Task 0 accuracy (verified by a baseline
  test). This is the substrate on which mitigations are compared.
- Interface mirrors existing tasks in `track_w/tasks/` (same sample/label
  API consumed by the pilots).

### 2. `track_w/continual/ewc.py` — `EWC`
- `estimate_fisher(wml, data_loader) -> dict[str, Tensor]`: diagonal Fisher
  information over `wml.parameters()` after Task 0, from squared gradients of
  the log-likelihood.
- `penalty(wml, fisher, theta_star, lam) -> Tensor`: `Σ_i F_i (θ_i − θ*_i)²`,
  added to the Task 1 loss.
- Respects **W-2** (penalty ranges over `parameters()`, which includes the
  codebook) and **W-1** (no mutation of another WML).
- `lam` (regularisation strength) is the single tunable; default chosen by a
  small sweep, documented in the result JSON.

### 3. `track_w/continual/rehearsal.py` (refactor, optional)
- Extract the existing rehearsal mixing logic from `run_w4_rehearsal` into a
  reusable `RehearsalBuffer` so `none`/`rehearsal`/`ewc` share one training
  scaffold. Pure refactor — behaviour identical, existing W4 tests stay green.

### 4. `scripts/track_w_pilot.py` — `run_w4_compare`
- `run_w4_compare(method: Literal["none","rehearsal","ewc"], task, steps,
  seed, **kw) -> dict` returning `{forgetting, acc0_before, acc0_after,
  acc1, method, ...}`.
- Existing `run_w4_shared_head` / `run_w4_rehearsal` remain (back-compat).

## Data flow (one comparison run)

```
Task0 train ──► acc0_before
   │ (ewc) estimate_fisher(), snapshot θ*
   ▼
Task1 train  loss = L_task1 + {0 | rehearsal_mix | ewc.penalty}
   ▼
re-eval Task0 ──► acc0_after
forgetting = (acc0_before − acc0_after) / acc0_before
```

## Testing & gate

- `tests/integration/track_w/test_w4_hard_split_baseline.py` — `none` on
  `HardSplitTask` forgets ≥ 0.50 (proves the task is genuinely hard).
- `tests/integration/track_w/test_gate_w4_ewc.py` — EWC forgetting measured
  on `HardSplitTask`; **assert it is reported**, and (soft) that it beats the
  `none` baseline. The < 0.20 threshold is a target, not a construction.
- `tests/integration/track_w/test_w4_method_comparison.py` — multi-seed
  (n≥5) `none`/`rehearsal`/`ewc`; asserts both mitigations strictly improve
  on `none`, and records which wins on the hard task.
- Markers: `@pytest.mark.slow` for multi-seed. Run:
  `uv run pytest tests/integration/track_w/ -m slow -k w4`.
- All numeric thresholds traced to `docs/superpowers/research/2026-05-30-w4-ewc-comparison.json`
  via `scripts/factcheck_audit.py`.

## Invariants

Load-bearing W-1..W-4 unchanged. EWC adds a penalty term only; it does not
touch routing (W-3) or WML identity (W-4). Existing `gate-w-passed` tests
must stay green (non-regression).

## Risks

| Risk | Mitigation |
|------|------------|
| EWC λ hard to tune → looks worse than rehearsal | That is an honest, publishable result; sweep λ and report the frontier. |
| Refactor of rehearsal breaks W4 gate | Keep `run_w4_rehearsal` behaviour bit-stable; run existing W4 tests before/after. |
| Hard task too hard (both methods fail) | Tune `HardSplitTask` overlap so `none`≈catastrophic but task is learnable in isolation (acc0_before > 0.6). |

## Decisions (resolved 2026-05-30)

- **λ (EWC strength)**: deferred to implementation. The plan sweeps a small
  grid and records the chosen value + frontier in
  `docs/superpowers/research/2026-05-30-w4-ewc-comparison.json`. Not fixed in
  this spec.
- **Substrate scope**: **MlpWML only** for this first hardening PR. LifWML
  comparison is an explicit follow-up (reuses the same `run_w4_compare`
  scaffold), out of scope here to keep the PR reviewable.
