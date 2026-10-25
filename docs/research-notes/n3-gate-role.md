# N-3 gate role: typing guard established, causal utility not established

**Status:** reopened 2026-10-25 — conclusion requalified (erratum below,
issue #71); originally closed 2026-04-21 under the title
"type-checker, not differential mechanism".
**Context:** nerve-wml Paper 1, Voie A investigation (post-audit repositioning).
**Scripts:** `scripts/ablation_n3.py`, `scripts/ablation_n3_predictive.py`, `scripts/ablation_n3_guard.py`
**Data:** `papers/paper1/figures/ablation_n3*.json`

## Question

The γ/θ gate invariant N-3 (`role==ERROR ⟺ phase==THETA`) is specified in
`nerve_core/invariants.py` as a computational transcription of the
Bastos-Friston 2012 canonical cortical microcircuit. Does it have a
measurable empirical effect on the nerve protocol, or is it purely a
formal specification?

## Three convergent ablations

### 1. `ablation_n3.py` — w2_hard canonical pipeline

Canonical runners `run_w2_hard_n{16,32,64}_multiseed` run with phase fixed
to `gamma=True, theta=False` throughout training; `wml.step(nerve, t)` is
never called. Toggle `strict_n3` via context-manager monkey-patch.

**Result:** Δ = 0.0000 exactly across 3 N-points × 5 seeds.
**Interpretation:** no ε letters are ever emitted → the invariant has
nothing to validate.

### 2. `ablation_n3_predictive.py` — full γ → θ → consolidate loop

Run 2 MlpWMLs with random Gaussian inputs, calling `wml.step(nerve, t)`
in γ phase (WMLs naturally emit ε via `emit_head_eps` when
`(h − h_prior).norm() > threshold_eps=0.30`), switch to θ phase and
collect ε trace via `DreamBridge.collect_eps_trace`, apply zero-delta
consolidation.

**Result:** Δ = 0 exact across 3 seeds. `theta_trace_len = 200` in
both conditions (200 ε letters collected per 200 ticks — the ε
emission path is exercised). **However** `apply_consolidation: false`
in all six cells of `ablation_n3_predictive.json`, with
`apply_err: "'MockNerve' object has no attribute '_transducers'"`:
the consolidation step never ran, so no learning signal existed that
the flag could have changed.
**Interpretation:** WMLs hard-code `phase=Phase.THETA` when emitting ε
(see `track_w/mlp_wml.py:124`). The invariant is therefore satisfied
structurally. strict=True vs strict=False have no observable difference
on a correct pipeline.

### 3. `ablation_n3_guard.py` — violation injection

Inject malformed Neuroletters (`role=ERROR, phase=GAMMA` — violation
of N-3) at varying rates v ∈ {0.0, 0.05, 0.10, 0.25, 0.50}, N=1000
letters, 3 seeds, measure catch rate (strict) and silent rate (open).

**Result:** strict catches N·v violations within binomial noise
(σ ≈ √(N·v·(1-v))); e.g. `n_caught: 70` at v = 0.05, seed 0
(`ablation_n3_guard.json`), mean 57.0 for an expected 50. Open lets
them pass silently. Both modes conform to their expected semantics.
**However** `n_received: 0` in all 30 cells: `listen()` returned
nothing, so reception was inactive during the test.
**Interpretation:** the guard is exercised at `send()`
(`scripts/ablation_n3_guard.py:80-87`) and catches violations; the
test says nothing about what a receiver would have done with them.

## Conclusion (requalified 2026-10-25)

- **Established:** the typing function of N-3. The guard is exercised
  at send time and catches malformed letters (ablation 3).
- **Not established by these ablations:** any causal utility of N-3
  for communication or learning. In ablations 1–3 the mechanism the
  flag could have influenced was inactive (no ε letters in ablation 1,
  consolidation never executed in ablation 2, `n_received: 0` in
  ablation 3), so Δ = 0 is uninformative rather than evidence of
  "no differential effect".

### Erratum (2026-10-25)

The original conclusion of this note — "N-3 is a formal correctness
contract (type-checker semantics), not a differential computational
mechanism" — over-read a Δ = 0 measured on an inactive mechanism.
Root cause: the ablation JSONs carry `n_received: 0`
(`ablation_n3_guard.json`) and `apply_consolidation: false`
(`ablation_n3_predictive.json`), which were not read as
disqualifying at the time. Testing the differential question requires
an ablation with `n_received > 0` and active consolidation,
pre-registered (L1) with a matched null. Paper 1 §Method and §Threats
to Validity, `CHANGELOG.md` and the README carry the same erratum.
Source of the correction: Hypneum Lab hypothesis review of
2026-10-24, revised 2026-10-25 (§2.2, annex 12 point M2).

This does **not** invalidate the Bastos-Friston 2012 ancestry:
biological γ/θ multiplexing is also a correctness-enforcing constraint
(spike-timing-dependent gating), not a free parameter that tunes
computational behaviour. The analogy holds at the level of structural
constraint.

## Implications

1. **Paper framing.** The γ/θ multiplexing claim is framed as a
   *typing guard whose mechanistic role is untested* in §Method and
   §Threats to Validity (erratum 2026-10-25). Historical framing as a
   *structural guarantee*: commits `dfbdbee` + follow-up integrating
   the three ablations.

2. **Downstream consumers** (`bouba_sens`, `dream-of-kiki`): `strict_n3`
   is the default and correct posture. Disabling it only makes sense
   during composition debugging where silent violations would mask bugs
   upstream.

3. **Genuine future work.** A differential γ/θ mechanism would require
   the gate itself to interact with learnable parameters (e.g.
   θ-phase-gated consolidation weight), which is an extension beyond
   N-3's current scope — not an ablation of it.
