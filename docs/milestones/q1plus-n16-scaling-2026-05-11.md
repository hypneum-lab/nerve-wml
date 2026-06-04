# Q1+ — GammaThetaMultiplexer at HardFlowProxyTask N=16 scaling

**Date pre-registered:** 2026-05-11 (BEFORE sweep launch on kxkm-ai)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau9-scaling-experiments.md
**N8 cross-reference:** Q1 verdict `tied` at N=2 (commit nerve-wml a6ddcba 2026-05-10).
**Status:** Pre-registered, not yet executed.

## H0 (to refute)

The Q1 `tied` verdict at HardFlowProxyTask N=2 generalizes to N=16. Either GTM matches latent-space baselines on aggregate (tied again at scale), OR PAC phase-coupling provides a scale-emergent advantage that did not surface at N=2.

## Methodology

- Same 4 architectures as Q1 (GTMBridge, RecursiveLinkBridge, MLPBridge, CrossAttentionBridge), same Bonferroni α=0.05/9=0.00556
- Same 5 seeds (0, 17, 42, 73, 101) for paired comparison vs N=2
- Same 3 metrics (round_trip_fidelity, mi_h, bandwidth_efficiency)
- Task change: HardFlowProxyTask **N=16** (n_classes=16 with XOR-on-noise structure ; existing tests/integration/track_w/test_w2_hard_scale.py confirms task at N=16 is well-defined)
- Wallclock estimate: 4 archs × 5 seeds × N=16 ~ 4× the N=2 cost. T13 ran in ~5min on electron-server CPU 8-core ; on kxkm-ai 28-core CPU expect ~30-60 min.
- Designed to launch on **kxkm-ai** CPU (not GPU — model still small enough that GPU offers no advantage and kxkm-ai GPU is occupied by Granite 30B).

## Decision criteria (pre-stated)

- **GTM-headline-at-scale:** GTM wins ≥6/9 comparisons at N=16 (Bonferroni-corrected) → Paper 2 §7.9 pivots from "convergent evidence" to "scale-emergent PAC advantage", ablation refresh recommended for v0.3 with non-trivial plasticity_schedule, A2 TMLR cover letter cites this.
- **Tied-stable:** 3-5 wins → §7.9 retains "convergent evidence" with N-invariance corollary (PAC advantage is task-property-driven, not scale-driven). N8 verdict robust across scale.
- **GTM-loses-at-scale:** GTM wins ≤2 OR loses ≥6 → §7.9 pivots to "PAC advantage degrades at scale", honest scope-limitation note ; Paper 2 narrative shifts to "PAC effective at small N=2 ; large-N requires different bridge".

## Result (executed 2026-05-11)

Sweep ran on kxkm-ai CPU (28-thread), wallclock **18 seconds total** for 20 runs (4 archs × 5 seeds × 800 steps each). Far below the 30-60min estimate ; tiny networks make CPU saturation trivial.

Results raw: `experiments/benchmark_multiplexer_vs_baselines/results_n16.json`.
Figure: `papers/paper2/figures/multiplexer_benchmark_n16.png`.

### Welch's t-test pairwise (Bonferroni α = 0.0056)

| Baseline | Metric | t | p | GTM mean | base mean | Outcome |
|----------|--------|---|---|----------|-----------|---------|
| RecursiveLink | rtf | -13.94 | 1.5e-4 | 0.6227 | 0.9908 | loss |
| RecursiveLink | mi_h | +31.66 | 1.1e-8 | 1.2330 | 0.4570 | win |
| RecursiveLink | bw_eff | -34.64 | 5.3e-10 | 0.1625 | 0.9125 | loss |
| MLP | rtf | -13.68 | 1.3e-4 | 0.6227 | 0.9871 | loss |
| MLP | mi_h | +24.50 | 8.7e-9 | 1.2330 | 0.4963 | win |
| MLP | bw_eff | -33.52 | 1.3e-9 | 0.1625 | 0.8250 | loss |
| CrossAttn | rtf | +6.14 | 3.3e-4 | 0.6227 | 0.4147 | win |
| CrossAttn | mi_h | +1.96 | 0.113 | 1.2330 | 1.0668 | tie |
| CrossAttn | bw_eff | -5.20 | 8.3e-4 | 0.1625 | 0.2750 | loss |

Wins / losses / ties: 3 / 5 / 1.

### Verdict

**`tied-stable`** per pre-stated decision criteria. Identical 3/5/1 breakdown to N8 Q1 N=2 baseline → the convergent-evidence narrative from Paper 2 §7.9 survives at N=16 scale. The N-invariance corollary (PAC advantage is task-property-driven, not scale-driven) is empirically supported for n_classes ∈ {12, 16}.

### Paper 2 update

dream-of-kiki commit (next) updated `docs/papers/paper2/outline.md` §7.9 with the cross-condition robustness paragraph.
