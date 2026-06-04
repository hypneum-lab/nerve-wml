# N14 — Latin Hypercube 7-dim coverage (milestone, pre-registered)

**Date pre-registered:** 2026-05-11
**Spec:** `HYPNEUM-PLANS/2026-05-11-niveau13-14-systematic-exploration.md`
**Pre-reg:** `HYPNEUM-PLANS/preregistrations/n14_latin_hypercube_7dim.md`
**Status:** Pre-registered, sweep NOT yet executed.

## Summary

Latin Hypercube uniform sampling of the 7-dim GTM design space :
`alphabet_size`, `n_symbols`, `Gumbel tau`, `code_dim`,
`plasticity_decay_type`, `gamma_hz`, `theta_hz`. N=50 LHC points × 5
seeds = 250 runs ; HardFlowProxyTask N=2, 800 steps ; ~21h wallclock
on kxkm-ai 4090 (CPU-bound).

## H0 (to refute)

The 7-dim design space contains at least one configuration that
produces statistically superior `mi_h` vs the N8/N9 default (Welch's
t-test, Bonferroni-corrected α = 0.05/50 = 0.001). I.e., the default
is NOT Pareto-optimal on `mi_h`.

## Statistical analysis

- Per LHC point : 5-seed mean of each metric.
- Welch's t-test (5 seeds vs 5 default seeds) on `mi_h` ; report
  `round_trip_fidelity` and `bandwidth_efficiency` for context.
- Bonferroni α = 0.001 (50 tests).
- "Winner" = significant mean-shift in favor of higher `mi_h`.

## Decision tiers (pre-stated)

- **N14-default-suboptimal** (≥1 winner) : tuned config exists ;
  feed forward to Paper 2 §X.Y or N15 Optuna seed.
- **N14-default-near-optimal** (0 winners) : robust to choice ;
  defensible "no extensive tuning" claim.
- **N14-many-winners** (≥10 winners) : default clearly suboptimal ;
  triggers N15 Bayesian Opt sprint with confidence.

## Compute / risk

- kxkm-ai 4090 CPU ; Granite 30B can stay UP.
- Risks : categorical `plasticity_decay_type` not LHS-friendly
  (use floor-on-continuous snap) ; snapping continuous → power-of-2
  reduces effective sample diversity ; Bonferroni α=0.001 with
  N=5 seeds requires Cohen's d ≥ 3.5 (massive effect) — small
  effects will be missed.

## Cross-reference

- Methodology sketch : `HYPNEUM-PLANS/2026-05-11-n13-n14-methodology-sketch.md`
- Sister sprint : `n13-ofat-gtm-internals-2026-05-11.md`
- Future follow-up : N15 Bayesian Opt (Optuna), N16 DoE Plackett-Burman.
- Implementation : reproduction artefacts will land at
  `experiments/n14_lhc_gtm/` once the sprint is executed.
