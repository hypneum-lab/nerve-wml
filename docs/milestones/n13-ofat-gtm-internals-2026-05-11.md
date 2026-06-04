# N13 — OFAT GTM internals (milestone, pre-registered)

**Date pre-registered:** 2026-05-11
**Spec:** `HYPNEUM-PLANS/2026-05-11-niveau13-14-systematic-exploration.md`
**Pre-reg:** `HYPNEUM-PLANS/preregistrations/n13_ofat_gtm_internals.md`
**Status:** Pre-registered, sweep NOT yet executed.

## Summary

One-Factor-At-A-Time exploration of 5 GTM internal hyperparameters
(`alphabet_size`, `n_symbols`, `Gumbel tau`, `code_dim`,
`plasticity_schedule_type`) at 4-5 levels each, 5 seeds per level,
HardFlowProxyTask N=2, 800 steps. Total = 105 runs ; ~9h wallclock
on kxkm-ai 4090 (CPU-bound for GTMBridge ~17K params).

## H0 (to refute)

At least 3 of the 5 hyperparams produce a statistically significant
per-level differential effect on the 3 paper metrics (`mi_h`,
`round_trip_fidelity`, `bandwidth_efficiency`) at Bonferroni-corrected
α = 0.0033 (15 hyperparam-metric pairs).

## Statistical analysis

- Per-axis Welch one-way ANOVA across levels.
- Post-hoc Tukey HSD pairwise comparisons.
- Effect sizes : η² (per-axis) and Cohen's d (per-pair).
- Bonferroni : α = 0.05 / 15 ≈ 0.0033.

## Decision tiers (pre-stated)

- **N13-rich-effects** (≥3 sig) : GTM is hyperparam-sensitive.
- **N13-modest-effects** (1-2 sig) : GTM mostly robust ; one axis
  matters.
- **N13-no-effects** (0 sig) : default values defensible ; honest
  scope statement.

## Compute / risk

- kxkm-ai 4090 CPU ; Granite 30B can stay UP.
- Risks : OFAT misses interactions (mitigated by N14) ;
  HardFlowProxyTask saturation may flatten all curves ; default
  values may be near-optimal a priori.

## Cross-reference

- Methodology sketch : `HYPNEUM-PLANS/2026-05-11-n13-n14-methodology-sketch.md`
- Sister sprint : `n14-latin-hypercube-7dim-2026-05-11.md`
- Implementation : reproduction artefacts will land at
  `experiments/n13_ofat_gtm/` once the sprint is executed.
