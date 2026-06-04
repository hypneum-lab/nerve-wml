# Renf 13 — harder routing (alphabet=128, K=9)

**Date:** 2026-05-20
**Host:** macm1 (Apple M1, 10 workers)
**Script:** `scripts/renf13_harder_routing.py`
**Wall-clock:** 343.8 s (~5.7 min)
**Status:** DONE_WITH_CONCERNS (K capped at 9, not 14)

## Goal

Renf 7 used `alphabet=64, K=7`; all arms (GTM, simple_gating, AKOrN)
saturated at accuracy=1.0, MI=2.22 bits, so the arms could not be
distinguished on the *coder* axis — only on the *synchrony* axis. This
script probes a harder regime to see whether (a) the saturation breaks
and (b) GTM dominates on accuracy when the task is hard.

## Constraint discovered

`GammaThetaConfig` enforces the Lisman-Idiart capacity bound `K ≤ 9`.
The brief requested `K=14`; the script caps at 9 instead and documents
this. Pushing alphabet to 128 (× 2 vs Renf 7) and K from 7 → 9 is the
strongest regime that does not violate the configured invariant.
Pushing past K=9 would require weakening the invariant — refused.

## Headline results (20 seeds, 200 steps, alphabet=128, K=9)

| Arm             | Accuracy        | MI (bits)        | Spectral entropy |
|-----------------|-----------------|------------------|------------------|
| gtm             | **1.000 ± 0.000** | **3.548 ± 0.004** | 2.411 ± 0.006 |
| simple_gating   | **1.000 ± 0.000** | **3.548 ± 0.004** | 3.655 ± 0.014 |
| akorn_best      | 0.510 ± 0.363   | 3.173 ± 0.266    | 1.981 ± 0.610 |
| null            | 0.019 ± 0.013   | 2.885 ± 0.002    | 2.376 ± 0.036 |

`mi_max_per_symbol = log2(128) = 7.0` bits (over K=9 symbols the per-batch
joint MI would be higher; the reported value is the per-symbol MI through
the Miller-Madow estimator).

## Interpretation

- **GTM and simple_gating both still saturate** at accuracy = 1.000 and
  MI = 3.548 with effectively zero variance. Pushing the task harder
  (alphabet 64 → 128, K 7 → 9) did **not** break their performance.
  Both remain perfect coders.
- **AKOrN does *not* saturate at K=9**: accuracy drops to 0.510 ± 0.363,
  and the std=0.363 indicates a strongly bimodal distribution — some
  seeds learn perfectly, others not at all. Renf 12 saw the same shape
  at smaller alphabet. This is the **first regime where AKOrN clearly
  underperforms the GTM / simple_gating pair on a coder metric**.
- **Spectral entropy still separates GTM from simple_gating** (2.411 vs
  3.655) — the GTM ablation signature survives the harder task.
- **Null arm collapses** to accuracy 0.019, as expected.

## Verdict on the hypothesis

**PARTIALLY CONFIRMED.**

- (a) "Do all three arms still saturate at MI=2.22?" — **REFUTED**.
  AKOrN drops to 0.51 / 3.17 bits; GTM and simple_gating saturate at
  the new ceiling (3.55 bits).
- (b) "Does GTM dominate on accuracy when the task is hard?" —
  **NEEDS_MORE_DATA**. At K=9 the task is hard enough to break AKOrN
  but **not** to separate GTM from simple_gating. Both saturate
  identically on accuracy + MI; the only separator is spectral entropy
  (signature axis, not coder axis). Pushing K beyond 9 would require
  relaxing the Lisman-Idiart invariant and was not done.

## Caveat

The brief asked for K=14, which is incompatible with `GammaThetaConfig`'s
configured invariant. K=9 was used instead. Two arms still saturate, so
the experiment does *not* falsify the no-coder-separation finding from
Renf 7 — it only narrows the regime where it holds. A follow-up run
with the invariant relaxed (e.g. theta cycles per symbol > 1) is the
natural next step.
