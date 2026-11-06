# NH-96 - the null that can fail (L0 draft)

**Revision:** 2026-11-06 — revised after the blind contradictor pass (PR #113, VERIFICATION.md):
H1 without `sym` and H0 stated per condition (section 1); paired null (section 2); sym P99
differs from asym P99 (section 3); one set of criteria, aligned on protocol section 5
(section 4); "0 by construction" holds for optimal policies only (section 5).
**Status:** L0 DRAFT, not frozen; open items TBD-12 to TBD-16 of `protocol.md` section 0 apply.

Source text (catalogue): "Nul qui peut echouer : si la tache asymetrique donne aussi
moins de 0,1, c'est l'architecture qui bloque." Problem: that sentence explains a
failure away instead of defining one. This document gives the calibrated null and
the rejection criteria. Follows `hypneum-nulls` (items 1-9).

## 1. Hypotheses

H1 (prediction, `asym` only; PI-1, 2026-11-01): D_asym meets the confirmation rule of
`protocol.md` section 5, case A (margin 0.30 or TBD-12; single rule TBD-16; the one-sided
paired sign test rejects at alpha 0.05), with D = mean over k seeds of `S_eng - S_nocomm`
(paired by seed; S = success of the frozen policy on n_eval independent episodes).
`sym` is not in H1.
H0 (null, `asym`): the message carries no usable information; within a seed both arms
are blind policies with the same expected success, the exact blind ceiling
B0 = 1/C (`blind_ceiling()`); any difference is binomial evaluation noise plus a seed
effect common to both arms.
Sym leak-check null (not a hypothesis of NH-96): both arms at B0 = 1.0. At sigma = 0 every
draw gives D = 0 exactly, so P99 = 0 and any positive D_sym has p = 1/(N+1); D_sym is read
only through the audit of `protocol.md` section 5, case D.

## 2. Construction of the null (same pipeline)

Paired null (PI-2, 2026-11-01): per seed, one latent p = clip(B0 + N(0, sigma_seed)) shared
by both arms, then one Binomial(n_eval, p)/n_eval per arm; D = mean over k seeds of the
paired difference; N = 10 000 draws; report observed D, null median, P95, P99 and
p = (hits+1)/(N+1). Code: `experiments/n10d/null.py`, `simulate_null` (stdlib, seconds).
In `asym` an exact blind policy has expectation 1/C, so the game itself does not require a
latent effect; the common latent is the form decided by the PI, and it cancels in the
paired difference except through the binomial variance. The former independent
arm-by-seed draw (`paired=False`) is reported for sensitivity only, never used for a decision.
Thresholds on ONE arm (protocol section 3, controls 2-4) use the single-arm null
`simulate_arm_null` (mean over k seeds of S_arm - B0), not this null of D.
The sym null uses its own B0 = 1.0; sigma measured on `asym` is not transported to it
(TBD-14). `sigma_seed` (TBD-9, closed; possible amendment TBD-14): measured on the
`nocomm` development seeds 9600-9607, by `sigma_seed.py` (one arm per call, D never
computed) before the freeze, then fixed in the manifest.
Limit, stated plainly: this surrogate null covers evaluation and seed noise, not
training dynamics. It calibrates the threshold for "above blind", not the size of
effects a trained pair can reach.

## 3. Computed before data (k=8 seeds, n_eval=500, K=16; null.py, rng 20261101)

Paired null of D (decision values) next to the former independent-arms null
(`paired=False`, sensitivity only; it reproduces the 2026-10-31 table), and the P99 of the
single-arm null (mean S_arm - B0) used by controls 2-4. Computed on macM1, 2026-11-06,
`python3 -B -m experiments.n10d.null` (output `~/scratch/n10d-verif-0611/null-table-0611/`).

| cond. | C | sigma_seed | paired median | paired P95 | paired P99 | indep. P95 | indep. P99 | single-arm P99 |
|---|---|---|---|---|---|---|---|---|
| asym | 2 | 0.00 | 0.000 | 0.018 | 0.026 | 0.019 | 0.026 | 0.018 |
| asym | 2 | 0.02 | 0.000 | 0.018 | 0.026 | 0.025 | 0.036 | 0.024 |
| asym | 2 | 0.05 | 0.000 | 0.018 | 0.026 | 0.046 | 0.063 | 0.044 |
| asym | 4 | 0.00 | 0.000 | 0.016 | 0.023 | 0.016 | 0.023 | 0.016 |
| asym | 4 | 0.02 | 0.000 | 0.016 | 0.023 | 0.023 | 0.033 | 0.023 |
| asym | 4 | 0.05 | 0.000 | 0.016 | 0.022 | 0.045 | 0.063 | 0.044 |
| asym | 8 | 0.00 | 0.000 | 0.012 | 0.017 | 0.012 | 0.018 | 0.012 |
| asym | 8 | 0.02 | 0.000 | 0.012 | 0.017 | 0.020 | 0.029 | 0.021 |
| asym | 8 | 0.05 | 0.000 | 0.012 | 0.017 | 0.043 | 0.061 | 0.043 |
| sym | 4 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sym | 4 | 0.02 | 0.000 | 0.003 | 0.005 | 0.010 | 0.015 | <= 0 |
| sym | 4 | 0.05 | 0.000 | 0.005 | 0.008 | 0.024 | 0.035 | <= 0 |

Consequences. (a) With the paired null the P99 of D no longer grows with sigma_seed: the
common latent cancels in the difference. (b) The sym P99 differs from the asym P99
(0.008 against 0.022 paired, 0.035 against 0.063 independent, at sigma = 0.05, C = 4),
and it is exactly 0 at sigma = 0: a sym bound is never borrowed from asym. (c) The margin
(0.30 or TBD-12) stays far above every P99 here, so the null does not bind H1; the margin
and the decision rule do. The table is illustrative (k=8); the real k comes from the rule
of protocol section 4.

## 4. Criteria that make NH-96 fail

One set of criteria, the same as `protocol.md` section 5 (TBD-16). NH-96 is rejected
(not "unlucky") if any holds, with the `asym` controls of protocol section 3 passed:
1. the rejection side of the single decision rule holds (TBD-16; under option (i): upper
   bound of the 95 % paired-bootstrap CI of D_asym < margin);
2. D_asym < 0.10: failed prediction. The sentence "the architecture
   blocks" is NOT a verdict; it is a new hypothesis for an amendment.
D_sym is not a rejection criterion (PI-1): D_sym above its paired P99 triggers the audit
of protocol section 5, case D. Non-rejection with the learning gate failed is
"non mesurable"; any other non-rejection is "non concluant"; neither is "supported".

## 5. Tautology check (item 4)

`msg_shuffle - nocomm` has a known answer (0 at ceiling); it is a pipeline control, not
evidence. On `sym` the blind ceiling is 1.0, so `eng - nocomm` there is 0 by construction
for optimal policies only; trained policies at finite budget can give D_sym != 0 without
any leak (protocol section 5, case D). Labelled "controle non informatif" for `sym`, kept
only as a leak check.
