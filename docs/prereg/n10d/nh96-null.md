# NH-96 - the null that can fail (L0 draft)

Source text (catalogue): "Nul qui peut echouer : si la tache asymetrique donne aussi
moins de 0,1, c'est l'architecture qui bloque." Problem: that sentence explains a
failure away instead of defining one. This document gives the calibrated null and
the rejection criteria. Follows `hypneum-nulls` (items 1-9).
Amended 2026-11-06 after the blind contradictor (protocol section 8: A5-A9, A18, A19),
then reconciled the same day with the parallel revision of PR #100 (protocol section 8,
"Reconciliation 2026-11-06"), a second time at its head cb7d66b (PI decisions of 17:05:
margin 0.30, CI rule, sym gate on `eng_sym` only, pilot reads `nocomm` only).
**Status:** L0 DRAFT, not frozen; the open items of `protocol.md` section 0 apply.

## 1. Hypotheses (asym only)

`sym` is out of H1 (PI decision, `brainstorm-2026-11-01.md` section 0, item 1); it is a
descriptive control arm (protocol section 3), never a term of H1 or H0.

H1 (prediction, asym): D_asym meets case A of `protocol.md` section 5 (margin 0.30 =
0.4 x (1 - 1/C), TBD-12; CI rule of TBD-16: lower bound of the 95 % paired CI of D_asym
>= margin and the one-sided paired sign test rejects at alpha 0.05), with D = mean over
k seeds of `S_eng - S_nocomm`, paired by seed, S = evaluation success of the frozen
policy on n_eval fresh episodes (protocol section 3, A8).
H0 (null, asym): the message carries no usable information. Within a seed both arms
are blind policies; neither can exceed the exact blind ceiling B0 = 1/C
(`blind_ceiling()`); per seed both arms share one latent success p <= B0, and any
difference is binomial evaluation noise.
No H0 is registered on `sym`. Its null (both arms at B0 = 1.0) is degenerate: at
sigma = 0 every draw gives D = 0 exactly, so P99 = 0 and any positive D_sym would get
p = 1/(N+1). This null is reported descriptively only and sets no threshold. D_sym is
read only through the sym check of `protocol.md` sections 3 and 5 (TBD-15): gate on
`eng_sym` alone (mean >= 0.95); case D iff the lower bound of the 95 % paired CI of
D_sym >= delta_sym = 0.10.

## 2. Construction of the null (same pipeline, paired by seed)

Per seed: one latent seed effect u ~ N(0, sigma_seed), shared by both arms (paired
null, PI decision `brainstorm-2026-11-01.md` section 0, item 2); latent success
p = clip(B0 + u, 0, B0), bounded above by B0 because no blind policy beats 1/C (A6);
then each arm is evaluated independently, Binomial(n_eval, p)/n_eval.
In `asym` an exact blind policy has expectation 1/C, so the game itself does not
require a latent effect; the common latent is the form decided by the PI, and it
cancels in the paired difference except through the binomial variance.
D = mean over k seeds of the paired difference; N = 10 000 draws; report observed D,
null median, P95, P99 and p = (hits+1)/(N+1). Code: `experiments/n10d/null.py`
(stdlib). `sigma_seed` (TBD-9, closed, not amended by TBD-14): measured on the
`nocomm`/`asym` development seeds 9600-9607 only, by `sigma_seed.py` (D never computed,
no k printed) before the freeze, then fixed in the manifest. It is a consistency check;
it does not size k (protocol section 4). A sigma measured on `asym` is never transported
to `sym`.
`simulate_null(..., paired=False)` keeps the former independent-arm draw as a
sensitivity check only; it is not the registered null and never enters a decision.

Level null (A9): `simulate_level_null` gives the null of one blind arm's mean success
over k seeds (a level, not a difference). Controls 2 and 4 of protocol section 3 use
its P99, written `L99`; they never reuse the P99 of D.

Limit, stated plainly: this surrogate null covers evaluation and seed noise, not
training dynamics, and not an arm-specific training effect. It calibrates the
threshold for "above blind", not the size of effects a trained pair can reach.
Because u is shared and p is capped at B0, sigma_seed nearly cancels in D (table
below); the margin 0.30, not the null, binds H1.

## 3. Computed before data (k=8 seeds, n_eval=500, K=16; null.py, rng 20261101/02)

Regenerated 2026-11-06 with the amended generator on macM1 (`python3 -m n10d.null`,
copy of `experiments/n10d/{null,spec}.py`, 2 min 15 s). This table supersedes the one
of PR #100 (e5fbe88), which was computed with the uncapped generator
(p = clip(B0 + u, 0, 1)) and is no longer the code (A6).

| condition | C | sigma_seed | D median | D P95 | D P99 | level L99 (B0) |
|---|---|---|---|---|---|---|
| asym | 2 | 0.00 | 0.000 | 0.018 | 0.026 | 0.518 (0.500) |
| asym | 2 | 0.02 | 0.000 | 0.018 | 0.026 | 0.512 (0.500) |
| asym | 2 | 0.05 | 0.000 | 0.018 | 0.026 | 0.508 (0.500) |
| asym | 4 | 0.00 | 0.000 | 0.016 | 0.023 | 0.266 (0.250) |
| asym | 4 | 0.02 | 0.000 | 0.016 | 0.022 | 0.261 (0.250) |
| asym | 4 | 0.05 | 0.000 | 0.015 | 0.022 | 0.256 (0.250) |
| asym | 8 | 0.00 | 0.000 | 0.012 | 0.017 | 0.137 (0.125) |
| asym | 8 | 0.02 | 0.000 | 0.012 | 0.016 | 0.132 (0.125) |
| asym | 8 | 0.05 | 0.000 | 0.011 | 0.016 | 0.128 (0.125) |
| sym | 4 | 0.00 | 0.000 | 0.000 | 0.000 | every draw 1.000 (1.000) |

Sensitivity (independent arms, `paired=False`, capped, asym, C=4): D P99 = 0.023,
0.026, 0.041 at sigma_seed = 0.00, 0.02, 0.05. The pre-amendment generator (independent,
uncapped) gave 0.023, 0.033, 0.062. The sym null is degenerate (P99 = 0 exactly): it
gives no test for D_sym, which is why sym is descriptive (protocol section 3, sym gate).
The table is illustrative (k=8); the real k comes from the rule of protocol section 4.

## 4. Criteria that make NH-96 fail

There is one decision rule, the table of protocol section 5 (A7); this section only
restates its rejection rows. With all asym controls of protocol section 3 passed:
1. Outcome C, failed prediction: the point estimate D_asym < 0.10 (TBD-6). "The
   architecture blocks" is NOT a verdict; it is a new hypothesis for an amendment.
2. Outcome B, weaker than predicted, NH-96 rejected: D_asym >= 0.10 and the upper
   bound U of the 95 % paired CI of D_asym < 0.30 (CI rule, TBD-16).
3. Outcome A, NH-96 supported: the lower bound L of the 95 % paired CI of D_asym
   >= 0.30 AND the one-sided sign test rejects at alpha = 0.05.
4. Anything else is "non concluant", never "refuted" nor "supported".
If the leak audit (control 4) fails the run set is "invalid"; if another asym control
fails it is "non mesurable"; neither is "supported".
D_sym enters no criterion (section 1).

## 5. Tautology check (item 4)

`msg_shuffle - nocomm` has a known answer (0 at the blind ceiling, protocol section 3,
control 2); it is a pipeline control, not evidence. On `sym` the blind ceiling is 1.0,
so `eng - nocomm` there is 0 only for optimal policies (A18): trained policies need
not reach it (e.g. a listener right on 8 of 16 identities and guessing among C = 4 on
the rest scores 0.5 + 0.5 * 0.25 = 0.625). `sym` is therefore labelled "controle non
informatif" for H1 and kept as a plumbing and leak control with its own gate.
