# NH-96 - the null that can fail (L0 draft)

Source text (catalogue): "Nul qui peut echouer : si la tache asymetrique donne aussi
moins de 0,1, c'est l'architecture qui bloque." Problem: that sentence explains a
failure away instead of defining one. This document gives the calibrated null and
the rejection criteria. Follows `hypneum-nulls` (items 1-9).

## 1. Hypotheses

H1 (prediction): D_asym >= 0.30 and D_sym <= P99 of its calibrated null (PI decision 2026-10-31, replaces the fixed 0.05), with D = mean over k seeds of
`S_eng - S_nocomm` (paired by seed, success = last-5-logs mean).
H0 (null): the message carries no usable information; both arms sit at the exact blind
ceiling B0 (asym 1/C, `blind_ceiling()`); any difference is evaluation noise plus an
arm-by-seed random effect.

## 2. Construction of the null (same pipeline)

Draw per seed and arm: p = clip(B0 + N(0, sigma_seed)), then Binomial(n_eval, p)/n_eval;
D = mean over k seeds of the paired difference; N = 10 000 draws; report observed D,
null median, P95, P99 and p = (hits+1)/(N+1). Code: `experiments/n10d/null.py`
(stdlib, seconds). `sigma_seed` (TBD-9, closed): measured on the `nocomm`
development seeds 9600-9607, by `sigma_seed.py` (nothing else read) before the freeze, then fixed in the manifest.
Limit, stated plainly: this surrogate null covers evaluation and seed noise, not
training dynamics. It calibrates the threshold for "above blind", not the size of
effects a trained pair can reach.

## 3. Computed before data (k=8 seeds, n_eval=500, asym, K=16; null.py, rng 20261101)

| C | sigma_seed | median | P95 | P99 |
|---|---|---|---|---|
| 2 | 0.00 | 0.000 | 0.019 | 0.026 |
| 2 | 0.02 | 0.000 | 0.025 | 0.035 |
| 2 | 0.05 | 0.000 | 0.046 | 0.063 |
| 4 | 0.00 | 0.000 | 0.016 | 0.023 |
| 4 | 0.02 | 0.000 | 0.023 | 0.033 |
| 4 | 0.05 | 0.000 | 0.044 | 0.062 |
| 8 | 0.00 | 0.000 | 0.012 | 0.017 |
| 8 | 0.02 | 0.000 | 0.020 | 0.029 |
| 8 | 0.05 | 0.000 | 0.043 | 0.061 |

Consequence (decided 2026-10-31): the sym bound is no longer a fixed 0.05. With
sigma_seed = 0.05 the P99 is about 0.06, so a fixed 0.05 would not be separable from
noise. The bound is the P99 of the null computed with the measured sigma_seed. The
margin 0.30 stays far above every P99 here, so the null does not bind H1_asym; the
margin does. The table is illustrative (k=8); the real k comes from the power formula.

## 4. Criteria that make NH-96 fail

NH-96 is rejected (not "unlucky") if any holds, with controls of protocol section 3 passed:
1. upper bound of the 95 % paired-bootstrap CI of D_asym < 0.30;
2. D_sym > null P99 (measured sigma_seed) with empirical p <= 0.01 (the twin is not symmetric,
   or the channel is used when redundant);
3. D_asym < 0.10: failed prediction. The sentence "the architecture
   blocks" is NOT a verdict; it is a new hypothesis for an amendment.
Non-rejection with the learning gate failed is "non mesurable", not "supported".

## 5. Tautology check (item 4)

`msg_shuffle - nocomm` has a known answer (0 at ceiling); it is a pipeline control, not
evidence. On `sym` the blind ceiling is 1.0, so `eng - nocomm` there is 0 by construction of the task: labelled "controle non informatif" for `sym`, kept only as a leak check.
