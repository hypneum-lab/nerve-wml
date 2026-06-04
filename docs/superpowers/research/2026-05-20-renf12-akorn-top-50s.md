# Renf 12 — AKOrN top cell at 50 seeds

**Date:** 2026-05-20
**Host:** macm1 (Apple M1, 10 workers)
**Script:** `scripts/renf12_akorn_top_50s.py`
**Wall-clock:** 553.2 s (~9.2 min)
**Status:** DONE

## Goal

Re-run Renf 1's top AKOrN cell (`n_oscillators=64, n_steps=32, lr=0.05`)
at 50 seeds (vs the original 5) to obtain a tight bootstrap CI on
synchrony, accuracy, and MI. Renf 1 reported synchrony
`0.4542 ± 0.1624` (n=5) — the std is large enough that the rank vs GTM
(synchrony ~0.20) could be a sampling artefact.

## Headline results (50 seeds, 200 training steps)

| Metric           | Mean   | Std    | Bootstrap CI95 (2000 resamples) |
|------------------|--------|--------|---------------------------------|
| Accuracy         | 0.3796 | 0.3063 | (not in report) |
| MI (bits)        | 1.5807 | 0.2918 | [1.5031, 1.6642] |
| Synchrony index  | 0.5310 | 0.2017 | [0.4746, 0.5838] |

Renf 1 reference (n=5): synchrony `0.4542 ± 0.1624`.

## Interpretation

- **Synchrony mean moved up** from 0.4542 → 0.5310 — about half a Renf-1
  std-dev higher. The CI95 `[0.4746, 0.5838]` does **not** contain the
  Renf 1 point estimate at the lower edge (0.4542 sits just outside the
  lower CI bound), so the original 5-seed estimate was on the low side.
- **The high-power CI still places AKOrN well above the GTM band
  (~0.20)**. CI95 lower bound is 0.4746, more than 2× the GTM mean.
  AKOrN's synchrony-spectrum signature is robust to power.
- **Accuracy is bimodal.** Mean 0.38 with std 0.31 means many seeds
  fail to learn the codes (some near 1.0, others near 0). This was hidden
  by the synchrony metric in Renf 1 — AKOrN converges in oscillator phase
  even when the decoder fails. The MI mean 1.58 bits (out of 6 = log2(64))
  echoes the same bimodality.

## Verdict on the hypothesis

**CONFIRMED.** The AKOrN top cell synchrony was not a 5-seed fluke; at
n=50 the CI95 still cleanly separates it from GTM. The original Renf 1
point estimate underestimated the mean slightly (`0.4542` vs the new
`0.5310`), but the spectral ordering claim survives.

A secondary finding: AKOrN's accuracy / MI distribution is heavy-tailed
and should be reported with the synchrony figure to avoid suggesting it
is a strong *coder*. It is a strong *phase-aligner*.
