# Renf 11 — seed-window robustness

**Date:** 2026-05-20
**Host:** macm1 (Apple M1, 10 workers)
**Script:** `scripts/renf11_seed_window.py`
**Wall-clock:** 1454.6 s (~24.2 min)
**Status:** DONE

## Goal

Renf 7 used seeds 0-49 to rank arms on the four synchrony-alternative
metrics (spectral_entropy, participation_ratio, effective_rank,
top3_dispersion). This script re-runs the same evaluation on three
seed windows — A=`range(0,50)`, B=`range(50,100)`, C=`range(1000,1050)`
— and asks: do the per-arm metric distributions differ across windows?
If any pair of windows gives p<0.05 on Mann-Whitney U, then seeds 0-49
were a lucky/unlucky window and Renf 7's ranking is fragile.

## Headline result

**Zero window-pair × arm × metric combinations cross the p=0.05 threshold.**
All 48 Mann-Whitney U tests (4 arms × 4 metrics × 3 window pairs) give
p > 0.29. The minimum p-value across the entire grid is 0.293
(simple_gating × top3_dispersion × A_vs_C).

## Per-window means — spectral entropy

| Arm           | Window A (0-49)  | Window B (50-99)  | Window C (1000-1049) |
|---------------|------------------|-------------------|----------------------|
| gtm           | 2.1729 ± 0.0089  | 2.1734 ± 0.0085   | 2.1742 ± 0.0074      |
| simple_gating | 3.4645 ± 0.0164  | 3.4641 ± 0.0164   | 3.4642 ± 0.0159      |
| akorn_best    | 1.9402 ± 0.5838  | 1.8876 ± 0.7166   | 1.8348 ± 0.6381      |
| null          | 1.9597 ± 0.1319  | 1.9804 ± 0.1244   | 1.9747 ± 0.1333      |

The arm rank (gtm low, akorn lower, simple_gating high) is **identical
across the three windows**. The gtm-vs-simple_gating gap is ~1.3 std
units; the AKOrN variance is large in every window (intrinsic to the
arm, not the window).

## Full p-value matrix

All 48 p-values (4 arms × 4 metrics × 3 window pairs) are above 0.29.
Highlights:

- gtm × spectral_entropy: A-B=0.860, A-C=0.634, B-C=0.764
- simple_gating × spectral_entropy: A-B=0.975, A-C=0.915, B-C=0.959
- akorn_best × spectral_entropy: A-B=0.839, A-C=0.556, B-C=0.617
- null × spectral_entropy: A-B=0.488, A-C=0.551, B-C=0.877

The same pattern holds for participation_ratio, effective_rank, and
top3_dispersion. See the full JSON for the per-metric matrix.

## Verdict on the hypothesis

**CONFIRMED.** Seeds 0-49 are not a lucky window. The Renf 7 ranking
(GTM and AKOrN low spectral entropy, simple_gating high, null in
between, AKOrN with the highest variance) reproduces on seeds 50-99
and seeds 1000-1049 with no statistically detectable difference. The
ablation finding generalises beyond the original seed range.

A side observation: AKOrN's std (0.58 - 0.72 on spectral entropy) is
~70× larger than GTM's std (0.008 - 0.009) in every window, confirming
that AKOrN's spread is an intrinsic property of the arm (oscillator
dynamics), not a seed-window artefact.
