# ADR 0001 — MI bias correction + permutation null for the MI(c_MLP;c_LIF)/H claim

> Status: accepted, 2026-06-04. Answers REVIEW.md item ② ("the strongest attack":
> plug-in MI/entropy estimator bias, Paninski 2003 / Panzeri 2007).
> Re-measurement script: `scripts/remeasure_mi_review_item2.py`.
> Estimators: `nerve_wml/methodology/mi_estimators.py`.
> Real data: `tests/golden/codes_mlp_lif.npz` (3 seeds, N=5000 eval samples,
> MLP/LIF trained 800 steps on HardFlowProxyTask).

## Context

The flagship cites **MI(c_MLP;c_LIF)/H = 0.91–0.96** with a small polymorphism gap as
evidence of substrate-agnosticism. The external review attacked the headline as a possible
**estimator artefact**: plug-in MI carries a *positive* finite-sample bias (Panzeri 2007:
a true-MI=0 source yields ~0.202 bit at N=20), so an inflated MI could be the **null
expectation**, not evidence of alignment. The review demanded a **bias-corrected estimator
(Panzeri/NSB)** plus a **permutation null-model** before the number is cited.

## What we measured

Estimator currently producing the headline: **plug-in, entropy-normalised MI/H(a) on the
argmax (post-VQ) discrete codes** (`mi_plugin_discrete` / `mi_argmax_onehot`). Sample size
**N = 5000** per seed. We added a **Panzeri-Treves** analytic bias correction
(`mi_panzeri_treves_discrete`) alongside the pre-existing Miller-Madow, KSG (Kraskov on the
**pre-VQ continuous embeddings**, already ~debiased) and MINE estimators, and ran a
**permutation null for every estimator** (shuffle the c_MLP↔c_LIF pairing).

### Table (mean over 3 seeds, on the real saved codes)

| Estimator | MI/H observed | Null (permutation) mean | Excess over null |
|---|---|---|---|
| Plug-in (discrete, headline) | **0.932** | 0.0068 | 0.925 |
| Panzeri-Treves (debiased, discrete) | **0.931** | 0.0016 | 0.929 |
| Miller-Madow (debiased, discrete) | 0.939 | — | — |
| MINE (neural, continuous) | 0.985 | — | — |
| **Kraskov-KSG (debiased, on pre-VQ continuous embeddings)** | **0.095** | 0.0011 | 0.094 |

Per-seed plug-in: 0.992 / 0.982 / 0.822. KSG/H per-seed: 0.032 / 0.068 / 0.184.
All p-values vs the permutation null are < 0.01 (every estimator, every seed). The
**VQ amplification ratio** (plug-in / KSG) is ≈ **9.8×**.

### Sampling-bias control (reproducing the Panzeri argument)

Two **independent** streams over an 8-symbol alphabet (true MI = 0), plug-in vs PT, in bits:

| N | plug-in null MI | Panzeri-Treves null MI |
|---|---|---|
| 20 | 1.395 | 1.273 |
| 50 | 0.804 | 0.514 |
| 100 | 0.424 | 0.166 |
| 500 | 0.073 | 0.003 |
| **5000** | **0.007** | **−0.000** |

This confirms the review's mechanism quantitatively: at small N the plug-in floor is huge
(>1 bit at N=20 even when the truth is 0). But the repo measures at **N=5000**, where that
floor collapses to 0.007 bit and Panzeri-Treves drives it to ~0.

## Decision / Verdict

The 0.91–0.96 figure **splits into two distinct questions**, and the honest answer differs:

1. **Is it a sampling-bias / null artefact (the literal review attack)?** — **No.**
   At N=5000 the plug-in positive bias is negligible (0.007 bit). Panzeri-Treves and
   Miller-Madow leave the discrete number essentially unchanged (0.931 / 0.939 vs 0.932),
   and the permutation null sits at ~0.005 with p < 0.01 across all seeds. On the discrete
   codes, **the number survives bias correction and is far above the permutation null.**
   The seed-to-seed stability is *not* the only thing holding it up; the large N is.

2. **Does "MI/H ≈ 0.93" mean what the headline implies (substrate-agnostic shared code)?**
   — **Substantially weaker than advertised.** The 0.93 is measured *after* vector
   quantisation (argmax over a ~7–27-symbol alphabet). On the **pre-VQ continuous
   activations**, the debiased KSG estimator reports MI/H ≈ **0.095** — a ~**10× drop**.
   The discretisation step itself manufactures most of the apparent alignment: collapsing
   each substrate to a handful of dominant symbols inflates normalised MI because H(a) also
   collapses. This is an *estimator/representation* artefact of a different kind than the
   one the review named (VQ amplification, not sampling bias), but it is real and it is
   internal to this repo's own data.

**Net:** the headline is **robust to the specific attack in REVIEW ②** (bias + null), but it
is **fragile to estimator/representation choice**: it holds on discretised codes and via
MINE, yet drops by ~10× under a debiased continuous estimator on the same activations.

## Consequences / recommended wording

When citing this number, it MUST be qualified as **MI/H on post-VQ discrete codes
(N=5000, bias-corrected, permutation p<0.01)**, and reported **alongside** the
continuous-embedding KSG value (≈0.095). Citing 0.91–0.96 bare, as substrate-agnostic
shared information, overstates it by ~10×. The polymorphism-gap claim inherits the same
caveat and should be re-derived on the continuous estimator before load-bearing use.

Paper text was **not** modified by this ADR; any claim edit must be done in lockstep and
tagged. This ADR is the falsifiability artefact REVIEW ③ asked for (permutation null +
bias-corrected estimator now both exist and are tested).
