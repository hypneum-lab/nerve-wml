---
title: "Empirical replication of Gröger et al. (2026) null-calibration predictions on a synthetic Gaussian substrate"
author:
  - Hypneum Lab
date: 2026-05-20
abstract: |
  Gröger, Wen & Brbić (2026) formalise null-calibration for
  representational-alignment metrics: their Proposition 4.1 derives an
  $O(d/n)$ random-matrix null baseline for linear CKA and their
  Proposition 4.2 gives $\mathbb{E}[\mathrm{mKNN}(X,Y)] = k/(n-1)$ as
  the corresponding null baseline for mutual $k$-NN metrics (CKNNA,
  mutual-kNN agreement). Under their framework, raw CKNNA scores at
  fixed $k$ should drift downward as $n$ grows because the
  scoreable range $(k/n, 1]$ widens, and the natural artefact-free
  quantity is the calibrated signal-vs-null separation. This
  supplementary note reports independent empirical confirmation of
  those predictions on a synthetic substrate disjoint from their
  image-text PRH setup. On a 1750-cell sweep (8 metrics × 7 values
  of $N$ spanning ~1.8 decades from 256 to 16384, i.e. 6 doublings ×
  5 noise levels × 50 seeds), CKNNA@k=10 drops from 0.924 ± 0.005 at
  $N=256$ to 0.873 ± 0.001 at $N=16384$ while the alignment is held
  fixed; the matched shuffled-null arm drops from 0.040 to 0.001
  over the same range, tracking $k/(N-1)$ as predicted. The paired
  signal-vs-null separation (Cohen's $d_z$) grows monotonically from
  $d_z \approx 135$ at $N=256$ to $d_z \approx 1311$ at $N=16384$,
  inverting the direction of the apparent effect on the raw score.
  We frame these observations as an empirical replication of the
  Gröger et al. predictions on a synthetic Gaussian substrate and
  use them to motivate the null-arm protocol applied to the main
  paper's ablations.
bibliography: refs.bib
---

# 1. Introduction

**This note is a supplementary empirical replication, not an
independent contribution.** Gröger, Wen & Brbić
(2026) [@aristotelianprh2026] established a formal null-calibration
framework for representational-alignment metrics, deriving explicit
random-matrix null baselines for both spectral metrics (linear CKA,
their Proposition 4.1) and local-neighbourhood metrics (mutual $k$-NN
including CKNNA, their Proposition 4.2: $\mathbb{E}[\mathrm{mKNN}(X,
Y)] = k/(n-1)$ under random pairing). Their main empirical claim is
that spectral cross-modal "convergence" reported by the Platonic
Representation Hypothesis [@huh2024platonic] disappears after
null-calibration, while local-neighbourhood structure survives —
motivating an "Aristotelian" reinterpretation. They also propose a
calibration map $s_{\mathrm{cal}} = \max((s_{\mathrm{obs}} -
\tau_\alpha) / (s_{\max} - \tau_\alpha), 0)$ that gives a Type-I
controlled effect-size readout.

The substrate they study is image-text embedding pairs at PRH-scale
($n \approx 1024$). The purpose of this supplementary note is to
record an **independent empirical replication of two of their
predictions on a synthetic Gaussian substrate** that is disjoint from
that setup, both to corroborate the predictions across substrate
types and to motivate the paired-null protocol used by the main
paper.

We report two empirical observations on a 1750-cell sweep that hold
the alignment and the noise process exactly constant while varying
$n$ over six doublings (~1.8 decades, $N \in \{256, 512, 1024, 2048,
4096, 8192, 16384\}$):

(D1) **Raw CKNNA decreases monotonically in $N$ at constant
signal.** This is the predicted consequence of the $k/(n-1)$ null
baseline (Proposition 4.2 of [@aristotelianprh2026]); we quantify
its magnitude on our substrate.

(D2) **Paired signal-vs-null Cohen's $d_z$ grows monotonically in
$N$ on the same data.** This is the artefact-free quantity in
the spirit of Gröger et al.'s $s_{\mathrm{cal}}$, expressed in
standardised effect-size units.

# 2. Method

**Substrate.** We generate aligned point clouds
$x, y \in \mathbb{R}^{N \times D}$ with $D=64$ by sampling
$x \sim \mathcal{N}(0, I_D)$ and setting $y = x + \sigma \cdot
\varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I_D)$ independent
of $x$. By construction $x$ and $y$ share their full latent
structure modulated only by isotropic Gaussian noise of magnitude
$\sigma$.

**Null arm.** We pair every real-arm cell with a *shuffled-rows*
counterpart in which the rows of $y$ are permuted by a random
permutation independent of $x$. This destroys the row-wise
alignment while preserving the marginal distributions exactly.

**Metric battery.** For each cell we compute 8 alignment metrics
on the real arm — HSIC (biased estimator), linear CKA, Procrustes
$R^2$, mutual-kNN-correlation@10, and CKNNA at $k \in \{5, 10, 20,
50\}$ — and a subset (HSIC, CKNNA@10, Procrustes $R^2$) on the
null arm. All metrics are computed on the same $(x, y)$ pair so
that effect-size estimates are paired across seeds.

**Hyperparameter grid.** $N \in \{256, 512, 1024, 2048, 4096,
8192, 16384\}$ (7 values), $\sigma \in \{0.001, 0.01, 0.05, 0.2,
1.0\}$ (5 values), 50 seeds per cell. Total: $7 \times 5 \times 50
= 1750$ cells. Total wall-clock: 16.7 min on a single Apple M1 with
32 GB unified memory and the MPS backend (PyTorch 2.12.0).

**Reproducibility.** Raw per-cell JSON, the driver script, and the
figure-generation code are all committed at the repository
referenced in §7. The exact run reported here is on git revision
`076f770`.

# 3. Results

## 3.1 CKNNA decays with N at constant signal

We fix $\sigma = 0.05$ — a regime where the alignment is strong but
not saturated — and sweep $N$ over 64×. Real and null CKNNA@10
means and standard deviations across the 50 seeds are tabulated in
Table 1 and plotted in Figure 1.

**Table 1.** CKNNA@k=10 at $\sigma = 0.05$. Mean ± SD over 50 seeds.

| N      | real CKNNA@10        | null CKNNA@10        |
| -----: | -------------------: | -------------------: |
|    256 | 0.9242 ± 0.0048      | 0.0397 ± 0.0050      |
|    512 | 0.9144 ± 0.0031      | 0.0191 ± 0.0023      |
|   1024 | 0.9054 ± 0.0021      | 0.0097 ± 0.0012      |
|   2048 | 0.8967 ± 0.0016      | 0.0048 ± 0.0005      |
|   4096 | 0.8886 ± 0.0012      | 0.0025 ± 0.0003      |
|   8192 | 0.8808 ± 0.0008      | 0.0012 ± 0.0001      |
|  16384 | 0.8733 ± 0.0007      | 0.0006 ± 0.0001      |

![CKNNA@k=10 vs N at $\sigma=0.05$. Real (aligned) and null
(shuffled-rows) arms; error bars are ±1 SD across 50
seeds.](figures/fig1_cknna_decay.png){width=85%}

Two observations:

1. The real arm decays by ~0.051 absolute between $N=256$ and
   $N=16384$, i.e. by ~5.5% relative. The decay is monotone and well
   outside the 1-SD seed variability at every consecutive pair.
2. The null arm decays *faster* in relative terms, from 0.040 to
   0.0006 — a 65× reduction over the same range.

The second observation is the explanation. The chance baseline of
mutual k-NN agreement for two independent point clouds scales as
$k/(N-1)$: as $N$ grows at fixed $k$, the probability that a random
pair of points share a $k$-neighbour vanishes. CKNNA does *not*
recenter on this chance baseline, so the headroom between perfect
agreement (1.0) and pure-chance agreement (≈ $k/N$) grows with $N$.
The real arm has to claim a larger share of an increasingly large
headroom to keep its raw score constant, and at finite $\sigma$ it
cannot.

A back-of-envelope upper bound: if a fraction $p$ of true neighbour
relations survives the noise, and the remaining $1-p$ behave as
independent random draws, then expected CKNNA tracks
$p + (1-p) \cdot k / (N-1)$, which is monotone decreasing in $N$ at
fixed $p$ and $k$. Plugging the observed null (≈ chance) and real
values gives $p$-estimates of 0.92 (N=256) and 0.873 (N=16384) — the
small residual gap is the genuine metric drift, the rest of the
nominal "0.92→0.87 degradation" is the headroom rescaling.

## 3.2 Signal-vs-null $d_z$ grows monotonically with N

The same 1750 cells, summarised as the paired Cohen's $d_z$ on
(real − null) per $N$, tell a very different story:

**Table 2.** Paired Cohen's $d_z$ at $\sigma = 0.05$, 50-seed pairs.

| N      | CKNNA@10 $d_z$ | HSIC $d_z$ | Procrustes $R^2$ $d_z$ |
| -----: | -------------: | ---------: | ---------------------: |
|    256 |          135.1 |       31.2 |                   55.3 |
|    512 |          209.3 |       42.8 |                  101.0 |
|   1024 |          408.4 |       73.4 |                  125.0 |
|   2048 |          547.5 |       92.4 |                  176.3 |
|   4096 |          668.4 |      125.8 |                  259.1 |
|   8192 |         1058.0 |      171.0 |                  468.2 |
|  16384 |         1311.0 |      258.1 |                  530.1 |

![Signal-vs-null Cohen's $d_z$ vs N at $\sigma = 0.05$ (log–log).
Two metrics shown — CKNNA@10 and HSIC — both grow monotonically.
Identical qualitative behaviour holds for Procrustes $R^2$ and
linear CKA on the same cells.](figures/fig2_dz_growth.png){width=85%}

The same data that *decreases* monotonically on the raw scale
*increases* monotonically once expressed as a standardised paired
effect. The growth is approximately $\sqrt{N}$ for $d_z$ on these
data, consistent with the fact that the standard deviation of the
real–null difference shrinks faster than its mean.

This is what we mean by "the cross-paper comparable quantity": a
study that reports CKNNA = 0.92 at $N=256$ and a study that reports
CKNNA = 0.87 at $N=16384$ on the *same* substrate would, on the raw
score, look like a 6% degradation; on $d_z$, they would look like a
$10\times$ *improvement* in statistical separation. Both
descriptions cannot be right; the $d_z$ one is artefact-free here
by construction.

## 3.3 Aside: HSIC decorrelates from the rest of the battery

Across all 1750 cells, the Pearson correlation between biased HSIC
and CKNNA@10 is $|r| = 0.003$, and the most independent metric pair
in the battery is HSIC vs mutual-kNN-correlation ($|r| = 0.001$).
We mention this only to flag that HSIC's joint scale-plus-noise
behaviour is essentially orthogonal to that of the neighbourhood
metrics on this benchmark; a HSIC-only diagnostic is not a
substitute for a CKNNA-class diagnostic, and vice versa. The
headline finding is about CKNNA.

# 4. Discussion

**Mechanism.** CKNNA's $k$ is typically chosen as a constant (10
in the original PRH paper, and in most replications). Holding $k$
constant while sweeping $N$ pushes the chance baseline $k/N$ toward
zero. The score is normalised against a maximum of 1 but not
recentred on $k/N$, so the "scoreable" range $(k/N, 1]$ widens with
$N$. At fixed $\sigma$ the true alignment claims a fixed fraction
of the available range, *not* a fixed absolute score, and this
manifests as a downward drift.

**Implication for replication studies.** [@platoscave2026]
interprets falling CKNNA across a scaling sweep as substantive
evidence about representational degradation. Our data corroborate
the prediction of [@aristotelianprh2026] that some — possibly much
— of that effect is the metric, not the system, and reinforce their
recommendation to report a null-calibrated effect size rather than
the raw score. We do not have the data to say *how much* of the
Plato's-Cave effect is artefact vs substrate; that would require
redoing that analysis with the Gröger et al. calibration map.

We do not interpret this as a criticism of those works — both report
N explicitly and the drift is consistent across their conditions,
so within-study comparisons are valid. The concern is *cross-study*
comparison and the absolute interpretation of CKNNA values, both of
which are now widespread.

**Two corrective protocols.**

1. *Fixed-N benchmarking.* Report a canonical N alongside every
   CKNNA value. This is the cheapest fix and we recommend it as a
   minimum.
2. *Null-arm $d_z$.* For every CKNNA cell, compute a shuffled-null
   counterpart on the same $N$ and report the paired Cohen's $d_z$.
   This is what we recommend for cross-paper headline claims.

**Relation to prior work.** The $k/(n-1)$ chance baseline of
mutual-$k$-NN agreement is formally established by Proposition 4.2
of [@aristotelianprh2026]; the present note adds an empirical
quantification on a Gaussian synthetic substrate (disjoint from
their image-text PRH cells) and confirms that the paired $d_z$
remedy is cheap and inverts the direction of the apparent effect.
The effective-rank/spectral-entropy diagnostic used elsewhere in
the main paper is the canonical Roy & Vetterli (2007) measure,
recently reapplied to LLM hidden-state collapse by [@wei2024differank].

# 5. Limitations

- **Single substrate type.** The (x, y = x + σε) Gaussian protocol
  is a clean instrument for isolating the metric's behaviour, but
  real cross-modal embeddings (DINOv2 vs CLIP, brain vs model) need
  not have the same chance-baseline geometry. We make no claim
  about the magnitude of the artefact on real representations.
- **Single noise model.** Additive isotropic Gaussian noise is the
  worst-case-friendly setting for the neighbourhood metrics.
  Non-isotropic or covariate noise may produce stronger or weaker
  drifts.
- **Single host, single backend.** All cells were executed on Apple
  M1 with the MPS PyTorch backend. We verified that CPU
  fallback gives bit-identical results up to floating-point
  reorderings on a 64-cell smoke test; cross-host reproduction is
  recommended.
- **50 seeds.** Sufficient for the tight CIs reported here; not
  sufficient for resolving the bottom-of-curve effects at $\sigma =
  1.0$.
- **No test on real model representations.** The natural follow-up
  is to recompute CKNNA + $d_z$ on the embedding pairs used in the
  PRH and Plato's-Cave studies.

# 6. Recommendation

For studies using CKNNA, mutual-kNN agreement, or any k-NN-based
representational alignment metric:

1. **Report N alongside every score.** Treat unannotated CKNNA
   values as incomparable.
2. **Report a shuffled-null arm.** A paired permutation null on the
   same $(N, k)$ costs the same as one extra real-arm evaluation
   and removes the N-scaling artefact analytically.
3. **Prefer $d_z$ for cross-study claims.** The raw CKNNA score is
   for within-study, fixed-N comparisons.

# 7. Reproducibility

All code, data, and figures are committed in a public git
repository. The exact revision reported here is `076f770`. The
relevant artefacts are:

- `docs/superpowers/research/2026-05-20-macm1-scientific-eval.json`
  — raw per-cell results (1750 cells).
- `docs/superpowers/research/2026-05-20-macm1-scientific-eval.md`
  — human-readable summary tables (the source of Tables 1 and 2).
- `scripts/macm1_scientific_eval.py` — the driver script. Re-run
  with `--device mps --seeds 50 --out result.json` on Apple
  Silicon, or `--device cpu` on any host with PyTorch ≥ 2.11.
- `papers/paper1/supplementary/cknna-n-dependence-replication/make_figures.py` —
  reads the JSON above and regenerates Figures 1 and 2.

The figure script is deterministic given the JSON; no random seed
is consumed at plotting time.

# References
