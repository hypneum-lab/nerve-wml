# NH-01 — Pre-registration amendment (L0 draft, target L1) 2026-10-30: size sweep N = 1 000, 2 000, 4 000

**Status:** L0 draft on a branch, not frozen, revision 4 (2026-10-31,
after three independent reviews, see §11). The file becomes binding
only through the freeze procedure of §9 : manifest in the tagged tree,
PR merged on green CI, annotated tag on the merge commit, Software
Heritage visit recorded in a separate registry. Nothing in this sweep
has been run. No cell computed before the conditions of §9.6 are met
counts.
**Amends:** `docs/milestones/2026-10-25-nh01-confirmatory-n1000.md`
(tag `prereg-nh01-confirmatory-20261025`, PR #86, merged) and, for the
rules on non-computable criteria, `docs/milestones/2026-10-27-nh01-correlated-patterns.md`
(tag `prereg-nh01-correlated-20261027`, PR #87, merged). Changes with
respect to those rules are listed explicitly in §2 and §4 ; nothing
else is a « reprise à l'identique » unless stated.
**Issue:** hypneum-lab/nerve-wml#68. **Investigator:** Clément Saillant
(sole investigator) ; drafting, implementation and run delegated to an
agent, the researcher merges and tags.
**Date drafted:** 2026-10-30, revised 2026-10-31. PI requests
2026-10-30 14:08, 2026-10-30 22:07, 2026-10-31 08:40 and 09:02 CET.
**Data status at drafting:** results at N = 1 000 are public (#86, #87,
provenance re-runs of 2026-10-30, exact reproductions). No cell at
N = 2 000 or 4 000 has been computed. The earlier runs inform the
predictions only and are never pooled : pilot N ≤ 256 (seeds 0–9, tag
`prereg-nh01-hopfield-dream-20261025`), confirmatory N = 1 000 (seeds
0–9, #86), correlated N = 1 000 (seeds 100–109, #87), timing probes
(seeds 999 and 998, discarded).

## 1. Question

Does the NH-01 Hopfield result at N = 1 000 (dreaming kernel at t = 10
multiplies the capacity estimate by ≥ 3 ; the median curve over t is
unimodal with an interior first maximum ; a Frobenius-matched uniform
shrink does not reproduce the gain at T = 0.2) also meet the same
pre-registered criteria at N = 2 000 and N = 4 000 ? Secondary : does
the matched shrink also fail to reproduce the gain at T = 0.1 and 0.3 ?

The sweep tests whether the criteria are met at each size. It does not
by itself explain a difference between sizes (§5).

## 2. Design (binding) and changes with respect to #86

| Item | Confirmatory (#86) | Size sweep |
|---|---|---|
| Size | N = 1 000 | **N ∈ {1 000, 2 000, 4 000}** |
| Seeds | 0–9 | **0–9 at every N** (PI decision 2026-10-31). The seed index is shared across sizes, but streams are keyed by (seed, N, …), so N = 2 000 and 4 000 are new draws, statistically independent of N = 1 000 ; at N = 1 000 they are the #86 draws on purpose (computational cross-host reproduction, **not** ten new statistical replications) |
| Dreaming time grid | {0, 1, 3, 5, 7, 10, 15, 20, 30, 100} | unchanged |
| Load grid | 24 values, 0.04 … 0.95 | unchanged ; global load index k = 0 … 23 |
| Conditions with all 19 arms | indices 0 = (T = 0, f = 0.10), 1 = (T = 0.2, f = 0.10) | unchanged, same indices |
| Extra conditions | none | **index 2 = (T = 0.1, f = 0.10), index 3 = (T = 0.3, f = 0.10)**, restricted to the arms `hebb`, `dream t = 10`, `shrink c(10)` (`condition_arms`) ; kept as secondary by PI decision 2026-10-31 |
| Retrieval threshold | 0.9 at T = 0, 0.8 at T = 0.2 | 0.9 at T = 0, **0.8 at T ∈ {0.1, 0.2, 0.3}** (`RETRIEVED_THRESH`) |
| Spectral shrink factor | omitted | omitted (`spectral = False`, binding : avoids 432 extra `eigvalsh` per seed task) |
| Censoring | below-grid rule for C1 only (#87, with an exception kept for « refuted ») | **general rule §4.1–4.3**, applied unconditionally (this changes `correlated.py:78`, which preserved « refuted ») |
| T = 0 implementation check | `np.isclose(…, 1)` on finite ratios | **bitwise equality of capacities with the expected count** (§4.7) |
| Task unit | one (N, seed) | one (N, seed, load) with the global load index ; bit-identity tested |
| Host | electron-server | **KXKM-AI CPU** (28 logical CPUs announced, 64 GB RAM, user `electron`, work dir `/ai/saisail/nh01-scale`) ; GPU and `gpu-swap` untouched |
| Provenance | partial (#86 review) | **§8** |

Everything else is identical to the confirmatory run : ±1 patterns
nested in α, Hebbian J, dreamed kernel J(1+t)(𝟙+tJ)⁻¹ by spectral
decomposition, diagonal set to 0, Frobenius-matched shrink J·c(t, α),
cue with exactly round(0.10·N) flipped bits, asynchronous sign dynamics
with the scale-invariant tie rule (T = 0, stop at the first sweep
without flip, cap 100 sweeps), Glauber dynamics at β = 1/T (40 sweeps,
overlaps averaged over the last 10), 32 probes per cell, capacity
estimator at the 0.5 crossing after the last load with r ≥ 0.5
(`run.py:capacity_from_curve`), seed-level percentile bootstrap (10 000
resamples, seed 20261025). c(t, α) is recorded per cell.

### 2.1 Code support (already on this branch)

`experiments/nh01_hopfield_dream/run.py:evaluate_task` accepts three
optional keys whose defaults leave every earlier output unchanged :
`alpha_indices` (global load index of each load of the task, used in
the cue and dynamics streams `stream(seed, N, 2|3, k, ci)`), `p_max`
(number of nested patterns drawn, fixed to round(0.95 N) for every
task of a seed) and `condition_arms`. `tests/unit/test_nh01_scale.py`
checks : per-load split bit-identical to the full task (independent
and correlated generators), a naive split without the global index
differs, appended restricted conditions leave the cells of conditions
0–1 unchanged, thresholds exist for every temperature.
`experiments/nh01_hopfield_dream/scheduling.py` implements the
deadline scheduler and the completeness check of §7.5 ;
`tests/unit/test_nh01_scheduling.py` tests them with a simulated clock
and also checks that `p_max` sets the shape of the pattern draw and
that a load above `p_max` is rejected. The test of default behaviour
compares two code paths of the current `run.py` ; the historical
reference is the bit-exact provenance re-run of #86 and #87, not this
test. `scale_stats.py:bootstrap_mean_ci_ext` (extended-real bootstrap
of §4.2.1, seed-ordered) and `provenance.py` (executed-code and
interpreter checks of §7.1 and §8) are tested in
`tests/unit/test_nh01_scale_stats.py` (censored counter-example of the
third review, −inf bounds, order invariance, per-size RNG seeds,
version check, late import caught in a real forked worker). The sweep
runner (`scale.py`) and its launch wrapper are written
before the freeze (§9.2) and must use these keys and this scheduler.

### 2.2 Why T ∈ {0.1, 0.3}

C3′ was decisive at one temperature (caveat 1 of #86). With the
normalisation J = ξᵀξ/N, at low load (α ≈ 0.04–0.10) the replica-
symmetric mean-field theory has a retrieval branch at both temperatures
(the first review reported m ≈ 0.997 at T = 0.1 and ≈ 0.95 at T = 0.3
for α = 0.10 from the replica-symmetric equations 8–9 of
Fachechi–Agliari–Barra ; indicative values, not recomputed here and
not used by any rule) ; with c(10) ≈ 0.92 the shrink acts as T_eff = T/c ≈ 0.11
and 0.33. This justifies the temperatures for the low-load part of the
grid only : it says nothing about retrieval from noisy cues within 40
sweeps, nor about higher loads, where the Hebbian curve is expected to
cross 0.5 below the grid floor at T = 0.3 in some seeds. That case is
handled by §4.3 (non calculable), not assumed away. C3″ is secondary
because at low temperature its success is strongly expected for c < 1
(it is a numerical comparison, not a mechanistic proof, §6).

## 3. Predictions (fixed before any N > 1 000 cell)

- P1 : for each N, R = α̂_c(dream, 10) / α̂_c(Hebb) at T = 0 has a 95 %
  CI lower bound ≥ 3 (N = 1 000 : 3.86 [3.79, 3.93]).
- P2 : for each N, the T = 0 median curve over t is unimodal within
  τ = 0.02 with an interior first maximum ; descriptively t\* ∈ [5, 20].
- P3 : for each N, S = α̂_c(shrink c(10)) / α̂_c(Hebb) at T = 0.2 has a
  CI upper bound < 1.1.
- P3″ (secondary) : the same at T = 0.1 and T = 0.3.
- P4 (secondary) : spurious final states at α ∈ {0.14, 0.20} reduced by
  ≥ 50 % under dream t = 10 vs Hebb.
- P5 (descriptive) : the dream α̂_c at t = 10, T = 0 changes by less
  than 0.03 between N = 1 000 and N = 4 000.

## 4. Decision function (binding once frozen)

Everything below is evaluated separately for each N, on the analysis
set of §7.5 (complete units only ; a size with an incomplete unit gets
« non concluant (incomplet) » for every criterion and is not analysed
further, §7.5). Labels are computed in the order
4.1 → 4.8 ; no later step changes an earlier label except where stated.

### 4.1 Capacity values and flags

For each seed s, arm a and condition, `capacity_from_curve` gives a
value and a flag : `ok` (crossing inside the grid), `below_grid` (no
load with r ≥ 0.5 ; the capacity lies in [0, 0.04), value stored 0 is
a convention, not a measurement) or `censored` (r ≥ 0.5 at the last
load ; capacity ≥ 0.95, stored 0.95 is a bound). Bounds :
below_grid → [0, 0.04) ; ok → the value ; censored → [0.95, +∞).

### 4.2 Ratio and difference criteria (C1, C3′, C3″)

For a criterion on X / Y (numerator arm X, denominator Hebb Y) :

1. A seed is **defined** iff X and Y are both `ok`. Every other flag
   combination (either side below_grid or censored) makes the seed
   **undefined** for this criterion. The rule is symmetric : no
   convention value (0 or 0.95) ever enters a point estimate.
2. **Mask.** The ratio R (or S) and, for C1, the paired difference D
   are computed on the **same** mask : the defined seeds. The estimand
   is the mean over defined seeds (complete-case population), stated as
   such in the results.
3. Let k be the number of undefined seeds. **k > 2 → label « non
   calculable »**, whatever the remaining seeds give, including a
   refutation.
4. **k ∈ {1, 2} → bound sensitivity**, exactly as in §4.2.1 below.
   A censored seed can only lower a label to « non concluant
   (censure) », never raise it.
5. **k = 0** → the label of §4.4–4.6 applies directly.

#### 4.2.1 Bound sensitivity (exhaustive)

Notation : `lab(v)` is the label rule of the criterion (§4.4 for C1
with its priority, §4.6 for C3′ and C3″) applied to a vector v of
per-seed values (R and D for C1, S for C3′/C3″), with the bootstrap
`scale_stats.py:bootstrap_mean_ci_ext` : 10 000 resamples, RNG seed
20261025 (or the per-size sequence of §4.8) reset at each call, the
same resampling indices as `run.py:bootstrap_mean_ci` for the same
number of seeds (results identical on finite data, tested), but
**every seed is kept** : this is an explicit exception to
`run.py:bootstrap_mean_ci`, which drops non-finite values, is unchanged
and is not used for §4.2.1. **Before any resampling, every vector is
sorted by increasing seed identifier** (L0, LOW, HIGH, the descriptive
quantities and the per-t curves ; t columns stay in grid order), so
that no result depends on the order in which seeds were collected
(order invariance tested). LOW and HIGH have the same number of seeds
and therefore use identical resampling indices.

```
input : defined seeds d (values from ok capacities), undefined seeds u
        (k = |u| ∈ {1, 2}), capacity bounds of §4.1 for every u :
        X ∈ [X_lo, X_hi], Y ∈ [Y_lo, Y_hi]
        (ok : X_lo = X_hi = value ; below_grid : [0, 0.04] ;
         censored : [0.95, +inf])
bounds per undefined seed, IEEE arithmetic with the conventions
        a / +inf = 0, a / 0 = +inf (a > 0) :
        ratio_lo = X_lo / Y_hi        ratio_hi = X_hi / Y_lo
        D_lo     = X_lo − Y_hi        D_hi     = X_hi − Y_lo
L0   = lab(values of d)                          # defined seeds only
LOW  = lab(values of d ∪ {ratio_lo, D_lo of every u})   # all k at once
HIGH = lab(values of d ∪ {ratio_hi, D_hi of every u})   # all k at once
if L0 == LOW == HIGH : label = L0
else                 : label = "non concluant (censure)"
report k, L0, LOW, HIGH and the bounds used
```

- **All k seeds are re-integrated simultaneously**, never one at a
  time. For C1 the lower ratio bound goes with the lower D bound (same
  completion), and the C1 priority (« confirmed » before « refuted »)
  is re-applied inside each `lab` call.
- **Why two completions suffice.** For fixed resampling indices each
  bootstrap mean is non-decreasing in every per-seed value, hence so
  are the percentile bounds. Every completion of the undefined seeds
  inside their bounds therefore gives CI bounds between those of LOW
  and HIGH ; a label shared by L0, LOW and HIGH is the label of every
  completion of the box and of the complete-case analysis.
- **Infinite bounds.** In LOW the only infinite values are D_lo = −inf
  (ratios are ≥ 0 and finite) ; in HIGH they are ratio_hi = +inf and
  D_hi = +inf. A vector therefore never holds both +inf and −inf and
  `inf − inf` is never computed (the helper raises if it would be). A
  resample mean is ±inf iff the resample holds an infinite value ;
  a percentile whose two neighbouring order statistics are finite is
  computed by `numpy.percentile` itself on the means with ±inf replaced
  by ± the largest finite float (so it is numpy's linear rule bit for
  bit, ties and signed zeros included) ; otherwise the infinite
  neighbour is returned (the lower one if the index is an integer or
  both neighbours are equal). Comparisons use the extended
  order : an upper bound +inf is not < 1.1 or < 0.1, a lower bound
  −inf is not ≥ 3, a lower bound +inf would be ≥ 2. NaN cannot occur.
  Example (third review) : nine seeds with S = 1 and one with X
  censored → HIGH gives (mean +inf, CI [1, +inf]) → not « holds » ;
  the historical helper would have dropped the seed and returned
  (1, 1, 1).
- **Initial label « non concluant ».** It is kept as « non concluant »
  only if LOW and HIGH are also « non concluant » ; any disagreement
  gives « non concluant (censure) ». A completion can never turn a
  non-concluding complete-case result into « confirmed » or « holds ».

Worked cases (from the second review ; both are now decided by the
text) :

- *C3′, k = 1.* Nine defined seeds, eight with S = 1 and one with
  S = 1.31 : L0 = non concluant (S = 1.034, CI [1, 1.103]). The tenth
  seed has X below_grid and Y = 0.05 ok, so S ∈ [0, 0.8]. HIGH (with
  0.8) gives CI [0.940, 1.093] → holds ; LOW (with 0) also holds. L0
  differs → **« non concluant (censure) »**, not « holds ».
- *C1, k = 2.* Eight defined seeds with X = 0.72, Y = 0.20 (R = 3.6)
  : L0 = confirmed. Two seeds have X censored and Y = 0.475 ok, so
  R ∈ [2, +inf]. LOW (both at 2, together) gives a lower CI bound 2.8
  < 3 → not confirmed. Labels differ → **« non concluant
  (censure) »**. (Adding one seed at a time would have kept
  « confirmed » ; that procedure is excluded.)

### 4.3 Curve criterion (C2′)

C2′ uses, for each seed, the ten capacities m_s(t), t ∈ T grid (t = 0
is Hebb), at T = 0. A seed enters C2′ iff all ten are `ok`. k > 2
excluded seeds → « non calculable » ; k ∈ {1, 2} → computed on the
remaining seeds, k reported. No bound substitution is provided for
C2′ (a design choice : the criterion is a shape check on the median of
the admissible seeds, §6).

### 4.4 C1_N (gain, T = 0, t = 10)

On the mask of §4.2 : « confirmed » if the CI lower bound of R ≥ 3 ;
otherwise « refuted » if the CI upper bound of D < 0.1 ; otherwise « non
concluant ». « Confirmed » has priority when both conditions hold (as
in `confirmatory.py`). Items 3–4 of §4.2 (with §4.2.1) then apply.

### 4.5 C2′_N (unimodal, interior first maximum)

m(t) = median over the C2′ seeds, t\* = first argmax, unimodality within
τ = 0.02 per step as in #86 (`confirmatory.py:unimodality`). « violated »
if not unimodal ; « non concluant » if t\* ∈ {0, 100} ; otherwise
**« holds »**, which means only : *the median curve is compatible with
unimodality and its first maximum is interior*. It does **not**
establish a strict optimum, the size of the drop after t\*, or
t\* ∈ [5, 20] ; those are descriptive (§4.8).

### 4.6 C3′_N (shrink at T = 0.2, t = 10)

On the mask of §4.2 : « holds » if the CI upper bound of S < 1.1 ;
« refutes » if the CI lower bound ≥ 2 ; otherwise « non concluant ».
Items 3–4 of §4.2 (with §4.2.1) then apply.

### 4.7 Implementation check (voids the run, not a scientific label)

At T = 0 the matched shrink must leave the dynamics unchanged. The
check runs **after** the completeness check of §7.5 and in two parts :

1. **Cell level, on every cell present** (complete or not) : for each
   (N, seed, α, t) where both cells exist, the condition-0 cell of
   `shrink c(t)` must equal the Hebbian cell bitwise on
   `retrieved_frac`, `other_stored_frac`, `spurious_frac`,
   `nonconverged_frac`, `sweeps` and `mean_overlap`.
2. **Capacity level, on complete units only** : the capacity of
   `shrink c(t)` equals the Hebbian capacity bitwise, with the same
   flag, for every t ≠ 0 ; expected count = 9 × (number of complete
   units), i.e. 90 per complete size and 270 for a complete sweep,
   asserted.

A comparison that **cannot be made** because a cell or unit is missing
is not a failure : it follows the « incomplet » branch of §7.5 for
that size only. A **difference observed** in any comparison that can
be made, at any size, voids the whole sweep (bug fixed, new amendment
and tag, voided run listed). The check is an identity of the dynamics
(sign invariance) : it can detect a bug, not refute anything.

### 4.8 Overall_N and secondary labels

- **Overall_N** is a function of C1_N, C2′_N and C3′_N only :
  « confirmé » iff C1_N = confirmed ∧ C2′_N = holds ∧ C3′_N = holds ;
  « réfuté » iff C1_N = refuted ∨ C2′_N = violated ∨ C3′_N = refutes ;
  otherwise « non concluant », with the reason (« non calculable : C… »,
  « censure : C… », or the non-concluding criterion). When both a
  refutation and a non-calculable criterion occur, « réfuté » is
  reported together with the non-calculable criterion.
- **C3″_N** (T = 0.1 and T = 0.3 separately) : same statistic, mask,
  §4.2 rules and thresholds as C3′ ; labels holds / refutes / non
  concluant / non calculable. **C3″ never enters Overall_N** : its
  non-calculability or failure changes only its own label.
- **P4** : own label as in #86, outside Overall_N.
- **Descriptive** (no label) : t\* ∈ [5, 20] ; per-seed t\* ; drop
  m(t\*) − m(100) with a bootstrap CI in which t\* is **re-selected in
  each resample** (selection-aware) ; R(N), dream and Hebb α̂_c per N.
- **Masks of the descriptive quantities**, fixed in advance and
  reported with their counts : R(N) on the C1 mask of that size ; the
  dream α̂_c (t = 10, T = 0) on the seeds whose dream capacity is `ok`
  ; the Hebb α̂_c on the seeds whose Hebb capacity is `ok` ; the drop
  on the C2′ seeds. Censored or below-grid values never enter a mean.
  More than 2 excluded seeds at a size → that quantity is « non
  calculable » at that size and no contrast involving it is computed.
- **Size contrasts** (descriptive) : R(4 000) − R(1 000) and the dream
  α̂_c difference (P5), each on the masks above, with bootstrap
  resamples drawn **independently for each size** : the RNG of each
  size is seeded with the sequence (20261025, N) (`seed` argument of
  `bootstrap_mean_ci_ext`), never the same
  indices at two sizes ; pairing of arms is kept within a size ; the
  seed index is not a pairing across sizes. Each contrast compares the
  two conditional populations defined by its masks ; if the masks
  differ in size between N, the contrast is read as a difference
  between those populations, not as an effect of N.
- No simultaneous-CI correction is applied ; the three sizes and every
  temperature are reported, none is selected.

« Démontre / réfute » are reserved to C1_N, C2′_N, C3′_N in the results
note. A level labelled « non calculable » or « non concluant
(censure) » is reported as such in every summary line, README row,
CHANGELOG entry and PR description, never as « confirmé (formal) ».

## 5. What a negative means

Each label states only what its statistic establishes :

- C1_N « refuted » : **the 95 % CI upper bound of the paired absolute
  gain D = α̂_c(dream, 10) − α̂_c(Hebb) is below 0.1 at that size.**
  It says the absolute gain is small ; it does not exclude R ≥ 3 (both
  can hold when the Hebbian capacity is small, e.g. Hebb 0.04 and
  dream 0.12), and it is not a statement about the ratio.
- C1_N non concluant or non calculable at some N : **the pre-registered
  gain is not established at that size**. This is not evidence of a
  finite-size effect : power, censoring or numerical differences can
  produce it. Attributing a difference to size requires the explicit
  size contrast of §4.8 with its uncertainty, on identical code and
  host.
- C3′_N « refutes » : **the 95 % CI lower bound of S is ≥ 2 at
  T = 0.2 at that size**, i.e. the matched shrink at least doubles the
  Hebbian capacity estimate. No equivalence with the dreaming gain is
  tested (the dream ratio may be much larger).
- C2′_N « violated » : the median curve over t is not unimodal within
  τ = 0.02 at that size.
- Non concluant : keep thresholds. More seeds only through a new,
  separately frozen replication on new seeds (no pooling, no
  sequential top-up of this sample).

## 6. Power and limits (declared)

From the #86 seeds at N = 1 000 : between-seed SD of R = 0.125, of S at
T = 0.2 = 0.0134. With 10 seeds and unchanged variance, 80 % power to
pass the CI bound needs a true distance to the threshold of about
(1.96 + 0.84) · SD / √10 : **0.111 for R** (true R ≥ 3.11) and **0.012
for S** (true S ≤ 1.088). With 8 seeds (k = 2) : 0.124 and 0.013. At
a true R of exactly 3, the probability of « confirmed » is ≈ 2.5 %.
These are **normal-approximation** sensitivity figures from N = 1 000
variances : they are not a calibration of the 10-seed percentile
bootstrap, nor of the decision after the censoring sensitivity of
§4.2.1 (which can only lower the probability of a positive label,
especially at k = 2), and not a guaranteed power at N = 4 000. C2′ has no error-rate or power
calibration (median of 10 curves, fixed τ) : it is a pre-registered
shape check, reported as such. The T = 0 shrink identity and the
Frobenius matching are constructions ; C3′/C3″ are numerical
comparisons of a finite dynamics, not proofs about every possible
uniform rescaling.

## 7. Compute plan, cost, stopping rule, completeness

### 7.1 Host and environment

KXKM-AI CPU only, `nice 19`, every BLAS/OpenMP pool set to one thread
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` = 1), checked at start
by the runner (`numpy.show_config()` and thread environment
recorded). Environment = the exact freeze of #86 (17 packages,
numpy 2.5.4, matplotlib 3.11.2), pinned by
`experiments/nh01_hopfield_dream/requirements-nh01-scale.txt` (wheel
sha256 from PyPI for the wheel installed in the #86 environment,
linux x86_64 cp312, no source distributions ; install with
`--require-hashes --only-binary :all:` ; checked to resolve and to
reproduce the #86 freeze in a throwaway venv on 2026-10-31).
**Approved by the PI (HITL review, 2026-10-31 09:02 CET) for file
SHA-256 `ec925b5f2232cd70758921a1b33c3b078d54d6559e2e5ae315601108eafc16e0`** ;
any change to the file voids this approval. The requirements file does
not pin the interpreter : the wrapper checks it separately with
`provenance.py:version_ok` (`sys.version_info[:3] == (3, 12, 15)` and
release level `final` ; tested). The repository `uv.lock` (numpy 2.4.4)
is not the sweep environment. The
number of logical and physical cores is recorded ; at most 24 worker
processes, so that ≥ 4 logical CPUs stay free.

### 7.2 Cells and memory

Per (N, seed) : 24 loads × (19 arms × 2 conditions + 3 arms × 2 extra
conditions) = 1 056 cells ; 30 units, **31 680 cells**, 32 probes each
(1 013 760 probes), 720 per-load tasks. Memory per worker at N = 4 000 :
19 kernels of 4 000² float64 (2.43 GB) + J, eigenbasis, patterns
(≤ 0.95 N rows) and the transposed copy of the dynamics (≈ 0.5 GB) =
≈ 2.9 GB before temporaries ; budget **3–4 GB per worker** (48–64 GB
at 16 workers, too much for a 64 GB host with services). Cap :
**≤ 12 workers while N = 4 000 tasks run**, ≤ 24 at N ≤ 2 000 ; the
final cap is set from the probe (§7.4) as ⌊(free RAM − 8 GB) / peak
RSS⌋, recorded before the freeze. A per-load task releases its arms
before returning (no overlap of two loads in one worker).

### 7.3 Cost model

From the #86 artefacts (electron-server, Xeon X5680 without AVX,
shared) : an N = 1 000 seed task took ≈ 2 420 s of wall time per
worker (reconstructed from the completion times ; contention
included). Split assumed : ≈ 200 s linear algebra (24 `eigh`, 216
kernel products and J construction, O(N³)) and ≈ 2 200 s dynamics
(Python loop over units ; between O(N) per sweep when per-step overhead
dominates and O(N²) when the 32 × N field products dominate). The extra
conditions add 144 cells of 40 Glauber sweeps : **+21 % to +28 %** of
the dynamics for mean T = 0 sweep counts of 20 to 5 (not +15 %).
Per seed task, single core at electron-server speed :

| N | linear algebra | dynamics (O(N) … O(N²)) | task | 10 seeds (core-h) |
|---|---|---|---|---|
| 1 000 | 0.2 ks | 2.7 ks | ≈ 0.8 h | ≈ 8 |
| 2 000 | 1.6 ks | 5.4 … 10.8 ks | 1.9 … 3.4 h | 19 … 34 |
| 4 000 | 13 ks | 10.8 … 43 ks | 6.6 … 15.6 h | 66 … 156 |

Total **≈ 93 … 198 core-hours**. With 12 workers on the N = 4 000
tasks and 24 on the others, **≈ 7 … 15 h wall** at electron-server
speed per core. KXKM cores are newer (AVX2) ; the split above is not
measured and the model is an order of magnitude, not a bound. The
binding figures are those of the probe.

### 7.4 Timing probe (declared, before the freeze)

On KXKM-AI : seed 999 (not a pre-registered seed), N = 2 000 and 4 000,
loads α ∈ {0.20, 0.50, 0.60, 0.95} (Hebbian transition, dream
transition, memory maximum), all four conditions, with the chosen
worker count running concurrently. Outputs discarded, no capacity
read ; recorded : per-phase times (J and `eigh`, kernels, dynamics per
condition), peak RSS per worker, projected wall time, chosen worker
caps. Written into §10 before the manifest. Only cost diagnostics are
read ; the protocol is not changed on any scientific output. If the
projection exceeds 48 h, the PI decides before the freeze (fewer extra
temperatures, or N = 4 000 on fewer seeds with the k-threshold of §4.2
rewritten for that count) ; thresholds never change.

### 7.5 Scheduling, wall-time cap, completeness

- **Scheduler** : `scheduling.py:run_with_deadline`, used by the
  runner with `max_in_flight` = the worker cap of the phase. Tasks are
  started one at a time, in the fixed order N = 4 000 (seeds 0–9,
  loads k = 0 … 23), then N = 2 000, then N = 1 000, and only when a
  worker slot is free : there is no pool-level queue of submitted but
  unstarted tasks (no `imap` over the whole list).
- **Deadline** : **72 h** on a monotonic clock from the first start.
  Once it has passed, no task is started ; every task not yet started
  is cancelled (never run) and listed ; tasks already running finish.
  The wall-time overshoot is bounded by the largest residual duration
  of the tasks running at the deadline (several may finish after it).
  Start and end times of every task are logged.
  `tests/unit/test_nh01_scheduling.py` checks these rules with a
  simulated clock (no start at or after the deadline, queue cancelled,
  in-flight bound, order).
- A unit (N, seed) is **complete** iff the set of its cell keys
  (N, seed, α, arm, t, T, f) **equals** the expected set
  (`scheduling.py:expected_cell_keys`, 1 056 keys), each present once
  (`unit_status` : no missing, duplicated or unexpected key) ; a count
  alone is not enough. A task that raises is a technical failure,
  logged, its unit incomplete.
- Only complete units are analysed. Incomplete units are archived, not
  analysed, never interpolated.
- **If any unit of a size is incomplete, every criterion of that size
  is « non concluant (incomplet) »** : no analysis on the subset of
  seeds that finished.
- No seed exclusion ; non-converged probes count as not retrieved ; no
  cell re-run after being read.
- Outputs : `results/nh01_scale/{cells.jsonl.gz, summary.json,
  figure.png, run_meta.json, run.log, probe/, provenance/,
  SHA256SUMS}` ; results note `docs/results/<date>-nh01-scale.md`.

## 8. Provenance (binding)

Written by the launch wrapper and the runner, which are both in the
tagged tree. Pre-run files are written and closed before the first
task ; post-run files after the last.

**Pre-run** (`provenance/pre/`) :
- `git rev-parse HEAD`, tag name, tag object id, target commit of the
  executing clone, which is a fresh clone **checked out at the tag** ;
  `git status --porcelain` (must be empty).
- The resolved chain **ref → tag object → commit** read from the
  GitHub API (`git/ref/tags/<tag>`, `git/tags/<object>`), raw responses
  **with HTTP headers** (server `Date`), checked against the local clone
  by the wrapper (abort on mismatch).
- The Software Heritage record of §9.6 (save request id, status
  `succeeded`, visit date, snapshot id, and the snapshot branch
  `refs/tags/<tag>` pointing to the tag), read from the SWH API before
  launch (abort if absent).
- Interpreter (`sys.executable`, version), working directory,
  command line, environment variables of §7.1, `uv pip freeze`,
  `numpy.show_config()`, host name, CPU model, logical and physical
  core counts, RAM, load average.
- **Executed-code check** (`provenance.py:check_loaded_modules`), run
  (a) in the parent after importing the runner, (b) in every worker
  process at start-up (pool initializer), (b′) **at the end of every
  task, in a `finally` block of the worker that ran it**
  (`provenance.py:run_checked`, one JSON line per task in
  `provenance/imports-<pid>.jsonl`), and (c) in the parent and in the
  analysis process at the end : for every
  module in `sys.modules` with a file, classify its resolved path as
  *environment* (under the `site-packages` of the venv built from the
  approved requirements file), *standard library* (under the recorded
  interpreter's stdlib path) or *repository* (under the clone root ;
  tested in that order, since a venv may sit inside the clone).
  Repository files
  : SHA-256 recorded and compared with `git show <tag>:<path>` ; any
  difference, or a repository module absent from the tag, aborts.
  Environment modules are counted (the venv's `uv pip freeze` is
  recorded and must equal the requirements file) ; standard-library
  modules are counted and their root recorded ; only repository files
  are hashed against the tag. Any module from another location is a
  violation. A violation found at (a), (b) or (c) aborts ; one found
  at (b′) fails the task (`ImportCheckError`), which makes its unit
  incomplete, and the run is reported as failed. A module imported
  late inside a task is caught by (b′) in that worker (tested with a
  real forked worker). The wrapper itself is hashed against the tag,
  and the interpreter is checked by `version_ok`.
- Start UTC timestamp.

**Post-run** (`provenance/post/`) : end UTC timestamp, exit status,
`git status --porcelain` (empty), completeness table of §7.5, SHA-256
of every output. `SHA256SUMS` covers outputs and provenance files.

**Confidence statement.** The tagger date is set by its author and is
not evidence of the push time. The order « tag published → run »
rests on two external records : the GitHub API response (server
`Date` header, tag visible) and the Software Heritage visit whose
snapshot contains the tag, both captured before the start. The
start and end times are written by the wrapper ; they show the order
observed by an honest wrapper, not a cryptographic proof of execution
time. The outputs are dated externally only by the post-run sequence
of §9.7 (hashes committed and **pushed**, then SWH save, then
verification that the visit snapshot contains that commit) ; a
`succeeded` status alone is not enough. The provenance re-runs of #86 and #87 (2026-10-30)
used a simpler wrapper (API JSON without headers, local times) and do
not reach this level ; they document the order observed by their
script only.

## 9. Freeze procedure

### 9.1 PI decisions (no freeze while any is open)

- [x] Extra temperatures T ∈ {0.1, 0.3} : **kept as secondary** (C3″,
      outside Overall_N) — PI 2026-10-31 08:40 CET.
- [x] Seeds : **0–9 at every N** ; N = 1 000 also serves as the
      reproduction of #86 — PI 2026-10-31 08:40 CET.
- [ ] Go on the KXKM-AI CPU booking and the worker caps of §7.2 (GPU
      services untouched) — deferred by the PI until E1-small and NH-81
      have finished (around 2026-11-02).
- [ ] Response to the probe projection if > 48 h — deferred likewise.

### 9.2 Code and environment

- [x] `run.py` options and thresholds, `scheduling.py`,
      `scale_stats.py`, `provenance.py`, their tests (§2.1), CI green.
- [ ] `experiments/nh01_hopfield_dream/scale.py` (runner, analysis of
      §4 with a unit test per branch of §4.2, §4.2.1 and §4.3–4.8 :
      k = 0, 1, 2, 3, 10 undefined seeds, every flag combination, the
      two worked cases of §4.2.1, label priority, infinite bounds,
      selection-aware drop CI, independent bootstrap across sizes,
      masks and counts, the two-part check of §4.7, completeness
      table) and the launch wrapper with the checks of §8.
- [x] Hash-pinned requirements file
      `experiments/nh01_hopfield_dream/requirements-nh01-scale.txt`
      (exact #86 freeze) — [x] **approved by the PI (HITL review,
      2026-10-31 09:02 CET, SHA-256 `ec925b5f…e0`)** ; void if the file
      changes.
- [ ] CI green ; independent review of runner, wrapper and this file.

### 9.3 Probe

- [ ] Timing probe of §7.4 on KXKM-AI after the PI go ; results
      written in §10.

### 9.4 Manifest (inside the tree that will be tagged)

- [ ] `docs/milestones/2026-10-30-nh01-scale-sweep-manifest.md`,
      committed on this branch **before** the merge : SHA-256 of this
      file, `run.py`, `confirmatory.py`, `scheduling.py`, `scale_stats.py`,
      `provenance.py`, `scale.py`, the wrapper, the tests, the
      requirements file and `uv.lock`.

### 9.5 Merge and tag

- [ ] PR merged on green CI ; the merge commit contains the manifest.
- [ ] Annotated tag `prereg-nh01-scale-<YYYYMMDD>` on that merge commit,
      message = SHA-256 of the manifest file as present in the tagged
      tree ; tag pushed.

### 9.6 External records (separate registry, never in a frozen file)

- [ ] Software Heritage save of the repository requested after the
      push, polled until `succeeded` ; the visit snapshot contains
      `refs/tags/<tag>` → tag object → merge commit.
- [ ] Registry `docs/milestones/registry/nh01-scale-freeze.json`
      (not listed in the manifest, committed afterwards on master in a
      commit that touches no file of the manifest) : tag, tag object,
      commit, tagger date, GitHub API responses with `Date` headers, SWH
      request id, visit date, snapshot id, revision SWHID
      (`swh:1:rev:…`).
- [ ] Launch only after the registry commit is pushed ; the wrapper
      re-reads both records (§8).

### 9.7 After the run (external dating of the outputs, in this order)

1. [ ] Commit `SHA256SUMS`, `summary.json`, `run_meta.json` and the
       provenance files on the results branch ; record the commit id.
2. [ ] **Push** that commit ; record `git ls-remote origin <branch>`
       showing the remote ref at that commit. No SWH request before the
       push has been confirmed.
3. [ ] Request the Software Heritage save of the repository ; poll
       until `succeeded` ; record request id and visit date.
4. [ ] Fetch the visit snapshot and **verify** that it contains the
       pushed branch pointing to the recorded commit (or a descendant
       whose history contains it), and that the `SHA256SUMS` blob of
       that commit in the archive equals the local file. Record the
       snapshot id, the revision SWHID and the raw API responses in the
       registry (§9.6 file, new entry).
5. [ ] Only when step 4 passes are the outputs reported as externally
       dated ; otherwise the results note says so.

## 10. Probe results (to be filled before the manifest)

_Empty : no probe has been run._

## 11. Responses to the reviews

The review reports are archived outside the repository with the PR
discussion ; each finding was checked against the source before the
revision that answers it.

### 11.1 First review (2026-10-30, Codex gpt-6.1-sol, blind) — answered in revision 2

| # | Finding (severity) | Response |
|---|---|---|
| 1 | Missing inputs (MAJEUR) | Partly an artefact of the review bundle (the earlier pre-registrations and the configurable `run.py` were in the repository but not in the bundle). The second pass is given the current branch tree. |
| 2 | Thresholds coherent (OK) ; override of `correlated.py:78` is a change | Stated as a change in §2. |
| 3 | C3″ veto ambiguity (BLOQUANT) | **Fixed** : §4.8, C3″ never enters Overall_N ; its non-calculability changes only its own label. |
| 4 | Asymmetric censoring, mask of D, C2′ (MAJEUR) | **Fixed** : §4.1–4.3 (flags and bounds, symmetric definition, common mask for R and D, bound sensitivity for k ∈ {1, 2}, C2′ rule). |
| 5 | C2′ « holds » read as a strict optimum (MAJEUR) | **Fixed** : §4.5 meaning restricted ; drop CI selection-aware and descriptive (§4.8). |
| 6 | C1 priority (MINEUR) | **Fixed** : §4.4. |
| 7 | « Non concluant ⇒ finite-size effect » (MAJEUR) | **Fixed** : §5. |
| 8 | Power (MAJEUR) | **Declared** : §6 (minimum distances 0.111 / 0.012, k = 2 figures, C2′ uncalibrated). |
| 9 | Seed namespaces OK ; bootstrap across sizes, multiplicity (MINEUR) | **Fixed** : independent resampling per size, no simultaneous correction declared, all sizes and temperatures reported, no sequential top-up (§4.8, §5). |
| 10 | `run.py` signature incompatible (MAJEUR) | **Rejected** : artefact of the bundle (the pilot-era `run.py` was supplied ; the merged `run.py` accepts `(n, seed, cfg)` and has `correlated_patterns`). The threshold lookup point was valid and is fixed (row 11). |
| 11 | Per-load split changes the streams ; thresholds for T = 0.1/0.3 (MAJEUR) | **Fixed in code** : `alpha_indices`, `p_max`, `condition_arms`, `RETRIEVED_THRESH` ; tests in `tests/unit/test_nh01_scale.py` (§2.1). |
| 12 | T = 0 identity is a construction (OK / MAJEUR if over-read) | **Fixed** : §4.7 (bitwise, counted, implementation check only) and §6. |
| 13 | Cost arithmetic plausible, split unmeasured (OK conditional) | Kept as order of magnitude ; split measured by the probe (§7.4). |
| 14 | +15 % optimistic ; spectral off implicit ; probe too narrow (MAJEUR) | **Fixed** : +21–28 %, `spectral = False` binding, probe at 4 loads with concurrency and per-phase times (§7.3–7.4). |
| 15 | Memory 48–64 GB at 16 workers (MAJEUR) | **Fixed** : ≤ 12 workers at N = 4 000, final cap from probe RSS, per-load release (§7.2). |
| 16 | 72 h stop and completeness (MAJEUR) | **Fixed** : §7.5 (order, check, completeness key, size-level « incomplet »). |
| 17 | Self-contradictory freeze (BLOQUANT) | **Fixed** : §9.4–9.6 (manifest in the tagged tree before merge ; SWHID and push evidence in a separate registry outside the manifest). |
| 18 | External dating (MAJEUR) | **Fixed** : §8 (HTTP `Date` headers, ref→object→commit check, SWH visit and snapshot before launch, post-run SWH save, confidence statement). |
| 19 | Executed-code identity (MAJEUR) | **Fixed** : §8 (wrapper in the tree, resolved `__file__` of every loaded module hashed against the tag, pre/post split). |
| 20 | T = 0.1/0.3 justification too general (MINEUR) | **Fixed** : §2.2 restricted to low load, finite-horizon and high-load caveats. |

### 11.2 Second review (2026-10-31, Codex gpt-6.1-sol, blind, on 076b523) — answered in revision 3

| # | Finding (severity) | Response |
|---|---|---|
| 1 | Bound sensitivity not total (BLOQUANT) | **Fixed** : §4.2.1 pseudo-code (simultaneous re-integration of the k seeds, D bounds X_lo − Y_hi / X_hi − Y_lo, coupled R/D completions, C1 priority inside each evaluation, initial « non concluant », infinite bounds, monotonicity argument) ; both counter-examples decided in the text (→ « non concluant (censure) »). |
| 2 | T = 0 check vs incompleteness ; wrong cross-reference (MAJEUR) | **Fixed** : §4.7 two-part check, missing comparison → « incomplet » branch, observed difference → whole sweep void, 90 per complete size / 270 ; §4 points to §7.5. |
| 3 | 72 h rule does not bound the queue (MAJEUR) | **Fixed** : §7.5 and `scheduling.py` (start only on a free slot, no start after the monotonic deadline, queue cancelled, running tasks finish, overshoot bound), tests with a simulated clock. |
| 4 | Negative wording exceeds the statistics (MAJEUR) | **Fixed** : §5 states what each label establishes (C1 refuted = CI upper bound of D < 0.1, not absence of a 3× ratio ; C3′ refutes = lower bound of S ≥ 2, no equivalence with the dream gain). |
| 5 | Tests do not prove `p_max` is used (MINEUR) | **Fixed** : `test_p_max_sets_the_pattern_draw` (shape of the draw) and `test_load_above_p_max_is_rejected` ; the default-path test is described as a comparison of two current code paths (§2.1). |
| 6 | Scope of the executed-code check (MINEUR) | **Fixed** : §8, checks in parent, every worker and analysis ; repository / environment / standard-library classification ; interpreter version checked separately (§7.1). |
| 7 | Masks of descriptive contrasts (MINEUR) | **Fixed** : §4.8 masks and counts per quantity, conditional-population reading, per-size RNG seeding for independence. |
| 8 | « Median of bounds not defined » (MINEUR) | **Fixed** : §4.3 wording. |
| 9 | Power is a normal approximation (MINEUR) | **Fixed** : §6. |
| 10 | m ≈ 0.997 / 0.95 not reproducible here (remark) | **Fixed** : §2.2 marks them as indicative, unused by any rule. |
| 11 | Old artefacts do not reach the new dating level (remark) | **Stated** : §8 confidence paragraph. |

### 11.3 Third review (2026-10-31, Codex gpt-6.1-sol, blind, on 556d48b) — answered in revision 4

| # | Finding (severity) | Response |
|---|---|---|
| F1 | Infinite bounds vs the historical bootstrap helper, which drops non-finite values ; NaN percentiles (BLOQUANT) | **Fixed** : `scale_stats.py:bootstrap_mean_ci_ext` (every seed kept, no `inf − inf`, extended percentiles, identical to the historical helper on finite data) ; `run.py:bootstrap_mean_ci` unchanged ; counter-example (9 × S = 1 + one censored seed) in the text (§4.2.1) and in a test. |
| F2 | Order of seeds in the bootstrap vectors not fixed (MAJEUR) | **Fixed** : every vector sorted by seed identifier before resampling (§4.2.1), enforced in the helper ; order-invariance test with the second-review vector. |
| F3 | Late imports in workers not covered (MAJEUR) | **Fixed** : check (b′) at the end of every task in a `finally` block, per PID (`provenance.py:run_checked`, §8) ; real forked-worker test. |
| F4 | `sys.version_info == (3, 12, 15)` always false (MAJEUR) | **Fixed** : `version_ok` uses `[:3]` and release level `final` (§7.1) ; tested. |
| F5 | Second SWH save may not contain the output hashes (MAJEUR) | **Fixed** : §9.7 order push → save → snapshot verification, with records ; §8 refers to it. |
| Row 19 remark | « every loaded module hashed against the tag » | **Fixed** : §8 says only repository files are hashed ; environment and standard library are classified and counted. |

### 11.4 Fourth review (2026-10-31, Codex gpt-6.1-sol, blind, targeted on F1–F5, on 6424410)

F2–F5 judged corrected. F1 : one new BLOQUANT — with an infinite
resample mean present, a percentile whose two neighbours are finite
and different raised an error instead of being interpolated.
**Fixed** after the review (commit following 8b0f56e) : numpy's linear
rule (same virtual index and lerp) is applied to finite neighbours ;
tests with heterogeneous finite values and an infinite bound of each
sign, checked against `numpy.percentile` with a finite sentinel ; the
review's expected values (+inf case : lower bound 0.9618469891270791 ;
−inf case : upper bound 1.0219528638394075) are reproduced. No other
BLOQUANT reported.

## References

As in the pilot pre-registration (Fachechi–Agliari–Barra 2019 ;
Hopfield–Feinstein–Palmer 1983 ; Amit–Gutfreund–Sompolinsky 1985 ;
Kanter–Sompolinsky 1987 ; Tononi–Cirelli 2014).
