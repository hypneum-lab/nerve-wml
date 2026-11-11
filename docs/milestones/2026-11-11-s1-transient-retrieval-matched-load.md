# Pre-registration (L0, draft) — S1: transient retrieval above capacity, dreamed against Hebbian couplings at matched relative load

**Status:** L0 draft written on 2026-11-11 at the request of the PI,
revised the same evening after two full readings (section 2). Nothing
is frozen, nothing has been run, no catalogue number is assigned.
Drafted by an agent; the PI decides whether it goes further. PI
decisions of 2026-11-11 (evening): the reading gate applies to this
test; a CPU run may happen before M3; its results are read only after
the method note is on arXiv, like Zuse-2.

## 1. Question, in plain words

A Hopfield memory above its capacity has no stable state near a stored
pattern. Clark (arXiv:2506.05303v3) shows that it can still help: from
a cue, the state first moves **towards** the pattern, then drifts away
slowly. Reading early beats the cue; waiting for convergence does not.

The lab's kernel `J(t) = J (1+t) (1 + tJ)^-1` (Fachechi, Agliari,
Barra; `dreaming_kernel` in `experiments/nh01_hopfield_dream/run.py`)
raises the capacity. The naive claim "dreaming widens the transient
window" is then a tautology: at a load between the two capacities the
dreamed network is simply below its own capacity and retrieves stably.

The only non-trivial comparison is **at the same relative distance
above each network's own capacity**. Is the transient gain larger,
smaller or the same in the dreamed network?

## 2. What is known (reading note, revised 2026-11-11 evening)

- Clark 2026, read on 2026-11-11 (library,
  `references/notes/arxiv-2506.05303.md`): graded units, no noise,
  Hebbian couplings; gain of about 0.1 to 0.15 in overlap at load 0.20
  from a cue at 0.5 to 0.75; the author names the dynamics of dreamed
  couplings as an open direction. The derivation was not checked.
- Gardner, Derrida, Mottishaw 1987, **read in full on 2026-11-11**
  (library, `references/notes/doi-10.1051_jphys_01987004805074100.md`,
  HAL copy). Facts, with PDF pages: the paper treats **parallel
  (synchronous) dynamics only** (eq. 1.1, p. 3; « only parallel
  dynamics », p. 2) and says itself that serial and parallel dynamics
  differ quantitatively at long times (p. 12). The first step is exact
  at N -> infinity: m(1) = 2 erf(m(0) / sqrt(alpha)) with the authors'
  erf(x) = (1/sqrt(2 pi)) int_0^x e^{-y^2/2} dy (eq. 3.14, p. 9); the
  threshold 2/pi is where the slope at the origin is 1, not a
  capacity: below it a single parallel step raises the overlap for
  any m(0) under the fixed point m_0(alpha). The second step is exact
  too, with the correlations kept (eq. 3.15, p. 9). At alpha = 0.4
  (N = 200, fig. 4b, p. 14) the overlap rises at t = 1 and falls
  afterwards (m(0) = 0.6 -> 0.66, 0.61, 0.35 at t = 1, 2, infinity,
  read at +/- 0.02): **the transient retrieval above capacity that S1
  wants to measure is documented there for parallel dynamics**. No load
  between 0.1 and 0.4, no time between 2 and infinity, no dreamed
  couplings. Reader's evaluation of eq. 3.14 on the S1 grid (parallel
  dynamics): gain at step 1 of +0.24 at r = 1.3, m(0) = 0.6, and
  between +0.09 and +0.31 over the whole grid.
- Fachechi, Agliari, Barra 2019, appendices **read in full on
  2026-11-11** (complement dated in the library note). Facts: alpha_c(t)
  is solved numerically at T = 0 (RS, fig. 3, p. 11); the only printed
  values are 0.138 (t = 0), about 0.4 (t = 1), about 0.8 (t = 5) and
  about 1.07 (t -> infinity); **alpha_c(10) is not printed** (read on
  the figure between 0.8 and 0.9; the fitted curve of the legend gives
  about 0.86). The paper says nothing of the regime above alpha_c(t)
  beyond « m = 0, spin glass » at equilibrium, nothing of a transient,
  nothing of « overfitting » or of a steeper landscape at large t; the
  internal fields tighten as t^-1 (p. 16). Appendix B proves the
  convergence of the discrete rule to the pseudo-inverse for C
  invertible, which requires P <= N (not written in the paper).
  Reader's calculation, outside the paper: the non-zero eigenvalues
  of J(t) are gamma (1 + t) / (1 + t gamma); dreaming compresses the
  spectrum towards 1, it does not rescale the kernel.
- Still not read: Amari and Maginu 1988 (paywalled, listed in
  `library/references/to-fetch/`); Kinzel 1986, Phys. Rev. B 33, 5086,
  the only prior art cited by GDM for **sequential** dynamics at T = 0,
  hence the closest to the update rule of S1; Agliari et al. 2023 has
  a full note of 2026-11-04 (large-t regime).
- Both directions remain defensible for the dreamed kernel. For a
  larger gain: slow regions are remnants of basins, and dreaming
  deepens basins and removes spurious states. For a smaller gain: at
  large t the kernel tends to the pseudo-inverse, where every pattern
  has the same stability and the fields are narrow.

**Tautology check (2026-11-11).** Three things are not open questions
and must not be sold as results: (a) at a load between alpha_c(0) and
alpha_c(t) the dreamed network retrieves stably, which is why the
comparison is made at matched relative load; (b) in parallel dynamics
the Hebbian gate of section 4 is passed by a margin of ten (GDM 1987,
eq. 3.14); the asynchronous value is unknown but a positive first-sweep
gain is expected; (c) with alpha_c(10) of about 0.86, the dreamed arm
at t = 10 sits at alpha of about 1.0, 1.1 and 1.4 for r = 1.15, 1.3,
1.6: **more patterns than neurons for r >= 1.3**, C singular, outside
every calculation of Fachechi 2019 and of the pseudo-inverse picture.
What is open is only the comparison of transient gains between
kernels at matched r, in asynchronous dynamics, where the loads stay
below 1. The design below is revised accordingly.

## 3. Design

- **Networks.** Binary units, zero diagonal, zero temperature,
  asynchronous updates in a random order. Kernels at t = 0 (Hebbian)
  and **t = 1 (primary dreamed arm, alpha_c about 0.4 printed by
  Fachechi 2019, so alpha stays between 0.46 and 0.64 over the whole r
  grid)**. Secondary, descriptive, bounded arms: **t = 3 at r = 1.15
  and 1.3 only** (alpha_c about 0.6 by the fitted curve of Fachechi
  2019, not printed, measured at step 0; r = 1.6 would reach alpha of
  about 0.96) and **t = 10 (`T_PRIMARY` of NH-01) at r = 1.15 only**,
  where alpha is about 1.0 (flagged: at the edge of P = N); t = 10 is
  dropped at r = 1.3 and 1.6 (P > N, section 2). Sizes N = 512 and
  2 048. **Decided by the PI on 2026-11-11 (evening): t = 1 primary,
  t = 3 and t = 10 secondary and bounded.**
- **Step 0 bis, analytic positive control of the code.** The same
  simulator run in **parallel** dynamics on the Hebbian kernel at
  N = 2 048, alpha = 0.179 (r = 1.3), cues 0.4, 0.6, 0.8, must
  reproduce m(1) = 2 erf(m(0) / sqrt(alpha)) of GDM 1987 (eq. 3.14:
  0.655, 0.843, 0.941) within 0.02, and the sign of m(2) - m(1) of
  eq. 3.15. A code that fails this is not used for the asynchronous
  test. Written down before step 1.
- **Step 0, calibration, not the test.** For each kernel and size,
  the capacity alpha_c(t, N) is measured with the estimator already
  pre-registered for NH-01 (`capacity_from_curve`: 0.5 crossing of the
  retrieval rate, retrieval threshold 0.9 from cues with 10 % of
  flipped units), on a load grid of step 0.01, 16 pattern sets. The
  three numbers per size are written down before step 1 is run.
- **Step 1, the test.** Relative loads r = alpha / alpha_c(t, N) in
  {1.15, 1.3, 1.6}; cue overlaps 0.4, 0.6, 0.8. Per cell: 16 pattern
  sets, 200 cues each. The overlap with the cued pattern is recorded
  every N / 10 single-unit updates, for 20 sweeps.
- **Measures per cue.** `gain = max_t m(t) - m(0)`; `t_opt` (time of
  the maximum, in sweeps); `final = m(20 sweeps)`;
  `loss_by_waiting = max_t m(t) - final`.
- **Primary cell.** r = 1.3, cue overlap 0.6, **t = 1 against t = 0**
  (revised 2026-11-11 evening, section 2). The other cells, including
  t = 10 at r = 1.15, are reported, not decided on.

## 4. Decision rule (to fix before any run)

Let `dG = mean gain(t = 1) - mean gain(t = 0)` in the primary cell,
with a 99 % bootstrap interval that resamples pattern sets (16
clusters), 10 000 draws, seed fixed.

0. **Gate.** In the Hebbian arm of the primary cell the 99 % interval
   of the mean gain lies above 0.02 at both sizes. Otherwise the
   transient regime is not reproduced in the lab's binary setting and
   the verdict is `non_test`.
1. **Larger with dreaming** iff the interval of dG lies above +0.05 at
   both sizes.
2. **Smaller with dreaming** iff it lies below -0.05 at both sizes.
3. **Same** iff it lies inside [-0.05, +0.05] at both sizes.
4. Otherwise `inconclusive`.

Sensitivity, reported and not decided on: the same reading with
alpha_c shifted by plus and minus 10 %, since a matched load is only
as good as the capacity estimate.

## 5. What each verdict changes

- **Larger:** dreaming helps beyond capacity too, in a regime the
  final-state grids of P8c cannot see; a per-step read-out becomes a
  standard measure of the Hopfield line.
- **Smaller:** dreaming buys capacity at the price of graceful
  degradation; to state as a limit of the kernel.
- **Same:** the transient is a property of relative load only; the
  dream changes where the capacity is, not what happens beyond it.
- **non_test:** the lab's binary asynchronous setting does not show
  the effect; nothing is said about dreaming.

## 6. Known weaknesses of this draft

- The margin of 0.05 is half of the effect size read on one published
  figure for a different model (graded, noiseless). It is a guess. The
  gate of 0.02 on the Hebbian arm is ten times under the parallel
  prediction of GDM 1987 and may be too lenient; the asynchronous
  value is not published, so it stays until step 0 bis and the
  calibration give a measured number.
- The dreamed kernel at matched r is compared at a different absolute
  load and a different spectrum (section 2); « same r » is one
  convention, « same alpha » the other, and the two can disagree. Both
  tables are reported; only the matched-r primary cell is decided on.
- alpha_c at finite N is a smooth crossing, not a sharp threshold;
  the matched load inherits that blur.
- At matched r the absolute number of patterns differs between
  kernels by a factor of several; cue noise and finite-size effects
  do not scale the same way. Two sizes mitigate, they do not remove.
- Zero temperature only. The Hopfield control of dream-of-kiki PR #100
  shows that at finite temperature the scale of the couplings matters;
  the dreamed kernel has a different norm, so a finite-temperature
  version would need a normalisation rule that is not chosen here.
- The author of the reference paper names this question as a next
  step and publishes his code.

## 7. Cost, and what is not done

CPU only, numpy; estimated minutes to a few hours [to measure on the
calibration step]. Not done: the code (per-step recording does not
exist in the NH-01 runner; step 0 bis to add); the reading of Amari
and Maginu 1988 (paywalled, PI decision on access) and Kinzel 1986
(identifier to resolve); the PI's choice of the primary dreamed arm is made (t = 1, 2026-11-11
evening). Decided by the PI on 2026-11-11
(evening): the reading gate applies; a CPU run is allowed before M3;
results are read after the method note.
