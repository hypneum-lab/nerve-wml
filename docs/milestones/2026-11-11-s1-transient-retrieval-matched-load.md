# Pre-registration (L0, draft) — S1: transient retrieval above capacity, dreamed against Hebbian couplings at matched relative load

**Status:** L0 draft written on 2026-11-11 at the request of the PI.
Nothing is frozen, nothing has been run, no catalogue number is
assigned. Drafted by an agent; the PI decides whether it goes further.

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

## 2. What is known (reading note, to complete before any freeze)

- Clark 2026, read on 2026-11-11 (library,
  `references/notes/arxiv-2506.05303.md`): graded units, no noise,
  Hebbian couplings; gain of about 0.1 to 0.15 in overlap at load 0.20
  from a cue at 0.5 to 0.75; the author names the dynamics of dreamed
  couplings as an open direction. The derivation was not checked.
- Not read: Gardner, Derrida, Mottishaw (first steps of binary
  parallel dynamics, growth of the overlap for load below 2/pi);
  Amari and Maginu; the capacity curve of the dreamed kernel as a
  function of t in Fachechi et al. 2019; Agliari et al. 2023 on the
  large-t regime.
- Both directions are defensible. For a larger gain: slow regions are
  remnants of basins, and dreaming deepens basins and removes spurious
  states. For a smaller gain: at large t the kernel tends to the
  pseudo-inverse, described as an overfitting regime, with a sharper
  landscape.

## 3. Design

- **Networks.** Binary units, zero diagonal, zero temperature,
  asynchronous updates in a random order. Kernels at t = 0 (Hebbian),
  1 and 10 (10 is `T_PRIMARY` of NH-01). Sizes N = 512 and 2 048.
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
- **Primary cell.** r = 1.3, cue overlap 0.6, t = 10 against t = 0.
  The other cells are reported, not decided on.

## 4. Decision rule (to fix before any run)

Let `dG = mean gain(t = 10) - mean gain(t = 0)` in the primary cell,
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
  figure for a different model (graded, noiseless). It is a guess.
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
exist in the NH-01 runner); the reading of the four references of
section 2; a decision of the PI on whether the reading gate applies
to this test and on whether it may run before M3.
