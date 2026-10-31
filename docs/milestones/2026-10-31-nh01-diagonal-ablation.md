# NH-01 — Pre-registration (L0 draft) 2026-10-31: diagonal ablation, N = 1 000

**Status:** L0 draft — NOT frozen. Tag `prereg-nh01-diagonal-20261031` has not
been created yet. Runs started before the annotated tag do not count. Software
Heritage save of the tagged commit is a PI act.
**Follows:** NH-01 pilot (tag `prereg-nh01-hopfield-dream-20261025`, PR #81);
NH-01 confirmatory run (tag `prereg-nh01-confirmatory-20261025`, PR #86,
verdict *confirmé*); NH-01 correlated-pattern follow-up (tag
`prereg-nh01-correlated-20261027`). This study is a targeted ablation of one
candidate cause; it is not an amendment to any of those registrations.
**Issue:** hypneum-lab/nerve-wml#68. **Investigator:** Clément Saillant (sole
investigator); implementation delegated to an agent, the researcher merges.
**Date drafted:** 2026-10-31 (dated by the GitHub push time of the tag, once
frozen).
**Data status at drafting:** the confirmatory run (seeds 0–9) and the
correlated-pattern run (seeds 100–109) are complete and published. **No cell
with seeds 200–209 has been computed.** The confirmatory results inform the
fixed values of t\* and the positive-control target; they do not enter the
ablation statistics.

## Scientific question

The confirmatory study (N = 1 000, seeds 0–9, T = 0, cue with
`round(0.10 · N) = 100` flipped bits, asynchronous sign dynamics, diagonal of
J set to 0 in all arms) found that the capacity of the dreaming arm is
non-monotone in t: it peaks at t\* = 7 (median α̂_c = 0.616, 9 of 10 seeds at
t\* = 7, 1 seed at t\* = 10) and loses 0.088 [0.082, 0.092] by t = 100.

The ReScience C submission (rescience-nh01) lists four unisolated candidate
causes for this finite-time optimum:

1. **Self-coupling removal** (diagonal set to 0 before dynamics);
2. Cue noise (100 flipped bits);
3. Asynchronous dynamics;
4. Finite N.

The original paper (Fachechi, Agliari, Barra 2019) keeps the diagonal in the
field sum. This study isolates **cause 1** by comparing, on the same seeds and
patterns, a `dream_offdiag` arm (diagonal zeroed, exact replica of the
confirmatory arm) against a `dream_diag` arm (diagonal kept in the update
fields), everything else held constant.

If the diagonal is the sole cause, D_diag (the drop in the kept-diagonal arm)
should be near zero while D_offdiag (positive control) reproduces the
confirmatory result.

## Design (binding when frozen)

Everything is as in the confirmatory pre-registration
`docs/milestones/2026-10-25-nh01-confirmatory-n1000.md` (N = 1 000; dreamed
kernel J(1+t)(𝟙+tJ)⁻¹ by spectral decomposition; t grid
{0, 1, 3, 5, 7, 10, 15, 20, 30, 100}; 24 loads {0.04, …, 0.95}; cue with
exactly 100 flipped bits; asynchronous sign dynamics at T = 0 with the
scale-invariant tie rule; cap 100 sweeps; 32 probes; capacity estimator at
the 0.5 crossing after the last load with r ≥ 0.5; seed-level percentile
bootstrap, 10 000 resamples, seed 20261025) **except**:

- **Conditions**: T = 0 only; the T = 0.2 (Glauber) arm is omitted. C3′ is
  not a question here (the diagonal has no effect on the Frobenius-matched
  shrink ratio, and the shrink inertness theorem holds independently).
- **Two arms** at each t: `dream_offdiag` (diagonal of J(t) set to 0 before
  the dynamics, exact replica of the confirmatory dream arm) and `dream_diag`
  (diagonal of J(t) **kept** in the update fields).
- **Seeds**: 200–209 (10 new seeds, disjoint from 0–9 and 100–109).
- **Spectral shrink factor**: omitted (secondary in the confirmatory run;
  not relevant here since there is no shrink arm).
- **No shrink arm**: this study does not repeat the Frobenius-matched control;
  it is not a question about spectral normalisation.

Code: `experiments/nh01_hopfield_dream/diagonal_ablation.py`. This file
imports the frozen functions `hebbian_kernel`, `offdiag`, `make_cues`,
`run_zero_temperature`, `capacity_from_curve`, `bootstrap_mean_ci`, `stream`,
`classify` from `run.py` verbatim, and the constants `N_CONF`, `T_GRID`,
`FLIPS` from `confirmatory.py`. Neither frozen runner is modified.

### Non-trivial aspects of the kept-diagonal arm

Two implementation details differentiate `dream_diag` from `dream_offdiag`
beyond the presence of J_ii(t) in the field:

1. **Tie-rule scale**: `run_zero_temperature` computes the zero-field tolerance
   as `TIE_REL · N · max|J|` where `TIE_REL = 1 × 10⁻¹²`. For `dream_diag`,
   `max|J|` includes the diagonal entries of J(t), which are positive and of
   order α = α(t) (they equal Σ_k v_ki² · wt_k, where wt_k > 0). For
   `dream_offdiag`, the diagonal is zero and `max|J|` is the off-diagonal
   maximum. The `dream_diag` tolerance is therefore (weakly) larger, which
   could marginally reduce the number of tie-resolution flips.
2. **Self-reinforcing bias**: when evaluating unit i, the field includes the
   term J_ii(t) · s_i. Because J_ii(t) > 0 (all eigenvalues of J are
   non-negative after the PSD clip), this term pushes the unit toward its
   current value, creating a state-dependent positive feedback absent in the
   confirmatory arm.

These differences are intrinsic to the theoretical question (they are exactly
what the original paper keeps), and both effects are expected to increase
stability. Pre-registering them here prevents post-hoc discussion about
whether they are confounds or the mechanism of interest.

### Seeding scheme

Patterns are generated as
`np.where(stream(seed, N, 1).random((p_max, N)) < 0.5, -1.0, 1.0)`
where `stream(seed, N, *ids) = np.random.default_rng([ROOT_SEED, seed, N, *ids])`
and `ROOT_SEED = 20261025`. Seeds 200–209 are numerically distinct from
0–9 and 100–109 in the RNG key; the namespace is therefore disjoint.

### Cell count

2 arms × 10 t-values × 24 loads × 10 seeds = **4 800 cells**.

(The PI's rough figure of 2 400 cells per arm counts each arm independently
over the full t grid including t = 0.)

## Predictions (fixed before any result with seeds 200–209)

- **Positive control (PC)**: D_offdiag on seeds 200–209 reproduces
  0.088 [0.082, 0.092] within bootstrap CI overlap. If it does not, the study
  stops and a deviation note is filed before the ablation statistics are
  examined (see stopping rule below).
- **P1 (diagonal is the cause)**: D_diag ≤ 0.020 (within C2′ tolerance),
  with the 95 % CI upper bound < 0.050.
- **P2 (diagonal is not the cause)**: D_diag ≥ 0.050, with the 95 % CI
  excluding 0.020.
- **P3 (descriptive)**: the full α̂_c(t) curves of both arms, the per-t median
  difference, and the count of right-censored cells (flag = `censored`,
  α̂_c = 0.95).

## Decision rule (binding when frozen)

Let D_arm = α̂_c(t\* = 7, arm) − α̂_c(t = 100, arm) per seed (t\* = 7 is
**fixed** from the confirmatory C2_unimodal result and is not re-picked on the
new data). Let ΔD = D_offdiag − D_diag. All intervals are 95 % seed-level
percentile bootstrap (10 000 resamples, seed 20261025).

### Positive-control check (gate — applied first)

D_offdiag (seeds 200–209) must have a bootstrap CI that overlaps with
[0.082, 0.092] (the confirmatory 95 % CI). If the CI does **not** overlap:

> **Study void**: the `dream_offdiag` replication failed. The run is listed as
> a deviation, no ablation verdict is issued, and the PI files a note before
> any statistical examination of `dream_diag`.

### C5 (diagonal ablation, primary)

- **diagonal_is_cause**: D_diag ≤ 0.020 AND CI upper bound of D_diag < 0.050
  AND CI of ΔD excludes 0 (lower bound > 0).
- **diagonal_is_not_cause**: D_diag ≥ 0.050 AND CI lower bound of D_diag > 0.020.
- **inconclusive**: otherwise.

ΔD is deliberately absent from diagonal_is_not_cause: if the diagonal is not
the cause, the two arms drop alike and ΔD is expected to contain 0; ΔD is
reported under C6 and does not enter that branch.

### Censoring guard (applied before C5)

If, in either arm, more than 2 of the 10 seeds are right-censored
(α̂_c = 0.95) at t = 7 or at t = 100, C5 is reported as inconclusive and a
note is filed; D is undefined under censoring.

**TBD by PI** (these thresholds block the freeze until confirmed):
- The 0.020 tolerance (currently set to match C2′) and the 0.050 upper
  threshold are *proposed* values. The PI must confirm them explicitly before
  the tag is created.

### C6 (secondary, CI of ΔD)

- **ΔD_positive**: CI lower bound of ΔD > 0 (offdiag drops more than diag).
- **ΔD_zero**: CI of ΔD contains 0.
- **ΔD_negative**: CI upper bound of ΔD < 0 (unexpected; diag drops more
  than offdiag).

« Démontre / réfute » are reserved to C5 in the results note.

## What this does NOT test

- Cue noise (f = 0.10 is fixed in both arms; zero-noise ablation is a
  separate study).
- Asynchronous dynamics (both arms use asynchronous sign dynamics; a
  synchronous-update ablation would require a separate pre-registration).
- Finite N (issue nerve-wml#90 covers the N sweep).
- T = 0.2 (thermal noise is not a question here).
- Correlated patterns (covered by the correlated-pattern pre-registration).
- The Frobenius-matched shrink arm (not relevant; there is no shrink-vs-diag
  question in this study).

## Pilots

None planned. The confirmatory study's `dream_offdiag` arm (seeds 0–9, tag
`prereg-nh01-confirmatory-20261025`, commit
`8a13a33c1c8057cf6014c50096b0596369728973`) serves as the pilot that established
t\* = 7 and the positive-control reference 0.088 [0.082, 0.092]. No new pilot
data is generated or examined before the ablation tag.

## Forking paths (declared)

- t\* = 7 comes from the confirmatory C2_unimodal.t\_star (9/10 seeds); it is
  a confirmed empirical result, not a model choice.
- The 0.020 C2′ tolerance was pre-registered in the confirmatory registration
  and is carried forward unchanged.
- The 0.050 decision threshold is **new to this registration** and is set here,
  before the run.
- The positive-control window [0.082, 0.092] is the confirmatory CI and was
  known before the tag.

## Compute plan, stopping rule, exclusions

- **Host**: electron-server only (8 × Xeon X5680, no AVX), venv
  `~/scratch/nh01/venv`, `nice 19`, 4 worker processes, single-threaded BLAS,
  `nohup`.
- **Wall-time cap**: 8 h. Beyond it, completed seeds are kept, missing seeds
  listed, C5/C6 are non concluant if fewer than 8 of 10 seeds completed.
- **Timing probe** (declared): seed 997 (not a pre-registered seed),
  α ∈ {0.20, 0.50}, output discarded.
- **Stopping rule**: if the positive-control check (gate) fails, the study
  stops before D_diag is examined; a deviation note is filed and no ablation
  verdict is issued.
- **Seed exclusion**: none. Non-converged probes count as not retrieved.
  No cell is re-run after being read.
- **Outputs**: `results/nh01_diagonal_ablation/{cells.jsonl, summary.json,
  run_meta.json, SHA256SUMS}`; `run.log` is the nohup redirect of stdout/stderr;
  results note `docs/results/2026-10-31-nh01-diagonal-ablation.md`.

## Cost

The confirmatory timing probe (2026-10-25, electron-server, load average ≈ 16)
measured **2.53 s per cell** (76 cells, 192.6 s; see
`results/nh01_confirmatory/probe.log`). This rate is used as the baseline
estimate; it is an estimate, not a guarantee.

4 800 cells at 2.53 s/cell ≈ 12 144 s total CPU time.
With 4 workers: ≈ **0.84 h wall time**; with a factor-2 contention margin:
≈ 1.7 h, well under the 8 h cap.

## Nothing runs before

PR merged on green CI; manifest entry; annotated tag created by the PI;
Software Heritage SWHID obtained; PI explicit go. The two decision-threshold
values marked TBD must be confirmed before the tag is created.

## References

As in the pilot pre-registration (Fachechi A., Agliari E., Barra A., *Dreaming
neural networks: forgetting spurious memories and reinforcing pure ones*,
Neural Networks 112 (2019) 24–40; Hopfield J.J., Feinstein D.I., Palmer R.G.,
*Unlearning has a stabilizing effect in collective memories*, Nature 304 (1983)
158–159; Amit D.J., Gutfreund H., Sompolinsky H., Phys. Rev. Lett. 55 (1985)
1530; Kanter I., Sompolinsky H., Phys. Rev. A 35 (1987) 380; Tononi G.,
Cirelli C., Cell 166 (2014) 1174–1189).
