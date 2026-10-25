# NH-01 — Pre-registration (L1) 2026-10-25: dreaming Hopfield kernel vs uniform shrink (`downscale`)

**Status:** L1 — frozen by the annotated tag `prereg-nh01-hopfield-dream-20261025` on the commit that adds this file. Runs started before the tag do not count. Software Heritage save of the tagged commit is a PI act (PROJECT-MANAGEMENT §4, §8) and is requested in the PR.
**Issue:** hypneum-lab/nerve-wml#68 (NH-01 « HopfieldDreamWML : rêve ≠ downscale »). Catalogue entry NH-01 (`.github`, `docs/research/2026-10-24-revue-hypotheses/synthese/02-catalogue-nouvelles-hypotheses.md`), review R1 (T1, H-R1.1, C6).
**Track:** matière (track M) — one of the two pilots counting for the S13 GO/NO-GO. PI decision 2026-10-25: L1 then CPU pilot the same day.
**Date drafted:** 2026-10-25 (dated by the GitHub push time of the tag; git dates are not evidence of time).
**Investigator:** Clément Saillant (Hypneum Lab, sole investigator). Implementation and runs delegated to an agent; the researcher merges.
**OSF:** none for NH-01 (L1 only; L3 would follow a positive pilot).
**Data status at drafting:** no NH-01 code and no run exist. Every number below is fixed before any result.
**Scope:** pilot (N ≤ 256, CPU). Not the N = 1 000 confirmatory run of the catalogue, and not the VQ-codebook part of NH-01 (MI/H ± 0.02, « fraction parasite du codebook VQ −50 % »), which needs the VQ substrate and is explicitly out of scope here.

## Statement (verbatim, EN)

Dreaming is Hebbian unlearning at budget t (the dreamed kernel), distinct from uniform `downscale` and from `recombine`. Operationally for this pilot: in a ±1 Hopfield network, the dreaming kernel of Fachechi, Agliari & Barra (2019) changes the attractor landscape (capacity, spurious attractors) in a way that no uniform multiplicative rescaling of the same Hebbian couplings, at matched synaptic budget, can reproduce.

## Background and the analytic fact the control must respect

Hebbian couplings for P = αN random patterns ξ^μ ∈ {±1}^N: J = (1/N) Σ_μ ξ^μ ξ^μᵀ. Dreaming kernel (Fachechi–Agliari–Barra, arXiv:1810.12217):

J(t) = J (1 + t) (𝟙 + tJ)⁻¹ , equivalently J_ij(t) = (1/N) Σ_{μν} ξ_i^μ [(1 + t)(𝟙 + tC)⁻¹]_{μν} ξ_j^ν with C_{μν} = (1/N) ξ^μ·ξ^ν.

Eigenvalues λ ≥ 0 of J map to λ(1 + t)/(1 + tλ): t = 0 is Hebb, t → ∞ is the projector (pseudo-inverse) rule. The paper's statistical mechanics gives a critical load α_c(t) that rises from ≈ 0.138 (Hebb, T = 0) towards 1.

Uniform shrink J_c = c·J, c > 0. Two facts are theorems, not measurements, and are declared here so that the control cannot be misread:

1. At zero temperature the asynchronous dynamics s_i ← sign(Σ_j J_ij s_j) is invariant under c > 0. The attractors, basins and retrieval statistics of J_c equal those of J exactly. A measured ratio ≠ 1 at T = 0 is an implementation bug.
2. At finite temperature with Glauber dynamics, J_c at inverse temperature β equals J at inverse temperature cβ: shrink is a temperature rescaling. Since the Hebbian retrieval region shrinks with temperature (Amit–Gutfreund–Sompolinsky), c < 1 cannot increase capacity.

The pilot therefore quantifies the dreaming side (how much, how fast in t, at these N) and verifies the shrink side numerically at T = 0 (exactness) and at one finite temperature (descriptive). The scientific content of « rêve ≠ downscale » on this substrate is: uniform downscale cannot alter the attractor set, dreaming does.

## Matched control (binding)

Uniform shrink J_c with c chosen so that ‖J_c‖_F = ‖J(t)‖_F after removing the diagonal from both kernels, i.e. c(t) = ‖offdiag J(t)‖_F / ‖offdiag J‖_F. One shrink arm per t > 0 of the grid, same patterns, same seed, same cues, same update order as the dreaming arm it is matched to.

Why Frobenius and not spectral: the Frobenius norm is the global synaptic budget (sum of squared synaptic strengths), which is what homeostatic downscaling (SHY) reduces and what a DR-0 budget in weight space measures. The spectral norm tracks only the top eigenvalue, which is precisely the one the dreaming kernel suppresses the most ((1 + √α)² → 1), so spectral matching would over-shrink relative to the synaptic budget actually removed. The spectral-matched c is computed and reported as a descriptive quantity only.

Diagonal: J_ii is set to 0 in every arm before the dynamics (standard for retrieval dynamics; the diagonal is a constant in the energy and does not affect the thermodynamics of the paper, but it does affect zero-temperature dynamics through the self-field).

## Design

- Patterns: i.i.d. uniform ±1, P = max(1, round(αN)).
- Arms: `hebb` (= dream at t = 0), `dream(t)` for t in the grid, `shrink(c(t))` for each t > 0.
- Sizes N ∈ {64, 128, 256}. Primary N = 256 (largest). Finite-size behaviour at 64 and 128 is reported, not judged.
- Load grid α ∈ {0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95} (24 values; grid resolution ≤ 0.05).
- Dreaming time grid t ∈ {0, 0.3, 1, 3, 10, 30, 100}. The catalogue prediction is stated at t = 10.
- Seeds: 10 (0–9), each seed fixes patterns, cues and update orders for every arm of that (N, α) cell (paired design). ≥ 5 required by PROJECT-MANAGEMENT §5.
- Cue: a stored pattern with a fraction f of bits flipped, exactly round(fN) flips. Primary f = 0.10 (cue overlap 0.80). Secondary f = 0 (stability of the stored pattern itself), descriptive.
- Probes per cell: min(P, 32) stored patterns (the first 32 by index), one cue each.
- Dynamics (primary, T = 0): asynchronous, one random permutation of the N units per sweep (fresh per sweep, shared by the probes of a cell), s_i ← sign(h_i) with ties (h_i = 0) resolved by keeping the current spin; stop at the first sweep with no flip, cap 100 sweeps (non-convergence flagged and counted as not retrieved).
- Dynamics (secondary, T = 0.2, descriptive): Glauber, P(s_i = +1) = 1/(1 + exp(−2βh_i)), β = 5, 40 sweeps, overlap averaged over the last 10 sweeps.
- Measures per probe: final overlap m with the cued pattern; class of the final state: `retrieved` (m ≥ 0.9; T = 0.2: mean m ≥ 0.8), `other_stored` (max_ν |m_ν| ≥ 0.9 for ν ≠ cued pattern), `spurious` (neither); energy E = −½ sᵀ J_arm s / N under the arm's own kernel and E_hebb under the Hebbian kernel; number of sweeps to convergence.
- Capacity estimator per (arm, N, seed): r(α) = fraction of probes `retrieved`; α̂_c = the α at which the piecewise-linear interpolation of r(α) on the grid crosses 0.5 (last α with r ≥ 0.5 and first following α with r < 0.5). r(0.95) ≥ 0.5 ⇒ α̂_c = 0.95, censored and flagged; r(0.04) < 0.5 ⇒ α̂_c = 0, flagged.
- Statistics: paired per-seed ratios and differences; mean over the 10 seeds with 95 % percentile bootstrap CI (10 000 resamples of seeds, seed-level resampling, bootstrap RNG fixed at 20261025).

## Predictions (fixed before any result)

From the catalogue (NH-01) transposed to the pilot sizes, at N = 256, f = 0.10, T = 0:

- P1: α̂_c(dream, t = 10) ≥ 3 × α̂_c(hebb).
- P2: α̂_c(dream, t) is non-decreasing in t over the grid.
- P3: α̂_c(shrink c(10)) < 1.1 × α̂_c(hebb) (theorem 1 predicts a ratio of exactly 1).
- P4 (secondary): the fraction of `spurious` final states at α ∈ {0.10, 0.14} is reduced by ≥ 50 % (relative) under dream t = 10 vs hebb.

## Decision rule (binding)

Evaluated at N = 256, f = 0.10, T = 0.

- C1 (gain). Statistic: mean over seeds of the paired ratio R = α̂_c(dream, 10) / α̂_c(hebb), and the paired difference D = α̂_c(dream, 10) − α̂_c(hebb). Confirmed if the lower bound of the 95 % CI of R ≥ 3. Refuted if the upper bound of the 95 % CI of D < 0.1 (catalogue: « gain < 0,1 »). Otherwise non concluant.
- C2 (monotonicity). Median over seeds of α̂_c(dream, t) on the t grid. Violated (refutes) if any consecutive decrease exceeds 0.05 (one grid step). Otherwise holds.
- C3 (uniform shrink). Statistic: mean over seeds of S = α̂_c(shrink c(10)) / α̂_c(hebb) with 95 % CI. « shrink < 1.1× » holds if the upper bound < 1.1. The hypothesis is refuted if the lower bound ≥ 2 (catalogue: « shrink uniforme ≥ 2× »). At T = 0 any S ≠ 1 stops the analysis as an implementation failure (see Stopping rule). S at T = 0.2 is reported, descriptive.
- Overall verdict: **confirmé** iff C1 confirmed and C2 holds and C3 holds; **réfuté** if C1 refuted or C2 violated or C3 refutes; **non concluant** otherwise. « Confirmé » requires the calibrated null (C3) as PROJECT-MANAGEMENT §6 demands.
- P4 receives its own label (holds / fails) and does not enter the overall verdict.

Words « démontre / réfute » are reserved to C1–C3 in the results note; everything else is descriptive.

## What a negative means

If C3 refutes (shrink ≥ 2×, impossible at T = 0 by theorem 1, so only an implementation failure could produce it): pillar B is a `downscale`. If C1 is refuted (dreaming gains < 0.1 at t = 10 and N = 256): the catalogue's transposition of α_c(t) to a finite-N, noisy-cue estimator fails and the N = 1 000 run is not justified without an estimator change. If non concluant: finite-size variance dominates; the follow-up is N = 1 000 with the same code, not a change of thresholds.

## Stopping rule and exclusions

- One pass over the full grid, started after the tag, on electron-server only (8 cores, no AVX, `nice 19`), ≤ 4 worker processes. No cell is re-run after being read. Wall-time cap 2 h: beyond it the run is stopped, completed cells are kept, missing cells are listed and the verdict is non concluant.
- Implementation checks before reading any capacity: unit tests pass (J(0) = J; spectral form equals the pattern-space closed form on a 2-pattern case; shrink invariance at T = 0 on one cell). If S(T = 0) ≠ 1 on any cell, the run is void, the bug is fixed and a new L1 tag (amendment) precedes any new run; the voided run is listed in the amendment.
- No exclusion of seeds. Non-converged probes count as not retrieved.

## Compute plan

numpy only (no torch, no MLX). ~19 000 cells (2 temperatures × 3 N × 24 α × 13 arms × 10 seeds, plus the f = 0 secondary), each a batch of ≤ 32 probes; expected minutes to tens of minutes on electron-server. Outputs: `results/nh01/cells.jsonl` (one record per cell), `results/nh01/summary.json` (capacities, ratios, CIs, verdict), `results/nh01/figure.png`. Deterministic per seed (`numpy.random.default_rng(seed)` with a declared sub-stream per (N, α, purpose)).

## Files

- Pre-registration: this file.
- Code: `experiments/nh01_hopfield_dream/run.py` (kernels, dynamics, estimator, bootstrap, figure), `tests/unit/test_nh01_hopfield_dream.py`.
- Results note: `docs/results/2026-10-25-nh01-pilot.md`.

## References

- Fachechi A., Agliari E., Barra A., « Dreaming neural networks: forgetting spurious memories and reinforcing pure ones », Neural Networks 112 (2019) 24–40, arXiv:1810.12217.
- Hopfield J. J., Feinstein D. I., Palmer R. G., « 'Unlearning' has a stabilizing effect in collective memories », Nature 304 (1983) 158–159.
- Amit D. J., Gutfreund H., Sompolinsky H., « Storing infinite numbers of patterns in a spin-glass model of neural networks », Phys. Rev. Lett. 55 (1985) 1530 (α_c ≈ 0.138 at T = 0).
- Kanter I., Sompolinsky H., « Associative recall of memory without errors », Phys. Rev. A 35 (1987) 380 (projector rule, diagonal removed).
- Tononi G., Cirelli C., « Sleep and the price of plasticity », Neuron 81 (2014) 12–34 (SHY, uniform downscaling).
- Catalogue references Barra arXiv:2605.13721 and Marinari arXiv:2603.09384 are cited by the issue and not re-verified here.
