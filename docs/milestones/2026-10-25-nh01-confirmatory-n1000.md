# NH-01 — Pre-registration amendment (L1) 2026-10-25: confirmatory run at N = 1 000

**Status:** L1 amendment — frozen by the annotated tag `prereg-nh01-confirmatory-20261025` on the commit that adds this file, pushed before the launch. Runs started before the tag do not count. Software Heritage save of the tagged commit is a PI act.
**Amends:** `docs/milestones/2026-10-25-nh01-hopfield-dream-vs-downscale.md` (pilot, tag `prereg-nh01-hopfield-dream-20261025`, verdict confirmé, PR #81 merged 2026-10-25 13:36 UTC). PI decision 2026-10-25 14:33 (QCM): confirmatory run at N = 1 000 with C2 re-stated and the finite-temperature arm made decisive for C3.
**Issue:** hypneum-lab/nerve-wml#68. **Investigator:** Clément Saillant (sole investigator); implementation and run delegated to an agent, the researcher merges.
**Date drafted:** 2026-10-25 (dated by the GitHub push time of the tag).
**Data status at drafting:** the pilot results (N ≤ 256) are public in `results/nh01/`; **no N = 1 000 cell has been computed** except the timing probe declared below, whose output is discarded. The pilot informs the predictions; it does not enter the confirmatory statistics.
**Runs excluded by this amendment:** none (the pilot stands as a pilot; its cells are not re-used).

## What changes with respect to the pilot (binding)

| Item | Pilot | Confirmatory |
|---|---|---|
| Size | N ∈ {64, 128, 256}, primary 256 | **N = 1 000** only |
| Seeds | 10 | **10** (0–9), paired design unchanged |
| Dreaming time grid | {0, 0.3, 1, 3, 10, 30, 100} | **{0, 1, 3, 5, 7, 10, 15, 20, 30, 100}** (refined around 3–30 ; 19 arms per cell) |
| Conditions | (T = 0, f = 0.10), (T = 0, f = 0), (T = 0.2, f = 0.10) | **(T = 0, f = 0.10)** and **(T = 0.2, f = 0.10)** ; the stability cue f = 0 is dropped (censored at the grid maximum in the pilot) |
| C2 | median α̂_c(dream, t) non-decreasing on the grid, tolerance 0.05 | **C2′ : unimodal with a finite interior optimum t\***, see below |
| C3 | ratio at T = 0 (theorem, exactness check) | **C3′ decided at T = 0.2** ; T = 0 stays an implementation check |
| Spurious loads (P4) | α ∈ {0.10, 0.14} | α ∈ {0.14, 0.20} (0.10 was a floor effect at N = 256) |
| Spectral shrink factor | reported | omitted (descriptive only ; two dense eigen-solves per arm at N = 1 000) |

Everything else is identical: ±1 patterns nested in α, Hebbian J, dreamed kernel J(1+t)(𝟙+tJ)⁻¹ by spectral decomposition, diagonal set to 0, Frobenius-matched shrink J·c(t), cue with exactly round(0.10·N) = 100 flipped bits, asynchronous sign dynamics with the scale-invariant tie rule (T = 0, stop at the first sweep without flip, cap 100 sweeps), Glauber dynamics at β = 5 (40 sweeps, overlaps averaged over the last 10), 32 probes per cell, load grid of 24 values from 0.04 to 0.95, retrieval thresholds 0.9 (T = 0) and 0.8 (T = 0.2), capacity estimator at the 0.5 crossing after the last load with r ≥ 0.5, seed-level percentile bootstrap (10 000 resamples, seed 20261025). Code: `experiments/nh01_hopfield_dream/confirmatory.py` calling the pilot primitives of `run.py` through a configuration object ; the pilot path is unchanged (checked bit-identical on the pilot smoke task).

## Why T = 0.2 for the decisive control

At T = 0 the matched shrink is inert by theorem (sign invariance), so C3 cannot discriminate there. At T = 0.2 (β = 5) the Hebbian network is still deep in its retrieval phase (pilot : α̂_c = 0.137 [0.132, 0.141] at N = 256 with the same estimator) while the thermal noise is large enough that the matched factor c(10) ≈ 0.92 at α ≈ 0.14 produced a measurable effect in the pilot (ratio 0.980 [0.973, 0.989], CI excluding 1). A lower temperature would make the control numerically inert again ; a higher one erodes the Hebbian retrieval phase itself. The same temperature is kept so that the confirmatory C3′ is a direct replication of the pilot's descriptive arm.

## Predictions (fixed before any N = 1 000 result, informed by the pilot)

- P1 : α̂_c(dream, t = 10) ≥ 3 × α̂_c(Hebb) at T = 0.
- P2 : the median curve t ↦ α̂_c(dream, t) at T = 0 is unimodal with its maximum at an interior t\* ∈ [5, 20], and α̂_c(t\*) − α̂_c(100) > 0 (finite optimum, not a plateau).
- P3 : at T = 0.2, α̂_c(shrink c(10)) / α̂_c(Hebb) < 1.1 (pilot : 0.98).
- P4 (secondary) : spurious final states at α ∈ {0.14, 0.20} reduced by ≥ 50 % under dream t = 10 vs Hebb.

## Decision rule (binding)

- **C1 (gain, unchanged)** — cell T = 0, f = 0.10, N = 1 000, t = 10 : mean over seeds of the paired ratio R = α̂_c(dream, 10) / α̂_c(Hebb) with 95 % CI, and of the paired difference D. Confirmed if the CI lower bound of R ≥ 3 ; refuted if the CI upper bound of D < 0.1 ; otherwise non concluant.
- **C2′ (unimodal, finite optimum)** — median over seeds m(t) of α̂_c(dream, t) on the grid (t = 0 is Hebb), t\* = argmax m(t) (first maximum). Unimodality within τ = 0.02 : m(t_{k+1}) ≥ m(t_k) − τ for t_k < t\* and m(t_{k+1}) ≤ m(t_k) + τ for t_k ≥ t\*. Label : **violated** if not unimodal (the curve rises again after its maximum or dips before it by more than τ) ; **non concluant** if unimodal but t\* is at a grid edge (t = 0 or t = 100 : no optimum inside the grid) ; **holds** otherwise. Reported alongside, not deciding : whether t\* ∈ [5, 20] (P2), the per-seed t\*, and whether the finite optimum is demonstrated (95 % CI of the paired per-seed difference α̂_c(t\*) − α̂_c(100) excluding 0 ; otherwise « plateau »).
- **C3′ (uniform shrink at T = 0.2)** — cell T = 0.2, f = 0.10, N = 1 000, t = 10 : mean over seeds of S = α̂_c(shrink c(10)) / α̂_c(Hebb) with 95 % CI. **Holds** if the CI upper bound < 1.1 ; **refutes** the hypothesis if the CI lower bound ≥ 2 (catalogue) ; **non concluant** otherwise. Implementation check : at T = 0 the same ratio must be exactly 1 on every seed ; otherwise the run is void (bug fixed, new amendment tag, voided run listed).
- **Overall** : **confirmé** iff C1 confirmed ∧ C2′ holds ∧ C3′ holds ; **réfuté** if C1 refuted ∨ C2′ violated ∨ C3′ refutes ; **non concluant** otherwise. P4 gets its own label and does not enter the overall verdict.

« Démontre / réfute » are reserved to C1, C2′, C3′ in the results note ; the rest is descriptive.

## What a negative means

C3′ refutes → uniform downscale reproduces the dreaming gain at finite temperature : pillar B is a `downscale` (catalogue rang 4 negative). C2′ violated → the dreaming time is not a budget with a single optimum ; the DR-0 reading of t fails as stated. C1 refuted at N = 1 000 → the pilot gain was a finite-size artefact. Non concluant → the thresholds (τ, 1.1, 3) are kept and the next step is more seeds, not new thresholds.

## Compute plan, stopping rule, exclusions

- Host : electron-server only (8 × Xeon X5680 without AVX, shared), venv `~/scratch/nh01/venv`, `nice 19`, 4 worker processes, single-threaded BLAS, `nohup`. Wall-time cap **8 h** : beyond it completed seeds are kept, missing seeds listed, verdict non concluant.
- Timing probe (declared) : one task with seed 999 (not a pre-registered seed) restricted to α ∈ {0.2, 0.5}, used only to project the cost ; its output is discarded and no capacity is read. Projection : see « Cost » below.
- Cells : 24 loads × 19 arms × 2 conditions = 912 per seed, 9 120 in total ; 32 probes each.
- No seed exclusion ; non-converged probes count as not retrieved ; no cell re-run after being read.
- Outputs : `results/nh01_confirmatory/{cells.jsonl.gz, summary.json, figure.png, run_meta.json, run.log, SHA256SUMS}` ; results note `docs/results/2026-10-25-nh01-confirmatory.md`.

## Cost

Timing probe run 2026-10-25 13:42–13:46 UTC on electron-server (load average ≈ 16 from other jobs, single process, `nice 19`) : 76 cells in 192.6 s, i.e. **2.53 s per cell** ; projected 2 312 s (≈ 39 min) per seed task and **≈ 1.6 h for 10 seeds with 4 workers** (≈ 3.2 h with a factor-2 contention margin), well under the 8 h cap. Decision before the tag : **10 seeds, full grid as declared above** ; no reduction.

## References

As in the pilot pre-registration (Fachechi–Agliari–Barra 2019 ; Hopfield–Feinstein–Palmer 1983 ; Amit–Gutfreund–Sompolinsky 1985 ; Kanter–Sompolinsky 1987 ; Tononi–Cirelli 2014).
