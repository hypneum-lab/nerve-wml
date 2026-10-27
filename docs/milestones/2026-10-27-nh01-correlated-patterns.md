# NH-01 — Pre-registration (L1) 2026-10-27: dreaming kernel vs matched shrink on correlated patterns, N = 1 000

**Status:** L1 — frozen by the annotated tag `prereg-nh01-correlated-20261027` on the commit that adds this file, pushed before the launch. Runs started before the tag do not count. Software Heritage save of the tagged commit is a PI act.
**Follows:** the NH-01 pilot (tag `prereg-nh01-hopfield-dream-20261025`, PR #81) and the N = 1 000 confirmatory run (tag `prereg-nh01-confirmatory-20261025`, PR #86, verdict confirmé on independent patterns). PI decision 2026-10-27 16:41 (« lance tout »): correlated-pattern follow-up on electron-server, ≤ 2 workers.
**Issue:** hypneum-lab/nerve-wml#68. **Investigator:** Clément Saillant (sole investigator); implementation and run delegated to an agent, the researcher merges.
**Date drafted:** 2026-10-27 (dated by the GitHub push time of the tag).
**Data status at drafting:** no correlated-pattern cell exists except the declared timing probe (seed 998, output discarded, no capacity read). Results of the pilot and of the confirmatory run (independent patterns, seeds 0–9) are known and inform the predictions.

## Question

Catalogue NH-01 (R1/C6, R7) anticipates that the dreaming kernel matters most on correlated data. Does the capacity gain of the dreaming kernel over Hebb, and its distinction from a matched uniform shrink, change when the stored patterns are correlated? In particular, does the gain decrease with correlation?

## Design (binding)

Everything is as in the confirmatory pre-registration `docs/milestones/2026-10-25-nh01-confirmatory-n1000.md` (N = 1 000 ; dreamed kernel J(1+t)(𝟙+tJ)⁻¹ by spectral decomposition, diagonal removed ; Frobenius-matched shrink J·c(t) ; t grid {0, 1, 3, 5, 7, 10, 15, 20, 30, 100} ; conditions T = 0 and T = 0.2 (Glauber, β = 5) with cues of exactly 100 flipped bits ; 32 probes ; scale-invariant tie rule ; thresholds 0.9 / 0.8 ; capacity estimator at the 0.5 crossing ; seed-level percentile bootstrap, 10 000 resamples) except:

- **Pattern generator.** For each (seed, level) one ancestor a ∈ {±1}^N is drawn uniformly; each pattern bit equals a_i with probability (1+m)/2 and −a_i otherwise, independently. The mean overlap of a pattern with the ancestor is m, the mean pairwise overlap is q = m². Patterns are nested in α as before.
- **Correlation levels** m ∈ {0, 0.15, 0.30} (q ∈ {0, 0.0225, 0.09}). m = 0 is the in-run independent-pattern control (same generator code path, new seeds), so the comparison across levels does not lean on the earlier runs. The levels are chosen so that Hebbian retrieval is degraded but not abolished on the grid : a back-of-envelope ancestor bias P·q·m < 1 gives α ≲ 0.04 at m = 0.30.
- **Load grid** extended downwards : {0.01, 0.02} ∪ the 24 previous values (26 loads), so that the Hebbian crossing stays measurable at m = 0.30.
- **Seeds** 100–109 (10 new seeds, disjoint from 0–9). For each seed the three levels use the same stream identifiers; levels are paired by seed index.
- Spectral shrink factor omitted. Records carry the level as `ancestor_m`.

Cells : 26 loads × 19 arms × 2 temperatures = 988 per (seed, level), 29 640 in total.

## Predictions (fixed before any result)

- P1 (per level) : C1, C2′ and C3′ of the confirmatory rule hold at every level.
- P2 (cross-level, primary new question) : the gain does **not** decrease with correlation ; it increases. Rationale : as t grows the dreamed kernel tends to the projector rule, whose stability does not depend on pattern correlations (Kanter–Sompolinsky), while the Hebbian capacity collapses with the ancestor bias. The ratio at t = 10 should therefore be larger at m = 0.30 than at m = 0.
- P3 (descriptive) : t\* does not move by more than one grid step with m.

## Decision rule (binding)

Per level ℓ ∈ {0, 0.15, 0.30}, on N = 1 000, cue 10 %, t = 10 :

- **C1ℓ** gain : as in the confirmatory run (CI lower bound of the paired ratio ≥ 3 → confirmed ; CI upper bound of the difference < 0.1 → refuted ; otherwise non concluant). Seeds whose Hebbian α̂_c is below the grid (no load with r ≥ 0.5) have an undefined ratio and are dropped from the ratio bootstrap ; if more than 2 of 10 seeds are in that case, C1ℓ is non concluant.
- **C2′ℓ** : unimodal median curve within τ = 0.02 with an interior maximum t\* (as in the confirmatory run).
- **C3′ℓ** : matched shrink ratio at T = 0.2 with CI upper bound < 1.1 → holds ; lower bound ≥ 2 → refutes ; otherwise non concluant. T = 0 exactness (ratio exactly 1) voids the run if it fails.
- **Overall per level** : confirmé iff C1ℓ ∧ C2′ℓ ∧ C3′ℓ ; réfuté if any refutes/violated ; non concluant otherwise.
- **C4 trend (cross-level)** : D = R(m = 0.30) − R(m = 0), paired by seed index, with R the T = 0, t = 10 ratio ; 95 % bootstrap CI over seeds. **gain_increases** if the CI lower bound > 0 ; **gain_decreases** if the CI upper bound < 0 ; **non concluant** otherwise. P2 is confirmed only by « gain_increases » ; « gain_decreases » refutes P2 and answers the PI's question positively. m = 0.15 is reported for the shape of the trend, not judged.

« Démontre / réfute » are reserved to C1ℓ, C2′ℓ, C3′ℓ and C4 in the results note.

## What a negative means

If C3′ℓ refutes at some m > 0 : on correlated data a uniform downscale reproduces the dreaming gain at finite temperature, and the confirmatory conclusion holds only for independent patterns. If C4 = gain_decreases : the dreaming advantage shrinks on structured memories, which weakens the catalogue's expectation that dreaming matters most on correlated data. If C1ℓ is non concluant at m = 0.30 because Hebb falls below the grid : the level was too strong for the estimator ; the next step is a finer low-load grid, not new thresholds.

## Compute plan, stopping rule, exclusions

- electron-server only, venv `~/scratch/nh01/venv`, `nice 19`, **2 worker processes** (host shared with kiki-flow and NH-03), single-threaded BLAS, `nohup`. Wall-time cap **12 h** : beyond it completed (seed, level) tasks are kept (cells are written per task), missing tasks listed, affected levels non concluant.
- Timing probe (declared) : seed 998, m = 0.30, loads {0.2, 0.5}, output discarded. Projection in « Cost » below.
- No seed exclusion ; non-converged probes count as not retrieved ; no cell re-run after being read.
- Outputs : `results/nh01_correlated/{cells.jsonl.gz, summary.json, figure.png, run_meta.json, run.log, probe.log, SHA256SUMS}` ; note `docs/results/2026-10-27-nh01-correlated.md`.
- Code : `experiments/nh01_hopfield_dream/correlated.py` (+ generator `correlated_patterns` in `run.py` ; pilot and confirmatory paths unchanged, pilot smoke task bit-identical).

## Cost

Timing probe run 2026-10-27 16:45–16:47 UTC on electron-server (load average ≈ 4, one process, `nice 19`) : 76 cells in 104.7 s, i.e. **1.38 s per cell** ; projected **≈ 5.7 h for 29 640 cells with 2 workers** (≈ 11.3 h with a factor-2 margin for the shared host), under the 12 h cap. Decision before the tag : **full grid, 10 seeds, 3 levels, no reduction** ; tasks are ordered level by level (m = 0, then 0.15, then 0.30), so a cap hit would cost the highest level first, which is then non concluant.

## References

As in the pilot ; plus Kanter I., Sompolinsky H., Phys. Rev. A 35 (1987) 380 (projector rule, stability independent of pattern correlations for linearly independent patterns).
