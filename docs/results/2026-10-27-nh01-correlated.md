# NH-01 correlated-pattern results — N = 1 000 (2026-10-27)

**Pre-registration:** `docs/milestones/2026-10-27-nh01-correlated-patterns.md`, L1 tag `prereg-nh01-correlated-20261027` on commit `df9a1e7`, pushed 2026-10-27 16:48:06 UTC. Issue hypneum-lab/nerve-wml#68.
**Run:** electron-server, launched right after the tag push (`started_utc` 16:48:06 UTC, launch command issued after the push returned ; host code files checked equal by SHA-256 to `df9a1e7` before launch — the host clone is shallow at `8a13a33`, hence `git_commit` in `run_meta.json`). 2 workers, `nice 19`, single-threaded BLAS, shared host. **22 866 s wall (6.35 h)** for 30/30 (seed, level) tasks, **29 640 cells**, 0 non-converged probe (max 80 sweeps). Probe projection : 5.7 h.
**Artefacts:** `results/nh01_correlated/{cells.jsonl.gz, summary.json, figure.png, run_meta.json, run.log, probe.log, SHA256SUMS}`. Every number below is read from `summary.json` or `cells.jsonl.gz`.

## Verdict (pre-registered rule ; 10 seeds 100–109 ; cue 10 % ; 95 % seed-level bootstrap CI)

| Level m (pairwise q) | C1 ratio at t = 10, T = 0 | C2′ t\* (per-seed) | C3′ shrink ratio at T = 0.2 | Overall |
|---|---|---|---|---|
| 0 (0) | **3.82 [3.77, 3.88]** confirmed | **7** (5–10), finite optimum 0.084 [0.079, 0.089] — holds | **0.979 [0.974, 0.984]** — holds | **confirmé** |
| 0.15 (0.0225) | **9.06 [8.84, 9.29]** confirmed | **7** (7–10), 0.088 [0.085, 0.090] — holds | **0.986 [0.979, 0.993]** — holds | **confirmé** |
| 0.30 (0.09) | **44.4 [42.6, 47.0]** confirmed — 44 (denominator interpolated between loads 0.01 and 0.02; magnitude not robust) | **10** (7–10), 0.103 [0.098, 0.108] — holds | 0.600 [0.200, 1.000] on 5/10 seeds — **non calculable**, see caveat 1 | **non concluant (C3′ non calculable)** |

Overall confirmation holds at m = 0 and m = 0.15 only. At m = 0.30 the tagged code labels C3′ « holds » and the level « confirme » in `summary.json` (computed on the 5 defined seeds) ; this note reports the level as « non concluant (C3′ non calculable) » (PI ruling 2026-10-30, review of #87), and the JSON label must not be cited.

T = 0 exactness check (shrink ratio exactly 1) passed at every level. P4 (spurious states at α ∈ {0.14, 0.20}) holds at every level (0 under dreaming).

**C4 trend (primary new question)** : paired difference R(m = 0.30) − R(m = 0) = **40.6 [38.7, 43.2]** → **gain_increases**. The pilot question « does the dreaming gain decrease with correlation? » is answered negatively under the pre-registered criterion : on this generator the gain **increases** with correlation, as predicted (P2).

## How to read the numbers

The ratio is dominated by the Hebbian denominator. The dreamed capacity at t = 10 barely moves with correlation — 0.611 [0.606, 0.616], 0.616 [0.612, 0.619], 0.636 [0.633, 0.639] for m = 0, 0.15, 0.30 — while the Hebbian capacity collapses : 0.160, 0.068, 0.014. The absolute gain (dream − Hebb) grows 0.451 → 0.548 → 0.621. This pattern is consistent with the projector-rule behaviour, in which the dreamed kernel is expected to remove the ancestor bias that impairs Hebbian retrieval ; the ancestor bias itself was not measured in this run. The median curves α̂_c(t) of the three levels nearly coincide from t = 5 upwards (figure, panel c, which plots the mean curves ; the C2′ statistics use medians) ; correlation only slows the rise at small t (t = 1 : 0.406, 0.349, 0.208).

## Caveats

1. **C3′ at m = 0.30 is not computable.** At T = 0.2 the Hebbian network is at the grid floor : α̂_c = 0 (below the lowest load 0.01) on 5 of 10 seeds and 0.012–0.015 on the other 5. The pre-registered below-grid rule applies to C1 only ; C3′ is computed on the 5 defined seeds, where the matched shrink gives ratio 1 on 3 and drops below the grid on 2 (ratio 0). The CI upper bound 1.0 < 1.1 would satisfy the threshold, but the cell cannot discriminate shrink from Hebb : the criterion is reported as **non calculable** and the level as non concluant ; it **should not be cited as evidence**. Levels m = 0 and 0.15 carry the C3′ evidence. A future amendment should apply the below-grid rule to C3′ as well and extend the load grid below 0.01 at strong correlation.
2. **The m = 0.30 ratio, 44 (denominator interpolated between loads 0.01 and 0.02; magnitude not robust), reflects a Hebbian α̂_c of 0.014 interpolated between loads 0.01 and 0.02** (at T = 0 no seed is below grid, so C1 is valid, but the ratio's magnitude is fragile to the grid). The absolute capacities above are the robust reading.
3. Single correlation structure (one ancestor, uniform mutation) ; hierarchical or multi-cluster correlations, which are closer to « structured » data, are not tested.
4. One finite temperature ; finite-size estimator (paired ratios) ; one shared host, no cross-host reproduction.
5. m = 0 reproduces the independent-pattern confirmatory run on new seeds (ratio 3.82 vs 3.86, t\* = 7 in both, shrink at T = 0.2 0.979 vs 0.975) — descriptive replication.

## What this changes for NH-01

The distinction « dreaming ≠ uniform downscale » survives pattern correlation at m ≤ 0.15 with full force and, at m = 0.30, on the dreaming side ; the finite-temperature shrink control is uninformative at m = 0.30 because Hebbian retrieval is essentially gone. The dreamed kernel's capacity is nearly correlation-independent, so its advantage over Hebb grows with correlation. The VQ-codebook / MI part of NH-01 remains untested. Closing decisions are the researcher's.
