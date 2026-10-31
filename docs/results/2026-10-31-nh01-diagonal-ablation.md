# NH-01 diagonal ablation — results, N = 1 000 (2026-10-31)

**Pre-registration:** `docs/milestones/2026-10-31-nh01-diagonal-ablation.md`, L1 tag `prereg-nh01-diagonal-20261031` (annotated tag object `d1b9e0e` on commit `6e1b006`, tagger 2026-10-31 16:38:17 UTC; Software Heritage save requested 16:38:19 UTC, snapshot `swh:1:snp:2bb4ee0f98f91afbdb855b4000c54501f0bf6ed0`, revision `swh:1:rev:6e1b0069c49750f7cb2a01dff832a257c6157a6e`). PR #97, issue #68. Thresholds 0.020 / 0.050 confirmed by the PI at 16:35 UTC, go at 16:37 UTC, both before the tag. Question: is the finite-time optimum t\* = 7 of the confirmatory run (`docs/results/2026-10-25-nh01-confirmatory.md`, drop 0.088 [0.082, 0.092] between t = 7 and t = 100) caused by the removal of the self-coupling J_ii from the update fields?
**Run:** electron-server, fresh clone at the tag, launched **16:39:15 UTC** (58 s after the tag, 56 s after the Software Heritage request), **901 s wall** for 10/10 seeds × 2 arms, **4 800 cells**, 4 workers, `nice 19`, single-threaded BLAS, shared host. 0 non-converged probe (max 68 sweeps), 0 right-censored estimate. Code hashes, environment and timings: `results/nh01_diagonal_ablation/launch_record.md`.
**Artefacts:** `results/nh01_diagonal_ablation/{cells.jsonl, summary.json, run_meta.json, run.log, launch_record.md, SHA256SUMS}`. Every number below is read from `summary.json`, except the retrieval and spurious-state counts, which are read from `cells.jsonl`.

## Verdict (pre-registered rule, N = 1 000, seeds 200–209, T = 0, cue with 10 % flipped bits, t\* = 7 fixed from the confirmatory data, 95 % seed-level bootstrap CI)

D = α̂_c(7) − α̂_c(100) per seed and arm; ΔD = D_offdiag − D_diag paired by seed.

| Step | Pre-registered rule | Result | Label |
|---|---|---|---|
| Positive-control gate | CI of D_offdiag must overlap the confirmatory [0.082, 0.092] | D_offdiag **0.086 [0.082, 0.090]** | passed |
| Censoring guard | at most 2 of 10 seeds censored (α̂_c = 0.95) per arm at t = 7 and t = 100 | 0 censored in both arms | passed |
| C5 (primary) | `diagonal_is_cause` if D_diag ≤ 0.020 and CI upper bound < 0.050 and CI lower bound of ΔD > 0; `diagonal_is_not_cause` if D_diag ≥ 0.050 and CI lower bound > 0.020; otherwise `inconclusive` | D_diag **0.048 [0.043, 0.054]** (per seed 0.033–0.067) | **inconclusive** |
| C6 (secondary) | sign of the CI of ΔD | ΔD **0.038 [0.032, 0.043]** (positive on 10/10 seeds, 0.021–0.048) | **ΔD_positive** |

Under the pre-registered rule the study **neither demonstrates nor refutes** that the self-coupling removal causes the finite-time optimum: D_diag lies 0.002 below the « not the cause » threshold and far above the « is the cause » tolerance. The threshold was fixed before the data and is not re-tuned here; the CI is given for a continuous reading.

C6 is informative. With the diagonal kept, the drop between t = 7 and t = 100 is smaller by 0.038 (44 % of D_offdiag) but a drop of 0.048 remains, with a CI that excludes 0. Removing J_ii therefore **contributes** to the fixed-endpoint drop D (the CI of ΔD excludes 0); whether it is the cause of the finite-time optimum is not decided (C5 inconclusive). The four candidates listed in the confirmatory note and in the ReScience article (self-coupling removal, cue noise, asynchronous dynamics, finite N) remain open.

## Descriptive results (not pre-registered)

Median α̂_c over the 10 seeds, for t = 0, 1, 3, 5, 7, 10, 15, 20, 30, 100:

| Arm | Medians | Peak |
|---|---|---|
| `dream_offdiag` (replica of the confirmatory dream arm, J_ii removed) | 0.156, 0.408, 0.569, 0.607, **0.615**, 0.611, 0.600, 0.583, 0.569, 0.528 | t = 7 |
| `dream_diag` (J_ii of J(t) kept in the fields) | 0.250, **0.614**, 0.535, 0.512, 0.490, 0.476, 0.471, 0.465, 0.462, 0.441 | t = 1 |

- **Seed replication of the confirmatory curve.** On fresh seeds 200–209 the `dream_offdiag` medians match the confirmatory medians (seeds 0–9: 0.158, 0.413, 0.571, 0.603, 0.616, 0.606, 0.595, 0.582, 0.569, 0.527) within 0.005 at every t (maximum 0.0049 at t = 15); the median peaks at t = 7, the per-seed argmax is t = 7 on 5 seeds, 10 on 3, 5 and 15 on one each (confirmatory: 7 on 9 seeds, 10 on one). This is the positive control seen as a curve, not a new test of C2′.
- **The kept-diagonal arm is not a shifted copy (post-hoc).** Its median curve peaks on the grid at t = 1 (0.614, argmax t = 1 on all 10 seeds; the grid has no point in ]0, 1[ or ]1, 3[, so the peak is only located somewhere in ]0, 3[), followed by a monotone decline to 0.441 at t = 100. The per-seed drop from each seed's own peak to t = 100 averages 0.166 in this arm against 0.087 in `dream_offdiag` (max-selected, upward-biased, same bias in both arms). The pre-registered statistic, with t\* = 7 fixed from the offdiag data, sits on the descending branch of this curve. Whether keeping J_ii moves the optimum earlier and deepens it was not tested; a test of t\* in the kept-diagonal arm would need its own pre-registration.
- **Baseline at t = 0.** With J_ii = P/N kept, the Hebbian estimate is 0.250 instead of 0.156. A positive self-coupling adds an inertia term α s_i to every field; at α ≤ 0.20 this yields more retrievals (320/320 probes at α = 0.14 against 272/320; 298/320 against 3/320 at α = 0.20). The two arms therefore start from different baselines.
- **End states at high load.** At t = 7 and α = 0.50, `dream_offdiag` retrieves 318/320 probes, `dream_diag` 129/320; at t = 100, 281/320 against 0/320. In the kept-diagonal arm the 191 probes classified « spurious » at t = 7, α = 0.50 end near the cue (mean overlap with the cued pattern 0.90, retrieval threshold 0.9, cue overlap 0.80, 6 sweeps; 0.87 at t = 100), not in distinct attractors: « spurious » is the classifier residual (not retrieved, not another stored pattern). Under the inertia term the dynamics freezes close to the cue, so α̂_c in this arm depends on the 0.9 threshold more than in the offdiag arm.

## Caveats

1. Inconclusive by a margin of 0.002 on the mean, with the whole CI inside [0.043, 0.054]: the result is stable, the label is a consequence of where the pre-registered threshold was set. No threshold is moved after the fact.
2. t\* = 7 was fixed from the confirmatory data, as pre-registered; it is not the optimum of the kept-diagonal arm (see above). A statistic defined from each arm's own peak was not pre-registered and is not used for the verdict.
3. D is a fixed-endpoint difference. In the kept-diagonal arm t = 7 is not the curve maximum, so ΔD is not a difference of optimum-to-plateau drops; it isolates the effect of the diagonal on the drop between two fixed t, not on the capacity.
4. Single shared host; numpy determinism per (seed, N, load, condition) stream; cross-host reproduction not done.
5. Not tested, as declared in the pre-registration: cue noise, asynchronous versus synchronous dynamics, finite N (issue #90), T = 0.2, correlated patterns, the Frobenius-matched shrink arm.
6. In the kept-diagonal arm the end states at high load sit just below the retrieval threshold (overlap ≈ 0.87–0.90 against 0.9), so α̂_c, and hence D_diag, is partly a threshold artefact of the estimator. Reported as a caveat; the threshold is the frozen one of `run.py` and is not re-tuned.

## What this changes for NH-01

The ReScience C article (rescience-nh01, tag `submission-20261031`) lists four unisolated causes for the t\* = 7 peak and is not modified. This study does not decide whether the self-coupling removal is the cause (C5 inconclusive); C6 shows that it contributes to the fixed-endpoint drop D. The four candidates remain open. The public wording of that statement, the follow-up to the authors of the original article, and the closing of any issue are the researcher's decisions.
