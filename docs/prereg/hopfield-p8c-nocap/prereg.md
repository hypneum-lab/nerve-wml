# P8c-nocap — Pre-registration (L0 draft) 2026-11-04: dreaming kernel vs Hebbian unlearning without dose cap

**Status:** L0 draft — NOT frozen. Planned freeze tag
`prereg-hopfield-p1bis-p8c-freeze-20261104` (one annotated tag shared with
`docs/prereg/hopfield-p1bis-ext/prereg.md`). Runs started before the tag push do not count. SWHID recorded in
`docs/prereg/hopfield-p1bis-ext/freeze-manifest.md` (single manifest for both
studies) at freeze.
**Investigator:** Clément Saillant (sole investigator); implementation and
drafting delegated to an agent; the researcher merges and tags.
**Study id:** `hopfield-p8c-nocap`. **Code:** `experiments/hopfield_dream_conf/`.
**Data status at drafting:** no cell with a confirmatory seed (32001–32020)
has been computed. Smoke cells with seeds ≥ 90 000 exist (listed under
"Pilots and smoke runs"); they are not analysed and never pooled.

**L2 deposit (annotation added 2026-11-08):** restricted Zenodo record
[10.5281/zenodo.24053908](https://doi.org/10.5281/zenodo.24053908) (Hypneum Lab lot of
2026-11-08, files `prereg-hopfield-p1bis-p8c-freeze-20261104.zip` SHA-256
`fae1997be8957b6bcc063c12ffbf24fc8dc7b5a162a65c2365da8c2bc1b89772` and `prereg-hopfield-p8c-relaunch-20261105.zip` SHA-256
`b561ce5b4a8b40b1688113be2e2212f3c5921fea4052dc613cfb1f3937b6fb75`) gives the freeze tags
`prereg-hopfield-p1bis-p8c-freeze-20261104` (commit `257e4698cadd6786d2381cf61d2eef4cdbeb4475`,
`swh:1:rev:257e4698cadd6786d2381cf61d2eef4cdbeb4475`) and `prereg-hopfield-p8c-relaunch-20261105` (commit
`db4eb5ba5644df8dc5622f53f249c19d52a9d82b`, `swh:1:rev:db4eb5ba5644df8dc5622f53f249c19d52a9d82b`) a verifiable deposit date,
level L2 of PROJECT-MANAGEMENT section 4. The deposit holds this file as it was
at the tags; anything dated after the tags in this live file is not in the deposit.

## Scientific question

Does the deterministic dreaming kernel of Fachechi, Agliari, Barra (2019),
J(t) = ξ^T (1+t)(𝟙 + tC)⁻¹ ξ / √N, recover stored patterns from noisy cues
better than stochastic Hebbian unlearning (HU; Hopfield, Feinstein, Palmer
1983; Benedetti et al. 2022) **when HU is allowed to reach its own best dose**?

The exploratory pilot P8c (GB300 slot, 2026-11-01) found the kernel ahead of
"matched" stochastic dreaming only at α = 0.3 with cue noise 0.2 (+0.917),
narrowly at α = 0.3, t = 1 (+0.033 [0.024; 0.041]), and at α = 0.5 (kernel
0.903 vs stochastic 0.000 at t = 2). The relecture of 2026-11-02 flagged that
the pilot's stochastic arm was truncated: it was never run beyond
‖ΔJ‖_F ≤ 0.5 ‖J_Hebb‖_F and was stopped at the kernel's budget, whereas HU is
known to reach α_c^HU ≈ 0.6 at its optimal dose D_in (Benedetti et al. 2022,
§ I; Benedetti et al. 2023, Fig. 1). The advantage may therefore be an
artefact of this dose cap. This study removes the cap.

## Hypothesis (primary, H1)

For at least one cell (N, α, q) with (N, α) ∈ {(4 096, 0.3), (4 096, 0.4),
(4 096, 0.5), (16 384, 0.5)} and cue noise q ∈ {0.2, 0.3}, the kernel at its best t recovers at least
0.10 more cues than HU at its best dose (cross-fitted, see below). H1 is
assessed cell by cell; the global verdict lists the cells.

## Design (binding when frozen)

### Network

P = round(α N) independent ±1 patterns; couplings scaled 1/√N; **no
clipping**; zero diagonal; float32 storage, float64 for the eigendecomposition
and Frobenius norms (pilot: same scale and no clipping; see "Changes").

### Arms (one cell = one (N, α, seed); all arms on the same patterns and cues)

| Arm | Definition |
|---|---|
| hebb | J₀ = ξ^T ξ / √N, zero diagonal |
| kernel | J(t) = ξ^T (1+t)(𝟙 + tC)⁻¹ ξ / √N, C = ξ ξ^T / N, computed from the exact eigendecomposition of C (float64); t ∈ T = {0.5, 1, 1.5, 2, 3, 5, 10, ∞} (t = ∞: ξ^T C⁻¹ ξ / √N); zero diagonal |
| stoch (HU) | from J₀: batches of b ≤ 64 random ±1 states, each relaxed to a fixed point by asynchronous zero-temperature dynamics on the frozen J, then J ← J − (1/(τ_d √N)) Σ_a s_a s_a^T, zero diagonal, τ_d = 100 (step = 1 % of a pattern's weight, as in the pilot). Batch size shrinks to 1 near each checkpoint (b = ⌈remaining/2⌉, capped at 64). ONE trajectory per cell |

**No dose cap.** The HU trajectory is checkpointed at the first crossing of
each relative budget r = ‖J − J₀‖_F / ‖J₀‖_F in R = {0.2, 0.3, 0.4, 0.5, 0.6,
0.7, 0.8, 0.9} ("grid" checkpoints) and of each kernel budget r_kernel(t),
t ∈ T ("matched" checkpoints, for the secondary contrast). It runs to
r = 0.9 (beyond D_fin for α ≤ 0.5 in our N = 512–4 096 smoke cells, where HU
recovery peaks then declines), unless the per-cell HU wall-time ceiling is
reached. A checkpoint is "measured" if the realised r is within 1 % of the
target (|r_realised/r − 1| ≤ 0.01; batches end at the first crossing, the
smoke cells show gaps ≤ 0.04 %).

### Dynamics

Asynchronous sign dynamics at T = 0: in each sweep the neurons are visited in
a fresh random order; s_i flips iff s_i h_i < 0 (h_i = 0 keeps s_i); local
fields are updated incrementally; a state stops when a full sweep has no flip;
cap 2 000 sweeps (non-converged states are counted as not recovered and
reported). Smoke cells (N = 512–16 384, seeds 99901, 99902, 90002): 0
non-converged cue relaxations and 0 non-converged dream states. The same rule is used for cue relaxation and for HU dream states
(implemented in `p8c_nocap.relax_batch`, numba, float32; a batch shares the
visiting order, each state follows ordinary asynchronous dynamics).

### Measures

- **Cue recovery (primary):** two disjoint sets of 200 stored patterns, A
  (selection) and B (evaluation), drawn once per cell. For each q ∈ {0.1,
  0.2, 0.3} and each set, cues with exactly round(q N) flipped spins (without
  replacement), fixed per cell and shared by every arm and checkpoint;
  recovered if the final overlap with the pattern is ≥ 0.95. Statistic per
  arm/checkpoint: fraction recovered in A and in B.
- **Descriptive:** fraction of 512 fixed stored patterns that are fixed points
  (all aligned fields), minimum normalised stability Δ_min on that subset,
  number of HU states D, mean max-overlap of visited states with the patterns,
  non-converged counts, r_kernel(t), eigenvalue range of C.

### Grid

(N, α) ∈ {(4 096, 0.3), (4 096, 0.4), (4 096, 0.5), (16 384, 0.5)} × 20
seeds = 80 cells.

**PI decision (2026-11-04, before any confirmatory datum), for cost:** the
relecture proposed N ∈ {4 096, 16 384} × α ∈ {0.3, 0.4, 0.5}. On kxkm-ai one
N = 16 384 cell needs ≈ 12 h of stochastic (HU) dreaming (smoke measurement
below), i.e. ≈ 3.5–4 days for the full N = 16 384 row. The PI kept the full
α grid at N = 4 096 and restricted N = 16 384 to α = 0.5 (the load where the
pilot's kernel advantage was largest and where the dose-cap artefact is most
plausible). No confirmatory or smoke score informed this choice; the smoke
cells were read for timing only. The driver refuses any other (N, α) pair in
`--confirmatory` mode (`p8c_nocap.CELLS`).

### Seeds and RNG

- **Confirmatory namespace:** seeds **32001–32020**.
- **Exploratory seeds of the pilot (never pooled):** 9511–9515 (main grid,
  hi-α, bigN), 9601–9620 (tfine); P8/P8b 9505–9509.
- **Smoke/pilot namespace:** seeds ≥ 90 000.
- RNG: `SeedSequence([20261104, 2, seed, N, round(100 α)]).spawn(4)` →
  patterns, cues/sets, HU stream (each batch i uses
  `default_rng([dream_seed, i])` for the initial states and numba seed
  `dream_seed + 7919 i` for the visiting order), cue-relaxation order. The
  driver refuses seeds < 90 000 unless `--confirmatory` is passed (only after
  the freeze tag).

### Determinism

Each cell is a pure function of (seed, N, α), single process, single-threaded
BLAS/numba. Within a batch (cue set or HU batch) all states share the random
visiting order of each sweep; this is a deliberate common-random-numbers
choice (each state still follows ordinary asynchronous dynamics; the shared
order removes order noise between arms and cue sets). Outputs other than timing do not depend on the number of
workers, the order of cells or the core type: checked on kxkm-ai on
2026-11-04 (smoke seed 99902, N = 2 048, α = 0.3, P-core vs E-core, 26/26
rows identical; a second N = 2 048 run after a code-only speed change gave
26/26 identical rows).

## Analysis plan (binding when frozen)

Code: `experiments/hopfield_dream_conf/analyse.py p8c <out_dir>` (reads every
`cells.jsonl` below `<out_dir>`) and `stats.py`.

### Primary contrast: best vs best, cross-fitted (removes the dose cap)

Per seed and cell (N, α, q):
1. t̂ = argmax over T of kernel recovery on set A (ties → smallest t);
2. r̂ = argmax over R of HU recovery on set A (ties → smallest r);
3. Δ = kernel recovery on set B at t̂ − HU recovery on set B at r̂.

Selection on A and evaluation on B make Δ free of the optimistic bias of
choosing the best t or r on the evaluation cues. A seed enters only if all
8 kernel points and all 8 grid HU checkpoints are measured; a test is
analysed only if ≥ 16 of 20 seeds enter (otherwise `not_measured`).

### Secondary contrasts

- **S1 matched budget, no cap (8 tests, q ∈ {0.2, 0.3}):** t̃ = argmax over
  T of (kernel − HU at r_kernel(t)) on A; Δ = same difference on B at t̃.
  This is the pilot's comparison with the 0.5 cap removed (t = 5, 10, ∞ now
  measured).
- **S2 q = 0.1 best vs best (4 tests):** continuity with the pilot's primary
  cue level.
- Descriptive: per-t and per-r curves, t̂ and r̂ distributions, D_steps,
  Δ_min, gate quantities.

### Inference

Exact one-sided sign-flip permutation tests on the 20 per-seed Δ (all 2^20
sign patterns, ties counted): p_sup for H1: mean Δ > 0; p_inf for
H1: mean Δ < 0.10 (on 0.10 − Δ). Holm step-down within each family at
one-sided level 0.025. Reported intervals (descriptive): 95 % percentile
bootstrap (10 000 resamples, seed 20261104) and Student t 95 %.

### Per-test labels (primary family: 8 tests = 4 (N, α) cells × q ∈ {0.2, 0.3})

- `success`: mean Δ ≥ 0.10 and Holm-adjusted p_sup < 0.025;
- `below`: Holm-adjusted p_inf < 0.025 (advantage established < 0.10;
  includes "HU equals or beats the kernel");
- `floor`: not `success` and both selected arms have mean recovery on B
  < 0.05 (no arm recovers; uninformative, replaces `below`);
- `inconclusive`: otherwise.

S1 uses the same labels (with the floor rule not applied; descriptive
companion). S2 uses the same labels.

### Gate (checked first; failure voids the study)

At N = 4 096, α = 0.3, q = 0.1: mean kernel recovery on B at t̂ ≥ 0.90 and
mean Hebb recovery on B ≤ 0.05 (pilot: 1.000 and 0.000). If it fails:
`void_gate_failed`, no verdict, deviation note before any primary statistic
is read.

### Decision rule (H1)

- **Avantage du noyau soutenu dans les cellules {…}:** gate passes and at
  least one primary test is `success`; the cells are listed; no
  generalisation beyond them.
- **Avantage du noyau non soutenu:** gate passes, no `success`, and every
  primary test that is not `floor` is `below` (at least one such test).
- **Non concluant:** otherwise (including all-`floor` or any
  `not_measured`/`inconclusive` primary test without a `success`).

Reading guide written now: if the pilot advantage was a dose-cap artefact we
expect `below` in the α = 0.3–0.4 cells (HU reaches recovery at r ≈ 0.4–0.8 in
the smoke cells) and possibly at α = 0.5. The words « démontre », « réfute »
and « confirmé » are not used.

### Exclusions and missing data

No seed is excluded for its result. Missing data arise only from unmeasured
checkpoints (budget gap > 1 %, HU ceiling) or unfinished cells. Rows are
written only when a cell completes; an interrupted cell is re-run from
scratch (deterministic). No cell is re-run after its rows have been read.

## Changes relative to "Priorités pour trancher" (rank 4) and why

Rank 4 proposed: α ∈ {0.3, 0.4, 0.5}, cue noise 0.2–0.3, N = 4 096 and
16 384, paired kernel vs stochastic test without dose cap, threshold
difference ≥ 0.10 with CI lower bound > 0, 20 seeds. Kept, with:

1. **Dose cap removed by a cross-fitted best-vs-best contrast** (HU free to
   choose its dose up to r = 0.9; kernel free to choose t), as recommended by
   `etat-art.md` (« vrai HU sans plafond de dose »). The budget-matched
   comparison is kept as secondary S1 without the cap.
2. **Asynchronous dynamics instead of the pilot's synchronous block dynamics,
   float32 instead of float16/bfloat16.** Reason: HU and its D_in are defined
   with asynchronous dynamics (Benedetti 2022/2023); synchronous updates can
   enter 2-cycles and the pilot's high non-convergence rates (up to 100 % at
   N ≥ 16 384) made several cells uninformative; CPU-only host, no need for
   half precision. Consequence: this is a protocol different from the pilot
   (no pooling, no direct numerical comparison).
3. **Cue levels:** q ∈ {0.2, 0.3} primary as requested; q = 0.1 (pilot
   primary) kept as secondary S2.
4. **t grid** {0.5, 1, 1.5, 2, 3, 5, 10, ∞} (pilot {0, 0.5, 1, 2, 5, 10, 30,
   ∞}; 1.5 and 3 added around the pilot's α = 0.5 transition (t ≈ 1.6–2.0),
   30 dropped (r_kernel(30) ≈ r_kernel(∞) − 0.02)).
5. **"CI lower bound > 0"** operationalised as Holm-adjusted exact sign-flip
   p < 0.025 over the 8 primary tests, plus mean ≥ 0.10; `below` and
   `floor` branches added.
6. **Host** kxkm-ai (PI decision 2026-11-04); GPU not used.
7. **N = 16 384 restricted to α = 0.5** (PI decision 2026-11-04, cost; see
   "Grid").

## Power (from the pilot variance)

Pilot per-seed sd of the paired kernel − stochastic difference (200 cues,
`out-p8c-*/cells.jsonl`): 0.003–0.028 where neither arm is at 0 or 1
(e.g. α = 0.3, t = 0.5, q = 0.1: mean 0.110, sd 0.028, n = 5; α = 0.5,
t = 1.8: 0.435, sd 0.026, n = 20). The cross-fitted best-vs-best Δ adds
selection noise (t̂, r̂ can differ between seeds), so the table covers sd up
to 0.30. Monte Carlo (normal Δ, one-sided t approximation, 20 000 replicates,
`python -m experiments.hopfield_dream_conf.power`), per-test power of
`success`, worst Holm step (0.025/8) / best step (0.025):

| true mean \ sd | 0.03 | 0.06 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| 0.10 | 0.507/0.507 | 0.507/0.507 | 0.507/0.507 | 0.236/0.482 | 0.082/0.297 |
| 0.15 | 1.000/1.000 | 1.000/1.000 | 0.987/0.987 | 0.616/0.850 | 0.239/0.570 |
| 0.20 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 0.901/0.983 | 0.487/0.807 |
| 0.30 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 0.901/0.988 |

Power of `below` (true advantage 0 / 0.03 / 0.05): ≥ 0.97 / 0.97 / 0.73 for
sd ≤ 0.06; 0.90/0.52/0.23 (worst step) at sd 0.10. With sd ≤ 0.10, 20 seeds
detect a true advantage ≥ 0.15 with probability ≥ 0.98 and establish its
absence (true 0) with probability ≥ 0.89. If the selection noise makes sd ≥ 0.20,
the study is underpowered for effects near 0.10–0.15 and an `inconclusive`
outcome is likely; this is accepted and stated now.

## Pilots and smoke runs (exploratory; listed, never pooled)

| Run | Code | Seeds | Role |
|---|---|---|---|
| P8b `rtx-p8-hopfield-mixtures` (2026-11-01) | `p8b_hopfield_fracbudget.py` | 9505–9509 | stochastic-arm implementation reused by P8c |
| P8c `gb300-p8c-dream-kernel` main, hi-α, bigN (2026-11-01 16:17–21:36Z) | `p8c_dream_kernel.py` (sha256 in its `deps-record.txt`) | 9511–9515 | α range, N range, kernel transition, cap artefact |
| P8c tfine (2026-11-01 20:24–21:13Z) | same | 9601–9620 | variance at the α = 0.5 transition (power) |
| Smoke kxkm-ai 2026-11-04 | this branch (pre-freeze working tree) | 99901 (N = 512, 2 048), 99902 (N = 2 048, determinism), 90002 (N = 4 096 and 16 384, α = 0.5, timing) | code runs, gaps, timing, determinism; scores not analysed; the smoke cells' HU-curve shape motivated R up to 0.9 |

Archive: `electron-server:~/archives/gb300-20261101/hypneum-20261101/pilots/`.

## Forking paths (declared)

- The 0.10 threshold, α grid, N grid, cue levels and 20 seeds come from the
  relecture's rank-4 proposal.
- R up to 0.9 and the 64-state batch were chosen on smoke seeds (99901) from
  the shape and cost of the HU trajectory, not from contrasts.
- The floor threshold 0.05 and the gate thresholds (0.90, 0.05) are set here.

## What this does NOT test

- Eigenvector dreaming, Daydreaming or other unlearning variants (Benedetti
  2023); clipped couplings; correlated patterns; finite temperature.
- Basin size beyond the three cue levels; N > 16 384.
- Whether either rule is "biological".

## Compute plan, stopping rule, outputs

- **Host:** kxkm-ai (i7-14700KF, 8 P + 12 E cores, 28 threads, AVX2, 62 GB
  RAM, ~41 GB free with the resident jobs), **CPU only**, working directory
  `/ai/saisail/p1bis-p8c-20261104`, venv from
  `experiments/hopfield_dream_conf/requirements-hopfield-conf.txt`
  (hash-pinned, Python 3.12.15), `nice -n 19`, `nohup`, launcher
  `scripts/hopfield_conf/kxkm_launch.sh`: lane A = N = 16 384 (8 workers,
  memory-bound; α = 0.5 only), lane B = P1bis-ext then N = 4 096
  (12 workers); ≤ 20 workers.
- **Duration (measured on kxkm-ai smoke cells, 2026-11-04, 3–6 concurrent
  smoke processes):** N = 2 048, α = 0.5: 56 s (P-core); α = 0.3 E-core
  114 s vs P-core 55 s (E-cores ≈ 2× slower). N = 4 096, α = 0.5: 583 s,
  D = 188 550 HU states. N = 16 384, α = 0.5 (seed 90002, HU ceiling set to
  1 200 s for the probe): Hebb + 8 kernel points + evaluations 755 s; HU at
  18.5 states/s (22 209 states, r = 0.073 reached); peak RSS 4.49 GB.
  Extrapolation (D ∝ N, observed 2 048 → 4 096): N = 16 384 needs ≈ 754 000
  HU states for α = 0.5 ≈ 11.3 h + kernel/evaluations ≈ 12–13 h per cell;
  α = 0.3 / 0.4 ≈ 0.66 / 0.83 of that. N = 16 384, α = 0.5: 20 cells ≈
  250 core-hours; lane A limited to 8 workers by memory (8 × 4.5 GB) →
  3 waves (8 + 8 + 4) **≈ 36–42 h wall**. N = 4 096: 60 cells ≈ 7.5
  core-hours, 12 workers (P/E mix) → ≈ 1 h after the P1bis lane.
- **HU ceiling per cell:** 18 h of HU time (≈ 1.5× the α = 0.5
  extrapolation); remaining checkpoints `not_measured`.
- **Wall-time cap:** 72 h after launch (the launcher creates the STOP file);
  finished cells are kept; tests with < 16 seeds are `not_measured`.
- **Stopping:** `touch /ai/saisail/p1bis-p8c-20261104/STOP`.
- **Outputs:** `results/hopfield_p8c_nocap/{n4096,n16384}/{cells.jsonl,
  run_info.json, SHA256SUMS}`, `results/hopfield_p8c_nocap/verdict.json`, logs;
  results note `docs/results/<date>-hopfield-p8c-nocap.md`.

## Nothing runs before

PR merged on green CI; freeze manifest; annotated tag pushed; SWHID recorded;
PI explicit go. The launcher checks the tag, a clean tree and the venv freeze.

## References

Fachechi A., Agliari E., Barra A., *Dreaming neural networks: forgetting
spurious memories and reinforcing pure ones*, Neural Networks 112 (2019)
24–40, DOI 10.1016/j.neunet.2019.01.006. Benedetti M., Ventura E., Marinari
E., Ruocco G., Zamponi F., J. Chem. Phys. 156 (2022) 104107, DOI
10.1063/5.0084219. Benedetti M., Carillo L., Marinari E., Mézard M.,
*Eigenvector dreaming*, arXiv 2308.13445 (2023). Hopfield J.J., Feinstein
D.I., Palmer R.G., Nature 304 (1983) 158–159. Kanter I., Sompolinsky H.,
Phys. Rev. A 35 (1987) 380.
