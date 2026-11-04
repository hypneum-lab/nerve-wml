# P1bis-ext — Pre-registration (L0 draft) 2026-11-04: dreaming vs two budget-matched controls, clipped Hopfield network

**Status:** L0 draft — NOT frozen. Planned freeze tag
`prereg-hopfield-p1bis-p8c-<YYYYMMDD>` (one annotated tag for this study and
the companion `docs/prereg/hopfield-p8c-nocap/prereg.md`, which share code; the
date is the UTC date of tag creation). Runs started before the tag push do not
count. Software Heritage save of the tagged commit and the SWHID are recorded
in `docs/prereg/hopfield-p1bis-ext/freeze-manifest.md` at freeze.
**Investigator:** Clément Saillant (sole investigator); implementation and
drafting delegated to an agent; the researcher merges and tags.
**Study id:** `hopfield-p1bis-ext`. **Code:** `experiments/hopfield_dream_conf/`.
**Data status at drafting:** no cell with a confirmatory seed (31001–31020)
has been computed. Smoke cells with seeds ≥ 90 000 exist (listed under
"Pilots and smoke runs"); they are not analysed and never pooled.

## Scientific question

In a Hopfield network with clipped couplings (|J_ij| ≤ A = 0.4) storing
P = 0.15 N random patterns, does Marinari dreaming (Hebbian unlearning of
relaxed states, τ_d = 100) stabilise more stored patterns than two controls
that change the couplings by **exactly the same Frobenius budget**
‖ΔJ‖_F = f · ‖J_Hebb‖_F: (d′) unlearning of random, non-relaxed states and
(g) symmetric Gaussian noise added then clipped?

This is the confirmatory version of the exploratory pilot P1bis (GB300 slot,
2026-11-01), which found b − d′ = +0.0444 [0.0437; 0.0452] and
b − g = +0.0439 [0.0425; 0.0455] at N = 2 000, f = 0.20 (9 seeds, one code),
with a plateau ρ = α from f ≥ 0.35.

**Relation to NH-01 and to the literature.** This study is on the "science
side" of NH-01 (is dreaming different from a matched non-specific weakening?),
but it does not test any claim of the NH-01 registrations (different network:
clipped couplings, stochastic dreaming, budget in Frobenius norm). The effect
is *not* presented as new: Hebbian unlearning is known to stabilise stored
patterns up to α_c^HU ≈ 0.6 at the optimal dose D_in (Benedetti, Ventura,
Marinari, Ruocco, Zamponi 2022, J. Chem. Phys., DOI 10.1063/5.0084219;
Benedetti, Carillo, Marinari, Mézard 2023, arXiv 2308.13445), and the clipped
model is that of Marinari, Rossi, Zamponi (code
`SaverioRo/hopfield-dreaming-clipping`, MIT). What this study adds is a
budget-matched comparison against two non-specific controls with clipped
synapses, written before the data.

## Hypothesis (primary, H1)

At f = 0.20, for every N ∈ {1 000, 2 000, 4 000}, the recognition rate ρ of
the dreaming arm exceeds that of each matched control by at least 0.02
(ρ units: retrieved patterns / N; the maximum is α = 0.15).

## Design (binding when frozen)

### Arms (definitions identical to the pilot driver `pilot_p1bis.py`)

| Arm | Definition |
|---|---|
| a | Hebb, clipped: patterns presented once, J_ij ← clip(J_ij + ξ_i ξ_j /(τ_l √N)), τ_l = 1, A = 0.4, zero diagonal |
| b | Marinari dreaming from J_a: random state → asynchronous zero-temperature relaxation → J ← clip(J − s s^T/(τ_d √N)), τ_d = 100. One trajectory per cell (cap 30 N steps). Checkpoint at the first crossing of each target f · ‖J_a‖_F; the scale c ∈ [0, 1] of the last step is bisected so that ‖J − J_a‖_F is within ±0.5 % of the target |
| d′ | m random ±1 states (m = number of dream steps of b at that checkpoint), not relaxed; J = clip(J_a + c M), M = −Σ s s^T /(τ_d √N) with zero diagonal; c bisected without cap to the same target |
| g | symmetric Gaussian matrix Z (zero diagonal, one draw per cell); J = clip(J_a + c Z), c bisected to the same target |

"Measured" rule (per seed, arm and f): realised budget within ±2 % of the
target (|realised/target − 1| ≤ 0.02) and the checkpoint reached. A paired
difference b − x is formed only for seeds where both b and x are measured.

### Grid

N ∈ {1 000, 2 000, 4 000} × α = 0.15 (P = round(α N)) × f ∈ {0.10, 0.20,
0.30} × 20 seeds. One cell = (N, seed); all f and arms are computed inside the
cell. 60 cells.

### Measures

- **ρ (primary)**: Marinari recognition rate — number of stored patterns
  whose asynchronous fixed point differs from them in fewer than 2 % of the
  neurons, **divided by N, not by P** (upstream convention of
  `hopfield.recognition_rate`, Marinari et al. Eq. 5; its maximum is
  α = 0.15, and the 0.02 threshold is in these units).
- **Recall from a noisy cue (secondary)**: for a fixed subset of
  min(P, 200) stored patterns, a cue with exactly round(q N) flipped spins
  (q ∈ {0.1, 0.2}, sampled without replacement), asynchronous relaxation,
  recovered if the final overlap with the pattern is ≥ 0.95. Same cues and
  same relaxation-order stream for every arm of a cell (common random
  numbers). q = 0.1 enters a secondary family; q = 0.2 is descriptive.
- **Controls**: ρ on P fresh (non-stored) patterns (`rho_fresh`, expected 0);
  realised/target budget ratio of every arm; mean and max overlap of dreamed
  states with the stored patterns (descriptive).

### Dynamics and code

Upstream `relax()` stops after 10⁵ sweeps **in total per call** (safety cap;
rows not yet relaxed when it is hit are returned as they are). Smoke check
(seed 90001, Hebb couplings, relaxation of all stored patterns as in ρ):
150/150, 300/300, 600/600 rows are fixed points at N = 1 000 / 2 000 /
4 000, i.e. the cap was not reached. The cap is kept unchanged (verbatim
upstream code).

Learning, dreaming steps and the asynchronous dynamics are the vendored
upstream module `experiments/hopfield_dream_conf/vendor/hopfield.py`
(Marinari, Rossi, Zamponi, MIT, commit
`7e4257489efd763560a6df0e4456da0fa428cc9d`, sha256
`dfaaeb7c1904a0eb2ab2ffa94b7211b36214dd7eef0182eb0d61959dcd9162af`, identical
to the file imported by the pilot), imported unchanged. The driver
`experiments/hopfield_dream_conf/p1bis_ext.py` reproduces the pilot driver
line by line for arms a, b, d′, g and adds only the f grid, the seed scheme
and the recall measure.

### Seeds and RNG

- **Confirmatory namespace:** seeds **31001–31020** (20 seeds).
- **Exploratory seeds that informed this design (never pooled):** P1
  9601–9609; P1bis 9611–9619 (GB300 slot 2026-11-01).
- **Smoke/pilot namespace:** seeds ≥ 90 000 only (see below).
- Cell RNG: `SeedSequence([20261104, 1, seed, N, round(1000 α)])` →
  one 32-bit base; legacy global NumPy stream seeded with base (patterns,
  dream initial states), numba stream base + 1, recognition-rate stream
  base + 2, Generator(base + 3) for d′/g, Generator(base + 4) for cues,
  numba stream base + 5 for cue relaxation. The driver refuses seeds < 90 000
  unless `--confirmatory` is passed (used only after the freeze tag).

### Determinism

Each cell is a pure function of (seed, N, α), computed in one process with
single-threaded BLAS and numba (`OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1`).
Outputs other than timing fields do not depend on the number of workers, the
scheduling order or the core type: checked on kxkm-ai on 2026-11-04 (smoke
seeds 99902–99903, N = 1 000: one worker pinned to a P-core vs two workers on
E-cores, 20/20 rows identical).

## Analysis plan (binding when frozen)

Code: `experiments/hopfield_dream_conf/analyse.py p1bis <out_dir>` and
`experiments/hopfield_dream_conf/stats.py`. It writes `verdict.json`.

### Statistic and inference

For each (N, f, control x ∈ {d′, g}) and each measured seed, the paired
difference Δ_s = ρ_b − ρ_x. A test is analysed only if ≥ 16 of 20 seeds are
measured (otherwise label `not_measured`).

- p_sup: exact one-sided sign-flip permutation p for H1: mean Δ > 0 (all
  2^n sign patterns, ties counted).
- p_inf: exact one-sided sign-flip p for H1: mean Δ < 0.02 (on 0.02 − Δ_s).
- Holm step-down inside each family, at one-sided level 0.025 (equivalent to
  the lower bound of a Holm-adjusted two-sided 95 % interval).
- Reported intervals (descriptive, not used for decisions): 95 % percentile
  bootstrap of the mean (10 000 resamples, seed 20261104) and Student t 95 %.

### Per-test labels

- `success`: mean Δ ≥ 0.02 **and** Holm-adjusted p_sup < 0.025;
- `below`: Holm-adjusted p_inf < 0.025 (effect established below 0.02);
- `inconclusive`: otherwise.

### Families

- **Primary (6 tests):** ρ, f = 0.20, x ∈ {d′, g}, N ∈ {1 000, 2 000, 4 000}.
- **Secondary S1 (12 tests):** ρ, f ∈ {0.10, 0.30}, both controls, three N;
  same labels and thresholds.
- **Secondary S2 (6 tests):** recall at q = 0.1, f = 0.20, both controls,
  three N; directional only: `success` = Holm p_sup < 0.025 (mean > 0),
  `below` = Holm p_inf < 0.025 with margin 0 (mean < 0).
- Everything else (q = 0.2, dreamed-state overlaps, n_steps, Hebb ρ by N,
  per-f curves) is descriptive.

### Gates (checked first; a failed gate voids the study)

- G1 (baseline replication): mean Hebb ρ at α = 0.15 lies in [0.035, 0.060]
  at every N (pilot: 0.047 ± 0.002 at N = 2 000).
- G2 (sanity): max ρ_fresh ≤ 0.005 over all rows (pilot: 0.000).

If a gate fails: label `void_gate_failed`, no H1 verdict, a deviation note is
filed before any primary statistic is read.

### Decision rule (H1)

- **H1 soutenue (supported):** gates pass and all 6 primary tests are
  `success`.
- **H1 non soutenue (not supported):** gates pass and at least one primary
  test is `below`.
- **Non concluant (inconclusive):** otherwise (including any `not_measured`
  primary test).

The words « démontre », « réfute » and « confirmé » are not used in the
results note; the labels above are the only verdict vocabulary.

### Exclusions and missing data

No seed is excluded on the basis of its result. A seed is missing for a test
only if a budget-matching rule fails (`ok = false`) or the cell did not
finish. No cell is re-run after its rows have been read. A cell interrupted by
the STOP file or a crash is re-run from scratch (deterministic, identical
output) before any analysis; its partial output is never written (rows are
written only when the cell completes).

## Changes relative to "Priorités pour trancher" (rank 3) and why

The relecture of 2026-11-02 (bilan, rank 3) proposed: f ∈ {0.10, 0.20,
0.30} × N ∈ {1 000, 2 000, 4 000}, α = 0.15, two matched controls, ρ and
noisy-cue recall, threshold b − d′ ≥ 0.02 with CI lower bound > 0 at
f = 0.20, 20 seeds. Kept as is, except:

1. **Primary family widened from "b − d′ at f = 0.20" to b − d′ *and* b − g
   at all three N (6 tests, Holm).** Reason: the pilot reported both controls
   and three N; requiring the effect against both controls and at every N is
   stricter than the single contrast, so this cannot make a positive verdict
   easier (power stays ≈ 1, see below).
2. **"CI lower bound > 0" operationalised as Holm-adjusted one-sided exact
   sign-flip p < 0.025.** Reason: per-seed ρ differences are discrete
   (steps of 1/N) and may have tiny variance; the exact permutation test
   needs no normality and is valid with ties; the Holm adjustment controls
   the family-wise error over the 6 tests.
3. **A `below` branch (effect < 0.02 established)** so that a null result is
   distinguishable from an inconclusive one.
4. **Recall from a noisy cue made a directional secondary family** (no pilot
   value exists to set a magnitude threshold).
5. **Host** kxkm-ai instead of electron-server (PI decision 2026-11-04).

## Power (from the pilot variance)

Pilot (9 seeds, `out-run/grid.jsonl`): at f = 0.20 the per-seed sd of the
paired differences is 0.0033/0.0041 (N = 1 000), 0.0013/0.0024 (2 000),
0.0013/0.0010 (4 000) for b − d′ / b − g; means 0.042–0.047. Monte Carlo
(normal differences, one-sided t approximation of the sign-flip test,
20 000 replicates, `python -m experiments.hopfield_dream_conf.power`), power
of `success` per test, worst Holm step (level 0.025/6) / best step (0.025):

| true mean \ sd | 0.0041 | 0.0082 | 0.0164 |
|---|---|---|---|
| 0.020 | 0.507/0.507 | 0.507/0.507 | 0.507/0.507 |
| 0.025 | 1.000/1.000 | 0.997/0.997 | 0.913/0.913 |
| 0.030 | 1.000/1.000 | 1.000/1.000 | 0.997/0.997 |
| 0.042 (pilot) | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 |

At the pilot effect and up to four times the pilot sd, the probability that
all six primary tests succeed is ≥ 0.99 if the pilot effect holds. Power of
`below` when the true effect is 0.01: 1.000 (sd 0.0041), 0.988/0.999
(sd 0.0082). A true effect exactly at 0.02 has a 50 % chance of
`success` by construction of the point-estimate rule; this is accepted.
The pilot sds come from 9 seeds of one code and may understate the
confirmatory variance; the table covers ×4.

## Pilots and smoke runs (exploratory; listed, never pooled)

| Run | Code | Seeds | Role |
|---|---|---|---|
| P1 `cpu-p1-hopfield-bounded` (GB300 slot, 2026-11-01) | `pilot_p1.py` + vendored `hopfield.py` | 9601–9609 | showed the step-count budget was mis-dosed; motivated f · ‖J_Hebb‖ |
| P1bis `cpu-p1bis-hopfield-fracbudget` (2026-11-01 14:47–15:09Z) | `pilot_p1bis.py` (sha256 in its `SHA256SUMS`) | 9611–9619 | set f = 0.20, the 0.02 threshold (≈ half of the pilot effect), the f grid, N and 20 seeds |
| Smoke kxkm-ai 2026-11-04 | this branch (pre-freeze working tree) | 90001 (N = 1 000, 2 000, 4 000); 99901 (N = 500); 99902–99903 (N = 1 000, determinism) | code runs, budget ratios in [0.995, 1.005], timing, determinism; scores not analysed |

Archive of the pilots: `electron-server:~/archives/gb300-20261101/hypneum-20261101/pilots/`.

## Forking paths (declared)

- The f grid, N grid, threshold 0.02 and the 20 seeds come from the
  relecture's rank-3 proposal, itself derived from the pilot.
- The Hebb window of gate G1 ([0.035, 0.060]) brackets the pilot value
  0.047 by ±25 %; it is a code-health check, not a hypothesis.
- No other parameter was tuned; A, τ_l, τ_d, the 2 % tolerance and the
  bisection tolerances are the pilot's.

## What this does NOT test

- The NH-01 claims (kernel dreaming, capacity estimator, uniform shrink).
- α ≠ 0.15, correlated patterns, unclipped couplings.
- A Daydreaming or eigenvector-dreaming arm (Benedetti 2023); a scoop risk on
  "dreaming + matched budget" is acknowledged and the literature is cited.
- Any claim beyond random independent patterns and this dynamics.

## Compute plan, stopping rule, outputs

- **Host:** kxkm-ai (i7-14700KF, 8 P-cores + 12 E-cores, 28 threads, AVX2,
  62 GB RAM), **CPU only** (`CUDA_VISIBLE_DEVICES=""`), working directory
  `/ai/saisail/p1bis-p8c-20261104`, venv from
  `experiments/hopfield_dream_conf/requirements-hopfield-conf.txt`
  (hash-pinned, Python 3.12.15), `nice -n 19`, `nohup`, launcher
  `scripts/hopfield_conf/kxkm_launch.sh` (lane B, 12 workers, before the
  P8c N = 4 096 lane).
- **Duration (measured):** smoke seed 90001 on kxkm-ai, one cell each:
  N = 1 000 6 s, N = 2 000 57 s, N = 4 000 328 s (P-core, machine otherwise
  idle). E-cores measured 2.1× slower. 20 seeds ≈ 7 800 core-s; with
  12 workers the N = 4 000 tail gives ≈ 15–25 min wall; budget 1 h.
- **Wall-time cap:** 6 h. At the cap the STOP file is created; finished cells
  are kept; tests with < 16 measured seeds are `not_measured`.
- **Stopping:** `touch /ai/saisail/p1bis-p8c-20261104/STOP` (checked between
  cells; the running cells finish or are terminated, nothing partial is
  written).
- **Outputs:** `results/hopfield_p1bis_ext/{grid.jsonl, control.jsonl,
  run_info.json, run.log, SHA256SUMS, verdict.json}` (copied from kxkm-ai);
  results note `docs/results/<date>-hopfield-p1bis-ext.md`.

## Nothing runs before

PR merged on green CI; freeze manifest with SHA-256 of every protocol, code
and environment file; annotated tag `prereg-hopfield-p1bis-p8c-<YYYYMMDD>`
pushed; SWHID recorded; PI explicit go. The launcher checks that the working
tree is clean and at the tag before starting.

## References

Marinari E., Rossi S., Zamponi F., *Dreaming improves memorization in a
Hopfield model with bounded synaptic strength* (code
github.com/SaverioRo/hopfield-dreaming-clipping @7e42574, MIT).
Benedetti M., Ventura E., Marinari E., Ruocco G., Zamponi F., *Supervised
perceptron learning vs unsupervised Hebbian unlearning*, J. Chem. Phys. 156
(2022) 104107, DOI 10.1063/5.0084219. Benedetti M., Carillo L., Marinari E.,
Mézard M., *Eigenvector dreaming*, arXiv 2308.13445 (2023). Hopfield J.J.,
Feinstein D.I., Palmer R.G., Nature 304 (1983) 158–159.
