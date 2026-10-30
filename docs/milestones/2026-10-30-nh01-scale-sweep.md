# NH-01 — Pre-registration amendment (L0 draft, target L1) 2026-10-30: size sweep N = 1 000, 2 000, 4 000

**Status:** L0 draft on a branch, not frozen. This file becomes binding
only at L1: PR merged on green CI, manifest (§8), annotated tag
`prereg-nh01-scale-<YYYYMMDD>` on the merge commit, pushed, and a
Software Heritage save whose revision SWHID (`swh:1:rev:…`) is recorded
here. The freeze waits for the PI ruling on #86 and #87, the open PI
decisions of §8 and the timing probe. Nothing in this sweep has been
run. Runs started before the tag push do not count.
**Amends:** `docs/milestones/2026-10-25-nh01-confirmatory-n1000.md`
(tag `prereg-nh01-confirmatory-20261025`, PR #86) and, for the rules on
non-computable criteria, `docs/milestones/2026-10-27-nh01-correlated-patterns.md`
(tag `prereg-nh01-correlated-20261027`, PR #87).
**Issue:** hypneum-lab/nerve-wml#68. **Investigator:** Clément Saillant
(sole investigator); drafting, implementation and run delegated to an
agent, the researcher merges and tags.
**Date drafted:** 2026-10-30. PI request 2026-10-30 14:08.
**Data status at drafting:** results at N = 1 000 are public (#86, #87,
provenance re-runs of 2026-10-30). No cell at N = 2 000 or 4 000 has been
computed. The earlier runs inform the predictions only and are not
pooled : pilot N ≤ 256 (seeds 0–9, tag
`prereg-nh01-hopfield-dream-20261025`), confirmatory N = 1 000 (seeds
0–9, #86), correlated N = 1 000 (seeds 100–109, #87), timing probes
(seeds 999 and 998, discarded).

## 1. Question

Does the NH-01 Hopfield result (dreaming kernel at t = 10 multiplies the
capacity by ≥ 3, with an interior optimum t\*, while a Frobenius-matched
uniform shrink does not) hold at N = 2 000 and N = 4 000, or does it
depend on the finite-size Hebbian estimate measured at N ≤ 1 000? And
does the shrink control stay inert over a range of finite temperatures,
not only at T = 0.2?

## 2. What changes with respect to the confirmatory run (binding)

| Item | Confirmatory (#86) | Scale sweep |
|---|---|---|
| Size | N = 1 000 | **N ∈ {1 000, 2 000, 4 000}** |
| Seeds | 0–9 | **0–9 at every N** (patterns drawn per (seed, N) stream, so sizes are independent ; N = 1 000 doubles as a cross-host reproduction of #86) |
| Dreaming time grid | {0, 1, 3, 5, 7, 10, 15, 20, 30, 100} | unchanged |
| Conditions, all 19 arms | (T = 0, f = 0.10), (T = 0.2, f = 0.10) | unchanged, same condition indices 0 and 1 |
| Extra temperatures | none | **T ∈ {0.1, 0.3}, f = 0.10**, condition indices 2 and 3, restricted to the arms `hebb`, `dream t = 10`, `shrink c(10)` |
| C3′ | decided at T = 0.2 | decided at T = 0.2 per N (unchanged) ; T = 0.1 and 0.3 give C3″ (secondary, see §4) |
| Non-computable criteria | rule for C1 only (#87) | **general rule (§4.5)** : no label « confirmé » on a level where a criterion cannot be computed |
| Task unit | one (N, seed) | one (N, seed, load) ; cells bit-identical to the per-seed task (unit test) |
| Host | electron-server | **KXKM-AI CPU** (28 cores, 64 GB RAM, user `electron`, work dir `/ai/saisail/nh01-scale`) |
| Provenance | partial (see #86 review) | **full trace, §6** |

Everything else is identical to the confirmatory run: ±1 patterns nested
in α, Hebbian J, dreamed kernel J(1+t)(𝟙+tJ)⁻¹ by spectral decomposition,
diagonal set to 0, Frobenius-matched shrink J·c(t), cue with exactly
round(0.10·N) flipped bits, asynchronous sign dynamics with the
scale-invariant tie rule (T = 0, stop at the first sweep without flip,
cap 100 sweeps), Glauber dynamics at β = 1/T (40 sweeps, overlaps
averaged over the last 10), 32 probes per cell, load grid of 24 values
from 0.04 to 0.95, retrieval thresholds 0.9 (T = 0) and 0.8 (T > 0),
capacity estimator at the 0.5 crossing after the last load with
r ≥ 0.5, seed-level percentile bootstrap (10 000 resamples, seed
20261025).

Because the dynamics streams are keyed by (seed, N, load index,
condition index), appending conditions 2 and 3 leaves conditions 0 and 1
bit-identical to the confirmatory runner. The N = 1 000 cells of
conditions 0 and 1 are therefore expected to equal #86's cells ; a
difference can only come from the host (BLAS, CPU) and is reported, not
corrected.

### Why T ∈ {0.1, 0.3}

C3′ was decisive at a single temperature (caveat 1 of #86). At T = 0.1
the shrink is expected to be close to inert (low thermal noise) ; at
T = 0.3 (β ≈ 3.3) the Hebbian network is still in its retrieval phase
at loads below ≈ 0.1 but the thermal noise is larger, so a shrink c(10)
≈ 0.92 acts as a stronger heating. The two points bracket T = 0.2
without entering the paramagnetic region. Only the three arms needed
for the shrink control run there, to keep the cost bounded.

## 3. Predictions (fixed before any N > 1 000 cell)

- P1 : for each N, α̂_c(dream, t = 10) ≥ 3 × α̂_c(Hebb) at T = 0. The
  ratio is expected to grow slightly with N (Hebb α̂_c 0.157 at
  N = 1 000, approaching 0.138 from above ; dream α̂_c ≈ 0.60–0.62).
- P2 : for each N, the median curve t ↦ α̂_c(dream, t) at T = 0 is
  unimodal with an interior maximum t\* ∈ [5, 20], and
  α̂_c(t\*) − α̂_c(100) > 0.
- P3 : for each N, at T = 0.2, α̂_c(shrink c(10)) / α̂_c(Hebb) < 1.1.
- P3″ (secondary) : the same ratio < 1.1 at T = 0.1 and T = 0.3.
- P4 (secondary) : spurious final states at α ∈ {0.14, 0.20} reduced
  by ≥ 50 % under dream t = 10 vs Hebb.
- P5 (descriptive) : the dream α̂_c at t = 10 changes by less than 0.03
  between N = 1 000 and N = 4 000.

## 4. Decision rule (binding once tagged)

Per size N ∈ {1 000, 2 000, 4 000}, cue 10 %, t = 10:

1. **C1_N (gain)** — cell T = 0 : mean over seeds of the paired ratio
   R = α̂_c(dream, 10) / α̂_c(Hebb) with 95 % CI, and of the paired
   difference D. Confirmed if the CI lower bound of R ≥ 3 ; refuted if
   the CI upper bound of D < 0.1 ; otherwise non concluant.
2. **C2′_N (unimodal, finite optimum)** — as in #86 : median curve
   m(t), t\* = first argmax, unimodality within τ = 0.02 ; violated if
   not unimodal ; non concluant if t\* ∈ {0, 100} ; holds otherwise.
   Reported alongside, not deciding : t\* ∈ [5, 20], per-seed t\*,
   finite optimum (95 % CI of α̂_c(t\*) − α̂_c(100) excluding 0).
3. **C3′_N (shrink at T = 0.2)** — mean of
   S = α̂_c(shrink c(10)) / α̂_c(Hebb) with 95 % CI. Holds if the CI
   upper bound < 1.1 ; refutes if the CI lower bound ≥ 2 ; otherwise
   non concluant. Implementation check : at T = 0 the ratio must be
   exactly 1 (bitwise equality of the capacity values, not
   `isclose`) on every seed, with the expected number of finite
   ratios ; otherwise the run is void.
4. **Overall_N** — confirmé iff C1_N confirmed ∧ C2′_N holds ∧ C3′_N
   holds, and §4.5 does not apply ; réfuté if C1_N refuted ∨ C2′_N
   violated ∨ C3′_N refutes ; non concluant otherwise.
5. **Non-computable criteria (general rule, new)** — a ratio is
   undefined for a seed when its denominator is below the grid (no
   load with r ≥ 0.5) or its numerator is censored at the grid top
   (r ≥ 0.5 at α = 0.95). For every criterion that uses a ratio
   (C1, C3′, C3″), seeds with an undefined ratio are dropped from the
   bootstrap and counted. **If more than 2 of 10 seeds are undefined,
   the criterion is labelled « non calculable »** and the level cannot
   be « confirmé » ; Overall_N is then « non concluant (non
   calculable) » unless another criterion refutes. This rule is applied
   unconditionally (it also overrides a « refuted » label computed on
   the remaining seeds), and before the overall label is formed. With
   1 or 2 undefined seeds the criterion is computed on the rest and
   the count is reported next to the label. The code implements this
   rule with a unit test per branch.
6. **C3″ (secondary, T ∈ {0.1, 0.3})** — same statistic and thresholds
   as C3′ at each temperature, same §4.5 rule ; labels holds / refutes
   / non concluant / non calculable ; does not enter Overall_N.
7. **Size trend (descriptive)** — R(N) and the dream and Hebb α̂_c per
   N with their CIs ; the difference R(4 000) − R(1 000) with a
   bootstrap CI over seed indices (sizes are independent draws, the
   seed index is a pairing label only).
8. **P4** gets its own label as in #86 and does not enter Overall_N.

« Démontre / réfute » are reserved to C1_N, C2′_N, C3′_N in the results
note ; everything else is descriptive. A level labelled « non
calculable » is reported as such in every summary line, README row,
CHANGELOG entry and PR description, never as « confirmé (formal) ».

## 5. What a negative means

C1 refuted or non concluant at N = 4 000 while confirmed at 1 000 → the
3× gain is a finite-size effect of the Hebbian estimate. C3′ refutes at
some N → the uniform shrink reproduces the gain at finite temperature at
that size (pillar B is a `downscale`). C2′ violated → the single-optimum
reading of the dreaming time fails at that size. Non concluant → keep
thresholds, add seeds (next amendment), never new thresholds.

## 6. Provenance (binding)

Lessons of the #86/#87 review: every run directory contains, written by
the runner or its launch wrapper, before the first cell:

- `git rev-parse HEAD` of the executing clone, the tag name, the tag
  object id and its target commit ; the executing clone is a fresh
  clone checked out **at the tag** (not a branch, not a working copy).
- `git status --porcelain` before and after the run, both empty.
- SHA-256 of every executed code file, and the same hashes recomputed
  from `git show <tag>:<file>` ; the run aborts if they differ.
- Python version and `uv pip freeze` of the environment ; numpy and
  BLAS configuration (`numpy.show_config()`), thread environment
  variables (all BLAS/OpenMP threads = 1).
- Host name, CPU model, core count, RAM, load average at launch.
- The tag record **fetched from the GitHub API** (`git/ref/tags/<tag>`
  and `git/tags/<object>`, raw JSON) with the fetch time, written
  before the launch time ; start and end UTC timestamps. GitHub does
  not expose the push time : it is bounded below by the tagger date
  and above by the API fetch time (the tag is visible on GitHub at
  that moment) ; the SWH save request and its SWHID, made before the
  launch, give the external timestamp.
- The timing-probe output (declared, discarded) and the exact launch
  command.

`SHA256SUMS` covers the outputs **and** the provenance files. The
results note quotes the tag-fetch time and the start time so that the
order tag → run is checked from the artefacts alone.

## 7. Compute plan, cost, stopping rule

- **Host:** KXKM-AI CPU only (28 cores, 64 GB), `nice 19`, BLAS
  single-threaded, uv venv pinned to Python 3.12.15 / numpy 2.5.4 (the
  versions of #86), work dir `/ai/saisail/nh01-scale`. The GPU and its
  `gpu-swap` services are not used and must not be disturbed.
- **Cells:** per (N, seed) 24 loads × (19 arms × 2 conditions + 3 arms
  × 2 extra temperatures) = 1 056 cells ; 10 seeds × 3 sizes = **31 680
  cells**, 32 probes each.
- **Memory:** at N = 4 000 one load holds 19 dense 4 000 × 4 000 float64
  kernels (≈ 2.4 GB) plus the eigenbasis ; the runner keeps at most
  these per task. Worker cap: **≤ 16 workers while N = 4 000 tasks
  run** (≈ 40 GB), up to 24 otherwise ; 4 cores always left free.
- **Cost model (from the N ≤ 1 000 timings on electron-server, Xeon
  X5680 without AVX, shared):** an N = 1 000 seed task took ≈ 2 400 s,
  of which ≈ 200 s linear algebra (24 `eigh` + 216 kernel products,
  O(N³)) and ≈ 2 200 s dynamics (Python loop over units, cost between
  O(N) when per-step overhead dominates and O(N²) when the 32 × N
  field products dominate). The extra temperatures add ≈ 15 %
  (6 Glauber cells of 40 sweeps per load). Per seed task,
  electron-server-equivalent single core:

  | N | linear algebra | dynamics (O(N) … O(N²)) | task | 10 seeds (core-h) |
  |---|---|---|---|---|
  | 1 000 | 0.2 ks | 2.5 ks | ≈ 0.75 h | ≈ 7.5 |
  | 2 000 | 1.6 ks | 5 … 10 ks | 1.8 … 3.2 h | 18 … 32 |
  | 4 000 | 13 ks | 10 … 40 ks | 6.4 … 15 h | 64 … 150 |

  Total **≈ 90 … 190 core-hours** at electron-server speed. Split per
  load, the 720 tasks fill the workers ; with 16 workers at N = 4 000
  and 24 otherwise the wall time is **≈ 5 … 11 h** at
  electron-server speed per core. KXKM cores are newer (AVX2), so the
  linear-algebra part is expected to be several times faster ; the
  binding projection is the probe below.
- **Timing probe (declared, before the tag):** on KXKM-AI, seed 999
  (not a pre-registered seed), N = 2 000 and 4 000, loads α ∈ {0.2,
  0.5}, all conditions ; output discarded, no capacity read ; projected
  wall time, peak RSS per worker and chosen worker count written in §8
  before tagging. If the projection exceeds 48 h, the PI decides
  before the tag (fewer extra temperatures, or N = 4 000 on 5 seeds
  declared as such) ; thresholds never change.
- **Wall-time cap:** 72 h. Beyond it completed (N, seed) units are
  kept, missing ones listed, the affected N labelled non concluant.
- No seed exclusion ; non-converged probes count as not retrieved ; no
  cell re-run after being read.
- **Outputs:** `results/nh01_scale/{cells.jsonl.gz, summary.json,
  figure.png, run_meta.json, run.log, probe.log, provenance/,
  SHA256SUMS}` ; results note `docs/results/<date>-nh01-scale.md`.

## 8. Before the tag (checklist, to be completed)

Open PI decisions (TBD by PI — no tag while any is open):

- [ ] Ruling on #86 / #87 (wording of m = 0.30 and of the mechanism
      sentence) ; if it changes a rule, mirror it here.
- [ ] Extra temperatures T ∈ {0.1, 0.3} kept, changed or dropped.
- [ ] Seed namespace : seeds 0–9 at every N (as drafted ; at N = 1 000
      they are the confirmatory seeds of #86 on purpose, for the
      cross-host reproduction ; at N = 2 000 and 4 000 the streams are
      new draws because they are keyed by N), or fresh seeds.
- [ ] Go on the KXKM-AI CPU booking (the GPU services stay untouched).

Freeze steps:
- [ ] Runner `experiments/nh01_hopfield_dream/scale.py` written on the
      confirmatory primitives, with unit tests: per-load split
      bit-identical to the per-seed task ; conditions 0–1 bit-identical
      to `confirmatory.py` at N = 1 000 ; §4.5 branches ; exact
      T = 0 check ; provenance block of §6.
- [ ] Independent review of runner and this file.
- [ ] Timing probe on KXKM-AI, results here (projection, RSS,
      workers).
- [ ] PR merged on green CI.
- [ ] Manifest `docs/milestones/2026-10-30-nh01-scale-sweep-manifest.md`
      : SHA-256 of this file, of the runner, of `run.py`,
      `confirmatory.py`, the tests and the lock file ; its own SHA-256
      recorded in the tag message.
- [ ] Annotated tag on the merge commit, pushed ; Software Heritage
      save requested, revision SWHID recorded here (in a follow-up
      commit that does not touch the frozen files).
- [ ] Tag record fetched from the GitHub API and written in the run
      directory, then launch.

## References

As in the pilot pre-registration (Fachechi–Agliari–Barra 2019 ;
Hopfield–Feinstein–Palmer 1983 ; Amit–Gutfreund–Sompolinsky 1985 ;
Kanter–Sompolinsky 1987 ; Tononi–Cirelli 2014).
