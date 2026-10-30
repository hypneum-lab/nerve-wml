# NH-01 confirmatory results — N = 1 000 (2026-10-25)

**Pre-registration:** `docs/milestones/2026-10-25-nh01-confirmatory-n1000.md`, L1 amendment tag `prereg-nh01-confirmatory-20261025` (annotated tag object `41135e4` on commit `a000f73`, created 13:47:06 UTC, pushed 13:47:08 UTC). Amends the pilot pre-registration (tag `prereg-nh01-hopfield-dream-20261025`, results `docs/results/2026-10-25-nh01-pilot.md`). Issue hypneum-lab/nerve-wml#68.
**Run:** electron-server, launched **13:47:22 UTC** (14 s after the tag push), i.e. after the freeze. Before launch, the SHA-256 of `run.py` and `confirmatory.py` on the host were checked equal to the tagged commit (the host clone is shallow at `8a13a33`, hence `git_commit = 8a13a33` in `run_meta.json`; the code files are those of `a000f73`). 4 worker processes, `nice 19`, single-threaded BLAS, shared host. **6 951 s wall (1.93 h)** for 10/10 seeds, **9 120 cells**, per-seed task 30–47 min, 0 non-converged probe (max 72 sweeps). Timing probe (seed 999, two loads, declared, output discarded) projected 1.6 h.
**Artefacts:** `results/nh01_confirmatory/{cells.jsonl.gz, summary.json, figure.png, run_meta.json, run.log, probe.log, SHA256SUMS}`. Every number below is read from `summary.json`, except the retrieval, spurious-state, convergence and matched-factor figures, which are read from `cells.jsonl.gz`.

## Verdict (amended rule, N = 1 000, 10 seeds, cue with 10 % flipped bits, 95 % seed-level bootstrap CI)

| Criterion | Pre-registered rule | Result | Label |
|---|---|---|---|
| C1 gain (T = 0, t = 10) | CI lower bound of α̂_c(dream)/α̂_c(Hebb) ≥ 3 ; refuted if CI upper bound of the difference < 0.1 | ratio **3.86 [3.79, 3.93]** ; Hebb 0.157 [0.155, 0.160] → dream 0.607 [0.604, 0.610] ; difference 0.450 [0.445, 0.454] | **confirmed** |
| C2′ unimodal, finite interior optimum (T = 0) | median curve unimodal within τ = 0.02, maximum strictly inside the t grid | medians 0.158, 0.413, 0.571, 0.603, **0.616**, 0.606, 0.595, 0.582, 0.569, 0.527 for t = 0, 1, 3, 5, 7, 10, 15, 20, 30, 100 ; **t\* = 7** ; strictly rising then strictly falling | **holds** |
| C3′ matched shrink at T = 0.2 (t = 10) | CI upper bound of α̂_c(shrink c(10))/α̂_c(Hebb) < 1.1 ; refutes if CI lower bound ≥ 2 | ratio **0.975 [0.967, 0.982]** (Hebb 0.126 [0.125, 0.128] → shrink 0.123) ; T = 0 exactness check : ratio exactly 1 on every seed and every t | **holds** |
| Overall | confirmé iff C1 ∧ C2′ ∧ C3′ | | **confirmé** |
| P4 spurious −50 % (secondary) | α ∈ {0.14, 0.20}, dream t = 10 vs Hebb | 0.159 → 0.000 (α = 0.14) ; 0.994 → 0.000 (α = 0.20) | holds |

Reported with C2′, not deciding : t\* ∈ [5, 20] as predicted (P2) ; per-seed t\* = 7 on 9 seeds, 10 on one ; finite optimum demonstrated, paired drop α̂_c(t\*) − α̂_c(100) = 0.088 [0.082, 0.092] (CI excludes 0 : not a plateau).

Under the pre-registered rule, the run **demonstrates** at N = 1 000 that (i) the Fachechi–Agliari–Barra dreaming kernel at t = 10 multiplies the empirical capacity by more than 3, (ii) the gain is maximal at a finite dreaming time t\* = 7 and decreases beyond it, and (iii) a uniform shrink matched on the Frobenius norm does not reproduce the gain at finite temperature (it lowers capacity by 2.5 %, consistent with shrink = heating). No refutation condition is met.

## Descriptive results

- **Finite temperature, dreaming side.** At T = 0.2 the dreaming gain persists : ratio 3.58 [3.53, 3.64] at t = 10, maximum 3.62 at t = 7 — the same optimum as at T = 0. The shrink ratio stays between 0.970 and 0.979 for every t ; all its CIs exclude 1 from below.
- **Size trend (pilot → confirmatory, T = 0, t = 10).** Ratio 3.13 (N = 64), 3.38 (128), 3.43 (256), 3.86 (1 000) ; Hebb α̂_c 0.211, 0.179, 0.175, 0.157 approaches the thermodynamic 0.138 from above while the dream α̂_c stays ≈ 0.60. The ratio grows mainly because the finite-size Hebbian estimate shrinks.
- **Retrieval and spurious states.** Dream (t = 7 or 10) retrieves nearly every probe up to α = 0.55 (319/320 at α = 0.50 for both ; 315/320 for t = 7 and 317/320 for t = 10 at α = 0.55) and crosses 0.5 near α = 0.60–0.62 ; Hebb and matched shrink fall from 0.84 at α = 0.14 to 0.006 (2/320) at α = 0.20, with spurious final states 0.16 → 0.99 over the same range (shrink identical to Hebb at T = 0 ; at T = 0.2 slightly more spurious states under shrink, 0.91 vs 0.85 at α = 0.14).
- **Matched factor.** c_fro(t = 10) ≈ 0.917 at α = 0.14.

## Caveats

1. The T = 0 shrink control is inert by theorem ; the discriminating control is the T = 0.2 arm (C3′). One temperature only : the result says nothing about temperatures near the Hebbian retrieval transition.
2. The capacity estimator (0.5 crossing over 32 noisy-cue probes) is a finite-size quantity ; absolute α̂_c values are not the replica-theory α_c(t). The ratios are paired within seed, which is what the criteria use.
3. P4 at α = 0.20 sits at the Hebbian ceiling (0.994) ; it reads as « dream removes spurious end states where Hebb has almost only spurious ones ».
4. Single host, shared (wall time is not a benchmark) ; numpy determinism per seed, cross-host reproduction not done.
5. Not tested : correlated patterns (where an interior optimum in t is predicted analytically), the VQ-codebook / MI part of NH-01, `recombine`, learning rules other than Hebb.

## What this changes for NH-01

With the pilot (N ≤ 256) and this run, the Hopfield part of NH-01 passes its three pre-registered criteria at N = 1 000 : on this substrate, dreaming is not a uniform downscale. The open parts of the catalogue entry (VQ-codebook spurious fraction at MI/H ± 0.02, correlated data) remain to be pre-registered separately. Closing the issue, the verdict table and any public wording are the researcher's decisions.
