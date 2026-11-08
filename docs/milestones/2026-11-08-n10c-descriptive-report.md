# N10-C — Descriptive report, 2026-11-08 (issue #63)

**Status:** DRAFT, for PI review. Descriptive only. Not frozen. Changes no registered provision.
**Refs:** issue #63 (this report), issue #62 (N10-C-bis freeze, undecided), PR #105 (stop declaration), PR #107 (interim amendment), closure declaration `n10c-procgen-multiagent-gtm-comm-closure-2026-11-05.md`.
**Date drafted:** 2026-11-08 (dated by the GitHub push time of this PR's first commit; git dates are not evidence of time).

## 0. Reading status of this report

- Every figure below is **exploratory**: one seed (0), one training host (macM1), one evaluation host (KXKM-AI), campaign stopped by resource decision (option (b) of the interim amendment, section 4.2, adopted 2026-11-04 about 19:10 CET; closure declaration, section 2.1).
- **No positive reading, no negative reading, no contrast, no interval and no test is made here.** The only interpretation rule applied is the frozen one (reading rules R1–R5 of `2026-10-25-n10c-bis-control-arms.md` and the frozen `analysis.py` at `99a7cf4`). That code returns `pilot-interval-unavailable` for one seed (section 4.2).
- The held-out results were read by the PI before the N10-C-bis freeze: the third declared blinding breach (interim amendment, section 4.1 item 1; closure declaration, section 3 for the 2026-10-30 reading). This report does not restore blinding. Any later decision on N10-C-bis or N10-D is non-blind with respect to these numbers.
- The task of the campaign was judged not to require communication on development records (closure, section 2.2 (a)); that is a recorded ground of the PI's decision, not a finding of this report.

## 1. What was frozen

| Item | Run 1 | Run 2 (the campaign reported here) |
|---|---|---|
| Tag (tag object -> commit) | `n10c-freeze-20261024` (`91e1dddc` -> `2ce4214b9e587328c3c4eab64dce4650409edd7d`) | `n10c-freeze-20261029` (`b14717b7` -> `99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb`) |
| SWHID | `swh:1:rev:2ce4214b9e587328c3c4eab64dce4650409edd7d` | `swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb` |
| Manifest | `manifest/stage2-manifest.md`, SHA-256 `1bddf9cae688…b230` | `manifest/stage2-manifest-fixc.md`, SHA-256 `68539232fb1d…3640` |
| Amendment | 2026-10-23, SHA-256 `0114dac7e277…d59b` | 2026-10-25, SHA-256 `dfb3b96f15376a5b…18e7` (full: `dfb3b96f15376a5bbfd2a7f5cc1ea7d8d9e29afb0dde1492f94e2f4a6ec618e7`) |
| Status | stopped 2026-10-25 about 08:50 (PI decision, trunk collapse found in the first two runs); **excluded from all analyses** | seed 0 only, stopped, see section 2 |

Sources: `run2-kxkm-campaign.json` (this PR) for run 2; the PI's `RUNBOOK.md` and `kxkm`/`macm1` `campaign.json` for run 1 (run 1 files are in the ES archive, section 6). Both `campaign.json` carry `tag_push_time: null`: the GitHub server-recorded push time of the tags was **not obtained** and is not reported here.

### 1.1 Commit mismatch to flag (issue #63, rapport §2.2)

Each frozen manifest records a **harness commit that is not the tag commit**; both manifests say "The freeze commit must be re-recorded after merge".

| Tag | Commit cited in the manifest | Tag commit | Relation (git) |
|---|---|---|---|
| `n10c-freeze-20261024` | `599ac25be1e78c3a30b067f76d206a81acc8cc72` | `2ce4214b9e587328c3c4eab64dce4650409edd7d` | cited commit is an ancestor; 1 commit between; the diff `599ac25..2ce4214` adds `stage2-manifest.md` (the manifest itself) and edits one line of the 2026-10-23 amendment |
| `n10c-freeze-20261029` | `b9aabefb3362d134818b3a06507c6d8189781fd2` | `99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb` | cited commit is an ancestor; 2 commits between (`7c9c411`, merge `99a7cf4`); the diff `b9aabef..99a7cf4` touches only `stage2-manifest-fixc.md` (6 lines) and the Status line of the 2026-10-25 amendment (2 lines) |

The runners were launched with `--freeze-tag` and `--swhid` of the **tag commit** (`campaign.json`), and the guards checked a clean tree at the tag. The mismatch is therefore a documentation gap in the manifest (cited commit predates the tag), not a difference in executed code on the evidence above (`git diff --stat`, files listed). It was not re-checked by rebuilding the manifest hashes against the tag tree.

## 2. What ran

### 2.1 Hosts and configuration (run 2)

| | macM1 | KXKM-AI |
|---|---|---|
| Role | training, seeds 0, 17, 42 (queue of 30) | evaluation of all synced runs; seeds 73, 101, 131 (never launched) |
| Backend | MLX Metal, Procgen arm64 (`backend: metal` in `runs.csv`) | x86-64 Procgen 0.10.7 build, CUDA |
| Campaign dir | `~/scratch/n10c-run2/campaign` | `/ai/saisail/n10c-run2/campaign` |
| Training preset | `fix-c-coin-pretrain`, 50,000,000 pair-steps per run, `n_workers` 1, shared trunk | — |

### 2.2 Training of seed 0 (macM1), per run

Times CET. Source per row: `status/<run>.json` and `runs/<run>/run.json` in the archive (extract: `results/n10c/seed0-eval-summary.json`, fields `started_at`, `finished_at`, `wall_time_s`, `restarts`, `pair_steps_done`).

| Run | Train start | Trained | Wall time (h) | Restarts | Pair-steps |
|---|---|---|---|---|---|
| `registered-gtm-s0` | 2026-10-29 06:52 | 2026-10-30 00:57 | 18.1 | 0 | 50,000,000 |
| `closed-gtm-s0` | 2026-10-30 00:57 | 2026-10-30 19:11 | 18.2 | 0 | 50,000,000 |
| `registered-nocomm-s0` | 2026-10-30 19:11 | 2026-10-31 13:33 | 18.4 | 0 | 50,000,000 |
| `closed-nocomm-s0` | 2026-10-31 13:33 | 2026-11-01 07:34 | 18.0 | 0 | 50,000,000 |
| `registered-dial-s0` | 2026-11-01 07:34 | 2026-11-02 01:54 | 18.3 | 0 | 50,000,000 |
| `closed-dial-s0` | 2026-11-02 01:54 | 2026-11-02 19:57 | 18.1 | 0 | 50,000,000 |
| `registered-eng-s0` | 2026-11-02 19:57 | 2026-11-03 14:22 | 18.4 | 0 | 50,000,000 |
| `closed-eng-s0` | 2026-11-03 14:22 | 2026-11-04 08:53 | 18.5 | 0 | 50,000,000 |
| `registered-mlp-s0` | 2026-11-04 08:53 | 2026-11-05 02:34 | 17.7 | 0 | 50,000,000 |

No run has `training_failure`, `liveness_flagged` or `liveness_abort` set (`seed0-eval-summary.json`).

### 2.3 Evaluation (KXKM-AI) and stop record

- Single-pass evaluation of the 9 runs on KXKM-AI: first `eval start` 2026-11-05 02:47:58 CET, last `eval done` (`registered-mlp-s0`) 02:54:35, then `gpu_guard_refused` (`free_gb` 21.72, `min_free_gb` 1000.0, `next_run` `registered-eng-s73`): no KXKM-AI training, no cross-play. Source: `results/n10c/run2-kxkm-attempts.jsonl` (copy of the archived `attempts.jsonl`).
- macM1: `registered-mlp-s0` trained 2026-11-05 02:34:00; runner exited on the STOP file at 02:34:02 (`stopped_by_STOP_file`); `registered-atoc-s0` never started (closure declaration, records M1–M5; macM1 `attempts.jsonl` was read by the declaration, not re-read here).
- Decisions: stop after seed 0 (deviation declaration 2026-11-04, signed 06:51 CET, PR #105); interim analysis (PR #107, merged 2026-11-04 18:16Z); stop for good (closure, 2026-11-05 about 06:33 CET).
- Run 1: `registered-gtm-s0` (macM1, trained 2026-10-25 07:33) and `registered-eng-s73` (KXKM-AI, trained 2026-10-25 00:35) were evaluated, `closed-eng-s73` had started. Recorded values in `n10c-raw-20261108/n10c-run/campaign/attempts.jsonl`: `R_intact` 0.0 and 0.005, `S` 0.0. These runs belong to the superseded tag and are excluded from all analyses; they are listed as records only.

### 2.4 Cell inventory

| Registered design | Executed |
|---|---|
| 10 cells x 6 seeds = 60 runs (2 hosts) | 9 runs: seed 0, 9 of 10 cells, trained and evaluated |
| `registered-atoc` cell | not started (STOP) |
| seeds 17, 42 (macM1), 73, 101, 131 (KXKM-AI) | not run |
| per-run-pretraining controls `prc-*-s101` (10) | not run |
| cross-seed cross-play (10 cells x 6 pairs) | not computed (needs two seeds) |

The issue text says "60 runs"; 9 exist.

## 3. What was measured, per the frozen rule

Held-out evaluation: 200 levels (1000–1199), one episode per level, `n = 1` per cell (one seed). Columns are the scalar fields of `runs/<run>/eval.json` and `runs/<run>/run.json`, extracted without transformation into `results/n10c/seed0-eval-summary.json`. `S = R_intact - R_shuffled` is the message-shuffle sensitivity of the frozen evaluator. Training success is the last point (50,000,000 pair-steps) of `success_curve` in `run.json`, on training levels 0–15.

| Run | R_intact | R_shuffled | S | completion_time | bandwidth_bits | train success (last point) | time_to_half |
|---|---|---|---|---|---|---|---|
| `registered-gtm-s0` | 0.440 | 0.390 | 0.050 | 45.98 | 7.745 | 0.96 | 1,000,000 |
| `closed-gtm-s0` | 0.435 | 0.400 | 0.035 | 42.68 | 6.701 | 0.91 | 1,000,000 |
| `registered-nocomm-s0` | 0.415 | 0.415 | 0.000 | 53.12 | 0.000 | 0.94 | 1,000,000 |
| `closed-nocomm-s0` | 0.375 | 0.375 | 0.000 | 43.08 | 0.000 | 0.86 | 1,000,000 |
| `registered-dial-s0` | 0.420 | 0.350 | 0.070 | 55.61 | 6.904 | 0.96 | 1,000,000 |
| `closed-dial-s0` | 0.450 | 0.375 | 0.075 | 44.44 | 5.152 | 0.91 | 1,000,000 |
| `registered-eng-s0` | 0.475 | 0.420 | 0.055 | 44.52 | 5.728 | 0.94 | 1,000,000 |
| `closed-eng-s0` | 0.465 | 0.380 | 0.085 | 42.30 | 5.518 | 0.86 | 1,000,000 |
| `registered-mlp-s0` | 0.400 | 0.385 | 0.015 | 46.66 | 9.816 | 0.91 | 1,000,000 |

Notes that follow from the files, without interpretation:

- `eval.json` of all 9 runs: `untrained: false`, `levels: [1000, 1200]`, `primary_pass: "intact"`, `codebook_usage: null`, 200 episodes intact and 200 shuffled.
- `time_to_half` equals the first logged point (1,000,000 pair-steps) for the 9 runs (`run.json`).
- `S` of the two `nocomm` runs is 0.000 and `bandwidth_bits` is 0.000: with no message, shuffling cannot change play. This control cannot fail at this setting; it is labelled « non falsifiable » (contrôle non informatif), not « passed ».
- The `R_intact` values were also computed independently from per-episode `returns_intact` in `eval.json` by the interim analysis (`runs.csv`, columns `check_R_intact`, `check_R_shuffled`, equal to the values above, ES `~/archives/n10c/interim-analysis-20261105/`).

### 3.1 Output of the frozen analysis code

`experiments/procgen_multiagent_gtm_comm/analysis.py` at `99a7cf4`, unmodified, run on the 9 runs through an adapter (file `results/n10c/frozen_analysis_output.json`, produced on 2026-11-05 by the interim analysis, SHA-256 `649d5d13…0a69` (see section 6)):

| Call | Result |
|---|---|
| `primary_report` (registered entry point) | error: seeds must be exactly `[0, 17, 42, 73, 101, 131]` |
| `_primary`, `_secondary` | error: cell `(atoc, registered)` not trained |
| `_by_host` | error: one host only |
| `_s_report` | per-cell means of `S` returned; intervals `null`; `positive_control: "uninterpretable"`, `interpretable: false`; `nocomm_zero_check` true for registered and closed |
| `_ni` | `dial`, `gtm`, `mlp` intervals `null`, `interpretable` `null` (denominator from one seed) |
| `pilot_category` | `pilot-interval-unavailable` |
| cross-play | not computed |

Quantities the frozen code cannot compute from one seed are **undefined** here, not replaced by another estimator (interim amendment, section 3).

### 3.2 Not reported on purpose

The between-arm differences, level-paired confidence intervals and McNemar p-values of the interim report (`exploratory_contrasts.csv`, computed outside the frozen code) are **not reproduced** here. They have no confirmatory value (interim amendment, section 3) and quoting them in a report whose rule is "no post hoc reading" would amount to a reading. They remain in the ES archive.

## 4. What is missing (deferred, not interpreted)

| Item | Why missing | Decided by |
|---|---|---|
| 51 of 60 registered runs, including the `registered-atoc` cell | campaign stopped 2026-11-04/05 | PI declarations, section 2.3 |
| Seed-level variance, host block, cross-play | needs the six seeds | not computable |
| N10-C-bis arms: mismatched message (`gtm_mis`), receiver warm-start (`warm`), VQ baseline | N10-C-bis not frozen, no run exists (issue #62; necessity gate G1–G3 of `2026-10-25-n10c-bis-control-arms.md`) | N10-C-bis only |
| N10-C-bis contrasts C1 (content), C2 (optimisation), C3 (GTM vs VQ) | same | N10-C-bis only |

## 5. Verdicts (lab vocabulary)

| Statement | Verdict | Basis |
|---|---|---|
| Registered N10-C estimands and pilot categories (six seeds, host block) | **non testé** | 1 of 6 seeds, 9 of 10 cells; frozen code returns `pilot-interval-unavailable` (section 3.1) |
| Cross-seed cross-play probe | **non testé** | needs two seeds |
| Cell `registered-atoc` | **non testé** | never started |
| Any statement that GTM communicates, coordinates or outperforms (R1) | **en attente** | only N10-C-bis can qualify it; it is not frozen; under R3, if it is never run N10-C is « uninterpretable as evidence of communication » and the embargo is permanent |
| Message content vs optimisation vs symbol structure (H-bis-1..3, arms `gtm_mis`, `warm`, VQ) | **en attente** | missing arms, section 4 |
| `nocomm` zero-sensitivity control (`S = 0`) | contrôle non informatif (non falsifiable) | no message by construction |
| Message-shuffle sensitivity S per cell | **non concluant** | descriptive values in section 3, no interval available, frozen positive-control check « uninterpretable » |
| Learning gate (training success well above its start) | not a verdict | training success 0.86–0.96 on levels 0–15; no learning-gate claim is made about held-out play |

No cell of this report is **confirmé**, **réfuté**, **nul TOST**, **rétracté**, **amendé** or **non vérifiable**. The classes « confirmé » and « réfuté » are not used because no registered test was run; « nul TOST » is not used because no equivalence margin was applied.

## 6. Archives and integrity

| Archive | Location | Content |
|---|---|---|
| Raw, runs 1 and 2 | `electron-server:~/hypneum-archives/n10c-raw-20261108/` and `macM1:/Volumes/T7/hypneum-archives/n10c-raw-20261108/` | `n10c-run/campaign`, `n10c-run2/campaign` (runs, status, trunks, attempts, console), `n10c-freeze/trunks`, `n10c-freeze-repo.{diff,txt}`; 179 files, 249 MB on ES |
| Development (gate, adam-study) | `…/n10c-dev-20261108/` (ES and T7) | 311 files, 699 MB on ES; not used in this report |
| Interim analysis of 2026-11-05 | `electron-server:~/archives/n10c/interim-analysis-20261105/` (`RAPPORT.md`, `runs.csv`, `frozen_analysis_output.json`, `exploratory_contrasts.csv`, `curves_summary.csv`, `episodes.csv`, `figures/`, `scripts/interim_analysis.py`) | |

Integrity, as stated and as checked:

- The `MANIFEST.md` of each raw/dev archive states "sha256: 178 files … 0 diff kxkm vs ES" (raw) and "310 files, 0 diff" (dev). **No `SHA256SUMS` file was found** in either archive directory (`ls -a`, `find -name '*SUM*'`); the sums themselves are not recoverable from the archive and were not independently recomputed for all files. `MIRROR.md` on the T7 states 2568 files compared by sha256 with 0 differences, last full pass 2026-11-08 11:31 CET.
- The interim-analysis directory is under `~/archives/n10c/` on ES, **outside** `~/hypneum-archives/`; I found no T7 copy in the mirror. It has therefore one known copy (ES) plus the files committed here. A second copy to the T7 is needed to honour the "ES and T7" rule.
- SHA-256 recomputed on ES for the interim files: `RAPPORT.md` `979eafba28fb7a690e7fb91e378354274b5ac53bc0ea034d0faf678be594ac56`; `runs.csv` `22d98675c4557420f98c00fc630dc3b220a7bc67e35ace84ea2641053e2bdfb4`; `frozen_analysis_output.json` `649d5d13ef859e3f21891df0e8abf1ef321c99c270f966a2a91fe966fa240a69`.
- SHA-256 of the archived `run.json`/`eval.json` of each of the 9 runs and of the last checkpoint are recorded in `results/n10c/seed0-eval-summary.json` (`sha256_eval_json`, `sha256_run_json`, `last_checkpoint_sha256`).

### 6.1 Files committed in this PR (`results/n10c/`, 56 KB in total)

| File | SHA-256 | Source |
|---|---|---|
| `seed0-eval-summary.json` | `e15bb9501d1f5545cd7309b88a84c8a93d7c8e0cc4b24f038eb0736ada06cd38` | scalar extract made on ES 2026-11-08 from `n10c-raw-20261108/n10c-run2/campaign/{runs/*/eval.json,runs/*/run.json,status/*.json}` (no per-episode lists, no curves) |
| `frozen_analysis_output.json` | `649d5d13ef859e3f21891df0e8abf1ef321c99c270f966a2a91fe966fa240a69` | `~/archives/n10c/interim-analysis-20261105/` (verbatim) |
| `run2-kxkm-attempts.jsonl` | `984f6d2f75b06fc9352b4f97a2109641575962e95ae00b30f942c14de0993d92` | `n10c-raw-20261108/n10c-run2/campaign/attempts.jsonl` (verbatim; outcome-bearing: carries `S` and `R_intact`) |
| `run2-kxkm-campaign.json` | `36cc876de8e532000a4890f7038d57905f99430f4d42ba72acd944c9b7d84a06` | `…/campaign.json` (verbatim) |

The run records (`run.json`, about 1 MB each, with training curves) and checkpoints stay in the archives only; per-episode lists stay in `eval.json` there.

## 7. Open points for the PI

1. Time of the tag push of both tags (GitHub / Software Heritage), not obtained by the campaign.
2. Copy `interim-analysis-20261105/` to the T7 (section 6).
3. Whether the manifest's cited commit should be annotated in an append-only record (section 1.1).
4. This report closes issue #63 only if the PI accepts it as the « descriptive report ». Issue #62 is not affected.

## LLM assistance

Drafted with Claude (Anthropic, Claude Code, Sonnet 5.5) on 2026-11-08: reading of the archived JSON/CSV files and of the declarations, extraction of the scalar summary, writing of this text. No model was run and no training or evaluation was launched. The numbers were copied from the files named in each section and cross-checked against `runs.csv` of the interim analysis; the PI has not yet verified them.
