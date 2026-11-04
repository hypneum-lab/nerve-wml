# N10-C — Deviation declaration 2026-11-04: campaign reduced to seed 0 (macM1), KXKM-AI half not launched

**Status:** DRAFT — declaration, not frozen. It changes no registered provision of N10-C. To be signed by the PI (section 6).
**Declares against:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` (FROZEN by tag `n10c-freeze-20261029`, commit `99a7cf4`, SWHID `swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb`), which amends the 2026-10-23 amendment and the locked `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md`; the declaration of 2026-10-30 (`…-amendment-2026-10-30.md`, DRAFT); and the N10-C-bis addendum `docs/milestones/2026-10-25-n10c-bis-control-arms.md` (DRAFT, not frozen).
**Date drafted:** 2026-11-04 (dated by the GitHub push time of its first commit; git dates are not evidence of time).
**Investigator:** Clément Saillant (Hypneum Lab, sole investigator).

The frozen amendments and the locked pre-registration are not modified. This document is append-only and only touches `docs/`.

## 1. Deviation: the campaign is stopped after seed 0, by resource decision, outcome-blind

**This stop is a resource decision taken before any outcome of the campaign of tag `n10c-freeze-20261029` was read. It is declared as a deviation because it leaves the registered design (6 seeds × 10 cells on two hosts) unexecuted.**

### 1.1 State of the campaign when the decision was taken (2026-11-04, about 06:35 CET)

Read from `campaign/attempts.jsonl`, `campaign/status/*.json` and `campaign/launcher.state` of `~/scratch/n10c-run2/` on macM1. Only status fields were read: run id, state, start and end times, wall time, `pair_steps_done`, restart counts, checkpoint names. No training-curve field, no held-out result and no checkpoint content were read (addendum, section « Blinding during N10-C »).

| # | run (macM1, seed 0) | train start (CET) | trained (CET) | wall time | restarts |
|---|---|---|---|---|---|
| 1 | `registered-gtm-s0` | 2026-10-29 06:52 | 2026-10-30 00:57 | 65 087 s | 0 |
| 2 | `closed-gtm-s0` | 2026-10-30 00:57 | 2026-10-30 19:11 | 65 604 s | 0 |
| 3 | `registered-nocomm-s0` | 2026-10-30 19:11 | 2026-10-31 13:33 | 66 136 s | 0 |
| 4 | `closed-nocomm-s0` | 2026-10-31 13:33 | 2026-11-01 07:34 | 64 844 s | 0 |
| 5 | `registered-dial-s0` | 2026-11-01 07:34 | 2026-11-02 01:54 | 66 005 s | 0 |
| 6 | `closed-dial-s0` | 2026-11-02 01:54 | 2026-11-02 19:57 | 64 994 s | 0 |
| 7 | `registered-eng-s0` | 2026-11-02 19:57 | 2026-11-03 14:22 | 66 272 s | 0 |
| 8 | `closed-eng-s0` | 2026-11-03 14:22 | in progress (checkpoint 45 006 848 at 06:22) | — | 0 |

Runs 1–7 are synced to KXKM-AI (`synced: true`), none is evaluated (`eval_done: false`): the KXKM-AI runner, which performs every evaluation, has not been launched. Remaining in the macM1 queue: `registered-mlp-s0`, `registered-atoc-s0` (seed 0), then the 20 runs of seeds 17 and 42. At the measured rate (about 18.2 h per run) the macM1 half would end around 2026-11-21.

**KXKM-AI half.** The 30 runs of seeds 73, 101 and 131 and the 10 per-run-pretraining control runs (`prc-*-s101`) were to start after E1-small (freeze procedure, amendment of 2026-10-25, section 5). E1-small ended on KXKM-AI (last run folder `run-2026-10-29`, no process on 2026-11-04). The N10-C KXKM-AI runner was never launched; the GPU has been idle since.

### 1.2 Decision and grounds

On 2026-11-04 at about 06:38 CET the PI chose, among four options put to him (finish seed 0 then stop; continue both hosts to the registered end; continue macM1 alone; stop now): **finish seed 0 then stop**. Grounds, all external to the outcomes:

1. The quarter plan of 2026-11-01 (`hypneum-lab/.github`, PR #24, « verrou de lecture »), accepted by the PI, allots 15 h of compute to the quarter, puts N10-D first, and judges that N10-C should have been relaunched with another design. Its machine table assumed macM1 free from about 2026-11-02; that assumption was wrong (30 runs, not 10, were queued on macM1). The plan is to be corrected on this point.
2. The KXKM-AI half was not launched after E1-small, so the registered analysis (six seeds, host as a block factor, cross-seed cross-play probe) could not be completed before the end of November on either option, and only by spending the whole month of GPU time on both hosts.
3. macM1 is also the second brain's host; its day model has been paused since 2026-10-29 06:52 for the campaign (amendment of 2026-10-25, macM1 launcher). Finishing seed 0 frees it around 2026-11-06 instead of 2026-11-21.

No outcome field of any run was read by the PI or by the session before or for this decision. The reading of 2026-10-30 (final training success of `registered-gtm-s0`, declared in the 2026-10-30 declaration) is not a ground of this decision and was not re-read.

### 1.3 Mechanism of the stop

- A watcher script (`~/scratch/n10c-run2/stop_after_atoc_s0.sh` on macM1, pid 36715, started 2026-11-04 06:39 CET, log `stop-watcher.log`) creates `campaign/STOP` as soon as `attempts.jsonl` records the start of `registered-atoc-s0`, the last seed-0 cell. The runner finishes that run, then exits with `EXIT_STOP` (`run_campaign.py`, `stop_requested`, event `stopped_by_STOP_file`) before `registered-gtm-s17`. The launcher wrapper lifts the day-model pause when the runner exits.
- Expected: `registered-mlp-s0` 2026-11-04 ≈ 08:40 → ≈ 2026-11-05 03:00; `registered-atoc-s0` → ≈ 2026-11-05 21:30; STOP event right after. The hourly sync (`47 * * * *`) copies the last runs to KXKM-AI within the hour.
- No run is interrupted: the stop happens between runs. The event line of `attempts.jsonl` and the exact times are to be appended to section 6 by the PI once recorded.

## 2. Affected cells

| cells | status | consequence |
|---|---|---|
| seed 0, 10 cells (6 registered arms, 4 closed arms) | trained on macM1 (8 done, 2 to finish) | evaluated on KXKM-AI by the registered procedure (section 3); descriptive report only |
| seeds 17 and 42, 20 runs (macM1) | not run | — |
| seeds 73, 101, 131, 30 runs (KXKM-AI) | not run (never launched) | — |
| per-run-pretraining controls `prc-*-s101`, 10 runs | not run | — |
| cross-seed cross-play probe, 10 cells × 6 fixed pairs | impossible (needs at least two seeds) | — |

Consequences for the registered analysis (amendment of 2026-10-23, estimands and confirmatory tests; amendment of 2026-10-25, section 4):

- **No confirmatory or estimation analysis of N10-C is possible from this campaign.** The registered design requires the six seeds with host as a block factor; a single seed on a single host gives no seed-level variance, no host block and no cross-play cell. No hypothesis of N10-C is tested, confirmed, disconfirmed or « read » from it (addendum, reading rules R1–R5).
- **What remains is a descriptive report of seed 0** (issue #63 « Évaluer N10-C (≈ 15 nov) : rapport descriptif sans lecture positive, archive ES »): the 10 held-out evaluations, reported cell by cell, without interval, test, pooling or ranking, labelled « one seed, one host, campaign stopped by resource decision ». The rule of the addendum that no positive reading of N10-C precedes N10-C-bis is unchanged; whether N10-C-bis is frozen at all is under the reading gate of the 2026-11-01 plan (issue #62) and is not decided here.
- The runs of this campaign are **not** reused as pilots, variance estimates or warm starts for any later registration unless that registration says so before its own freeze.

## 3. Evaluation of the seed-0 runs on KXKM-AI (registered procedure, CLI only)

All evaluations run on KXKM-AI with the x86-64 Procgen build (amendment of 2026-10-25, section « Hosts »). The registered runner at the frozen tag evaluates synced macM1 runs before training its own seeds (`run_campaign.py`, `run`: « KXKM evaluates (never trains) macM1 runs once synced here »). To evaluate seed 0 **without starting any KXKM-AI training run**, the runner is launched with a `--min-free-gb` larger than the card's memory: the evaluations of the 10 synced seed-0 runs need no GPU guard and complete; the first KXKM-AI training run (`registered-gtm-s73`) is then refused by the GPU guard (`gpu_guard_refused`, `EXIT_GPU`) and nothing is trained. This is a command-line setting of the frozen runner, as the `--min-free-gb 4` relaunch of 2026-10-25 was; the code at the tag is unchanged.

`kxkm/start.sh` cannot be used for this: its own synchronous `--dry-run` pre-check prints « GPU guard: WOULD REFUSE » for a `--min-free-gb` above the free memory and exits before starting. The runner command of `start.sh` is therefore launched directly, in the same tmux session and with the same console log:

```
cd /ai/saisail/n10c-run2/repo && export CUDA_HOME=/ai/saisail/n10c-bench/cuda_home
CMD="env PYTHONPATH=. /ai/saisail/n10c-run2/.venv/bin/python -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign \
  --campaign-dir /ai/saisail/n10c-run2/campaign --host kxkm \
  --freeze-tag n10c-freeze-20261029 --swhid swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb \
  --min-free-gb 1000"
$CMD --dry-run          # every guard but the GPU one must print ok
tmux new-session -d -s n10c "$CMD 2>&1 | tee -a /ai/saisail/n10c-run2/campaign/console.log"
```

Conditions: the 10 seed-0 status files are present in `/ai/saisail/n10c-run2/campaign/status/` with `status: trained` (after the sync of `registered-atoc-s0`, ≈ 2026-11-05 22:47); ComfyUI (not ours) may still hold GPU memory; expected duration about 10 × 6.7 min. The results are written by the runner into the run records and archived to electron-server (`/home/electron/archives/n10c`, cron `17 * * * *`). **They are not read** by the PI or by any summarising tool until the addendum's blinding condition is met or until the PI states, in a later declaration, how the seed-0 descriptive report of issue #63 is read.

## 4. Resources and records

- macM1: heavy lock `~/scratch/.heavy.lock` and `~/.mtplx/paused` are released by the launcher at `EXIT_STOP`; if the wrapper is killed, `rm ~/.mtplx/paused` by hand (runbook). Internal disk at 97 % on 2026-11-04 (28 GB free); the campaign holds 1.1 GB.
- KXKM-AI: no service was stopped for this campaign's KXKM-AI half (never launched); `restore-services.sh` is not needed.
- Records to keep with the campaign: `attempts.jsonl` (with the `stopped_by_STOP_file` event), `status/*.json`, `stop-watcher.log`, `console.log`, `sync.log`, this declaration. Runbook: `~/Documents/Projets/hypneum-n10c/RUNBOOK.md` (GrosMac) and `/home/electron/n10c-pull/RUNBOOK.md` (electron-server).

## 5. What this declaration does not change

Hypotheses, arms, estimands, reading rules, blinding rule, the N10-C-bis addendum and the N10-D work under the reading gate (PRs #100, #102; issues #103, #104). It does not decide whether N10-C is relaunched with another design; that belongs to the roadmap v2 arbitration of the 2026-11-01 plan.

## 6. PI signature and records to append

- [ ] Decision time confirmed (2026-11-04, about 06:38 CET) and signed.
- [ ] `stopped_by_STOP_file` event line and times of `registered-mlp-s0` / `registered-atoc-s0` appended here.
- [ ] Evaluation command of section 3 confirmed and its `attempts.jsonl` lines appended (or another procedure chosen).
- [ ] Machine table of the 2026-11-01 plan (`.github` PR #24) corrected: macM1 busy until the stop, not until 2026-11-02.
