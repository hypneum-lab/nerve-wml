# N10-C — Closure declaration 2026-11-05: campaign stopped for good, exploratory, task to be redesigned for N10-D

**Status:** DECLARED — decisions of the PI on 2026-11-05 at about 06:33 CET, recorded after the first reading of the interim results. Not frozen. **The decision of section 2 is outcome-informed (not blind)**; section 2.2 says which of its grounds predate the reading.
**Declares against:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` (FROZEN by tag `n10c-freeze-20261029`, tag object `b14717b`, commit `99a7cf4`, SWHID `swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb`); the N10-C-bis addendum `docs/milestones/2026-10-25-n10c-bis-control-arms.md`; the declaration of 2026-10-30 (`…-amendment-2026-10-30.md`); the morning declaration of 2026-11-04 (`…-deviation-2026-11-04.md`, signed 06:51 CET); and the interim-analysis amendment of 2026-11-04 (`…-amendment-2026-11-04-interim.md`, PR #107), whose sections 4.3 and 8 ask for this record.
**Date drafted:** 2026-11-05 (dated by the GitHub push time of its first commit; git dates are not evidence of time).
**Investigator:** Clément Saillant (Hypneum Lab, sole investigator).

The frozen amendments, the locked pre-registration and the earlier declarations are not modified. This document is append-only and only touches `docs/`. Nothing in it is a confirmation of any hypothesis; under option (b) of the interim amendment (section 4.2), every figure of this campaign is exploratory.

## 1. Records (section 8 of the interim amendment)

Times are CET (UTC+1). Sources are cited; outcome fields are not reproduced here except where section 2.2 needs them as grounds, and then only the figures the PI quoted.

| # | Record | Time | Source |
|---|---|---|---|
| R1 | Interim amendment merged on GitHub (PR #107, merge commit `5ac2a82`) | 2026-11-04 18:18:16Z (19:18:16 CET) | GitHub API, `mergedAt` |
| R2 | `registered-mlp-s0` synced to KXKM-AI and trained (trigger preconditions met: 9 status files, `eval_done` false, no `eval.json`, HEAD `99a7cf4` = tag, clean, no runner) | 2026-11-05 02:47:56 | `kxkm-ai:/ai/saisail/n10c-run2/interim-eval-trigger.log` |
| R3 | `--dry-run` of the frozen command: « GUARDS: all pass »; GPU guard « 22.1 GiB free, need 1000.0 -> WOULD REFUSE » (expected) | 2026-11-05 02:47:57 | same log |
| R4 | Single-pass evaluation of the 9 seed-0 runs: first `eval start` (`registered-gtm-s0`) 02:47:58, last `eval done` (`registered-mlp-s0`) 02:54:35, then event `gpu_guard_refused` (no KXKM-AI training, no cross-play cell); trigger done 02:54:57 | 2026-11-05 02:47:57 → 02:54:57 | `kxkm-ai:/ai/saisail/n10c-run2/campaign/attempts.jsonl` (fields `at`, `run`, `phase`, `state` only), trigger log; runner exit code 3 (`EXIT_GPU`) in `interim-eval-runner.rc` |
| R5 | Runs with `eval_done: true`: `registered-gtm-s0`, `closed-gtm-s0`, `registered-nocomm-s0`, `closed-nocomm-s0`, `registered-dial-s0`, `closed-dial-s0`, `registered-eng-s0`, `closed-eng-s0`, `registered-mlp-s0` (9 of 9) | — | same `attempts.jsonl` |
| R6 | **First reading of the interim results** by the PI, in a Claude Code session; the report (`kxkm-ai:/ai/saisail/n10c-run2/interim-analysis-20261105/RAPPORT.md`, file time 06:24:53) was relayed in chat | 2026-11-05, about 06:25 | session record; file mtime |
| R7 | Time and channel of the « 0.96 » reading of 2026-10-30 | see section 3 | — |
| R8 | Decision to stop N10-C for good, with its grounds | 2026-11-05, about 06:33 | section 2 |

Still open after this record: the `stopped_by_STOP_file` event line of the macM1 runner (pid 75468), the end time of `registered-mlp-s0` on macM1 and the removal of the hourly sync cron: macM1 was unreachable over Tailscale and the LAN when this record was drafted. They are appended in a later commit. The covariance correction of the 2026-10-30 declaration (section 3 there) is moot under section 2: no further analysis of this campaign is planned; if one is ever made, the decision is non-blind and is recorded with it.

## 2. Decision: N10-C is stopped for good

### 2.1 Decision

On 2026-11-05 at about 06:33 CET, after reading the interim report, the PI decided:

- **N10-C is stopped definitively, with no resumption of any kind:** not `registered-atoc-s0`, not the macM1 queue (seeds 17 and 42), not the KXKM-AI half (seeds 73, 101, 131, and the per-run-pretraining controls). Section 6 of the interim amendment (conditions of resumption) will not be used. The macM1 `STOP` file stays in place.
- **The task is to be redesigned for N10-D.** No N10-C run, figure or curve is carried into N10-D as a pilot, a threshold, a sample-size input or a warm start unless the N10-D registration says so explicitly before its own freeze (interim amendment, section 4.1, item 2).
- **The campaign of tag `n10c-freeze-20261029` is reported as exploratory and stopped** (interim amendment, section 4.2, option (b)). Its 9 evaluated seed-0 cells (one seed, one host) are the whole of its data. No verdict, positive or negative, is drawn on GTM communication from it.

### 2.2 Grounds, split by date relative to the reading

**Explicit statement required by section 4.3 of the interim amendment.** Every ground of this decision that is taken from the interim analysis is **informed by the results** and the decision is therefore **not blind**. It is reported as an outcome-informed decision.

**(a) Ground that predates the reading of 2026-11-05.**

- **The task did not require communication** — BLOQUANT (3) of the review of 2026-10-30. In the development records, the content-free arm `gtm_mis` reached 0.81 team success at 1M pair-steps, above `gtm` (0.54–0.73 on the development seeds); a scripted no-message policy reaches about 43 %; team success stays close to the per-agent own-coin rate. This is why the PI decided on 2026-10-30, at about 15:10 CET, the **necessity gate** before any N10-C-bis freeze, with an asymmetric N10-D on failure (addendum, section « Necessity gate », G1–G3; PR #91, merged 2026-10-30 19:33:19Z). Its sources are development records only (seeds 91xx, learnability gate, B3 results), not outcomes of the registered campaign. Caveat already declared on 2026-10-30: that gate decision itself was taken **after** the « 0.96 » reading of section 3 and is reported as a non-blind design decision, although it does not rest on that figure.
- The morning declaration of 2026-11-04 (stop after seed 0, by resource decision, outcome-blind) already made a confirmatory or estimation analysis of N10-C impossible from this campaign.

**(b) Grounds that postdate the reading (outcome-informed, not blind).** Each of the following was read in the interim report of 2026-11-05 and is therefore informed by the results:

1. **`nocomm` too good:** with no message, the pair reaches 0.94 training success and 0.415 held-out success, at the level of the communicating arms (control at risk of being tautological).
2. **`time_to_half` degenerate:** it equals the first logged point (1,000,000 pair-steps) for the 9 runs, so the registered sample-efficiency metric carries no information on this task as warm-started.
3. **Generalisation gap of 0.40 to 0.54** between training success (levels 0–15) and held-out success: the held-out measure is dominated by generalisation to new levels, not by coordination.
4. **The effects themselves:** `gtm − nocomm` +0.025 (registered) and +0.060 (closed), point estimates from one seed; the message-shuffle sensitivity S of `gtm` is compatible with 0. The `gtm − nocomm` contrasts were computed outside the frozen code (interim report, section 5); the frozen `analysis.py` cannot compute seed-level intervals from one seed. All of them are exploratory.

Grounds (b)1–4 are consistent with ground (a), but they are not independent of the results: any reader must treat the decision as made after, and partly because of, the look at the seed-0 held-out data.

### 2.3 What this decision does not do

- It does not exclude, replace, re-train or re-evaluate any run (interim amendment, section 5).
- It does not change hypotheses, estimands, thresholds or reading rules R1–R5 of the addendum; under R3, N10-C is reported as uninterpretable as evidence of communication, with its exploratory estimates.
- It does not freeze or launch N10-C-bis. The N10-C-bis necessity gate and the N10-D design are decided in their own documents, and every such decision is **non-blind with respect to the seed-0 held-out results** (interim amendment, section 4.1, item 1).

## 3. The « 0.96 » reading of 2026-10-30 (open decision 1 of the 2026-10-30 declaration)

**Best estimate.** The final training success of `registered-gtm-s0` (last point of `success_curve` in `run.json`, 0.96 at 50,000,000 pair-steps; `run.json` has no separate final-success field) was **first displayed to an agent on 2026-10-30 at 13:12:11 CET**, and **first shown to the PI at 13:12:25 CET**, in the chat of an interactive Claude Code session on the PI's laptop (GrosMac). The PI had asked at 13:11:35 CET « des premier résultat ? » (status of E1-small and N10-C); the session answered with a status of both campaigns that quoted « finit à 0,96 ».

**Evidence (Claude Code transcripts on GrosMac, read-only):**

| Time (CET) | Event | Transcript entry |
|---|---|---|
| 2026-10-30 00:57:40 | `run.json` of `registered-gtm-s0` written on macM1 (end of training) | deviation of 2026-11-04, section 1.1; KXKM-AI copy keeps this mtime |
| 13:11:35 | PI asks for first results | session `53298cbc…`, user message |
| 13:11:47 | Session lists the run directory (`run.json` present); no field read | tool result |
| 13:11:55 | `head -c 1500` of `run.json` displayed: start of `success_curve` only (0.68 at 1M, 0.86 at 2M, 0.96 at 3M, …), **not** the final value | tool result |
| 13:12:11 | Python extraction prints `final [50000000, 0.96]` and the last 10 curve values | tool result |
| 13:12:25 | Status relayed to the PI in chat: « 0,68 à 1 M, 0,96 à 3 M … finit à 0,96 » | assistant message |
| 14:08:52 | Next PI message in that session (« vas y »): latest time by which the PI had the message | user message |
| 14:09:55 | Figure written to the session memory note | tool call |
| 14:15:41 | Figure pasted into the review session of the N10-C-bis design (session `0f498106…`), where the independent review flagged it as a blinding breach | user message |
| ≈ 15:10 | PI decision on the necessity gate (after the reading) | addendum, « Necessity gate » |

**Correction to the 2026-10-30 declaration.** That declaration says the figure was « read and quoted during the PI's review of the N10-C-bis design ». It was in fact first read about an hour earlier, in a campaign-status exchange, and then quoted in that review (14:15:41 CET). Both precede the necessity-gate decision.

**Uncertainty and bounds.**

- **Agent first saw the final value:** 13:12:11 CET (tight; the 13:11:55 display showed 0.96 only as an early curve point at 3M).
- **PI first saw it:** not earlier than 13:12:25 CET and not later than 14:08:52 CET on the transcript evidence; most likely within minutes of 13:12:25, since the PI had just asked the question.
- **Absolute lower bound:** 00:57:40 CET (file written). Between 00:57:40 and 13:11:55 no GrosMac Claude Code transcript contains the value or reads `run.json`; the only background job touching the run directory (a stall watcher) read checkpoint modification times only. The hourly sync to KXKM-AI did not cover the run-2 directory before the macM1 crontab was repointed at about 20:36 CET (first sync at 20:47 CET), an automated copy with no reader.
- **Not checked:** sessions on other hosts (macM1 local sessions, Codex CLI) and any direct terminal look by the PI on macM1; these cannot be excluded from the transcripts and are the residual uncertainty. macM1 was unreachable when this was drafted, so its `sync.log` and local `attempts.jsonl` were not re-read.

## 4. Records still to append

- [ ] macM1: `stopped_by_STOP_file` event line, end time and final status of `registered-mlp-s0`, `launcher.state`, `~/.mtplx/paused` lifted, heavy lock free.
- [ ] macM1: hourly sync cron (`47 * * * *`, `sync-to-kxkm.sh`) removed, crontab backed up.
- [ ] Machine table of the 2026-11-01 plan (`.github` PR #24) corrected (morning declaration, section 6).

## 5. Signature

PI, 2026-11-05 — Clément Saillant, decisions given in session at about 06:33 CET, after the first reading of the interim results (about 06:25 CET). Pending the PI's review and merge of this pull request.
