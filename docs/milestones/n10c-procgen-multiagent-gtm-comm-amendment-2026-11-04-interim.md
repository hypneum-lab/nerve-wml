# N10-C — Deviation amendment 2026-11-04 (evening): early stop of macM1 after `registered-mlp-s0` and declared interim analysis of the seed-0 runs

**Status:** DECLARED — signed by the PI on 2026-11-04 (section 8), with his decisions of about 19:10 CET; written and pushed to GitHub **before** any evaluation of a run of tag `n10c-freeze-20261029`. Not frozen. Decision on inference: the whole campaign is exploratory (section 4.2, option (b)).
**Declares against:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` (FROZEN by tag `n10c-freeze-20261029`, tag object `b14717b`, commit `99a7cf4`, SWHID `swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb`), which amends the frozen 2026-10-23 amendment and the locked `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md`; the N10-C-bis addendum `docs/milestones/2026-10-25-n10c-bis-control-arms.md` (DRAFT, section « Blinding during N10-C », binding); the declaration of 2026-10-30 (`…-amendment-2026-10-30.md`, DRAFT); and the morning declaration of the same day, `docs/milestones/n10c-procgen-multiagent-gtm-comm-deviation-2026-11-04.md` (signed 06:51 CET).
**Date drafted:** 2026-11-04, about 19:10 CET (dated by the GitHub push time of its first commit; git dates are not evidence of time).
**Investigator:** Clément Saillant (Hypneum Lab, sole investigator).

The frozen amendments, the locked pre-registration and the earlier declarations are not modified. This document is append-only and only touches `docs/`, so the N10-C launch guard (HEAD a descendant of the tag commit whose intervening commits only touch `docs/` or the manifest folder) still holds.

## 1. Decision

**This interim analysis was not pre-registered.** Neither the locked pre-registration, nor the amendments of 2026-10-23 and 2026-10-25, nor the N10-C-bis addendum provide for any interim look at N10-C outcomes; the addendum forbids reading any held-out result of an N10-C run before the N10-C-bis freeze tag. It is declared here as a deviation, before it happens.

- **Who:** the PI, Clément Saillant.
- **When:** 2026-11-04 at 19:02 CET.
- **What:** (1) a clean pause of the macM1 half of the campaign of tag `n10c-freeze-20261029` after the run in progress; (2) an interim analysis (« pré-analyse », the PI's word) of the held-out evaluations of the seed-0 runs whose training has ended.
- **Why:** the PI asked for a pre-analysis of the runs already trained. No outcome of any run of this campaign had been read when the decision was taken (section 2.3).
- **Mechanism:** a file `campaign/STOP` was created in `~/scratch/n10c-run2/` on macM1 at **2026-11-04T19:02:45+0100** (its content is that timestamp). The runner checks the file between runs only (`run_campaign.py`, `run`, `stop_requested`): it finishes `registered-mlp-s0`, writes the event `stopped_by_STOP_file` and exits with `EXIT_STOP` (4) before `registered-atoc-s0`. No run is interrupted, so no infrastructure restart is consumed. The launcher wrapper lifts the day-model pause at exit.

**Relation to the morning declaration.** The morning declaration (section 1.3) planned to stop after `registered-atoc-s0`, the last seed-0 cell, through a watcher (`stop_after_atoc_s0.sh`, pid 36715). The STOP of 19:02 comes one run earlier: `registered-atoc-s0` is **not trained** unless the campaign is resumed (section 6). The watcher only creates the STOP file when `registered-atoc-s0` starts and exits without acting when the runner pid 75468 is gone; it will therefore exit when the runner exits and plays no further role. The morning plan's « 10 seed-0 runs » becomes 9 seed-0 runs.

## 2. State of the campaign at the STOP

Read-only, on 2026-11-04 between 19:04 and 19:10 CET. Only liveness fields were read: run id, `phase`, `state`, timestamps, `status`, `synced`, `eval_done`, restart counters, checkpoint names, process and GPU lists. No `eval.json`, no `run.json`, no training-curve field and no checkpoint content was opened.

### 2.1 macM1 (`~/scratch/n10c-run2/campaign`, seeds 0, 17, 42)

Runner pid 75468 alive (`--host macm1 --freeze-tag n10c-freeze-20261029 --swhid swh:1:rev:99a7cf4…`), `launcher.state`: « running: command pid 75468, day model paused, 85% free ». Pins of `campaign.json`: `freeze_tag` `n10c-freeze-20261029`, `swhid` `swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb`.

| # | run (seed 0) | train start (CET) | trained (CET) | status | synced to KXKM-AI | restarts (infra / liveness) |
|---|---|---|---|---|---|---|
| 1 | `registered-gtm-s0` | 2026-10-29 06:52:44 | 2026-10-30 00:57:40 | trained | yes | 0 / 0 |
| 2 | `closed-gtm-s0` | 2026-10-30 00:57:41 | 2026-10-30 19:11:08 | trained | yes | 0 / 0 |
| 3 | `registered-nocomm-s0` | 2026-10-30 19:11:10 | 2026-10-31 13:33:27 | trained | yes | 0 / 0 |
| 4 | `closed-nocomm-s0` | 2026-10-31 13:33:29 | 2026-11-01 07:34:14 | trained | yes | 0 / 0 |
| 5 | `registered-dial-s0` | 2026-11-01 07:34:16 | 2026-11-02 01:54:22 | trained | yes | 0 / 0 |
| 6 | `closed-dial-s0` | 2026-11-02 01:54:23 | 2026-11-02 19:57:39 | trained | yes | 0 / 0 |
| 7 | `registered-eng-s0` | 2026-11-02 19:57:40 | 2026-11-03 14:22:14 | trained | yes | 0 / 0 |
| 8 | `closed-eng-s0` | 2026-11-03 14:22:15 | 2026-11-04 08:53:16 | trained | yes | 0 / 0 |
| 9 | `registered-mlp-s0` | 2026-11-04 08:53:18 | expected ≈ 2026-11-05 03:00 | running | no | 0 / 0 |
| 10 | `registered-atoc-s0` | — | — | not started (STOP) | — | — |

Seeds 17 and 42 (20 runs): not started. The hourly sync (macM1 cron `47 * * * *`, `sync-to-kxkm.sh`, last line « synced 0 run(s) » at 18:47) will copy `registered-mlp-s0` to KXKM-AI at the first :47 after its end (≈ 2026-11-05 03:47).

### 2.2 KXKM-AI (`/ai/saisail/n10c-run2/campaign`, seeds 73, 101, 131)

Checked with `ssh kxkm-ai` (user `electron`), read-only, 2026-11-04 19:04 CET.

- **No N10-C process.** No `run_campaign` process, no tmux server for user `electron`. The only GPU compute process is ComfyUI (`/ai/ComfyUI2/comfyenv/bin/python`, pid 712018, 652 MiB; not ours); 1 381 MiB of 24 564 MiB in use.
- **The KXKM-AI runner of this campaign has never been started:** the campaign directory holds only `runs/`, `status/` and `trunks/` (no `campaign.json`, no `attempts.jsonl`, no `console.log`).
- **Runs trained on KXKM-AI: 0 of 30** (seeds 73, 101, 131) and **0 of 10** per-run-pretraining controls (`prc-*-s101`). The registered trunks are in place (`trunks/seed-73`, `seed-101`, `seed-131`, the 10 `seed-101-<condition>-<arm>` control trunks, and the macM1 seeds).
- **Synced macM1 runs:** 8 status files (runs 1–8 of the table above), each `status: trained`, `synced_from: macm1`, `eval_done: false`, `last_checkpoint: ckpt_50000000.safetensors`. **No `eval.json` exists** under `runs/` (count by file name only).
- Repository `/ai/saisail/n10c-run2/repo` at commit `99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb` (tag `n10c-freeze-20261029`, exact match); `.venv` links to `/ai/saisail/n10c-run/.venv`.
- The run-1 directory `/ai/saisail/n10c-run/campaign` (tag `n10c-freeze-20261024`, stopped 2026-10-25, excluded) still holds its `STOP` (2026-10-25 08:38) and is not concerned.

### 2.3 Blinding up to the decision

Up to 19:02 CET on 2026-11-04, no held-out result of any run of tag `n10c-freeze-20261029` existed (no evaluation had run; section 2.2) and none could have been read. The only outcome-related field read on this campaign is the training success of `registered-gtm-s0`, read on 2026-10-30 and declared then (2026-10-30 declaration, section 1). The decision of section 1 is therefore outcome-blind with respect to held-out results, and non-blind with respect to that one training-curve value, which is not a ground of it.

## 3. What is evaluated, where, with what code

**Interim set (9 runs):** `registered-gtm-s0`, `closed-gtm-s0`, `registered-nocomm-s0`, `closed-nocomm-s0`, `registered-dial-s0`, `closed-dial-s0`, `registered-eng-s0`, `closed-eng-s0`, `registered-mlp-s0`. `registered-mlp-s0` is included whatever its training outcome (`trained`, or `training_failure` evaluated at its last finite checkpoint, frozen rule). If it ends in a state that cannot be evaluated, the interim set is the 8 others and this is recorded in section 8. **No other run is evaluated** for this interim analysis.

**Where:** on KXKM-AI only, the sole evaluation host (x86-64 Procgen 0.10.7 build, amendment of 2026-10-25, « Hosts »), in `/ai/saisail/n10c-run2/campaign`, after the sync of `registered-mlp-s0`.

**Code:** the frozen runner at commit `99a7cf4` (tag `n10c-freeze-20261029`), unmodified, with the procedure of section 3 of the morning declaration: the runner command of `kxkm/start.sh` launched directly in tmux session `n10c` (not through `start.sh`, whose dry-run pre-check refuses), with `--freeze-tag n10c-freeze-20261029 --swhid swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb --min-free-gb 1000`. With that setting the runner evaluates the synced seed-0 runs (queue order `SEEDS = (0, 17, 42, 73, 101, 131)`: the seed-0 runs come first; seeds 17 and 42 have no status file on KXKM-AI and are skipped), then reaches `registered-gtm-s73`, whose training is refused by the GPU guard (`gpu_guard_refused`, `EXIT_GPU`, 3). **No KXKM-AI training run starts and no cross-play cell is computed** (cross-play needs two seeds and is only reached after the queue loop).

**Single pass.** The evaluation is launched once, after the 9 status files are present on KXKM-AI (expected ≈ 2026-11-05 03:47, then about 9 × 6.7 min). The 8 runs already synced are **not** evaluated earlier in a separate pass: one look, not two. Each run is evaluated once by the runner (`eval_done`); evaluation failures are retried by the runner within `MAX_EVAL_ATTEMPTS = 3`, never by hand.

**Trigger.** The launch is automated on KXKM-AI by a one-shot trigger (`/ai/saisail/n10c-run2/interim_eval_trigger.sh`, started with `nohup`, timestamped log `/ai/saisail/n10c-run2/interim-eval-trigger.log`). It polls the KXKM-AI status files, reads only `status`, `synced_from` and `eval_done`, and launches the command above when the preconditions below hold; it then exits. It gives up, without launching, if `registered-mlp-s0` is not synced by **2026-11-05 09:00 CET**; the evaluation is then rescheduled by a later record of section 8.

**Preconditions, in order:**

1. This document is signed by the PI (section 8) and **merged on GitHub**; the push and merge times are the evidence that it precedes the evaluation.
2. On KXKM-AI, `status/` holds exactly the 9 status files of the interim set, all `trained` or `training_failure`, `eval_done: false`; `runs/` holds no `eval.json`. (`registered-mlp-s0` is synced only after its training ended, and the macM1 runner exits on the STOP file right after; its `stopped_by_STOP_file` event is recorded afterwards in section 8.)
3. The KXKM-AI repository HEAD is the tag commit `99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb`, with a clean tree.
4. `--dry-run` of the same command prints `ok` for every guard except the GPU one.

**What the evaluation writes.** The runner writes `runs/<run_id>/eval.json` (`harness/evaluate.py`, `EvalRecord`) and an event per run in `attempts.jsonl` that **carries `S` and `R_intact`** (`run_campaign.py`, `evaluate`). From the launch on, the KXKM-AI `attempts.jsonl` and `console.log` are outcome-bearing files; any later liveness monitoring of the campaign reads them only through a field filter (`run`, `phase`, `state`, timestamps). The electron-server archive pull (`/home/electron/archives/n10c`, cron `17 * * * *`) copies the evaluated runs as usual.

**What the interim analysis reads and reports (PI decision 2026-11-04, about 19:10 CET: full scope, all of it exploratory):**

- for each run of the interim set, the scalar fields of `eval.json`: `R_intact`, `R_shuffled`, `S`, `completion_time`, `bandwidth_bits`, with `untrained` / `levels` as provenance, and the per-episode lists when needed;
- the training records of `run.json`, training curves included (`coin_rate_curve`, `entropy_curve`, training success, `liveness`);
- `analysis.py` at the frozen tag, run on the interim set as it is, with the intervals and tests it produces. With one seed and one host it has no seed-level variance, no host block and no cross-play cell: any quantity the frozen code cannot compute from one seed is reported as **undefined**, never replaced by another estimator, interval or test chosen after the look (that would be a new deviation);
- every figure is reported cell by cell (arm × condition, seed 0, host macM1) and labelled « interim, one seed, one host, not pre-registered, exploratory ». **No interval, test, category or contrast of this analysis has any confirmatory value**, whatever its value; none is a pilot category in the sense of the registration.

## 4. Consequences for inference

### 4.1 What is already settled

The morning declaration (section 2) already states that no confirmatory or estimation analysis of N10-C is possible from this campaign as stopped (one seed, one host). The interim analysis adds three things:

1. **Blinding breach (third), declared in advance.** The PI reads held-out results of N10-C runs before the N10-C-bis freeze tag, which the addendum (« Blinding during N10-C », binding) forbids. After the breaches of 2026-10-25 and 2026-10-30, this is the third. Every later decision on N10-C-bis (whether to freeze it, its necessity gate, its open decisions) and on N10-D under the reading gate of the 2026-11-01 plan is **non-blind with respect to the seed-0 held-out results** and must be declared as such where it is recorded. The N10-C-bis hypotheses, arms and contrasts stay those of its first commit (`c910298`).
2. **The 9 evaluated cells become exploratory** for any purpose: they cannot enter a confirmatory test, a pilot category, a variance estimate, a sample-size or threshold choice of a later registration, or a warm start, unless that registration says so before its own freeze.
3. **Reading rules R1–R5 of the addendum are unchanged.** No sentence of the form « GTM-comm communicates / coordinates / outperforms » follows from the interim analysis; the embargo applies to it.

### 4.2 Rule for what comes after (written before any data are seen)

Three options were considered:

- **(a)** the rest of the campaign stays confirmatory for the cells and seeds not looked at, and the final analysis treats them apart;
- **(b)** the whole campaign of tag `n10c-freeze-20261029` is reclassified as exploratory, including any run trained after a resumption;
- **(c)** an alpha-spending correction (group-sequential boundary) applied to the final confirmatory tests.

**Adopted by the PI on 2026-11-04 (about 19:10 CET): (b).** The whole campaign of tag `n10c-freeze-20261029` is exploratory: the 9 seed-0 runs of the interim set, `registered-atoc-s0` and every other run if the campaign is resumed, the evaluations, intervals and tests of the interim analysis and of any later analysis of this campaign. Grounds (as drafted before the decision):

- Option (a) has almost nothing to protect. Seed 0 touches 9 of the 10 cells; the registered estimators are per-arm across six seeds, with host as a block factor. Removing seed 0 leaves 5 seeds with unbalanced host blocks (2 macM1, 3 KXKM-AI), which is a different estimator, not the registered one. The only cells not looked at would be `registered-atoc` and the cross-play cells, and the cross-play cells involve seed 0 through the fixed pairs `(131, 0)` and `(0, 17)`. Moreover, any resumption would be decided after the look, so the set of runs that exists at the end would itself depend on the interim results.
- Option (c) does not fit. No spending function was registered; choosing one now is itself an analysis choice made at the look. With one seed the interim analysis has no test, so the alpha « spent » is undefined. Alpha-spending controls repeated testing on accruing data; it does not repair design decisions (stop, continue, scope) that may follow the look.
- The campaign has already lost its confirmatory status on other grounds: two declared blinding breaches (2026-10-25, 2026-10-30), the stop after seed 0 of the morning declaration, the open covariance question of 2026-10-30 (section 3 there), and R3 of the addendum if N10-C-bis is not run.
- Option (b) is the most honest reading and costs nothing that (a) or (c) would credibly preserve. A confirmatory N10-C claim, if the PI still wants one, requires a **new freeze**: new annotated tag and Software Heritage SWHID, a registered analysis that includes the covariance decision, and a seed namespace stated in that freeze that excludes seed 0 and every seed of this campaign that has been looked at.

Under (b), runs trained after a resumption are reported as exploratory, with the same frozen code and the same reporting format, and are never pooled with a later confirmatory campaign.

### 4.3 Decisions taken after the interim analysis

Any decision taken after the interim analysis to **resume, stop for good, extend, restrict or re-scope** the campaign is recorded in a new dated declaration that states its grounds. It may not depend on the interim results without a new declared deviation: if any ground refers to an interim result, the declaration says so and the decision is reported as outcome-informed. This holds under every option of section 4.2; under (b) it does not change the exploratory status of the data, but it keeps the record of why each run exists.

## 5. Prohibited because of the interim analysis

On the basis of the interim results, none of the following may be changed for N10-C, for its resumption or for the N10-C-bis runs:

- hyperparameters, training options (`fix-c-coin-pretrain`), PPO settings, budget (50 000 000 pair-steps), per-host settings;
- thresholds of every kind: liveness guard, learnability gate, pilot categories, equivalence margin Δ, reading rules R1–R5, failure and exclusion rules;
- code at the tag (`99a7cf4`), the trunk pins (`campaign/trunk_pins.py`), the guards, the evaluator and the held-out levels;
- the set of cells, arms, conditions, seeds, the run order of the frozen queue and the per-run-pretraining control;
- the disposition of any run: no run is excluded, replaced, re-trained or re-evaluated because of its interim result.

A change of any of these after the interim analysis is a new deviation and, under section 4.3, outcome-informed unless its grounds are shown to be independent of the results.

## 6. Conditions of resumption

The macM1 half may be resumed only if all of the following hold:

1. **A dated PI declaration** (appended to this file or a new file) records the decision to resume, its scope (for instance `registered-atoc-s0` only, to complete seed 0, or the full macM1 queue) and its grounds, under section 4.3.
2. **The previous runner has exited** (event `stopped_by_STOP_file` in `attempts.jsonl`, `launcher.state` no longer « running ») and the day-model pause has been lifted (`~/.mtplx/paused` absent).
3. **The PI removes the STOP file:** `rm ~/scratch/n10c-run2/campaign/STOP`. Nobody else removes it.
4. **Relaunch through the frozen launcher**, with the base directory of run 2 set explicitly (the default `BASE` of `start.sh` is the run-1 directory `~/scratch/n10c-run`, which belongs to the excluded campaign):

   ```
   cd ~/scratch/n10c-run2/repo
   BASE=~/scratch/n10c-run2 zsh experiments/procgen_multiagent_gtm_comm/campaign/macm1/start.sh \
     --freeze-tag n10c-freeze-20261029 --swhid swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb
   ```

   with the same environment as the launch of 2026-10-29 (including `HEAVY_LOCK` if it was set then). `start.sh` refuses if its dry-run guard reports « WOULD REFUSE » or if a runner is already alive. The repository must stay at the tag commit or a descendant that touches only `docs/` or the manifest folder.
5. **Restart rules unchanged:** at most 2 infrastructure restarts per run (`MAX_INFRA_RESTARTS = 2`, so at most 3 attempts; after that a training failure evaluated at its last finite checkpoint, never replaced), and at most 1 liveness restart, counted apart (`MAX_LIVENESS_RESTARTS = 1`). A STOP between runs is not an infrastructure failure; a runner killed during a run counts as one infrastructure restart on resume.
6. **Stop point:** if the resumption is limited (for instance to `registered-atoc-s0`), a new stop mechanism must be set up before the launch: the watcher of the morning declaration exits with the runner pid 75468 and does not act on a new runner.
7. **Evaluation of any run trained after the resumption** follows section 3 (KXKM-AI only, frozen runner, `--min-free-gb 1000` unless the KXKM-AI half is launched by a separate declaration), in a single pass per declared batch.

Under option (b), adopted in section 4.2, the runs of a resumption are exploratory runs of tag `n10c-freeze-20261029`, never confirmatory runs.

## 7. What this amendment does not change

Hypotheses, arms, estimands, blinding rule (except the breach declared in section 4.1), reading rules, the N10-C-bis addendum, the morning declaration (except its stop point, section 1), and the N10-D work under the reading gate.


## 8. PI signature, decisions and records to append

**Signature.** PI, 2026-11-04 — Clément Saillant, instruction given in session at about 19:10 CET, with the decisions below, before any evaluation of a run of tag `n10c-freeze-20261029`.

**Decisions of the PI (2026-11-04, about 19:10 CET):**

- [x] Option of section 4.2: **(b)**, the whole campaign of tag `n10c-freeze-20261029` is exploratory.
- [x] Scope of the interim reading: **full** (section 3): scalar fields of `eval.json`, training curves of `run.json`, `analysis.py` with its intervals and tests, all declared exploratory, without any confirmatory value.
- [x] Covariance correction of the 2026-10-30 declaration (section 3 there), which was to be decided « before any held-out N10-C result is read »: **not decided now**. It will be decided **after** the interim reading and is therefore a **non-blind** decision, reported as such. Under option (b) it bears on the exploratory analysis only.
- [ ] Time and channel of the « 0.96 » reading of 2026-10-30 (open decision 1 of that declaration): **still open**, not decided here.

**Records to append, each in its own later commit:**

- [ ] Pull request number and merge time of this amendment on GitHub.
- [ ] `stopped_by_STOP_file` event line on macM1, end time and final status of `registered-mlp-s0`.
- [ ] Time of the sync of `registered-mlp-s0` to KXKM-AI.
- [ ] Launch time of the KXKM-AI evaluation (trigger log), its `--dry-run` output (guard lines only), the `gpu_guard_refused` event, and the list of the 9 runs with `eval_done: true`, **without their results**.
- [ ] Date and time at which the PI first reads the interim results.
- [ ] Decision on the covariance correction (after the reading, non-blind).
- [ ] Decision on resumption (section 6) and its grounds (section 4.3).

**Closure (appended 2026-11-05).** The records above (PR number and merge time, sync of `registered-mlp-s0`, evaluation launch and the 9 runs with `eval_done: true`, first reading of the interim results), the time and channel of the « 0.96 » reading of 2026-10-30, and the PI decision of 2026-11-05 (about 06:33 CET) to stop N10-C for good with no resumption — an outcome-informed decision — are recorded in `docs/milestones/n10c-procgen-multiagent-gtm-comm-closure-2026-11-05.md`. The macM1 records (`stopped_by_STOP_file`, end of `registered-mlp-s0`) are in its section 1.
