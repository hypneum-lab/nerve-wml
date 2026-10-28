# N10-C — Deviation and pre-registration amendment 2026-10-25: campaign stop, trunk collapse, training fix and learnability gate

**Status:** DRAFT (not frozen, not filed). Needs a new freeze tag, Software Heritage snapshot and stage-2 manifest before any confirmatory run.
**Amends:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md` (frozen by tag `n10c-freeze-20261024`, commit `2ce4214`), which amends `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md`.
**Date drafted:** 2026-10-25.
**Code:** branch `fix/n10c-trunk-collapse` (`experiments/procgen_multiagent_gtm_comm/harness/{options,train,rollout,vec_env,pretrain,devledger,gate_report}.py`, `campaign/{guards.py,queue.py,run_campaign.py}`, `campaign/macm1/{start,pretrain,day_model_paused}.sh`, tests `tests/unit/n10c/test_{trunk_fix,gate_report,macm1_launcher,liveness_control}.py`).
**Records:** the gate and pre-training records of sections 4.1–4.2 are committed, byte-identical, in `docs/superpowers/research/2026-10-25-n10c-gate/` (`index.json`: SHA-256, ledger id, archive location and code of each record; checkpoints and console logs stay in the archive).
**Depends on it:** the N10-C-bis control-arms addendum (`docs/milestones/2026-10-25-n10c-bis-control-arms.md`, PR #61, code #80) was merged into master on 2026-10-25 as a DRAFT, not frozen. By PI decision of 2026-10-25 19:50 it is aligned with this amendment in this pull request: its appended section « Alignment with the N10-C amendment of 2026-10-25 » makes its control arms use `fix-c-coin-pretrain`, the seed trunks, the guards and the liveness rule, and flags its provisions made inapplicable by the stop and the fix (including a declared blinding breach).

The frozen 2026-10-23 amendment and the locked 2026-05-11 file are not modified. This document is append-only. Where it conflicts with them, it governs once filed.

## 1. Deviation: the campaign was stopped after outcome inspection

**This stop was outcome-informed and is declared as a deviation.**

- The campaign under tag `n10c-freeze-20261024` started on 2026-10-24 (KXKM-AI 14:22:40, first run `registered-eng-s73` ; macM1 14:23:47, first run `registered-gtm-s0`).
- Two runs completed training (50M pair-steps) and their registered held-out evaluation (`campaign/attempts.jsonl`):
  - `registered-eng-s73` (KXKM-AI): trained 2026-10-24 14:22 → 2026-10-25 00:35, evaluated 00:40, held-out team success `R_intact = 0.005` ;
  - `registered-gtm-s0` (macM1): trained 2026-10-24 14:23 → 2026-10-25 07:33 (65,387 s), evaluated on KXKM-AI 08:33, `R_intact = 0.0`.
  The PI and the analysis inspected these outcomes, the training records (near-zero training success, no learning) and the checkpoints.
- Two runs were in progress and were interrupted: `closed-gtm-s0` (macM1, started 2026-10-25 07:33) and `closed-eng-s73` (KXKM-AI, started 08:33).
- On 2026-10-25 at about 08:50 (CET), **by PI decision, after this inspection**, both hosts were stopped (KXKM-AI by SIGINT, macM1 by SIGTERM). `STOP` files were left in both campaign directories and the frozen campaign will not be resumed.
- The stop decision was made after looking at outcomes (training success, held-out team success and checkpoint diagnostics) of registered runs. It is therefore **not** an outcome-blind failure rule of the 2026-10-23 amendment (Failure, exclusion and replacement), and it is reported as such.

**Disposition of the runs of tag `n10c-freeze-20261024`.** The 2 completed runs and the 2 interrupted runs are **excluded from every confirmatory or estimation analysis** of N10-C. They are reported in full (run records, checkpoints, the audit below) as a failed configuration. None of them is reused, pooled or used as a pilot variance estimate for the new freeze.

## 2. Audit evidence (measured on the checkpoints)

Scripts: `diag_ckpt.py`, `diag_trunk.py`, `diag_env.py` (macM1 `~/scratch/n10c-audit/`, to be archived with the run records). An independent code audit of the frozen tag (Codex, read-only) found no defect that drops the team reward, reverses the PPO objective or disables the optimizer, and ranked reward starvation first.

1. **The IMPALA trunk dies early on both hosts.** On real CoinRun-coop frames of the training levels, the positive fraction of the residual stream after the third (last) IMPALA block is **0**, at most **11 of 256** h units are active on any frame, and the policy logits and the value are **identical across frames**. This holds for both completed runs (GTM-comm on macM1, the engineered reference sender on KXKM-AI), so it does not depend on the arm, the host or the backend. It is reached within the first 1M pair-steps (the first checkpoint).
2. **Trigger: a team-only sparse reward.** On the training levels, a uniform random pair earns the team reward in about **0.5–2.5 %** of cooperative episodes. A scripted "right / right-jump" policy reaches about **43 %** through the real wrapper, so the task is solvable and the wrapper is not broken. With the rare reward, the policy gradient carries almost no task signal; the value and entropy terms, under Adam, push the last-block pre-activations negative until the ReLUs are dead, after which no gradient reaches the trunk (dead-ReLU collapse).
3. **Secondary defects (do not by themselves explain the failure):**
   - actions of a *finished* agent (frozen, action ignored by the environment) still enter the PPO policy loss, the entropy bonus and the advantage statistics (`coop_env.py:200-201`, `harness/train.py:447-460, 223-228` at the frozen tag) ;
   - MLX Adam uses `eps = 1e-8` (its default) where PPO baselines use `1e-5` ; with near-zero gradients the smaller epsilon makes larger relative steps ;
   - there is no liveness or learnability check before or during a run: the smoke runs and tests validated mechanics, not learning.

## 3. Training fix (PI choice C, made 2026-10-25 08:55, before any development result)

The PI asked for two fix families, decided by a learnability gate, and then chose, **before any gate result existed**, variant **C = (1) + (2) + guards** as the fix of this amendment. Variants A and B are kept as disclosed ablations, not decision inputs.

Every option is a named, registered preset of `harness.train.TrainOptions` (`PRESETS`). Nothing else can be selected. A registered run must use the preset named by `campaign.guards.REGISTERED_TRAIN_OPTIONS`, which the new freeze sets to `fix-c-coin-pretrain`. The frozen configuration stays selectable as `frozen-20261024` (the default until the new freeze), so tag `n10c-freeze-20261024` remains reproducible.

| Preset | Own-coin reward | Pre-trained trunk | Adam eps | Finished-agent mask | Liveness guard |
|---|---|---|---|---|---|
| `frozen-20261024` (tag of 2026-10-24) | 0 | no | 1e-8 | no | off |
| `fix-a-coin` (A, ablation) | +0.5 | no | 1e-5 | yes | abort |
| `fix-b-pretrain` (B, ablation) | 0 | yes | 1e-5 | yes | abort |
| **`fix-c-coin-pretrain` (C, PI choice)** | **+0.5** | **yes** | **1e-5** | **yes** | **abort** |

### 3.1 Reward (changes the training reward, not the outcome measure)

- **Training reward of agent i at a step** = team reward (+1 on the step where the second agent reaches its coin within Δt = 20 steps of the first, unchanged) **+ 0.5 if agent i reached its own coin on that step**. An agent can collect its own coin at most once per cooperative episode (it is then finished).
- **The outcome measures are unchanged**: team success (team return 0/1), the success curve, time-to-half, the estimands and the held-out evaluation all use the team reward only. The individual coin reward is a training signal and is never a policy input.
- Consequence for interpretation: a team now receives training signal for one agent reaching its coin even when the pair fails, so part of the learning is single-agent navigation. This is a change of the training objective and is reported as such. The estimand (team success with message vs. reference arms) is unchanged.

### 3.2 Trunk pre-training (warm start)

- `harness/pretrain.py`: the IMPALA trunk is trained by PPO on **single-agent CoinRun, training levels 0–15 only** (`num_levels=16`, `start_level=0`, `easy`), reward 1 for the coin, the registered PPO hyper-parameters and Adam eps 1e-5, with a temporary policy / value head that is discarded.
- **Budget: 1,000,000 single-agent steps** per pre-training run (64 environments × horizon 128).
- Seeding: `SeedSequence([seed, 99])`. The trunk depends on the run seed only: **all arms and both conditions of one seed start from the same pre-trained trunk** (common random numbers across arms). Because the GPU runs are not bit-reproducible and the hosts' settings differ (section 4.2), the registered trunk of a seed is the one produced by its registered pre-training run on the seed's fixed host, identified by its SHA-256. Exception (PI decisions 2026-10-25, 12:07–12:16 and 19:12): the declared per-run-pretraining control of section 3.4 pre-trains one trunk per run of seed 101, seeded `SeedSequence([seed, 99, arm_code, condition_code])` (`harness.pretrain.pretrain_seed_entropy`, `pretrain --per-run ARM CONDITION`). The other parameters (message encoder / decoder, fusion, heads) are initialised exactly as before from the run's own stream.
- The trunk file's SHA-256 is recorded in every run record (`init_trunk_sha256`) and must be listed in the new manifest. Held-out levels 1000–1199 are never used.
- Pre-training adds no partner, no message and no team reward: it cannot by itself produce coordination or communication.

### 3.3 Guards

- **Adam eps = 1e-5** (bias correction stays off, the MLX default, as in the frozen tag).
- **Finished-agent mask.** The live mask is the agent's state *before* the action. Finished agents are excluded from the PPO policy loss, the entropy bonus and the advantage normalisation statistics (masked means over the full minibatch, invariant to micro-batching). Their forward pass is kept, so the message they send to a live partner still receives gradient through the partner's loss. Their value loss is kept as well: the value target of every agent is its own GAE return of the shaped per-agent training reward (team reward plus that agent's own-coin reward, section 3.1), so after an agent has finished its later targets contain only team reward (its own-coin reward is then 0).
- **In-run trunk-liveness guard.** A probe on 128 pair-steps (256 agent frames) of the current rollout, at step 0 and after the first update ending at or after every 100,000 pair-steps (for a 1M-step run: 10 probes after step 0, 11 including it), logs: the positive fraction of the last IMPALA block's residual stream ; the fraction of the 256 h units active on at least one frame ; the overall h nonzero fraction ; the across-batch std of the logits and of the value. The probe is **flagged** if the block-3 positive fraction or the live-h fraction is below 1 %, or a std is ≤ 1e-6. In `abort` mode (all fix presets) a flagged probe after step 0 stops the run, writes a checkpoint and records `liveness_abort` ; the step-0 probe is informational only.
- **Outcome-blind rule for an in-run abort (PI decision 2026-10-25, 12:07–12:16; replaces the proposal of the first draft, which was "training failure, no restart").** A liveness abort under the registered configuration restarts the run **once from scratch**; if the restarted run aborts again, it is a **training failure**: it is not replaced and is evaluated at its last checkpoint (as for a non-finite loss, frozen rule). Implemented in the campaign runner (`campaign/run_campaign.py`, `campaign/queue.py`): the decision reads only the liveness flag of the training record, never an outcome. The first abort is recorded as attempt outcome `liveness_abort`, the attempt is kept aside (`runs/<run_id>.liveness-abort-1`, never evaluated, reported), the status becomes `liveness_restart` and the counter `liveness_restarts` (at most 1) is kept apart from `infra_failures` (at most 2 infrastructure restarts, unchanged); the second abort is a `training_failure` (`liveness_abort: true`) evaluated at the checkpoint the abort wrote. The transition is saved in the status file before the attempt is moved aside (the move is idempotent on resume), and a `running` status whose run directory already holds the `run.json` of an aborted attempt is recorded as that abort on resume, not as an infrastructure failure, so an interruption can never grant a second restart. If a restarted run instead exhausts its infrastructure restarts, the frozen rule applies as before, except that the checkpoints of the aborted attempt (kept under `.liveness-abort-1`) are never used. The pre-freeze gate (section 4) exists so that this should not happen.
- Per-run logs added to `run.json`: `liveness` (every probe), `coin_rate_curve` (per-agent own-coin rate over the last 100 completed episodes, both agents), `entropy_curve` (mean and minimum policy entropy of live agents per rollout), `train_options`, `init_trunk_sha256`.

### 3.4 Declared per-run-pretraining control (PI decisions 2026-10-25, 12:07–12:16 and 19:12)

The shared trunk of section 3.2 makes the arms of a seed start from the same trunk (common random numbers). A declared control measures how much that choice matters, descriptively:

- **Control seed: 101** (a KXKM-AI seed, `codes.PER_RUN_TRUNK_SEED`). All **10 runs** of that seed (6 arms × `registered`, 4 arms × `closed`) are trained **a second time**, each from its **own** pre-trained trunk. The registered runs of seed 101 are unchanged and keep the seed's shared trunk.
- **Seeding:** the own trunk of run (arm, condition) is pre-trained with `SeedSequence([101, 99, arm_code, condition_code])` (`codes.ARM_CODES`, `codes.CONDITION_CODES`); the shared trunk keeps `SeedSequence([seed, 99])`. Budget, levels and PPO settings as in section 3.2; registered pre-training on the seed's host (KXKM-AI, its per-host settings), `pretrain --registered --per-run ARM CONDITION`, guarded (control seed only, an N10-C run, exact budget, real Procgen).
- **Run ids:** `prc-<condition>-<arm>-s101` (`queue.control_queue`), distinct from the 60 registered run ids, which are unchanged; the frozen 60-run list and its order are unchanged.
- **Place in the queue (PI decision 2026-10-25 19:50):** on KXKM-AI, **interleaved and paired in time**: each control run `prc-<condition>-<arm>-s101` is trained immediately after its shared-trunk registered run `<condition>-<arm>-s101` (`queue.control_partner`), so the pair runs back to back: their difference in time and host state is reduced, not removed, and is reported; the registered order of the 60 runs is unchanged, the KXKM-AI runs of seed 131 start after the 20 runs of seed 101. They are trained, evaluated (held-out, as any run) and archived like the registered runs; the campaign is complete only when they are evaluated. They exist only when the registered preset uses a pre-trained trunk (none under the frozen preset). An earlier draft of this section placed them after all registered KXKM-AI runs; the PI chose the paired order before any run.
- **Provenance:** each control trunk is pinned in the stage-2 manifest (`guards.REGISTERED_CONTROL_TRUNK_SHA256[(arm, condition)]`, file `trunks/seed-101-<condition>-<arm>/trunk.safetensors`); the training boundary refuses any other trunk; the run record carries `init_trunk_sha256` and `trunk_mode: "per_run"` (registered runs: `"shared"`).
- **Analysis role:** descriptive only. For each of the 10 (arm, condition) cells, the held-out team success of the control run is reported next to the registered seed-101 run with the shared trunk (difference, no interval, no test). Control runs are **never pooled** into the confirmatory or estimation analysis, the pilot categories or the cross-seed cross-play (whose cells use the 60 registered runs only), and they do not replace a registered run.

## 4. Pre-freeze learnability gate (new development rule)

Before the new freeze tag, the adopted preset must pass a learnability gate:

- **Runs:** arms `gtm` and `eng`, condition `registered`, development seed **9101** (not a registered seed), **≤ 1,000,000 pair-steps** each, training levels 0–15 only, logs every 40,000 pair-steps, liveness probe and checkpoint every ~100,000 pair-steps. A **second development seed, 9102**, was added for C by the PI at 12:07 (see below), with its own pre-trained trunk.
- **Hosts (PI decisions 2026-10-25, all made before the corresponding results):** the trunk pre-training ran on macM1. At 09:10 the PI moved the gate runs (C and the ablations A and B, both arms) to **KXKM-AI**, with the KXKM per-host settings (micro-batch 512 pairs, 8 workers, `--clear-cache`), one run at a time. At 09:18 the PI added **one C check on macM1** (arm `gtm`, macM1 per-host settings: no micro-batching), because the campaign uses both hosts and backends (CUDA, Metal). Rows are never compared across hosts.
- **Pass criterion (fixed here before the gate results were examined; ratified by the PI at 12:07 on the condition of a second development seed, see section 6):** for both arms, (i) no liveness probe flagged after step 0 and no abort ; (ii) the team success over the last 100 completed episodes at 1M pair-steps is at least **5 %**, twice the upper end (2.5 %) of the uniform-random baseline ; (iii) the per-agent coin rate rises above its first logged value. A failure of C is **reported and the campaign is not frozen**; the PI decided that a failed C is **not** replaced by switching to A or B. The check is implemented by `harness/gate_report.py` (`verdict`), which also verifies the record's arm, seed, condition and preset, the probe schedule and the curve endpoints.
- **Accounting:** development-ledger kinds `gate_dev` (per arm) and `pretrain_dev` (in total, arm `trunk`), separate from the exhausted smoke allowance, in `manifest/smoke_ledger.json` (`harness/devledger.py`). The caps were raised by successive PI decisions, each made before the results of the runs it allowed; none is retroactive:

  | PI decision (2026-10-25) | `gate_dev` gtm | `gate_dev` eng | `pretrain_dev` | max steps per `gate_dev` run | runs it allowed |
  |---|---:|---:|---:|---:|---|
  | first draft (`a070f11`, 09:09) | 3 | 3 | 2 | 1M | C, A, B per arm (rows 1–6) |
  | 09:10 (`06dd573`) | 4 | 4 | 2 | 1M | + the slot spent by the 0-step crash of 09:05 (rows 8–10, 12) |
  | 09:18 and 09:50 (`f700935`) | 7 | 4 | 2 | 1M | macM1 C check (row 7), its rerun (11), B rerun (13) |
  | 12:07–12:10 (`3ebc98f`) | 10 | 7 | 4 | 1M | second dev seed of C (rows 14, 15, 20, 21), C vs C-bc on seed 9101 (16–19) |
  | 12:16 (`f4a4778`) | 10 | 7 | 5 | 1M | validation of the micro-batched pre-training (row 22) |
  | 13:33 (`72e0890`) | 20 | 13 | 8 | 10M | the Adam bias-correction study of section 4.3 (in progress) |

  The two hosts (and, on KXKM-AI, the working copies `repo`, `repo-head`, `repo-fix` and `repo-adam` of the successive code versions) kept separate copies of the ledger; they are merged by run id into the committed file, each entry tagged in its `note` with host and preset (from row 14 also the code). Every gate run, including crashed ones, is ledgered and disclosed below. Gate runs never touch held-out levels or any result of a full-length run, and are excluded from all analyses.

### 4.1 Development runs of this amendment (ledger)

All times 2026-10-25 (CET), from `started_at` in `manifest/smoke_ledger.json`. The pass criterion above was committed at 09:09 (`f53d509`), before the first gate result (09:26). Rows 1–13 were listed in the first version of this table; rows 14–22 were started later the same day and are appended in start order.

| # | start | host | kind | seed | arm | preset | steps | outcome |
|---|---|---|---|---|---|---|---:|---|
| 1 | 09:05 | macM1 | pretrain_dev | 9101 | trunk | — | 0 of 1,000,000 | infrastructure crash at 0 steps (`cmake` not on PATH for the Procgen build) |
| 2 | 09:05 | macM1 | gate_dev | 9101 | gtm | fix-a-coin | 0 of 1,000,000 | same crash |
| 3 | 09:05 | macM1 | gate_dev | 9101 | eng | fix-a-coin | 0 of 1,000,000 | same crash |
| 4 | 09:06 | macM1 | pretrain_dev | 9101 | trunk | — | 1,000,000 | completed (trunk shared by every seed-9101 gate run that uses one) |
| 5 | 09:13 | KXKM-AI | gate_dev | 9101 | gtm | fix-a-coin (ablation) | 1,000,000 | completed |
| 6 | 09:26 | KXKM-AI | gate_dev | 9101 | gtm | **fix-c-coin-pretrain** | 1,000,000 | completed |
| 7 | 09:27 | macM1 | gate_dev | 9101 | gtm | **fix-c-coin-pretrain** | 0 of 1,000,000 (no checkpoint) | stopped by hand (SIGTERM, 09:47): 19 GB footprint next to the resident second-brain day model, swap nearly exhausted, no step logged in 20 min |
| 8 | 09:38 | KXKM-AI | gate_dev | 9101 | gtm | fix-b-pretrain (ablation) | ~606,000 of 1,000,000 | infrastructure crash (CUDA illegal memory access), most likely caused by our own unit tests run on the same card at 09:46 (GPU OOM) |
| 9 | 09:46 | KXKM-AI | gate_dev | 9101 | eng | **fix-c-coin-pretrain** | 1,000,000 | completed |
| 10 | 09:59 | KXKM-AI | gate_dev | 9101 | eng | fix-a-coin (ablation) | 1,000,000 | completed |
| 11 | 10:00 | macM1 | gate_dev | 9101 | gtm | **fix-c-coin-pretrain** | 1,000,000 | completed (rerun of 7, day model paused 09:59–10:21, PI decision 09:50) |
| 12 | 10:11 | KXKM-AI | gate_dev | 9101 | eng | fix-b-pretrain (ablation) | 1,000,000 | completed |
| 13 | 11:40 | KXKM-AI | gate_dev | 9101 | gtm | fix-b-pretrain (ablation) | 1,000,000 | completed (rerun of 8, PI decision 09:50; no other process of ours on the card) |
| 14 | 12:07 | KXKM-AI | pretrain_dev | 9102 | trunk | — | 0 of 1,000,000 | crashed at the first update: MLX CUDA `cudaGraphAddKernelNode ... invalid argument` on the full 2048-sample PPO minibatch (pre-training had no micro-batching then); led to the micro-batch fix `f4a4778` |
| 15 | 12:08 | macM1 | pretrain_dev | 9102 | trunk | — | 1,000,000 | completed (616 s); trunk copied to KXKM-AI for rows 20–21 |
| 16 | 12:14 | KXKM-AI | gate_dev | 9101 | gtm | **fix-c-coin-pretrain** | 1,000,000 | completed (Adam comparison, PI 12:10; code `3ebc98f`) |
| 17 | 12:26 | KXKM-AI | gate_dev | 9101 | eng | **fix-c-coin-pretrain** | 1,000,000 | completed (same) |
| 18 | 12:39 | KXKM-AI | gate_dev | 9101 | gtm | fix-c-coin-pretrain-bc (comparison) | 1,000,000 | completed (same) |
| 19 | 12:51 | KXKM-AI | gate_dev | 9101 | eng | fix-c-coin-pretrain-bc (comparison) | 1,000,000 | completed (same) |
| 20 | 13:04 | KXKM-AI | gate_dev | 9102 | gtm | **fix-c-coin-pretrain** | 1,000,000 | completed (second dev seed of C, PI 12:07; code `06dd573` + allowance patch) |
| 21 | 13:16 | KXKM-AI | gate_dev | 9102 | eng | **fix-c-coin-pretrain** | 1,000,000 | completed (same) |
| 22 | 13:28 | KXKM-AI | pretrain_dev | 9102 | trunk | — | 1,000,000 | completed (374 s; validation of the micro-batched pre-training `f4a4778`, PI 12:16; its trunk is used by no gate run) |

Runs 1–3, 7, 8 and 14 consumed ledger slots without a result (infrastructure). Under the caps of 12:07–12:16, `gate_dev` gtm used 10 of 10 (rows 2, 5, 6, 7, 8, 11, 13, 16, 18, 20), eng 7 of 7 (rows 3, 9, 10, 12, 17, 19, 21) and `pretrain_dev` 5 of 5 (rows 1, 4, 14, 15, 22). Code of each completed run: `index.json` of the records (KXKM-AI rows 5, 6, 9, 10, 12: `06dd573`; rows 13, 20, 21: `06dd573` plus the allowance change; rows 16–19: `3ebc98f`; row 22: `f4a4778`; macM1 rows 4, 11, 15: the branch files in an uncommitted working copy). Runs of the Adam study (section 4.3), started from 13:35, are ledgered under the caps of 13:33 and listed in section 4.3; their ledger entries are merged into the committed file (five at 14:00, the rest at 19:26, including the outcome of the C eng seed-9103 run that was still running at 14:00 ; the final outcome of the part-B C run and the part-B C-bc entry on 2026-10-26).

### 4.2 Gate results

Generated from the committed records by `python -m experiments.procgen_multiagent_gtm_comm.harness.gate_report table docs/superpowers/research/2026-10-25-n10c-gate/index.json` (the test `test_committed_gate_cells_match_index_and_amendment` checks that every row below is what the records give). Team success = mean team return over the last 100 completed training episodes (the success curve); coin rate = per-agent own-coin rate over the last 100 completed episodes (3 decimals: the rate moves in steps of 0.005); liveness minima are over the 10 probes after step 0 (11 probes per run including step 0); "flagged" is recomputed from the probe metrics; "criterion" = (i)–(iii) of section 4 for that run's own arm, seed and preset. The uniform-random team success on these levels is 0.5–2.5 %.

| # | host | seed | arm | preset | team success at 1M (max) | coin rate first → 1M | min block-3 positive fraction | min live-h fraction | flagged / abort | criterion | wall time |
|---|---|---|---|---|---|---|---:|---:|---|---|---:|
| 6 | KXKM-AI | 9101 | gtm | **C** | **0.54** (0.72) | 0.217 → 0.595 | 0.072 | 0.75 | no / no | PASS | 736 s |
| 9 | KXKM-AI | 9101 | eng | **C** | **0.76** (0.76) | 0.235 → 0.785 | 0.075 | 0.86 | no / no | PASS | 740 s |
| 11 | macM1 | 9101 | gtm | **C** | **0.39** (0.44) | 0.208 → 0.455 | 0.085 | 0.73 | no / no | PASS | 1,291 s |
| 5 | KXKM-AI | 9101 | gtm | A | 0.82 (0.82) | 0.185 → 0.860 | 0.115 | 0.57 | no / no | PASS | 747 s |
| 10 | KXKM-AI | 9101 | eng | A | 0.87 (0.88) | 0.219 → 0.880 | 0.099 | 0.67 | no / no | PASS | 741 s |
| 12 | KXKM-AI | 9101 | eng | B | 0.00 (0.01) | 0.206 → 0.005 | 0.037 | 0.92 | no / no | FAIL | 730 s |
| 13 | KXKM-AI | 9101 | gtm | B | 0.00 (0.01) | 0.181 → 0.000 | 0.057 | 0.82 | no / no | FAIL | 729 s |
| 16 | KXKM-AI | 9101 | gtm | **C** | **0.64** (0.66) | 0.218 → 0.655 | 0.067 | 0.84 | no / no | PASS | 733 s |
| 17 | KXKM-AI | 9101 | eng | **C** | **0.63** (0.66) | 0.275 → 0.705 | 0.076 | 0.83 | no / no | PASS | 739 s |
| 18 | KXKM-AI | 9101 | gtm | C-bc | 0.61 (0.67) | 0.210 → 0.665 | 0.094 | 0.68 | no / no | PASS | 743 s |
| 19 | KXKM-AI | 9101 | eng | C-bc | 0.70 (0.78) | 0.260 → 0.770 | 0.103 | 0.70 | no / no | PASS | 740 s |
| 20 | KXKM-AI | 9102 | gtm | **C** | **0.73** (0.73) | 0.219 → 0.765 | 0.076 | 0.86 | no / no | PASS | 738 s |
| 21 | KXKM-AI | 9102 | eng | **C** | **0.84** (0.86) | 0.175 → 0.865 | 0.067 | 0.83 | no / no | PASS | 737 s |

Pre-training (all 1,000,000 single-agent steps, trunk alive by the probe thresholds; coin rate at the first log, 106,496 steps → 1M): row 4 (macM1, seed 9101, 615 s, 5,302 episodes) 0.40 → 0.63, trunk SHA-256 prefix `0c27f2daa0a9`, recorded as `init_trunk_sha256` in rows 6, 9, 11, 12, 13 and 16–19 ; row 15 (macM1, seed 9102, 616 s, 8,379 episodes) 0.46 → 0.85, prefix `b6805dee76c2`, rows 20–21 ; row 22 (KXKM-AI, seed 9102, micro-batch 512 pairs and `--clear-cache`, 374 s, 9,135 episodes) 0.51 → 0.88, prefix `b5955d1baffa`: the micro-batched pre-training runs on CUDA, where row 14 crashed. Rows 15 and 22 used the same seed on different hosts and settings and give different trunks.

**Gate verdict for C: PASS.** On seed 9101, both arms on KXKM-AI (rows 6, 9, and again 16, 17) and the gtm check on macM1 (row 11) meet (i) no flagged probe after step 0 and no abort, (ii) team success at 1M ≥ 5 % and (iii) a rising coin rate; so do both arms of the second development seed 9102 on KXKM-AI (rows 20, 21), which the PI required at 12:07 for the ratification of the gate.

**GPU non-determinism (reported, not a decision input).** The KXKM-AI runs are not bit-reproducible: MLX has no deterministic convolution backward on CUDA. Rows 16–17 repeat rows 6 and 9 (same seed, trunk, preset and host settings; code `3ebc98f` only adds the `-bc` preset) and do not reproduce their values: team success at 1M gtm 0.54 → 0.64, eng 0.76 → 0.63. The two repeats differ from the first runs by 0.10 and 0.13 in team success. This is one repeat per arm, not an estimate of the run-to-run spread, but it shows that single-run differences of about 0.1 between rows are not interpretable.

Observations, not decision inputs:

- **In these runs, the own-coin reward is what separated learning from non-learning.** B (pre-trained trunk, team-only reward) keeps the trunk alive by the probe thresholds but did not learn in either of its two runs (one seed, 1M of 50M pair-steps): the coin rate falls to about 0 (eng 0.206 → 0.005, gtm 0.181 → 0.000) and on `eng` the value is nearly constant across frames (std ≈ 0.001), whereas every run with the own-coin reward (A, C, C-bc) learned. This does not establish that B cannot learn with more steps or other seeds. The thresholds of the liveness guard do not detect this form of failure; the gate criterion (ii)–(iii) does.
- A ablation exceeds C at 1M on both arms of seed 9101 (one run each; differences 0.28 on gtm and 0.11 on eng, against repeat differences of 0.10–0.13 above). The PI's choice of C was made before these results and is not revisited.
- **C vs C-bc (Adam bias correction, rows 16–19):** gtm 0.64 vs 0.61, eng 0.63 vs 0.70 at 1M; both pass. The differences (0.03 and 0.07) are smaller than the repeat differences above; **the question is not decided**: the PI asked for a deeper verification (section 4.3).
- **macM1 vs KXKM-AI for C/gtm (0.39 vs 0.54 and 0.64):** one run on macM1, different per-host settings (no micro-batching on macM1) and backends; not interpretable as a host effect. Peak process memory on macM1 was 11.0 GB (KXKM-AI ≈ 7 GB). The first macM1 attempt (row 7) stalled at a 19 GB footprint next to the second-brain day model; the frozen campaign's macM1 run had trained 18 h next to the same service. The memory budget of macM1 with the pre-trained-trunk path is settled by the PI decision of section 6.

### 4.3 Adam bias-correction study (PI decision 2026-10-25 13:33, before its results; done, decided: no bias correction)

Development runs on KXKM-AI (KXKM per-host settings, `gate_dev` / `pretrain_dev`, code `f4a4778` with the allowance change of 13:33, working copy `repo-adam`), C (`fix-c-coin-pretrain`) vs C-bc (`fix-c-coin-pretrain-bc`), training levels 0–15 only:

- **(C) mechanism:** seed 9101, arm gtm, C and C-bc, 200,000 pair-steps each, a checkpoint every 10,000 pair-steps, for an offline analysis of the update magnitudes and of trunk liveness early in training.
- **(A) seeds:** three more development seeds 9103, 9104 and 9105, each pre-trained on KXKM-AI with the micro-batched pre-training, then C and C-bc on gtm and eng at 1M pair-steps (12 runs, 3 pre-trainings).
- **(B) length:** seed 9101, arm gtm, C and C-bc at 10,000,000 pair-steps (the 13:33 cap for `gate_dev` runs).

**Parts A and C, status at 19:26 (records in `docs/superpowers/research/2026-10-25-n10c-gate/kxkm/adam-study/`, listed under `adam_study` in `index.json` with SHA-256, ledger id and archive location; run.json and pretrain.json only, the checkpoints stay on KXKM-AI):**

- **(A) done:** 3 pre-trainings (completed, trunks alive) and the 12 runs. Table generated by `python -m experiments.procgen_multiagent_gtm_comm.harness.gate_report adam docs/superpowers/research/2026-10-25-n10c-gate/index.json` (checked by `test_committed_adam_rows_match_amendment`). "Last-200k mean" = mean of the team-success curve at the 5 logs after 800k pair-steps; block-3 = mean over the 10 probes after step 0.

  | seed | arm | C at 1M | C-bc at 1M | C last-200k mean | C-bc last-200k mean | C-bc − C | mean block-3 positive fraction C / C-bc | criterion C / C-bc |
  |---|---|---:|---:|---:|---:|---:|---|---|
  | 9103 | eng | 0.84 | 0.80 | 0.748 | 0.822 | +0.074 | 0.076 / 0.108 | PASS / PASS |
  | 9103 | gtm | 0.76 | 0.62 | 0.698 | 0.624 | -0.074 | 0.066 / 0.089 | PASS / PASS |
  | 9104 | eng | 0.71 | 0.77 | 0.696 | 0.786 | +0.090 | 0.070 / 0.103 | PASS / PASS |
  | 9104 | gtm | 0.58 | 0.66 | 0.578 | 0.726 | +0.148 | 0.060 / 0.102 | PASS / PASS |
  | 9105 | eng | 0.88 | 0.80 | 0.796 | 0.826 | +0.030 | 0.073 / 0.136 | PASS / PASS |
  | 9105 | gtm | 0.89 | 0.85 | 0.778 | 0.792 | +0.014 | 0.079 / 0.122 | PASS / PASS |

  Summary: C-bc >= C on the last-200k mean in 5 of 6 cells, mean difference +0.047; mean block-3 positive fraction higher with bias correction in 6 of 6. At the single 1M point, C is ahead in 4 of 6 cells. All 12 runs meet the gate criterion of section 4 and no probe was flagged. These are 3 seeds × 1M of 50M pair-steps, with run-to-run differences of about 0.1 on repeats (section 4.2): **not decisive** for the registered preset.
- **(C) mechanism done** (two 200k runs, seed 9101, gtm; 20 checkpoints each, at the first update ending at or after every 10,000 pair-steps, 16,384 to 200,000): block-3 positive fraction at the probes 0 / ~100k / 200k is 0.064 / 0.087 / 0.081 for C and 0.064 / 0.111 / 0.124 for C-bc; team success at 200k 0.25 vs 0.27. The relative parameter change between consecutive checkpoints is **1.5–2× larger without bias correction** over the first 200k pair-steps (computed on the archived checkpoints with `mech.py`, KXKM-AI `/ai/saisail/n10c-dev/mech.py`, SHA-256 in `index.json`; not recomputable from the committed records).
- **(B) done (2026-10-25 22:32).** The two 10M runs were first launched at 16:27 and refused at once by the training CLI, whose development cap was still 1M for every kind (no step, no ledger entry); after the `gate_dev` cap was raised to 10M (`b5d76a9`), C ran 18:24–20:28 and C-bc 20:28–22:32 (seed 9101, gtm, 7,434 s and 7,431 s). Records under `adam_study` (part B) in `index.json`. Team success (last 100 episodes) at 2 / 4 / 6 / 8 / 10M pair-steps: C 0.90 / 0.99 / 0.96 / 0.91 / 0.87, C-bc 0.93 / 0.87 / 0.98 / 0.98 / 0.94; mean of the last 5 logs (9.2–10M) 0.920 vs 0.924. Coin rate first log → 10M: C 0.44 → 0.875, C-bc 0.24 → 0.95. Both trunks alive (live-h fraction at the 10M probe 0.988 and 0.992; minimum over the 100 probes after step 0: live-h 0.785 and 0.660, block-3 positive fraction 0.063 and 0.054), no probe flagged, no abort. One seed per preset: no decisive evidence favouring either preset at 10M.

**Decision (PI, decision round of 2026-10-25 about 20:15 and 2026-10-26 22:02; the exact time of this item is in the PI's record — at 20:15 the C run of part B was close to its end (it finished at 20:28) and the C-bc run had not started): no Adam bias correction for the new freeze.** The registered preset stays `fix-c-coin-pretrain` (bias correction off, as in the frozen tag and in the gate). `fix-c-coin-pretrain-bc` remains a development-only comparison preset: it is never the registered preset and is not used by any registered run. The study is closed.

## 5. New freeze, archive and manifest

This amendment cannot reuse tag `n10c-freeze-20261024`. Before any run of the new configuration: (step-by-step plan, open decisions and estimates: `docs/milestones/n10c-freeze-plan-2026-10-28.md`)

1. the PI's ratification of the gate criterion and of the gate result of C (12:07, on the condition of a second development seed, met by rows 20–21) is recorded in the stage-2 manifest ;
2. `campaign.guards.REGISTERED_TRAIN_OPTIONS` is set to `fix-c-coin-pretrain` (or to the bias-correction variant, only if the PI so decides after section 4.3 and before the freeze) ;
3. a new stage-2 manifest records the presets, the pre-training budget and seeding, the trunk files' SHA-256 (one per registered seed, produced by registered pre-training runs on the seed's fixed host, plus the 10 per-run trunks of the declared control of section 3.4), the gate results and this ledger, and the per-host throughput including pre-training ;
4. a **new annotated freeze tag** (`n10c-freeze-<date>`), a **new Software Heritage snapshot (SWHID)** and the manifest SHA-256 in the tag message, under the 2026-10-23 commitment rules ; the campaign launcher's pins (`campaign.json`) are regenerated ;
5. runs started before the new tag's GitHub push time are excluded and reported.

All other provisions of the 2026-10-23 amendment (arms, conditions, seeds and hosts, 50M pair-steps, evaluation, estimands, failure and exclusion rules) are unchanged, except the in-run abort rule of section 3.3, the declared control of section 3.4 (10 additional 50M-pair-step runs on KXKM-AI, about +33 % of its training time, outside the estimands) and the pre-training compute (one 1M-step pre-training per registered seed plus one per control run, small next to a 50M-pair-step run).

## 6. Decided by the PI (2026-10-25)

- **macM1 memory (11:54):** during every macM1 run of the new configuration (pre-training and training), the second-brain day model is paused (`touch ~/.mtplx/paused`, oMLX backend on :8081 stopped) and resumed when the run ends. The registered macM1 per-host settings (no micro-batching) are unchanged. `campaign/macm1/day_model_paused.sh` owns the pause (it refuses if the flag already exists, checks free memory before the command starts and lifts the pause on every exit path); `campaign/macm1/start.sh` (campaign runner) and `campaign/macm1/pretrain.sh` (pre-training) run through it.
- **Gate (12:07):** the gate criterion and the gate result of C are ratified, on the condition of a second development seed of C on both arms. Done: seed 9102, rows 15, 20 and 21, PASS (section 4.2).
- **In-run abort (12:07–12:16):** a liveness abort restarts the run once from scratch; a second abort is a training failure (section 3.3, implemented in the campaign runner).
- **Pre-training (12:07–12:16, 19:12):** registered runs use one pre-training per registered seed, shared by all arms and conditions of that seed (common random numbers, section 3.2). In addition, a **declared control** on one KXKM-AI seed: every run of that seed, all arms, gets its own pre-training (one per run) instead of the shared trunk. At 19:12 the PI fixed the control: seed 101 (KXKM-AI), all 10 runs, own trunk `SeedSequence([seed, 99, arm_code, condition_code])`; specified in section 3.4 (run ids, provenance, descriptive role only). At 19:50 the PI placed each control run right after its shared-trunk registered run (paired in time, section 3.4).
- **Adam bias correction (12:10, 13:33):** first a C vs C-bc comparison on seed 9101 (rows 16–19, not conclusive), then the deeper study of section 4.3, before any of its results. **Decided (decision round 2026-10-25 about 20:15 / 2026-10-26 22:02, see section 4.3): no bias correction for the freeze**; `fix-c-coin-pretrain` unchanged, `fix-c-coin-pretrain-bc` development-only (section 4.3).
- **Control queue and N10-C-bis (19:50):** the control runs are interleaved, each right after its shared-trunk registered run of seed 101 (section 3.4) ; the N10-C-bis addendum is aligned with this fix in this pull request (its control arms use `fix-c-coin-pretrain`, the seed trunks, the guards and the liveness rule ; appended alignment section of the addendum).
- **Freeze procedure (2026-10-28, about 18:00; plan `docs/milestones/n10c-freeze-plan-2026-10-28.md`):** the trunk SHA-256 of section 5 item 3 come from **pre-freeze registered pre-trainings** (`harness.pretrain --pre-freeze`: the exact registered procedure of each registered seed and of the 10 control units of seed 101, on the seed's fixed host with its settings, from a clean commit on origin, ledger kind `pretrain_registered`; outcome-blind, no coop task, no held-out level) run before the new tag; an unusable pre-training gets one restart, then the PI decides. The 16 trunk files are committed in the freeze commit (`manifest/trunks/`) so that the tag and its Software Heritage snapshot contain them; their SHA-256 are pinned in code (`campaign/trunk_pins.py`) and listed in the new stage-2 manifest (`stage2-manifest-fixc.md`). The freeze rewrites the Status line of this amendment only; the 2026-10-23 amendment and `stage2-manifest.md` stay byte-identical. Order: macM1 pre-trainings then its campaign right after the tag; KXKM-AI pre-trainings, E1-small, then the N10-C campaign; N10-C-bis frozen after the N10-C tag.

## 7. Open decisions for the PI

None open from this amendment at the time of writing; the decisions of 2026-10-26 22:02 on N10-C-bis (gate of the new arms, blinding statement, VQ mask) are recorded in the N10-C-bis addendum (section « Further PI decisions, 2026-10-26 »).
