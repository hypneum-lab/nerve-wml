# N10-C — Deviation and pre-registration amendment 2026-10-25: campaign stop, trunk collapse, training fix and learnability gate

**Status:** DRAFT (not frozen, not filed). Needs a new freeze tag, Software Heritage snapshot and stage-2 manifest before any confirmatory run.
**Amends:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md` (frozen by tag `n10c-freeze-20261024`, commit `2ce4214`), which amends `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md`.
**Date drafted:** 2026-10-25.
**Code:** branch `fix/n10c-trunk-collapse` (`experiments/procgen_multiagent_gtm_comm/harness/{train,rollout,vec_env,pretrain,devledger}.py`, `campaign/guards.py`, tests `tests/unit/n10c/test_trunk_fix.py`).
**Depends on it:** draft PR #61 (N10-C-bis control arms addendum) is written against the frozen 2026-10-23 configuration. It must be rebased on this amendment and its control arms must use the training options adopted here.

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
- Seeding: `SeedSequence([seed, 99])`. The trunk depends on the run seed only: **all arms and both conditions of one seed start from the same pre-trained trunk** (common random numbers across arms). The other parameters (message encoder / decoder, fusion, heads) are initialised exactly as before from the run's own stream.
- The trunk file's SHA-256 is recorded in every run record (`init_trunk_sha256`) and must be listed in the new manifest. Held-out levels 1000–1199 are never used.
- Pre-training adds no partner, no message and no team reward: it cannot by itself produce coordination or communication.

### 3.3 Guards

- **Adam eps = 1e-5** (bias correction stays off, the MLX default, as in the frozen tag).
- **Finished-agent mask.** The live mask is the agent's state *before* the action. Finished agents are excluded from the PPO policy loss, the entropy bonus and the advantage normalisation statistics (masked means over the full minibatch, invariant to micro-batching). Their forward pass is kept, so the message they send to a live partner still receives gradient through the partner's loss ; their value target (the team return) is kept.
- **In-run trunk-liveness guard.** A probe on 128 pair-steps (256 agent frames) of the current rollout, at step 0 and after the first update ending at or after every 100,000 pair-steps, logs: the positive fraction of the last IMPALA block's residual stream ; the fraction of the 256 h units active on at least one frame ; the overall h nonzero fraction ; the across-batch std of the logits and of the value. The probe is **flagged** if the block-3 positive fraction or the live-h fraction is below 1 %, or a std is ≤ 1e-6. In `abort` mode (all fix presets) a flagged probe after step 0 stops the run, writes a checkpoint and records `liveness_abort` ; the step-0 probe is informational only.
- **Outcome-blind rule for an in-run abort (proposed).** A liveness abort under the registered configuration is a **training failure**: the run is not restarted or replaced, it is evaluated at its last checkpoint (as for a non-finite loss, frozen rule). The pre-freeze gate (section 4) exists so that this should not happen.
- Per-run logs added to `run.json`: `liveness` (every probe), `coin_rate_curve` (per-agent own-coin rate over the last 100 completed episodes, both agents), `entropy_curve` (mean and minimum policy entropy of live agents per rollout), `train_options`, `init_trunk_sha256`.

## 4. Pre-freeze learnability gate (new development rule)

Before the new freeze tag, the adopted preset must pass a learnability gate:

- **Runs:** arms `gtm` and `eng`, condition `registered`, development seed **9101** (not a registered seed), **≤ 1,000,000 pair-steps** each, training levels 0–15 only, logs every 40,000 pair-steps, liveness probe and checkpoint every ~100,000 pair-steps.
- **Hosts (PI decisions 2026-10-25, all made before the corresponding results):** the trunk pre-training ran on macM1. At 09:10 the PI moved the gate runs (C and the ablations A and B, both arms) to **KXKM-AI**, with the KXKM per-host settings (micro-batch 512 pairs, 8 workers, `--clear-cache`), one run at a time. At 09:18 the PI added **one C check on macM1** (arm `gtm`, macM1 per-host settings: no micro-batching), because the campaign uses both hosts and backends (CUDA, Metal). Rows are never compared across hosts.
- **Pass criterion (to be ratified by the PI before the freeze; fixed here before the gate results were examined):** for both arms, (i) no liveness probe flagged after step 0 and no abort ; (ii) the team success over the last 100 completed episodes at 1M pair-steps is at least **5 %**, twice the upper end (2.5 %) of the uniform-random baseline ; (iii) the per-agent coin rate rises above its first logged value. A failure of C is **reported and the campaign is not frozen**; the PI decided that a failed C is **not** replaced by switching to A or B.
- **Accounting:** development-ledger kind `gate_dev` (cap 4 per arm and 7 for `gtm`, see `harness/devledger.py`; separate from the exhausted smoke allowance) and `pretrain_dev` (cap 2 in total), in `manifest/smoke_ledger.json`. The two hosts kept separate working copies of the ledger during the gate; they were merged by run id into the committed file. Every gate run, including crashed ones, is ledgered and disclosed below. Gate runs never touch held-out levels or any result of a full-length run, and are excluded from all analyses.

### 4.1 Development runs of this amendment (ledger)

All times 2026-10-25 (CET), from `started_at` in `manifest/smoke_ledger.json`. The pass criterion above was committed at 09:09 (`f53d509`), before the first gate result (09:26).

| # | start | host | kind | arm | preset | steps | outcome |
|---|---|---|---|---|---|---:|---|
| 1 | 09:05 | macM1 | pretrain_dev | trunk | — | 0 of 1,000,000 | infrastructure crash at 0 steps (`cmake` not on PATH for the Procgen build) |
| 2 | 09:05 | macM1 | gate_dev | gtm | fix-a-coin | 0 of 1,000,000 | same crash |
| 3 | 09:05 | macM1 | gate_dev | eng | fix-a-coin | 0 of 1,000,000 | same crash |
| 4 | 09:06 | macM1 | pretrain_dev | trunk | — | 1,000,000 | completed (trunk shared by every gate run that uses one) |
| 5 | 09:13 | KXKM-AI | gate_dev | gtm | fix-a-coin (ablation) | 1,000,000 | completed |
| 6 | 09:26 | KXKM-AI | gate_dev | gtm | **fix-c-coin-pretrain** | 1,000,000 | completed |
| 7 | 09:27 | macM1 | gate_dev | gtm | **fix-c-coin-pretrain** | 0 of 1,000,000 (no checkpoint) | stopped by hand (SIGTERM, 09:47): 19 GB footprint next to the resident second-brain day model, swap nearly exhausted, no step logged in 20 min |
| 8 | 09:38 | KXKM-AI | gate_dev | gtm | fix-b-pretrain (ablation) | ~606,000 of 1,000,000 | infrastructure crash (CUDA illegal memory access), most likely caused by our own unit tests run on the same card at 09:46 (GPU OOM) |
| 9 | 09:46 | KXKM-AI | gate_dev | eng | **fix-c-coin-pretrain** | 1,000,000 | completed |
| 10 | 09:59 | KXKM-AI | gate_dev | eng | fix-a-coin (ablation) | 1,000,000 | completed |
| 11 | 10:00 | macM1 | gate_dev | gtm | **fix-c-coin-pretrain** | 1,000,000 | completed (rerun of 7, day model paused 09:59–10:21, PI decision 09:50) |
| 12 | 10:11 | KXKM-AI | gate_dev | eng | fix-b-pretrain (ablation) | 1,000,000 | completed |
| 13 | 11:40 | KXKM-AI | gate_dev | gtm | fix-b-pretrain (ablation) | 1,000,000 | completed (rerun of 8, PI decision 09:50; no other process of ours on the card) |

Runs 1–3, 7 and 8 consumed ledger slots without a result (infrastructure). The gtm allowance of 7 covers rows 2, 5, 6, 7, 8, 11 and 13 (PI decisions 09:10, 09:18, 09:50, each before the result it allowed). The B rerun (13) used the gate code of commit `06dd573` plus the allowance change of `f700935`, as the other KXKM rows did.

### 4.2 Gate results

Team success = mean team return over the last 100 completed training episodes (the success curve); coin rate = per-agent own-coin rate over the last 100 completed episodes; liveness minima are over the probes after step 0 (11 probes per run). The uniform-random team success on these levels is 0.5–2.5 %.

| Run (#) | host | arm | preset | team success at 1M (max) | coin rate first → 1M | min block-3 positive fraction | min live-h fraction | flagged / abort | wall time |
|---|---|---|---|---|---|---:|---:|---|---:|
| 6 | KXKM-AI | gtm | **C** | **0.54** (0.72) | 0.22 → 0.60 | 0.072 | 0.75 | no / no | 736 s |
| 9 | KXKM-AI | eng | **C** | **0.76** (0.76) | 0.24 → 0.79 | 0.075 | 0.86 | no / no | 740 s |
| 11 | macM1 | gtm | **C** | **0.39** (0.44) | 0.21 → 0.46 | 0.085 | 0.73 | no / no | 1,291 s |
| 5 | KXKM-AI | gtm | A | 0.82 (0.82) | 0.19 → 0.86 | 0.115 | 0.57 | no / no | 747 s |
| 10 | KXKM-AI | eng | A | 0.87 (0.88) | 0.22 → 0.88 | 0.099 | 0.67 | no / no | 741 s |
| 12 | KXKM-AI | eng | B | 0.00 (0.01) | 0.21 → 0.005 | 0.037 | 0.92 | no / no | 730 s |
| 13 | KXKM-AI | gtm | B | 0.00 (0.01) | 0.18 → 0.00 | 0.057 | 0.82 | no / no | 729 s |

Pre-training (row 4, macM1, 615 s, 5,302 episodes): single-agent coin rate 0.40 at the first log (106,496 steps) → 0.63 at 1M; trunk alive. Trunk SHA-256 prefix `0c27f2daa0a9`, recorded as `init_trunk_sha256` in rows 6, 9, 11, 12 and 13.

**Gate verdict for C (to be ratified by the PI): PASS.** Both arms on KXKM-AI and the gtm check on macM1 meet (i) no flagged probe after step 0 and no abort, (ii) team success at 1M ≥ 5 % (0.54, 0.76 and 0.39), and (iii) a rising coin rate.

Observations, not decision inputs:

- **The own-coin reward is what unblocks learning.** B (pre-trained trunk, team-only reward) keeps the trunk alive by the probe thresholds but does not learn: on both arms the coin rate falls to about 0 (eng 0.21 → 0.005, gtm 0.18 → 0.00) and on `eng` the value is nearly constant across frames (std ≈ 0.001). The thresholds of the liveness guard do not detect this form of failure; the gate criterion (ii)–(iii) does.
- A ablation exceeds C at 1M on both arms. This is one seed and 1M of 50M pair-steps; the PI's choice of C was made before these results and is not revisited.
- **macM1 vs KXKM-AI for C/gtm (0.39 vs 0.54):** one seed per host, different per-host settings (no micro-batching on macM1) and backends; not interpretable as a host effect. Peak process memory on macM1 was 11.0 GB (KXKM-AI ≈ 7 GB). The first macM1 attempt (row 7) stalled at a 19 GB footprint next to the second-brain day model; the frozen campaign's macM1 run had trained 18 h next to the same service. The memory budget of macM1 with the pre-trained-trunk path is settled by the PI decision of section 6.

## 5. New freeze, archive and manifest

This amendment cannot reuse tag `n10c-freeze-20261024`. Before any run of the new configuration:

1. the PI ratifies the gate criterion and the gate result of C ;
2. `campaign.guards.REGISTERED_TRAIN_OPTIONS` is set to `fix-c-coin-pretrain` ;
3. a new stage-2 manifest records the presets, the pre-training budget and seeding, the trunk files' SHA-256 (one per registered seed, produced by registered pre-training runs on the seed's fixed host), the gate results and this ledger, and the per-host throughput including pre-training ;
4. a **new annotated freeze tag** (`n10c-freeze-<date>`), a **new Software Heritage snapshot (SWHID)** and the manifest SHA-256 in the tag message, under the 2026-10-23 commitment rules ; the campaign launcher's pins (`campaign.json`) are regenerated ;
5. runs started before the new tag's GitHub push time are excluded and reported.

All other provisions of the 2026-10-23 amendment (arms, conditions, seeds and hosts, 50M pair-steps, evaluation, estimands, failure and exclusion rules) are unchanged, except that the pre-training adds compute per seed (one 1M-step pre-training per registered seed, small next to 10 runs × 50M pair-steps).

## 6. Decided by the PI (2026-10-25)

- **macM1 memory (11:54):** during every macM1 run of the new configuration (pre-training and training), the second-brain day model is paused (`touch ~/.mtplx/paused`, oMLX backend on :8081 stopped) and resumed when the run ends. The registered macM1 per-host settings (no micro-batching) are unchanged. Before the new freeze, the macM1 launcher must do the pause, check free memory before each run and lift the pause afterwards (not yet implemented).

## 7. Open decisions for the PI

1. Ratify the gate pass criterion (section 4) and the gate outcome of C.
2. Ratify the in-run abort rule (liveness abort = training failure, not replaced).
3. Pre-trained trunk shared by all arms of a seed (common random numbers) vs. one pre-training per run.
4. Whether to also enable Adam bias correction (off in MLX by default and in the frozen tag ; not changed here).
5. Rebase of draft PR #61 (N10-C-bis) on this amendment.
