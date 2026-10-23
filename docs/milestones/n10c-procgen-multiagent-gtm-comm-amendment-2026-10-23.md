# N10-C — Pre-registration amendment 2026-10-23: estimation reframing, engineered reference sender and side-channel control

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Amends:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md` (pre-registered 2026-05-11, commit `1640c5c`)
**Date drafted:** 2026-10-23
**OSF registration:** no N10-C-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N10-C training run exists. The only code is the CoinRun-coop environment (`experiments/procgen_multiagent_gtm_comm/coop_env.py`) and a smoke test. Caveflyer-coop and Maze-coop are not implemented.

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed. Where it conflicts with the locked file, this amendment governs. The locked file labels its research hypothesis "H0" ; this amendment calls it the **research hypothesis**.

## Summary of the change

1. **Estimation / pilot reframing (PI decision 2026-10-23). This changes the interpretation of the study.** N10-C becomes an estimation (pilot) study. It reports effect estimates with confidence intervals and makes **no superiority verdict**. The C-headline / C-tied-stable / C-loses labels and their win-count thresholds are **not applied**. Seeds and the registered arms are unchanged.
2. **Team reward without step penalty.** Success is the only reward ; completion time is reported separately.
3. Add an **engineered reference sender** arm and a **no-communication** arm, and report each arm's **normalized improvement** over no-communication.
4. Add a **side-channel control**: a side-channel-closed training condition and an evaluation-time message-shuffle probe. Its result is reported only as message-disruption sensitivity.
5. Give an executable specification of the CoinRun-coop environment, the 16-bit message for every arm, the engineered sender's inputs, the evaluation protocol, randomness, and failure / exclusion / replacement rules. Correct the compute figures.

## Rationale

- **Power.** With 5 seeds per arm and Bonferroni correction the registered win-count tests have little power. Fixed-df (df = 8) two-sided t calculations for two independent normal groups of 5 with equal variances:

  | Family | Comparisons | Per-test α | Power at d = 1 | d for 80 % power |
  |---|---:|---:|---:|---:|
  | Registered | 27 | 0.001852 | 2.6 % | 3.59 |
  | Single contrast, unadjusted | 1 | 0.05 | 28.6 % | 2.02 |

  d is the standardized mean difference. No seed-level variance estimate exists, so power at the registered threshold cannot be computed. The PI decided on 2026-10-23 to keep 5 seeds and to report estimates instead of verdicts.
- **Reference and side channels.** Chauhan & Bera, "Emergence, Not Bandwidth: Physical Coupling and the Limits of Learned Multi-Agent Communication", [arXiv:2609.34373](https://arxiv.org/abs/2609.34373) (2026-09-28), argue that the value of learned communication in rate-limited Dec-POMDPs is governed by physical coupling and side channels. Applied to N10-C:
  - A GTM difference against broadcast-MLP / ATOC / DIAL at 16 bits/step does not show how much any learned channel improves over no communication, relative to a channel whose content is known.
  - If agents can coordinate through paths other than the message, team-reward differences between channel architectures may not reflect communication. The kickoff docstring describes a low-dim "team summary" observation (the partner's distance-to-coin estimate) that the 2026-05-11 kickoff code did not implement. On 2026-10-23 the PI decided to implement it on purpose as a **declared side channel** and to close it in the control condition.
- **Reward.** The kickoff code paid −0.1 per step and +1 on team success. A team succeeding at step 13 or later then scored below a team whose agent died at step 1, so team return could reward failure over coordination.

## Environment specification (CoinRun-coop, frozen)

Implemented by `ProcgenCoopEnv` (`experiments/procgen_multiagent_gtm_comm/coop_env.py`) at the commit recorded in the configuration manifest.

- **Instances.** Two procgen CoinRun instances (`distribution_mode="easy"`) play the **same level**: both use the same level seed, derived from (run stream seed, cooperative-episode index). Neither agent's image shows its partner.
- **Levels.** Training: `num_levels=16`, `start_level=0`. Held-out evaluation: the 200 levels with indices 1000–1199, one episode per level, each forced with `num_levels=1`, `start_level=1000+j`.
- **Team reward.** +1 on the step where the second agent reaches its coin, if this happens within Δt = 20 steps of the first agent's coin ; 0 on every other step. No step penalty. An episode's team return is therefore 1 (team success) or 0, and its theoretical range is [0, 1].
- **Episode semantics.** An agent whose own procgen episode ends (coin, death or timeout) is *finished*. It is frozen: its actions are ignored, its observation stays at its last pre-termination frame and its individual reward is 0. The cooperative episode ends at the first of: team reward earned ; the Δt window expires ; an agent finishes without a coin ; all agents finished. This replaces the kickoff code, which ended the cooperative episode as soon as either agent's procgen episode ended, including at the first coin, so the partner could never act on the first coin.
- **Declared team summary** (`team_summary=True`, `TEAM_SUMMARY_DIM = 2`). Procgen exposes no agent or coin positions, so the summary carries the partner's coin timing instead of a distance-to-coin estimate. For agent i: `[1, f]` if its partner reached its coin inside the open window, else `[0, 0]`. f = (number of later steps on which a coin by agent i still earns the team reward) / Δt, so f > 0 exactly while that is possible. The summary is part of the step observation and is computed after that step's reward update. A summary returned with `done=False` can be acted on at the next step ; one returned with `done=True` is terminal. It is reset with the episode.
- **Policy inputs.** Each agent's policy and value network receive only its own image, the received message and, in the registered condition, its team summary. Rewards (team or individual), done flags and partner actions are **not** inputs.
- **Stub.** The procgen stub is rejected unless `allow_stub=True`, which is used only by the smoke test. Every scientific run must report `using_stub == False`. A run on the stub is an infrastructure failure.
- Unit tests: `tests/unit/test_procgen_coop_team_summary.py`.

### Caveflyer-coop and Maze-coop

The registered definitions ("keep distance ≤ X", "visit all 4 quadrants") need agent positions, which procgen does not expose without privileged internal state. Their definitions are therefore part of the required configuration manifest, under these constraints:

- the reward rule uses only signals exposed by procgen's public gym3 interface (rewards, episode ends, images) ;
- the episode team return is bounded, with a stated theoretical range, and successful completion scores above every failure ;
- the side-channel inventory and the engineered sender are specified to the same level of detail as for CoinRun-coop ;
- an implementation with unit tests exists at the manifest commit.

An environment for which no such definition is filed before the first N10-C run is **dropped from every analysis and condition before any run**. This is reported as a deviation. n_env denotes the number of environments retained.

## Message specification (all arms)

Every arm transmits, per agent and per step, a vector m ∈ {0, 1}¹⁶ (16 bits) at evaluation. The message an agent computes from its own observation at step t is delivered to its partner at the same step t and concatenated to the partner's hidden state before the policy and value heads. A finished agent keeps sending messages computed from its frozen observation.

- **GTM-comm:** K = 4 symbols from an alphabet of A = 16 (4 × 4 = 16 bits). The transmitted message is the 4 hard Gumbel-softmax indices. The receiver rebuilds the GTM code from the indices, with the straight-through gradient of `GTMBridge`.
- **Broadcast MLP:** 16 encoder logits, hard-thresholded at 0, with a straight-through estimator.
- **ATOC-style cross-attention:** the sender emits 16 bits exactly as the MLP arm does. The receiver's cross-attention attends over the received binary messages, never over peer hidden states.
- **DIAL:** 16 DRU units. In training, the DIAL discretise/regularise unit of Foerster et al. (2016), σ(m + 𝒩(0, σ_n²)) with σ_n = 2 fixed here. At evaluation, a hard threshold at 0. DIAL is the only arm whose training-time messages are not hard bits. This is disclosed.
- **Engineered reference sender:** see below.
- **No-communication:** constant all-zero message in training and evaluation (0 bits of information).
- **Enforcement:** the message bus asserts shape (16,) and values in {0, 1} for every delivered message (at evaluation for all arms ; in training for all arms except DIAL) and logs bits per step. A violation is an implementation error, handled as an infrastructure failure (fix, then rerun from scratch), never as an outcome-based exclusion. *(PI confirmation item 4 for K and A.)*

## New arms

### Engineered reference sender

A fixed, hand-designed 16-bit sender. The receiver uses the same per-agent network and PPO setup as the other arms. Its inputs are **local to the sender and obtainable without privileged procgen state**: the sender's own coin event (from its own environment's reward signal), the wrapper's window counter, and the sender's own action history. For CoinRun-coop:

- bit 0: 1 if the sender reached its coin inside the currently open Δt window ;
- bits 1–5: the remaining number of scoring steps (0–20, as in the team summary's numerator), unsigned ;
- bits 6–15: n = clip(#right − #left actions since the episode start, −512, 511), 10-bit two's complement. Right = procgen actions {6, 7, 8}, left = {0, 1, 2}.

**Disclosed:** n is a proxy for horizontal progress, not the agent's position (walls and falls break the link). In the registered condition, bits 0–5 duplicate information also present in the receiver's team summary. This arm is a **reference with known content**. It is neither an optimum nor an upper bound, and other arms may exceed it.

### No-communication (floor)

The same agents with the all-zero message, as above.

## Side-channel control

**Inventory for CoinRun-coop (frozen):**

1. the declared team summary (open or closed) ;
2. the shared team reward, which acts only through learning and is never a policy input. It is the task and cannot be removed ;
3. episode-termination coupling (the episode ends when the partner dies or the window expires). It is observed only as termination, with no action after it ;
4. the shared level seed. This is common knowledge, not partner state ;
5. partner rendering: none (separate instances).

**Conditions:**

- *Registered condition:* `team_summary=True`. The declared channel is open, as the 2026-05-11 design described. Flag: because the kickoff code never implemented the channel, opening it is a change to the implemented environment, recorded here before any run.
- *Side-channel-closed condition:* `team_summary=False`. Paths 2–4 remain and are inherent to the task.

**Arms run in the side-channel-closed condition:** GTM-comm, DIAL, engineered reference, no-communication.

## Evaluation protocol and randomness (frozen)

- **Independent arm streams.** The registered seeds {0, 17, 42, 73, 101} are seed labels. Run (arm a, seed s, condition c) draws all of its training randomness (initialisation, PPO sampling, training level seeds) from `numpy.random.SeedSequence([s, a, c])`, with fixed integer codes for arms and conditions listed in the manifest. Training randomness is therefore independent across arms, and trained arms are analysed as **independent groups**.
- **Fixed evaluation set.** Every model is evaluated on the same 200 held-out levels. Action sampling at evaluation on level j uses `numpy.random.default_rng([20261023, j])`, identical for every model and pass. All estimands are defined **conditional on this fixed evaluation set**. The only random unit in the analysis is the trained model (seed).
- **Metric 1, team reward** = the model's mean episode team return over the 200 levels = its team success rate.
- **Completion time** (reported separately, descriptive) = the mean number of steps to team success over successful episodes.

### Message-shuffle probe (evaluation only)

For every trained model (arm × seed × condition):

1. **Intact pass:** one episode on each of the 200 evaluation levels. Every message each agent emits is recorded.
2. **Shuffled pass:** the same 200 levels and the same evaluation action generators. A fixed derangement π of the 200 levels is drawn once with `numpy.random.default_rng(20261023)` and shared by all models. At step t of level j, the message delivered to agent i is the message emitted at step min(t, T_π(j) − 1) by the same sender role in the intact pass of level π(j), where T is that episode's length. Messages computed during the shuffled pass are not delivered.

For each model, S = R_intact − R_shuffled, with R its mean team return over the 200 levels. Given the fixed evaluation set, derangement and action generators, S is a deterministic function of the trained model. The donor construction is therefore not resampled.

## Estimation plan (replaces the registered statistical analysis and decision criteria)

### Interval methods

- **Difference of two arms:** Welch CI, difference in seed means ± t(1 − a/2, ν) · √(s₁²/5 + s₂²/5), with ν the Welch–Satterthwaite df and a the stated per-interval error rate.
- **One arm (S):** one-sample t CI over the 5 seed values, mean ± t(1 − a/2, 4) · s/√5.
- **Ratios (normalized improvement):** Fieller interval (below).
- All intervals assume approximately normal seed-level values. With 5 seeds this assumption cannot be checked well, and intervals are read as pilot estimates. No bootstrap is used.

### Primary estimand (registered condition, per environment e)

The research hypothesis is "GTM team reward exceeds the best baseline by ≥ 10 % of dynamic range". With success-only rewards, team reward is a success rate with theoretical range [0, 1], so the **dynamic range is 1** (for Caveflyer-coop and Maze-coop: the theoretical range stated in the manifest). GTM exceeds the best baseline by at least a given amount if and only if it exceeds every baseline by that amount. The primary estimands are therefore the **three simultaneous contrasts**

θ_e,b = (R̄_GTM,e − R̄_b,e) / range_e, for b ∈ {broadcast MLP, ATOC-style, DIAL},

each with a Welch CI at level 1 − 0.05/3 (98.33 %) within the environment. range_e is a known constant, not an estimate. No selection of a "best" baseline from the data is involved.

- **Changes flagged:** the registered text used the median across seeds and did not define "dynamic range" or "best baseline". The mean is used because the Welch interval is defined for means.
- **Reference value (SESOI):** Δ = 0.10, the registered "10 % of dynamic range". It was set by the PI on 2026-05-11 as the smallest coordination gain worth a Paper 2 section. It was not derived from a power analysis.

### Pilot category per environment (mutually exclusive, exhaustive, applied in this order)

Using the three 98.33 % intervals [L_b, U_b]:

1. **pilot-excludes-SESOI** if U_b < Δ for at least one baseline b. Wording: "a gain of Δ or more over every baseline is not supported by this pilot."
2. **pilot-positive** if L_b ≥ Δ for all three baselines. Wording: "the pilot estimates are consistent only with gains of at least Δ over every baseline ; superiority is not established, because this is not a confirmatory test."
3. **pilot-inconclusive** otherwise.

If L_b ≥ Δ for all b, then U_b ≥ Δ for all b, so 1 and 2 cannot both hold. None of the categories is a superiority, equivalence or "matches" verdict. The registered "on at least 2 of 3 tasks" aggregate is reported only as the number of environments in each category. "Pilot-positive on ≥ 2 environments" is still not a superiority claim.

### Secondary contrasts (descriptive)

- **Registered set:** GTM vs {broadcast MLP, ATOC-style, DIAL} × 3 metrics × n_env environments (27 when n_env = 3), registered condition. Each contrast is a difference in seed means with a Welch CI at level 1 − 0.05/k (k = 27 → 99.815 %).
- **Directions.** Team reward: positive (GTM − comparator) favours GTM. Sample efficiency: defined as comparator − GTM, so positive favours GTM. For bandwidth utilisation no favourable direction was registered: it is reported signed and never counted as favouring either arm.
- **Metric definitions completed here:**
  - *Sample efficiency* is a **capped time-to-threshold endpoint**, not a censored-survival analysis. Training team success rate is measured every 1M steps as the mean over the last 100 completed training episodes. The endpoint is the first evaluation step at which it reaches 0.5 (50 % of the theoretical maximum of 1 ; for other environments, 50 % of the stated range above its minimum). A run that never reaches it within 50M steps is assigned the cap, 50M. The report counts separately runs that reach the threshold at the final 50M evaluation and runs that never reach it.
  - *Bandwidth utilisation* = plug-in entropy, in bits (≤ 16), of the 16-bit message distribution over all held-out evaluation steps.
- For each set, the number of intervals entirely above 0, entirely below 0 and including 0 is reported **without a label**. An interval including 0 is never described as a tie or "matches".

### Normalized improvement (descriptive, both conditions)

NI_arm,e = (R̄_arm,e − R̄_NOCOMM,e) / (R̄_ENG,e − R̄_NOCOMM,e), for every arm other than ENG and NOCOMM. By definition NI = 1 for ENG and 0 for NOCOMM ; these are not estimated.

- **Interval:** 95 % Fieller interval. With numerator a, denominator b, v_X = s_X²/5 for each arm X, Var(a) = v_arm + v_NOCOMM, Var(b) = v_ENG + v_NOCOMM and Cov(a, b) = v_NOCOMM (the shared NOCOMM mean), the interval is the set of ρ with (a − ρb)² ≤ t² · (Var(a) − 2ρ Cov(a, b) + ρ² Var(b)). t is the 97.5 % quantile at the smaller Welch–Satterthwaite df of the numerator and denominator differences.
- **Denominator rule:** NI is **undefined** if b² ≤ t² · Var(b) (the Fieller set is then unbounded) or if b = 0.
- NI may exceed 1 or be negative, because the engineered arm is not an upper bound. It carries no wording rule.

### Message-disruption sensitivity (descriptive)

- For each arm × condition × environment, the mean of the 5 seed-level S values is reported with a one-sample t CI at level 1 − 0.05/n_env (98.33 % for n_env = 3).
- **Wording.** S is reported only as the arm's **sensitivity to message disruption** ("the arm's team reward drops when partner messages are replaced by messages from another level"). It is not an estimate of the reward that flows through the channel. No difference between arms, including GTM's difference from any baseline, is attributed to communication on the basis of S.
- **Positive control.** If the engineered reference's side-channel-closed interval for S does not lie entirely above 0, the probe is reported as **insensitive** on that environment, and no arm's S on it is interpreted.
- For the no-communication arm the probe is a no-op, and S = 0 is checked as a sanity test.

## Failure, exclusion and replacement

- **Infrastructure failure** (crash, out-of-memory, hardware fault, preemption, a run on the stub, a message-bus violation, or a non-finite value traced to a software bug fixed without changing the registered configuration): the run is restarted from scratch with the same seed stream and configuration, at most twice. All attempts are logged.
- **Training failure** (divergence or non-finite loss under the registered configuration, or still failing after two restarts): the run is not replaced. It is evaluated at its last checkpoint with finite parameters. If there is none, it is evaluated at initialisation (an untrained policy).
- No run is excluded or replaced on the basis of its outcome. Restarts and training failures are reported per arm.
- If a bug that affects an arm's registered configuration is found after runs have started, all runs of every affected arm are repeated, and this is reported as a deviation.

## Superseded provisions of the locked file

- **Decision criteria** (C-headline ≥ 18/27 wins and reward gain ≥ 10 % on ≥ 2 envs ; C-tied-stable 9–17 ; C-loses ≤ 6): not applied. They are replaced by the categories above. This also removes the unlabeled 7–8 win range and the unresolved "≥ 18 wins without the reward-gain condition" case.
- **Welch's t-test with Bonferroni α = 0.00185 as a verdict:** replaced by the adjusted intervals. p-values are not used to label outcomes.
- **"Seeds 0, 17, 42, 73, 101" as shared random streams:** they are labels of arm-specific streams, as above.
- **Risk-factor provision "pre-allocate compute budget for 10-seed re-run if 5-seed shows wide IQR":** withdrawn. The seed set is fixed at 5. Additional seeds belong only to a separate, future confirmatory registration, whose test does not reuse the pilot data.
- **Compute-budget statement "4 parallel runs … ~9 days":** withdrawn (see Compute).
- **"Bandwidth measurement … document the measurement method per arch":** replaced by the message specification above.

## Confirmatory follow-up

Any superiority claim requires a separate, future confirmatory pre-registration with a powered N. Its sample size is computed from the pilot's seed-level variance (the upper one-sided 80 % confidence limit of the pooled SD). The pilot data are not reused in its confirmatory test.

## Configuration manifest (required, immutable, part of this registration)

Before the first N10-C run on any registered environment, the PI files a dated manifest on OSF as an update to this registration. It may contain only:

1. the commit of `coop_env.py`, the PPO-MA harness and the message bus used for every run, and the integer codes for arms and conditions ;
2. the per-agent network, PPO and encoder/decoder sizes of every arm, as implemented at that commit ;
3. the Caveflyer-coop and Maze-coop definitions (reward rule and range, side-channel inventory, engineered sender, tests), or a statement that an environment is dropped ;
4. the measured training throughput (steps/s, one job and concurrent jobs) on the target hardware, the resulting compute estimate including evaluation, and the scope adopted under PI confirmation item 1.

Development that informs items 1–3 may use only the stub, smoke runs of at most 1M steps, and unit tests. It never uses held-out levels 1000–1199 or results of a full-length run. The manifest cannot be changed after filing. Runs started before it is filed are excluded from all analyses and reported.

## Effect on hypotheses

The research hypothesis (GTM team reward exceeds the best baseline by ≥ 10 % of dynamic range at 16 bits/step on ≥ 2 of 3 tasks) is kept as the **primary estimand and its reference value**. "Dynamic range" and "best baseline" are now defined, the seed summary is the mean instead of the median, and the reward has no step penalty. It is no longer tested for a superiority verdict at 5 seeds. **Flag for the PI:** this changes how every N10-C outcome may be interpreted and worded in Paper 2 §9, and must be filed as a change of interpretation.

## Compute

The registered sweep is 60 runs × about 14 h = **840 GPU-h** (35 days on one GPU). This amendment adds:

- registered condition: engineered reference + no-communication, 2 arms × 3 envs × 5 seeds = 30 runs ;
- side-channel-closed condition: 4 arms × 3 envs × 5 seeds = 60 runs ;
- evaluation: intact and shuffled passes over 200 levels for 150 trained models.

That is **90 additional runs = 1,260 GPU-h, 1.5× the registered training compute**. The **total is 150 runs = 2,100 GPU-h, 2.5× the registered compute, about 87.5 days sequential** on one RTX 4090, before evaluation overhead. The registered "4 parallel runs, ~9 days" assumes four-fold throughput on one GPU. That has not been measured: concurrent jobs share one GPU, so the real speed-up is unknown until the throughput measurement in the manifest is made. The 14 h per run is itself the registered estimate (50M steps at about 1 ms/step) and is unverified.

**Scope options (PI confirmation item 1).** Reductions are whole arms, conditions or environments only, decided before filing:

| Option | Runs | Training GPU-h |
|---|---:|---:|
| A. Full design as above | 150 | 2,100 |
| B. Drop DIAL from the side-channel-closed condition | 135 | 1,890 |
| C. Side-channel-closed condition on CoinRun-coop only | 110 | 1,540 |
| D. CoinRun-coop only, all arms and both conditions (n_env = 1) | 50 | 700 |

This draft registers **option A**. If it cannot be funded, the PI selects another option and edits this section before filing.

## PI confirmation items (to settle before filing)

1. Compute scope (option A as drafted, or B / C / D).
2. Primary estimand on seed means (was median), with dynamic range = the theoretical range of the team return.
3. The estimation reframing as worded above (PI decision 2026-10-23).
4. GTM message format K = 4 symbols × A = 16 (16 bits).
5. Success-only team reward (no step penalty).

If the PI changes any item, the text above is edited before filing, never after data are seen.

## Optional modern baselines (not added)

MA-JEPA ([arXiv:2609.33563](https://arxiv.org/abs/2609.33563)) and SCALE-COMM ([arXiv:2605.27532](https://arxiv.org/abs/2605.27532)) are recent multi-agent communication / world-model baselines. They are **not** added by this amendment.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N10-C metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for GTM-comm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no contrast.

## References

- Chauhan, M., Bera, A. (2026). Emergence, Not Bandwidth: Physical Coupling and the Limits of Learned Multi-Agent Communication. arXiv:2609.34373.
- Foerster, J., Assael, Y. M., de Freitas, N., Whiteson, S. (2016). Learning to Communicate with Deep Multi-Agent Reinforcement Learning. NeurIPS 2016.
- Kaplowitz, Obahor, Schroeder de Witt (2026). MA-JEPA: Joint-Embedding World Models for Multi-Agent Reinforcement Learning. arXiv:2609.33563.
- Abouelyazid, Hammad (2026). SCALE-COMM: Shared, Contrastively-Aligned Latent Embeddings for MARL Communication. arXiv:2605.27532.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
