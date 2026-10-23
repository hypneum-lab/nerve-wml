# N10-C — Pre-registration amendment 2026-10-23: estimation reframing, engineered reference sender and side-channel control

**Status:** DRAFT — not yet frozen or registered (committed by a frozen git tag, then registered on OSF referencing the tag ; see Filing)
**Amends:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md` (pre-registered 2026-05-11, commit `1640c5c`)
**Date drafted:** 2026-10-23 (revised the same day for the second round of PI decisions: items 7-10 of the summary; draft edits only, nothing filed)
**OSF registration:** no N10-C-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N10-C training run exists. The only code is the CoinRun-coop environment (`experiments/procgen_multiagent_gtm_comm/coop_env.py`) and a smoke test. Caveflyer-coop and Maze-coop are not implemented.

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed. Where it conflicts with the locked file, this amendment governs. The locked file labels its research hypothesis "H0" ; this amendment calls it the **research hypothesis**.

## Summary of the change

1. **Estimation / pilot reframing (PI decision 2026-10-23). This changes the interpretation of the study.** N10-C becomes an estimation (pilot) study. It reports effect estimates with confidence intervals and makes **no superiority verdict**. The C-headline / C-tied-stable / C-loses labels and their win-count thresholds are **not applied**. The registered arms are unchanged ; the seed set is extended by one seed (item 7).
2. **Team reward without step penalty.** Success is the only reward ; completion time is reported separately.
3. Add an **engineered reference sender** arm and a **no-communication** arm, and report each arm's **normalized improvement** over no-communication.
4. Add a **side-channel control**: a side-channel-closed training condition and an evaluation-time message-shuffle probe. Its result is reported only as message-disruption sensitivity.
5. Give an executable specification of the CoinRun-coop environment, the 16-bit message for every arm, the engineered sender's inputs, the evaluation protocol, randomness, and failure / exclusion / replacement rules. Correct the compute figures.
6. **Scope option D (PI decision 2026-10-23):** CoinRun-coop only, all arms, both conditions. Caveflyer-coop and Maze-coop are not run, and the registered "on at least 2 of 3 tasks" target is withdrawn. **Changed 2026-10-23 (second round): 60 runs (6 seeds), on two hosts, see items 7-8.** The earlier wording "50 runs, on macM1 only" is superseded.
7. **Six seeds instead of five (PI decision 2026-10-23, CHANGE).** The registered seeds {0, 17, 42, 73, 101} plus one new seed, **131**, fixed now by an outcome-blind rule (see Seeds and hosts). Total 6 seeds × (6 registered-condition arms + 4 closed-condition arms) = **60 runs**. The power table and every N / df statement below are recomputed for 6 seeds per arm.
8. **Two hosts, host fixed per seed (PI decision 2026-10-23, CHANGE).** macM1 (Apple M1 Max, MLX Metal) trains seeds {0, 17, 42} ; KXKM-AI (RTX 4090, MLX CUDA) trains seeds {73, 101, 131}. All evaluations run on KXKM with the x86-64 Procgen build. The two Procgen builds are **not bit-identical**, and macM1-trained models are evaluated on the x86 rendering (disclosed). The analysis stays Welch / independent groups ; a by-host summary is added, descriptive and untested.
9. **Reproducibility statement (PI decision 2026-10-23, CHANGE).** Runs are statistically, not bitwise, reproducible from the seed, on both backends (conv backward in MLX 0.31.1 is nondeterministic). See Reproducibility.
10. **Design additions from the 2026-10-23 literature check (descriptive or evaluation-only, no new test):** a cross-seed cross-play probe (see Cross-seed cross-play probe).

## Rationale

- **Power.** The 2026-10-23 design has **6 seeds per arm** (item 7 ; the first draft had 5). With Bonferroni correction the registered win-count tests have little power. Fixed-df (df = 2n − 2 = 10) two-sided t calculations for two independent normal groups of n = 6 with equal variances, computed by the same method as the first draft (noncentral t ; the 5-seed values, df = 8, are given for comparison and reproduce the first draft's table):

  | Family | Comparisons | Per-test α | Power at d = 1, n = 5 → **n = 6** | d for 80 % power, n = 5 → **n = 6** |
  |---|---:|---:|---:|---:|
  | Registered | 27 | 0.001852 | 2.6 % → **4.0 %** | 3.59 → **3.02** |
  | Primary estimand (this design) | 3 | 0.016667 | 14.0 % → **18.4 %** | 2.51 → **2.19** |
  | Secondary set (this design, n_env = 1) | 9 | 0.005556 | 6.3 % → **8.9 %** | 3.03 → **2.59** |
  | Single contrast, unadjusted | 1 | 0.05 | 28.6 % → **34.7 %** | 2.02 → **1.80** |

  d is the standardized mean difference. The sixth seed raises power only modestly. No seed-level variance estimate exists, so power at the registered threshold cannot be computed. The PI decided on 2026-10-23 to report estimates instead of verdicts, and (second round) to add one seed (6 instead of 5) because the extra compute is available on a second host. The seed-to-host assignment is a block factor, so the pooled-variance power above is an idealisation : host effects, if any, add to the seed-level variance and lower the power. The first draft's decision "keep 5 seeds" is superseded.

- **Reference and side channels.** Chauhan & Bera, "Emergence, Not Bandwidth: Physical Coupling and the Limits of Learned Multi-Agent Communication", [arXiv:2609.34373](https://arxiv.org/abs/2609.34373) (v1 2026-09-28 ; v2 2026-09-29, an ICLR 2027 submission under review, [OpenReview OAmBy25HHg](https://openreview.net/forum?id=OAmBy25HHg)), argue that the value of learned communication in rate-limited Dec-POMDPs is governed by physical coupling and side channels. Applied to N10-C:
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

### Caveflyer-coop and Maze-coop (not run under scope option D)

The registered definitions ("keep distance ≤ X", "visit all 4 quadrants") need agent positions, which procgen does not expose without privileged internal state. Under scope option D (PI decision 2026-10-23, see Compute), Caveflyer-coop and Maze-coop are **not run** under this registration. They are removed from every analysis and condition by this amendment, before any run. n_env denotes the number of environments retained: **n_env = 1** (CoinRun-coop).

If they are run later, that requires a separate amendment, filed before any of their runs, whose definitions satisfy at least these constraints:

- the reward rule uses only signals exposed by procgen's public gym3 interface (rewards, episode ends, images) ;
- the episode team return is bounded, with a stated theoretical range, and successful completion scores above every failure ;
- the side-channel inventory and the engineered sender are specified to the same level of detail as for CoinRun-coop ;
- an implementation with unit tests exists at the commit named in that amendment.

## Message specification (all arms)

Every arm transmits, per agent and per step, a vector m ∈ {0, 1}¹⁶ (16 bits) at evaluation. The message an agent computes from its own observation at step t is delivered to its partner at the same step t and concatenated to the partner's hidden state before the policy and value heads. A finished agent keeps sending messages computed from its frozen observation.

- **GTM-comm:** K = 4 symbols from an alphabet of A = 16 (4 × 4 = 16 bits). The transmitted message is the 4 hard Gumbel-softmax indices. The receiver rebuilds the GTM code from the indices, with the straight-through gradient of `GTMBridge`.
- **Broadcast MLP:** 16 encoder logits, hard-thresholded at 0, with a straight-through estimator.
- **ATOC-style cross-attention:** the sender emits 16 bits exactly as the MLP arm does. The receiver's cross-attention attends over the received binary messages, never over peer hidden states.
- **DIAL:** 16 DRU units. In training, the DIAL discretise/regularise unit of Foerster et al. (2016), σ(m + 𝒩(0, σ_n²)) with σ_n = 2 fixed here. At evaluation, a hard threshold at 0. DIAL is the only arm whose training-time messages are not hard bits. This is disclosed.
- **Engineered reference sender:** see below.
- **No-communication:** constant all-zero message in training and evaluation (0 bits of information).
- **Enforcement:** the message bus asserts shape (16,) and values in {0, 1} for every delivered message (at evaluation for all arms ; in training for all arms except DIAL) and logs bits per step. A violation is an implementation error, handled as an infrastructure failure (fix, then rerun from scratch), never as an outcome-based exclusion. *(K and A decided by the PI on 2026-10-23.)*

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

## Seeds and hosts (PI decisions 2026-10-23, second round: CHANGES)

**Six seeds.** The seed set is {0, 17, 42, 73, 101, 131}. The five registered seeds are unchanged. The sixth seed, **131**, is fixed now, before any N10-C run, by an outcome-blind rule : *the smallest prime greater than 101 whose last digit is 1, the last digit of the largest registered seed* (103, 107, 109, 113, 127 end in other digits ; 111 = 3·37 and 121 = 11² are not prime). The rule uses no result of any kind (no N10-C training run exists). The seed is a label, as for the other five : each run draws its randomness from `numpy.random.SeedSequence([s, a, c])`. Total : 6 seeds × (6 registered-condition arms + 4 side-channel-closed arms) = **60 runs**, 36 in the registered condition and 24 in the closed condition. **Flag :** this changes the registered seed count (5 → 6) ; the provision "the seed set is fixed at 5" of the first draft is superseded.

**Host is a block factor, fixed per seed.**

| Host | Seeds | Hardware and software | Procgen build |
|---|---|---|---|
| macM1 | 0, 17, 42 | Apple M1 Max, 32 GB, MLX 0.31.1 (Metal) | 0.10.7 arm64 build with the archived patch |
| KXKM-AI | 73, 101, 131 | RTX 4090 24 GB, MLX 0.31.1 + mlx-cuda-12 0.31.1 (CUDA 12), Python 3.10 | official Procgen 0.10.7 x86-64 wheel |

- All arms and both conditions of a seed run on that seed's host. Each host therefore trains the same set of arm × condition cells for its 3 seeds, so the host is balanced across arms and conditions (a block factor). It is confounded with the seed labels by construction.
- On KXKM the PPO update uses gradient accumulation (`micro_batch_pairs = 512`, `clear_cache` on), a harness option that computes the same registered minibatch gradient in micro-batches (accumulated gradient equal to the full-batch gradient to relative 1e-5 in the unit tests ; evidence on the branch `feat/n10c-grad-accum`, PR #53, `bench/kxkm/README.md`). The registered PPO hyperparameters are unchanged. macM1 runs use no accumulation. Both are recorded per run (`micro_batch_pairs`, `clear_cache`, `n_workers`) in the stage-2 manifest.
- **All evaluations run on KXKM with the x86-64 Procgen build** : the 200 held-out levels, intact and shuffled passes, and the cross-seed probe. macM1-trained checkpoints are copied to KXKM for evaluation. **Disclosed :** the two Procgen builds are **not bit-identical**. On CoinRun (easy, one level, fixed seed), levels 0-15 and 2000-2003 over 300 steps, the frame / reward / first-flag stream hashes differ on 20 of 20 levels between the macM1 arm64 build and the KXKM x86-64 build, although each machine is deterministic across reruns. The initial frames and the early episode are identical on some levels (levels 0 and 1 : same per-frame hashes at steps 0, 1 and 299, same pixel sums) and diverge late in an episode on others (level 5 : identical at steps 0 and 1, different at step 299, after an episode reset). Hence macM1-trained models are evaluated on a rendering (x86-64) that is not the one they were trained on, and seed-for-seed reproducibility of Procgen across the two hosts cannot be assumed. The evidence is `reports/bench-m3-kxkm-es.md` (frames equality paragraph) and its frames files `frames_macm1.json`, `frames_kxkm.json`, `frames.py`, `frames2.py`, which are to be archived with the stage-2 manifest. Any systematic effect of this train / evaluation rendering mismatch is absorbed in the by-host summary and is not corrected.
- **Analysis.** Unchanged : Welch / independent groups over the 6 seed values of each arm, pooling the three macM1 and three KXKM seeds. No host term enters any registered interval, and no test of a host effect is made. With 3 seeds per host a host effect and seed-to-seed variation cannot be separated ; any host effect inflates the seed-level variance and therefore widens the intervals, which is conservative for the pilot categories.
- **By-host summary (descriptive, untested).** For each primary and secondary contrast, the point estimate computed on each host's 3 seeds is reported next to the pooled estimate, together with the per-host seed means of every arm. No interval, p-value or label is attached to a by-host figure, and a by-host difference is never described as significant or as an effect of the hardware.

## Evaluation protocol and randomness (frozen)

- **Independent arm streams.** The seeds {0, 17, 42, 73, 101, 131} (the five registered seeds and the new sixth seed, see Seeds and hosts) are seed labels. Run (arm a, seed s, condition c) draws all of its training randomness (initialisation, PPO sampling, training level seeds) from `numpy.random.SeedSequence([s, a, c])`, with fixed integer codes for arms and conditions listed in the manifest. Training randomness is therefore independent across arms, and trained arms are analysed as **independent groups**.
- **Fixed evaluation set.** Every model is evaluated on the same 200 held-out levels. Action sampling at evaluation on level j uses `numpy.random.default_rng([20261023, j])`, identical for every model and pass. All estimands are defined **conditional on this fixed evaluation set**. The only random unit in the analysis is the trained model (seed).
- **Deterministic evaluation of discrete messages:** at evaluation GTM-comm transmits argmax symbols (no Gumbel noise), and every other arm its hard-thresholded bits.
- **Metric 1, team reward** = the model's mean episode team return over the 200 levels = its team success rate.
- **Completion time** (reported separately, descriptive) = the mean number of steps to team success over successful episodes. If a model has no successful evaluation episode, its completion time is reported as unavailable.

### Message-shuffle probe (evaluation only)

For every trained model (arm × seed × condition):

1. **Intact pass:** one episode on each of the 200 evaluation levels. Every message each agent emits is recorded.
2. **Shuffled pass:** the same 200 levels and the same evaluation action generators. A fixed derangement π of the 200 levels is drawn once with `numpy.random.default_rng(20261023)` and shared by all models. At step t of level j, the message delivered to agent i is the message emitted at step min(t, T_π(j) − 1) by the same sender role in the intact pass of level π(j), where T is that episode's length. Messages computed during the shuffled pass are not delivered.

For each model, S = R_intact − R_shuffled, with R its mean team return over the 200 levels. Given the fixed evaluation set, derangement and action generators, S is a deterministic function of the trained model. The donor construction is therefore not resampled.

### Cross-seed cross-play probe (evaluation only, descriptive; PI decision 2026-10-23, ADDITION)

Added after the 2026-10-23 literature check. Chauhan & Bera v2 (arXiv:2609.34373v2, ICLR 2027 submission under review) report that learned protocols reach high self-play team reward but collapse when agents trained in different runs are paired (self-play 0.980 against 0.144 across seeds, in their setting). Kim (arXiv:2606.10582) reports a related lack of mutual intelligibility between independently trained populations. N10-C adds the corresponding evaluation-only probe, with **no new training, no new test and no new contrast** :

- For each arm × condition and each seed s in the fixed seed order (0, 17, 42, 73, 101, 131), agent role 0 is taken from the model of seed s and agent role 1 from the model of the next seed in that order (131 pairs with 0), both from the same arm and condition. The pairing is fixed here, before any run, and is the same for every arm and condition. Where an arm's two roles share one set of parameters, each role simply uses the parameters of its own seed.
- One pass over the same 200 held-out levels, the same evaluation action generators as the intact pass, on KXKM, intact messages. The cross-seed team reward R_cross(s) is the mean team return over the 200 levels.
- Reported per arm and condition : the six R_cross values, their mean, and the descriptive gap between the mean of the six R_intact values and the mean of the six R_cross values. The engineered and no-communication arms are included, as references (their messages do not depend on the learned sender, so a gap there reflects policy mismatch only). No confidence interval, p-value, category or between-arm comparison is computed from the probe, and a gap is never described as a loss of communication.
- Cost : about half of one model evaluation per pass (one pass instead of the intact plus shuffled pair). The probe is part of the stage-2 evaluation code and is recorded in the manifest.

## Estimation plan (replaces the registered statistical analysis and decision criteria)

### Interval methods

- **Difference of two arms:** Welch CI, difference in seed means ± t(1 − a/2, ν) · √(s₁²/6 + s₂²/6), with ν the Welch–Satterthwaite df (at most 10 and at least 5 for two arms of 6 seeds ; **changed from 8 and 4 with 5 seeds**) and a the stated per-interval error rate.
- **One arm (S):** one-sample t CI over the 6 seed values, mean ± t(1 − a/2, **5**) · s/√6 (df = 5 ; was 4 with 5 seeds).
- **Ratios (normalized improvement):** Fieller interval (below).
- If an interval cannot be computed (both arms have zero seed variance ; for a one-arm interval, the 6 seed values have zero variance ; or a required df is undefined), it is reported as **unavailable** with the point estimate only. A zero-width interval is never reported. Identical seed outcomes, for example six runs capped at 50M, are not read as certainty about the population.
- **Interval levels are unchanged by the sixth seed** : 98.33 % (primary, 3 contrasts), 99.444 % (secondary set, 9 contrasts), 95 % (S, Fieller). Only the critical values change, through the df above (t at 0.975 : 2.228 at df = 10, 2.571 at df = 5).
- All intervals assume approximately normal seed-level values, so stated simultaneous coverage is approximate. With 6 seeds this assumption cannot be checked well, and intervals are read as pilot estimates. No bootstrap is used.

### Primary estimand (registered condition, CoinRun-coop)

The research hypothesis is "GTM team reward exceeds the best baseline by ≥ 10 % of dynamic range". With success-only rewards, team reward is a success rate with theoretical range [0, 1], so the **dynamic range is 1**. GTM exceeds the best baseline by at least a given amount if and only if it exceeds every baseline by that amount. The primary estimands are therefore the **three simultaneous contrasts**

θ_e,b = (R̄_GTM,e − R̄_b,e) / range_e, for b ∈ {broadcast MLP, ATOC-style, DIAL},

each with a Welch CI at level 1 − 0.05/3 (98.33 %). Here e = CoinRun-coop and range_e = 1 is a known constant, not an estimate. No selection of a "best" baseline from the data is involved.

- **Changes flagged:** the registered text used the median across seeds and did not define "dynamic range" or "best baseline". The mean is used because the Welch interval is defined for means.
- **Reference value (SESOI):** Δ = 0.10, the registered "10 % of dynamic range". It was set by the PI on 2026-05-11 as the smallest coordination gain worth a Paper 2 section. It was not derived from a power analysis.

### Pilot category (mutually exclusive, exhaustive, applied in this order)

Using the three 98.33 % intervals [L_b, U_b]:

0. **pilot-interval-unavailable** if any of the three intervals cannot be computed (both arms have zero seed variance, for example all-zero success rates, or a required df is undefined). The point estimates are reported, and nothing beyond them is stated.
1. **pilot-excludes-SESOI** if U_b < Δ for at least one baseline b. Wording: "a gain of Δ or more over every baseline is not supported by this pilot."
2. **pilot-positive** if L_b ≥ Δ for all three baselines. Wording: "the pilot estimates are consistent only with gains of at least Δ over every baseline ; superiority is not established, because this is not a confirmatory test."
3. **pilot-inconclusive** otherwise.

If L_b ≥ Δ for all b, then U_b ≥ Δ for all b, so 1 and 2 cannot both hold. None of the categories is a superiority, equivalence or "matches" verdict. Because only CoinRun-coop is run (option D), the registered "on at least 2 of 3 tasks" target is **withdrawn**: no cross-environment aggregate is formed, and only the CoinRun-coop category is reported. It is never described as evidence about the other two tasks.

### Secondary contrasts (descriptive)

- **Registered set:** GTM vs {broadcast MLP, ATOC-style, DIAL} × 3 metrics × n_env environments, registered condition. With n_env = 1 this is k = 9 contrasts (27 in the locked three-environment design). Each contrast is a difference in seed means with a Welch CI at level 1 − 0.05/9 (99.444 %).
- **Directions.** Team reward: positive (GTM − comparator) favours GTM. Sample efficiency: defined as comparator − GTM, so positive favours GTM. For bandwidth utilisation no favourable direction was registered: it is reported signed and never counted as favouring either arm.
- **Metric definitions completed here:**
  - *Sample efficiency* is a **capped time-to-threshold endpoint**, not a censored-survival analysis. Training team success rate is measured every 1M steps as the mean over the last 100 completed training episodes. The endpoint is the first evaluation step at which it reaches 0.5 (50 % of the theoretical maximum of 1). A run that never reaches it within 50M steps is assigned the cap, 50M. The report counts separately runs that reach the threshold at the final 50M evaluation and runs that never reach it.
  - *Bandwidth utilisation* = plug-in entropy, in bits (≤ 16), of the 16-bit message distribution over all held-out evaluation steps.
- For each set, the number of intervals entirely above 0, entirely below 0 and including 0 is reported **without a label**. An interval including 0 is never described as a tie or "matches".

### Normalized improvement (descriptive, both conditions)

NI_arm,e = (R̄_arm,e − R̄_NOCOMM,e) / (R̄_ENG,e − R̄_NOCOMM,e), for every arm other than ENG and NOCOMM. By definition NI = 1 for ENG and 0 for NOCOMM ; these are not estimated.

- **Interval:** 95 % Fieller interval. With numerator a, denominator b, v_X = s_X²/6 for each arm X, Var(a) = v_arm + v_NOCOMM, Var(b) = v_ENG + v_NOCOMM and Cov(a, b) = v_NOCOMM (the shared NOCOMM mean), the interval is the set of ρ with (a − ρb)² ≤ t² · (Var(a) − 2ρ Cov(a, b) + ρ² Var(b)). t is the 97.5 % quantile at the smaller Welch–Satterthwaite df of the numerator and denominator differences.
- **Interpretability rule:** NI is reported only if the denominator's interval b ± t · √Var(b) lies entirely above 0 (this also guarantees a bounded Fieller set). Otherwise the interval is reported as **unavailable**: this is a property of the data, not a statement that the population NI is undefined.
- NI may exceed 1 or be negative, because the engineered arm is not an upper bound. It carries no wording rule.

### Message-disruption sensitivity (descriptive)

- For each arm × condition, the mean of the 6 seed-level S values is reported with a one-sample t CI at level 1 − 0.05/n_env, that is 95 % with n_env = 1.
- **Wording.** S is reported only as the arm's **sensitivity to message disruption** ("the arm's team reward drops when partner messages are replaced by messages from another level"). It is not an estimate of the reward that flows through the channel. No difference between arms, including GTM's difference from any baseline, is attributed to communication on the basis of S.
- **Positive control.** If the engineered reference's side-channel-closed interval for S does not lie entirely above 0, the probe is reported as **insensitive**, and no arm's S is interpreted. If that interval is unavailable (for example, six identical S values), the positive control is reported as **uninterpretable**, and no arm's S is interpreted either.
- For the no-communication arm the probe is a no-op, and S = 0 is checked for every seed as a deterministic sanity test. No interval is computed for it.

### By-host summary (descriptive, untested)

See Seeds and hosts. For every primary and secondary contrast and for S, the per-host point estimates (3 seeds each) are reported beside the pooled estimate. They carry no interval and no label, and are not used in any pilot category.

## Reproducibility (PI decision 2026-10-23: CHANGE)

On both backends, Metal (macM1) and CUDA (KXKM), the backward pass of the convolutions in the IMPALA trunk of MLX 0.31.1 is **nondeterministic**. Measured with the same seed, same mode, repeated :

- initialisation, forward pass and the first rollout are bit-identical across repeats ;
- the gradients of every tensor outside the trunk convolutions (fc, heads, GTM) are bit-identical ; the 22 convolution weight and bias gradients in `trunk.blocks` differ, by about 1e-7 relative on Metal and 1.4e-5 to 4.5e-5 relative per gradient on CUDA (cuDNN), with and without gradient accumulation, for the MLP and GTM arms ;
- after one update the parameters differ (Adam turns a tiny gradient difference into a step of up to 2·lr per entry), and after 8 updates the rollouts have diverged (max parameter difference 1.1e-1 to 1.9e-1 on CUDA) ;
- MLX 0.31.1 has **no deterministic switch** for convolution backward on CUDA or on Metal (only MLX_DISABLE_COMPILE, MLX_ENABLE_TF32, MLX_USE_CUDA_GRAPHS, MLX_CUDA_CONV_CACHE_SIZE, none of which controls it).

Consequences, stated before any run :

- Runs are **statistically, not bitwise, reproducible from the seed**. The seed fixes the initialisation, the sampling streams and the training levels, not the trajectory.
- An infrastructure-failure restart with the same seed stream (see Failure, exclusion and replacement) **will not retrace the same trajectory** as the failed attempt. This is accepted : a restart is a new realisation of the same registered configuration, and all attempts are logged.
- A reader who reruns the archived configuration should expect seed-level results that differ by training noise, and should not expect bit-identical checkpoints.

Evidence : `experiments/procgen_multiagent_gtm_comm/bench/kxkm/repeat/` (`analysis.txt`, `grad_localisation.txt`, per-config `*.info.json` and `*.params.npz`) and `bench/kxkm/README.md`, scripts `bench/kxkm/repeat_control.py` and `grad_repeat.py`, on the branch `feat/n10c-grad-accum` (PR #53). The Metal measurement is in the tests `test_same_mode_repeat_control` and `test_nondeterminism_is_localised_to_conv_backward` of the same branch.

## Failure, exclusion and replacement

- **Infrastructure failure** (crash, out-of-memory, hardware fault, preemption, a run on the stub, a message-bus violation, or a non-finite value traced to a software bug fixed without changing the registered configuration): the run is restarted from scratch with the same seed stream and configuration, at most twice. All attempts are logged.
- **Training failure** (divergence or non-finite loss under the registered configuration, or still failing after two restarts): the run is not replaced. It is evaluated at its last checkpoint with finite parameters. If there is none, it is evaluated at initialisation (an untrained policy).
- No run is excluded or replaced on the basis of its outcome. Restarts and training failures are reported per arm.
- If a bug that affects an arm's registered configuration is found after runs have started, all runs of every affected arm are repeated, and this is reported as a deviation.

## Superseded provisions of the locked file

- **Decision criteria** (C-headline ≥ 18/27 wins and reward gain ≥ 10 % on ≥ 2 envs ; C-tied-stable 9–17 ; C-loses ≤ 6): not applied. They are replaced by the categories above. This also removes the unlabeled 7–8 win range and the unresolved "≥ 18 wins without the reward-gain condition" case.
- **Welch's t-test with Bonferroni α = 0.00185 as a verdict:** replaced by the adjusted intervals. p-values are not used to label outcomes.
- **"Seeds 0, 17, 42, 73, 101" as five shared random streams:** they are labels of arm-specific streams, as above, and the set is extended to six seeds by adding 131 (PI decision 2026-10-23).
- **"All runs on one machine" (implicit in the registered single-GPU design and in the first draft's macM1-only plan):** replaced by the two-host plan with the host fixed per seed (see Seeds and hosts).
- **Risk-factor provision "pre-allocate compute budget for 10-seed re-run if 5-seed shows wide IQR":** withdrawn. The seed set is fixed at 6 (the five registered seeds and seed 131). Additional seeds belong only to a separate, future confirmatory registration, whose test does not reuse the pilot data.
- **Compute-budget statement "4 parallel runs … ~9 days":** withdrawn (see Compute).
- **Caveflyer-coop and Maze-coop, and the "on ≥ 2 of 3 tasks" target:** not run and withdrawn under scope option D (see Compute).
- **"Bandwidth measurement … document the measurement method per arch":** replaced by the message specification above.

## Confirmatory follow-up

Any superiority claim requires a separate, future confirmatory pre-registration with a powered N. Its sample size is computed from the pilot's seed-level variance (the upper one-sided 80 % confidence limit of the pooled SD). The pilot data are not reused in its confirmatory test.

## Filing (two stages, commitment by frozen git tag; PI decisions 2026-10-23)

- **Stage 1** : this amendment text, that is the research hypothesis, the environment, message and side-channel specifications, the estimands, the analysis, failure and exclusion rules, the superseded provisions and the adopted scope.
- **Stage 2** : the configuration manifest below, which fixes the implementation details that stage 1 leaves open (architectures, adapters and objectives, including the GTM symbol adapter, and the MLX PPO harness).

**Commitment point (PI decision 2026-10-23, 22:17, CHANGE ; replaces "the manifest is filed on OSF before any run").** Runs may start **before** the OSF filing. The commitment point is a **frozen annotated git tag** (name of the form `n10c-freeze-<date>`) :

1. The tag points at the exact commit whose tree contains **both** the stage-1 text (this amendment, final) **and** the stage-2 manifest, together with the code. Its message contains the manifest path and the manifest's SHA-256 (`manifest: <path>` and `manifest-sha256: <hex>`).
2. The tag is pushed to GitHub (`hypneum-lab/nerve-wml`) and mirrored to Gitea.
3. The repository state at the tag is submitted to Software Heritage ("Save Code Now"), which gives an independent third-party archival timestamp and a SWHID.
4. The OSF registration is filed **afterwards** and must reference the tag name, the commit and the SWHID.
5. **Evidentiary timestamps** : the GitHub server-recorded push time of the tag and the Software Heritage archival time. Git author and committer dates are **not** evidence of time (they are set by the client).
6. **Runs started before the tag's push time (as recorded by GitHub) are excluded from all analyses and reported.** The campaign launcher refuses to start unless the tag exists on origin, is annotated, points at HEAD (or at an ancestor with only `docs/` and manifest-folder changes since), the tree is clean, and the SHA-256 in the tag message equals the committed manifest's. The manifest status line then reads `Status: FROZEN (tag <tag>)`.

The two-stage logic is kept : stage 1 and stage 2 are both in the tagged tree, and stage 2 cannot change any stage-1 provision. If the manifest cannot be written within the stage-1 rules, stage 1 is amended and a new tag is created before any run. No PI confirmation item remains open for N10-C.

## Configuration manifest (stage 2, required, immutable once tagged, part of this registration)

Before the first N10-C run, the PI commits a dated manifest and freezes it with the tag above ; the OSF registration then references that tag. It may contain only:

1. the commit of `coop_env.py`, the MLX PPO-MA harness, the message bus and the evaluation code (including the shuffle and cross-seed probes) used for every run, and the integer codes for arms and conditions ;
2. the per-agent network, PPO and encoder/decoder sizes of every arm, and the GTM symbol adapter, as implemented at that commit ;
3. the statement that Caveflyer-coop and Maze-coop are not run under this registration (scope option D) ;
4. the host record : the seed-to-host mapping ; for each host the measured training throughput (pair-steps/s with the GTM and message-bus overhead included), `n_workers` and the resulting compute estimate including evaluation ; the Procgen build of each host (macM1 : the arm64 build patch SHA-256 and the toolchain versions ; KXKM-AI : the official x86-64 wheel and its hash) ; the MLX and mlx-cuda versions ; `micro_batch_pairs` and `clear_cache` for KXKM ; and the nondeterminism statement of the Reproducibility section.

Development that informs items 1–2 and 4 may use only the stub, smoke runs of at most 1M steps (at most 3 per arm, all disclosed in the manifest with their outcomes), throughput measurements on random or untrained policies, and unit tests. It never uses held-out levels 1000–1199 or results of a full-length run. The manifest cannot be changed after the tag is pushed. Runs started before the tag's GitHub push time are excluded from all analyses and reported.

## Effect on hypotheses

The research hypothesis (GTM team reward exceeds the best baseline by ≥ 10 % of dynamic range at 16 bits/step on ≥ 2 of 3 tasks) is kept as the **primary estimand and its reference value on CoinRun-coop only**. The "on ≥ 2 of 3 tasks" part is **withdrawn** (scope option D): N10-C makes no statement about Caveflyer-coop or Maze-coop. "Dynamic range" and "best baseline" are now defined, the seed summary is the mean instead of the median, and the reward has no step penalty. It is no longer tested for a superiority verdict ; the study has 6 seeds per arm (item 7 of the summary). **Flag for the PI:** this changes how every N10-C outcome may be interpreted and worded in Paper 2 §9, and must be filed as a change of interpretation.

## Compute

**Registered design.** 60 runs × about 14 h = **840 GPU-h** (35 days on one RTX 4090). The 14 h per run (50M steps at about 1 ms/step) is the registered estimate and was never measured. The registered "4 parallel runs, ~9 days" assumed four-fold throughput on one GPU and is withdrawn.

**Full amended design (not adopted).** The amendment adds the engineered reference and no-communication arms in the registered condition (2 arms × 3 envs × 6 seeds = 36 runs) and the side-channel-closed condition (4 arms × 3 envs × 6 seeds = 72 runs), on top of the 72 registered-design runs (4 arms × 3 envs × 6 seeds) : 180 runs in total, plus intact and shuffled evaluation passes for every trained model. (**Changed 2026-10-23:** the counts of the first draft, with 5 seeds, were 60 + 30 + 60 = 150.)

**Scope options.** Reductions are whole arms, conditions or environments only. Run counts are for 6 seeds (the first draft's counts for 5 seeds were 150, 135, 110 and 50) ; the hours use the first-round planning figures (14 h/run on the RTX 4090 as registered, about 16 h/run on macM1) and are given only to compare options :

| Option | Runs | At the registered 14 h/run (RTX 4090) | At about 16 h/run on macM1 (continuous) |
|---|---:|---:|---:|
| A. Full design as above | 180 | 2,520 h | about 2,880 h ≈ 120 days |
| B. Drop DIAL from the side-channel-closed condition | 162 | 2,268 h | about 2,592 h ≈ 108 days |
| C. Side-channel-closed condition on CoinRun-coop only | 132 | 1,848 h | about 2,112 h ≈ 88 days |
| D. CoinRun-coop only, all arms and both conditions (n_env = 1) | 60 | 840 h | about 960 h ≈ 40 days on macM1 alone ; **adopted on two hosts, see below** |

**Adopted: option D (PI decision 2026-10-23).** CoinRun-coop only : 6 arms × 6 seeds in the registered condition (36 runs) and 4 arms × 6 seeds in the side-channel-closed condition (24 runs), **60 runs**, plus intact and shuffled evaluation passes (and the cross-seed probe) for the 60 trained models. The registered "on at least 2 of 3 tasks" target is withdrawn (see Effect on hypotheses). **Changed 2026-10-23, second round :** the first draft adopted 50 runs on macM1 alone ; that is superseded by the two-host plan below, and every statement that all runs use macM1 is withdrawn.

**Hardware, framework and hosts (PI decisions 2026-10-23).** The runs use MLX, the lab's framework, on two hosts with the seed-to-host assignment given under Seeds and hosts (macM1 : seeds 0, 17, 42 ; KXKM-AI : seeds 73, 101, 131). The MLX PPO harness is part of the stage-2 manifest. The registered PPO hyperparameters are unchanged. KXKM uses gradient accumulation (`micro_batch_pairs = 512`, `clear_cache` on) for memory headroom on the shared 24 GB card (the unaccumulated registered minibatch peaks at about 23 GB of 24.5 GB) ; this does not change the update (see Seeds and hosts).

**Compute per host (end-to-end, GTM and message-bus overhead included).**

| Host | Runs | Throughput | Per run (50M pair-steps) | Total |
|---|---:|---:|---:|---:|
| KXKM-AI (RTX 4090) | 30 | 1,290 pair-steps/s (micro 512 + clear_cache) | about 10.8 h | about 324 h ≈ 13.5 days |
| macM1 (M1 Max) | 30 | about 790 pair-steps/s | about 17.6 h | about 528 h ≈ 22 days (**the bottleneck**) |

- The two hosts run in parallel, so the campaign wall time is set by macM1 : about 22 days of continuous compute, plus the evaluation of its last models. KXKM has about 8.5 days of slack in that window, which the evaluations use.
- **Evaluation** (200 levels, intact and shuffled passes) takes about 6.7 min per model on KXKM, for all 60 models about 6.7 h, plus about half of that again for the cross-seed probe. macM1-trained checkpoints are evaluated on KXKM after being copied there (see Seeds and hosts).
- **Measurements.** macM1 : the neural-network work alone is about 1,800 samples/s (8.2 s per 16,384 samples, 3 epochs, minibatch 2048 ; 15.4 h per run for 2 agents × 50M steps = 10⁸ samples), archived in `experiments/procgen_multiagent_gtm_comm/bench/README.md` with `bench_ppo_mlx.py`, the lock files and two repetitions (1,807 and 1,789 samples/s) ; the end-to-end training throughput measured with the harness on 2026-10-23 is about 790 pair-steps/s (776-810 across the Task 9 runs), hence about 17.6 h per run. KXKM : bench evidence in `bench/kxkm/README.md` (branch `feat/n10c-grad-accum`, PR #53), 1,286-1,295 pair-steps/s at micro 512 with `clear_cache`. The first-round planning figure of about 16 h per run on macM1 is superseded by 17.6 h.
- The environment, throughput and manifest records are part of the stage-2 manifest.

**Risk: GPU contention.** macM1 is shared with the second-brain services, which also use its GPU, and KXKM is a shared GPU host. For the campaign the GPU services `gpu-worker` and `whisper-dex` on KXKM are **stopped**, and restored afterwards with `/ai/saisail/n10c-run/restore-services.sh`. Contention can still make runs slower than the estimate. A slower run changes no registered quantity (step budgets, seeds, evaluation protocol). A run killed or corrupted by contention is an infrastructure failure under the rules above ; it is restarted from scratch and, by the Reproducibility statement, does not retrace the failed attempt.

**Reproducibility note: two Procgen builds.** *macM1* : Procgen 0.10.7 was built from source for arm64, with a local build patch : the Xcode 26.5 SDK forced through `SDKROOT` / `DEVELOPER_DIR` ; Qt 5.15.19 found through `PROCGEN_CMAKE_PREFIX_PATH` ; and `-Werror` removed from `procgen/CMakeLists.txt`, because clang 21 flags two unused variables in `chaser.cpp` and `starpilot.cpp`. The patch changes the build configuration only, not game code. It is archived as `experiments/procgen_multiagent_gtm_comm/bench/procgen-0.10.7-arm64-build.patch` (SHA-256 `df9bbfa1d454ee51cbea1b8af47fbb1966a79f4a075357ba2a4c95acca834e7f` ; build details in `bench/README.md`). *KXKM-AI* : the official Procgen 0.10.7 x86-64 wheel, whose hash the manifest records. **The two builds are not bit-identical** (frames equality check, see Seeds and hosts), so the equivalence of the arm64 build with the official builds, which was unchecked in the first draft, is now known to fail at the bit level. It is not known whether the difference has any effect on results. All evaluations use the x86-64 build.

## PI decisions (2026-10-23)

1. The estimation reframing as worded above.
2. Implement the declared team summary on purpose and close it in the side-channel-closed condition (see Rationale).
3. Compute scope: option D (CoinRun-coop only, all arms, both conditions) with MLX, as described under Compute ; **changed in the second round : 60 runs on two hosts (item 9 below), not 50 runs on macM1 alone.** The "≥ 2 of 3 tasks" target is withdrawn.
4. Two-stage filing: this text now (stage 1), the configuration manifest as a second locked stage before any run (stage 2),  **Changed 22:17 : commitment by frozen annotated tag (GitHub push time + Software Heritage), OSF registration afterwards ; runs may start before the OSF filing.** **Changed 22:17 : commitment by frozen annotated tag (GitHub push time + Software Heritage), OSF registration afterwards ; runs may start before the OSF filing (see Filing).**
5. Primary estimand on seed means (was median), with dynamic range = the theoretical range of the team return.
6. GTM message format K = 4 symbols × A = 16 (16 bits/step).
7. Success-only team reward (no step penalty).

**Second round (2026-10-23), each flagged as a change :**

8. **Six seeds** : the registered {0, 17, 42, 73, 101} plus 131, fixed by the outcome-blind rule under Seeds and hosts. 60 runs. Power table and all N / df statements recomputed.
9. **Two hosts, host fixed per seed** : macM1 seeds {0, 17, 42}, KXKM-AI seeds {73, 101, 131}. All evaluations on KXKM with the x86-64 Procgen build, non-identity of the two builds disclosed. Welch / independent groups kept ; a descriptive, untested by-host summary added.
10. **Reproducibility statement** : statistically, not bitwise, reproducible ; an infrastructure restart does not retrace the failed trajectory.
11. **Compute** : KXKM 30 runs × about 10.8 h ≈ 324 h ; macM1 30 runs × about 17.6 h ≈ 528 h ≈ 22 days (bottleneck) ; evaluation about 6.7 min per model on KXKM ; `gpu-worker` and `whisper-dex` stopped on KXKM for the campaign. Option D kept.
12. **Cross-seed cross-play probe** (evaluation only, descriptive), from the literature check.
13. **Commitment by frozen git tag (22:17, CHANGE)** : runs may start before the OSF filing ; commitment point = annotated tag `n10c-freeze-<date>` on the tree holding stage 1 and the stage-2 manifest, pushed to GitHub (mirrored to Gitea), archived at Software Heritage, OSF filed afterwards referencing tag, commit and SWHID ; GitHub push time and SWH time are the evidence, git dates are not ; earlier runs are excluded.

No PI confirmation item remains open for N10-C. If the PI changes any item, the text above is edited before filing, never after data are seen.

## Optional modern baselines (not added)

MA-JEPA ([arXiv:2609.33563](https://arxiv.org/abs/2609.33563)) and SCALE-COMM ([arXiv:2605.27532](https://arxiv.org/abs/2605.27532)) are recent multi-agent communication / world-model baselines. They are **not** added by this amendment.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N10-C metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for GTM-comm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no contrast.

**Added 2026-10-23 (literature check, all four N10/N11 amendments).** Scalar PAC indices are many-to-one : different coupling geometries and phase polarities can give the same value. Besides the Tort modulation index and the preferred phase, any exploratory PAC diagnostic therefore also reports the **circular moments** of the coupling or the **phase-binned amplitude histogram** (Keshavarzi & Goswami, bioRxiv follow-up posted 2026-09-21, [doi:10.64898/2026.09.15.751789](https://doi.org/10.64898/2026.09.15.751789)). Exploratory only, no contrast, no test.

## References

- Foerster, J., Assael, Y. M., de Freitas, N., Whiteson, S. (2016). Learning to Communicate with Deep Multi-Agent Reinforcement Learning. NeurIPS 2016.
- Kaplowitz, Obahor, Schroeder de Witt (2026). MA-JEPA: Joint-Embedding World Models for Multi-Agent Reinforcement Learning. arXiv:2609.33563.
- Abouelyazid, Hammad (2026). SCALE-COMM: Shared, Contrastively-Aligned Latent Embeddings for MARL Communication. arXiv:2605.27532.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
- Chauhan, M., Bera, A. (2026). Emergence, Not Bandwidth: Physical Coupling and the Limits of Learned Multi-Agent Communication. arXiv:2609.34373v2 (2026-09-29), ICLR 2027 submission under review (OpenReview OAmBy25HHg). Cited for the cross-seed cross-play probe.
- Kim (2026). Drawing with Strangers: Population Scaling Drives Zero-Shot Mutual Intelligibility in Emergent Sketching. arXiv:2606.10582. Cited as supporting the cross-seed probe.
- Keshavarzi, M., Goswami, S. (2026). Computing the geometry of phase–amplitude coupling: A new moment-based framework. bioRxiv. doi:10.64898/2026.09.15.751789 (posted 2026-09-21).
