# N10-C — Pre-registration amendment 2026-10-23: optimal-sender reference and side-channel control

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Amends:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md` (pre-registered 2026-05-11, commit `1640c5c`)
**Date drafted:** 2026-10-23
**OSF registration:** no N10-C-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N10-C training run exists. The only code is the CoinRun-coop environment kickoff (`experiments/procgen_multiagent_gtm_comm/coop_env.py`, with a stub fallback) and a smoke test. Caveflyer-coop and Maze-coop are not implemented.

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed.

## Summary of the change

1. Add an **engineered-optimal sender** reference arm and a **no-communication** arm. Report each learned channel's **gap to the engineered reference**, not only GTM against the learned baselines.
2. Add a **side-channel control**: a side-channel-closed training condition and an evaluation-time message ablation. Together they separate reward that flows through the 16-bit channel from reward obtained by other coupling paths.

The registered H0, the 27-comparison family, its Bonferroni α and the C-headline / C-tied-stable / C-loses thresholds are unchanged.

## Rationale

Chauhan & Bera, "Emergence, Not Bandwidth: Physical Coupling and the Limits of Learned Multi-Agent Communication", [arXiv:2609.34373](https://arxiv.org/abs/2609.34373) (2026-09-28), argue that the value of learned communication in rate-limited Dec-POMDPs is governed by physical coupling and side channels. They also give information-theoretic optima at a fixed bit budget. Applied to N10-C, this raises two problems:

- A GTM win over broadcast-MLP / ATOC / DIAL at 16 bits/step does not show how close any learned channel comes to what 16 bits/step can deliver. All four arms could be far from the optimum.
- If agents can coordinate through paths other than the message (shared reward, partner information in observations), team-reward differences between channel architectures may not reflect communication at all. The kickoff environment's docstring describes a low-dim "team summary" observation channel (the partner's distance-to-coin estimate). That would be such a path, although the current `_observe()` returns raw images only.

## Methodology delta

### New arms

- **Engineered-optimal sender (reference):** a hand-designed sender that transmits exactly 16 bits/step per agent. Its message is a fixed-length quantised encoding of task-relevant state that the receiver does not observe (for CoinRun-coop, the sender's progress toward the coin and whether it has reached it). The encoding for each environment is specified from the environment definition and recorded in a dated note **before any training run**. The receiver uses the same per-agent network and PPO setup as the other arms. This arm is an *empirical upper reference*, not a proof of optimality. Where the analysis of Chauhan & Bera can be instantiated for an environment, the analytic optimum is reported next to it.
- **No-communication arm (floor):** the same agents with the message replaced by a constant zero vector during training and evaluation.

### Side-channel control

- **Side-channel inventory (pre-run):** for each environment, list every path by which an agent can get information about its partners other than the message. Examples are partner rendering in the observation, the "team summary" channel, and shared-observation features. Record the list in the dated note before training.
- **Conditions:**
  - *Registered condition*: the environment as implemented for the primary sweep. The primary 27-comparison analysis runs here, as registered.
  - *Side-channel-closed condition*: every inventoried path is removed, so the only remaining coupling is the shared team reward. If the inventory for an environment is empty, the two conditions coincide for that environment and this is reported.
- **Message ablation at evaluation:** for every trained arm in both conditions, re-evaluate on the 200 held-out levels with messages replaced by messages drawn from a different episode (shuffled). This needs no retraining.
- **Arms run in the side-channel-closed condition:** GTM-comm, DIAL, engineered-optimal sender, no-communication. Each runs on 3 environments × 5 seeds.

## Analysis-plan delta

- **Primary family (unchanged):** GTM vs {broadcast MLP, ATOC-style cross-attention, DIAL} × 3 metrics × 3 envs = **27** comparisons, Bonferroni α = 0.05/27 ≈ **0.00185**, registered condition.
- **New descriptive quantities** (median and bootstrap 95 % CI over seeds, per env and per arm; not part of any hypothesis test):
  - Normalised gap to reference: `G = (R_arm − R_nocomm) / (R_engineered − R_nocomm)`, reported in both conditions. It is reported as undefined if `R_engineered − R_nocomm` has a CI that includes 0.
  - Channel-attributable reward: `ΔR_msg = R(intact messages) − R(shuffled messages)`, reported in both conditions.
- **Pre-stated reading:**
  - On any environment where GTM-comm's side-channel-closed `ΔR_msg` has a CI that includes 0, a GTM advantage on that environment is **not** described as a communication advantage.
  - A C-headline outcome is reported together with `G` for every arm. A C-headline wording ("multi-agent comm SOTA") is not used if the engineered reference is not available for that environment.

## Effect on hypotheses

H0 (GTM team reward exceeds the best baseline by ≥ 10 % of dynamic range at 16 bits/step on ≥ 2 of 3 tasks) is **unchanged**. The numeric decision thresholds are unchanged.

**Flag for the PI:** the "pre-stated reading" adds conditions on how the C-headline and C-tied-stable outcomes may be *worded*. It does not change the labels or thresholds, but it is an addition to the interpretation rules and should be filed as such.

## Compute delta

The registered sweep is 60 runs × about 14 h ≈ 840 h. This amendment adds:

- Registered condition: engineered sender + no-comm, 2 arms × 3 envs × 5 seeds = 30 runs.
- Side-channel-closed condition: 4 arms × 3 envs × 5 seeds = 60 runs.
- Message-ablation evaluations: inference only, negligible.

That is about **90 additional runs ≈ 1,260 h**, or about 2.5× the registered training compute. The cost is not small. If it cannot be funded, the side-channel-closed arm set should be narrowed **before filing**, not after data are seen.

## Optional modern baselines (not added)

MA-JEPA ([arXiv:2609.33563](https://arxiv.org/abs/2609.33563)) and SCALE-COMM ([arXiv:2605.27532](https://arxiv.org/abs/2605.27532)) are recent multi-agent communication / world-model baselines. They are **not** added by this amendment. Adding either would enlarge the primary comparison family and change its α, which requires a separate amendment.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N10-C metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for GTM-comm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no hypothesis test.

## References

- Chauhan, M., Bera, A. (2026). Emergence, Not Bandwidth: Physical Coupling and the Limits of Learned Multi-Agent Communication. arXiv:2609.34373.
- Kaplowitz, Obahor, Schroeder de Witt (2026). MA-JEPA: Joint-Embedding World Models for Multi-Agent Reinforcement Learning. arXiv:2609.33563.
- Abouelyazid, Hammad (2026). SCALE-COMM: Shared, Contrastively-Aligned Latent Embeddings for MARL Communication. arXiv:2605.27532.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
