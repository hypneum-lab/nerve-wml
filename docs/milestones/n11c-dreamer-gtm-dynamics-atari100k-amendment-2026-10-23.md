# N11-C — Pre-registration amendment 2026-10-23: estimation reframing, ITC comparator and DreamerV3 rationale

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI in two stages, see Filing). The compute option is **pending** (see Compute).
**Amends:** `docs/milestones/n11c-dreamer-gtm-dynamics-atari100k-2026-05-11.md` (pre-registered 2026-05-11, commit `3340440`). The dev kickoff `docs/milestones/n11c-dev-kickoff-dreamerv3-port-2026-05-11.md` is not affected.
**Date drafted:** 2026-10-23
**OSF registration:** no N11-C-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N11-C code or run exists in this repository (no `experiments/dreamer_gtm_dynamics_atari100k/`). The CartPole dev-kickoff smoke has not been committed.

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed. Where it conflicts with the locked file, this amendment governs. The locked file labels its research hypothesis "H0" ; this amendment calls it the **research hypothesis**.

## Summary of the change

1. **Estimation / pilot reframing (PI decision 2026-10-23). This changes the interpretation of the study.** N11-C becomes an estimation (pilot) study. It reports effect estimates with confidence intervals and makes **no superiority verdict**. The C'-headline / C'-tied-stable / C'-loses labels and their win-count thresholds are **not applied**. Seeds, games, arms and the registered compute are unchanged.
2. **Replace the IRIS comparator with the published ITC agent** (Simulus with ITC decoding, run as a full-agent comparator). IRIS is kept as the fallback if the ITC code cannot be run.
3. **State why DreamerV3**, and not Dreamer 4, is the target architecture.
4. Freeze a common Atari-100k protocol for all arms, independent per-arm random streams, and a required immutable configuration manifest.
5. Withdraw the locked file's reduced-seed and "top-2 promising games" provisions. Correct the compute statement. The compute option is pending a throughput measurement (PI decision 2026-10-23).

## Rationale

- **Power.** With 5 seeds per arm and game and Bonferroni correction the registered per-game tests have little power. Fixed-df (df = 8) two-sided t calculations for two independent normal groups of 5 with equal variances:

  | Family | Comparisons | Per-test α | Power at d = 1 | d for 80 % power |
  |---|---:|---:|---:|---:|
  | Registered per-game | 12 | 0.004167 | 5.0 % | 3.17 |
  | Single contrast, unadjusted | 1 | 0.05 | 28.6 % | 2.02 |

  d is the standardized mean difference. No seed-level variance estimate exists, so power at the registered 10 % threshold cannot be computed. The PI decided on 2026-10-23 to keep 5 seeds and to report estimates instead of verdicts.
- **Comparator.** The locked pre-registration uses IRIS (Micheli et al. 2023) as the token-based "alternative SOTA". ITC (Kim, Sun, Kim, Park, Song, "Identifiable Token Correspondence for World Models", [arXiv:2605.16457](https://arxiv.org/abs/2605.16457), 2026-05-15) treats token-based transformer world-model decoding as a structured assignment problem. It reports gains on Atari-100k over the prior token-based state of the art. Its Atari agent is Simulus with ITC decoding ([§4.4](https://arxiv.org/html/2605.16457v3#S4.SS4)).
- **Target architecture.** Hafner, Yan, Lillicrap, "Training Agents Inside of Scalable World Models" (Dreamer 4), [arXiv:2509.24527](https://arxiv.org/abs/2509.24527) (2025-09-29), was public before the 2026-05-11 lock but is not discussed in it. It uses a transformer world model trained with shortcut forcing. The choice of DreamerV3 should be justified on the record.

## Methodology delta

### Comparator arm (frozen)

- **Removed from the comparison set:** IRIS-transformer-dynamics as a module inside DreamerV3. Published IRIS scores may be quoted as context and are not part of any contrast.
- **Added: the published ITC Atari agent** (Simulus with ITC decoding), run from the official code at the commit recorded in the manifest, with its published agent hyperparameters, under the **common protocol** below, on the same 6 games with the same 5 seed labels. It is labelled "**full-agent comparator**" in all reporting. GTM-vs-ITC contrasts compare whole agents (DreamerV3 with a GTM transition predictor versus Simulus+ITC). They are not comparisons of matched dynamics modules.
- **Not used:** a module-level port of ITC decoding into the DreamerV3 RSSM. It would be a new, unvalidated adaptation, not the published comparator.
- **Comparability:** if the official ITC code cannot be configured to the common protocol, the ITC arm is still run, under its own published protocol, and every GTM-vs-ITC result is described only as a comparison of **different agent-and-environment packages**, never as agent performance under equal conditions. Whether the common protocol can be applied is recorded in the manifest before any Atari-100k run.
- **Reproduction check (reported, never an exclusion criterion):** per-game ITC means from this study are reported next to the published ITC numbers.
- **Fallback, decided before any Atari-100k run on availability only:** the official ITC code is smoke-run on Pong for 10k environment steps. If it is not publicly available, or does not complete this smoke without error, the comparator is the official **IRIS** agent (the registered alternative, run as a full agent under the same rules). The outcome is recorded in the manifest. Smoke runs are excluded from all analyses.

### Why DreamerV3 and not Dreamer 4 (rationale, no design change)

- **Interface:** the research hypothesis concerns replacing the RSSM transition predictor `(h_t) → ẑ_{t+1}`. Dreamer 4 does not use the RSSM transition-predictor MLP that it targets, so the registered intervention has no direct counterpart there.
- **Benchmark fit:** Atari-100k is a standard sample-efficiency benchmark for DreamerV3, with published reference scores. The control arm is therefore anchored to an established baseline.
- **Compute and tooling:** the registered budget is a single RTX 4090 with an open PyTorch port (`dreamerv3-torch`, per the dev kickoff). Dreamer 4 is designed for a much larger compute scale. The target hardware for N11-C is under review (see Compute) ; this does not change the choice of DreamerV3.
- **Consequence for claims:** any N11-C result is a statement about RSSM-based world models (DreamerV3 class). It must not be extrapolated to transformer world models such as Dreamer 4 or ITC beyond what the full-agent comparison itself shows.

### Common Atari-100k protocol (all arms, frozen)

- **Budget:** 100k agent steps = 400k emulator frames (frame skip 4), counted from the first environment interaction, including random warm-up steps.
- **Emulator:** ALE v5 games, minimal action set, `repeat_action_probability = 0` (no sticky actions), maximum 108,000 frames per episode, up to 30 no-op starts. Observation preprocessing (resolution, colour) is part of each agent and follows its published configuration.
- **Training:** life loss is not a terminal signal for the agent's returns.
- **Evaluation:** after the full budget, 100 evaluation episodes with the final checkpoint. Episode k uses environment seed `numpy.random.SeedSequence([20261023, g, k])` (g the game code), identical for every arm and seed. Actions are chosen by each agent's published evaluation mode (DreamerV3: the mode of the policy distribution). The score is the mean undiscounted episode return. Human-normalised score = (agent − random) / (human − random) × 100, with the per-game random and human reference scores used by the DreamerV3 Atari-100k evaluation, listed in the manifest.

### Unchanged

Control arm (standard DreamerV3 MLP transition), 6 games (Pong, Breakout, Asterix, MsPacman, Seaquest, BeamRider), seeds {0, 17, 42, 73, 101}, human-normalised score, official DreamerV3 hyperparameters.

## Estimation plan (replaces the registered statistical analysis and decision criteria)

### Primary estimand (aggregate across games)

Per arm, A_arm = (1/6) Σ_g HNS̄_g,arm, where HNS̄_g,arm is the mean over 5 seeds of the human-normalised score on game g. The estimand is the relative gain θ = A_GTM / A_MLP − 1. This is the aggregate named in the registered research hypothesis.

- **Interval:** 95 % Fieller interval for the ratio ρ = A_GTM / A_MLP, then θ = ρ − 1. With a = A_GTM and b = A_MLP, v_a = (1/36) Σ_g s²_g,GTM / 5 and v_b likewise (s²_g,arm is the seed variance on game g), arms independent, the interval is the set of ρ with (a − ρb)² ≤ t² · (v_a + ρ² v_b). t is the 97.5 % quantile at df = min(ν_a, ν_b), where ν_a = v_a² / Σ_g [(s²_g,GTM / 180)² / 4] is the Satterthwaite df of v_a, and ν_b likewise. If exactly one of v_a and v_b is 0, t uses the df of the other. If both are 0, neither the relative-gain interval nor the absolute-difference interval below is computed: both are reported as unavailable, with their point estimates only. The interval assumes approximately normal seed-level scores. With 5 seeds per game this cannot be checked well, and it is read as a pilot estimate. No bootstrap is used.
- **Interpretability rule:** human-normalised scores can be negative, and a ratio of two negative aggregates reverses meaning. θ is interpreted only if the baseline's interval b ± t · √v_b lies entirely above 0, which also guarantees a bounded Fieller set. Otherwise the category is *pilot-interval-unavailable*. This is a property of the data, not a statement that the population estimand is undefined. The absolute difference A_GTM − A_MLP is always reported, with interval (a − b) ± t · √(v_a + v_b) at the same t when that interval can be computed. An unavailable relative-gain interval and an unavailable absolute-difference interval are reported separately.
- **Reference value (SESOI):** Δ = 0.10. The registered research hypothesis says "exceeds baseline … score by ≥ 10 %". It is read here as a relative gain of 10 % of the baseline's aggregate score. *(Decided by the PI on 2026-10-23.)* The value was set on 2026-05-11 as the smallest sample-efficiency gain worth a Paper 2 section. It was not derived from a power analysis.

### Pilot categories (mutually exclusive, exhaustive, applied in this order)

For an interval [L, U] and reference Δ > 0:

0. **pilot-interval-unavailable** if the relative-gain interval is not reported under the interpretability rule, or an interval cannot be computed (zero seed variance in both arms, or an undefined df). The absolute difference (with its interval if available, otherwise its point estimate only) is then the reported result, and nothing about relative gain is stated.
1. **pilot-excludes-SESOI** if U < Δ. Wording: "a gain of Δ or more is not supported by this pilot."
2. **pilot-positive** if L ≥ Δ. Wording: "the pilot estimate is consistent only with gains of at least Δ ; superiority is not established, because this is not a confirmatory test."
3. **pilot-inconclusive** otherwise (L < Δ ≤ U).

Categories 1 and 2 cannot both hold because L ≤ U. None is a superiority, equivalence or "matches" verdict.

### Secondary contrasts (descriptive)

- The 12 per-game contrasts (6 games × {MLP, full-agent comparator}) are each reported as a difference in mean human-normalised score (GTM − comparator; positive favours GTM). Each has a two-sided Welch CI at level 1 − 0.05/12 (99.583 %). A contrast whose interval cannot be computed (zero seed variance in both arms) is reported as unavailable with its point estimate.
- The number of intervals entirely above 0, entirely below 0 and including 0 is reported **without a label**. An interval including 0 is never described as a tie or "matches".
- If the IRIS fallback is used, the comparator is named as IRIS in every table.

### Superseded provisions of the locked file

- **Decision criteria** (C'-headline ≥ 9/12 ; C'-tied-stable 5–8 ; C'-loses ≤ 4), including the "matches SOTA Dreamer/IRIS" wording: not applied. They are replaced by the categories above.
- **Welch's t-test with Bonferroni α = 0.00417 as a verdict:** replaced by the adjusted intervals. p-values are not used to label outcomes.
- **Compute-budget provisions** "Reduced subset: 6 games × 3 archs × 3 seeds", "Realistic minimum: … 1 seed. Then add seeds for top-2 promising games", and the risk-factor statement on "reducing to 1–3 seeds": **withdrawn**. All 3 arms × 6 games × 5 seeds are required. Seeds are never added, and games never selected, on the basis of results.

## Randomness

The registered seeds {0, 17, 42, 73, 101} are seed labels. Run (arm a, game g, seed s) draws all of its randomness (initialisation, environment seeds, exploration) from `numpy.random.SeedSequence([s, a, g])`, with fixed integer codes listed in the manifest. Randomness is therefore independent across arms, and arms are analysed as independent groups.

## Failure, exclusion and replacement

- **Infrastructure failure** (crash, out-of-memory, hardware fault, preemption, or a non-finite value traced to a software bug fixed without changing the registered configuration): the run is restarted from scratch with the same seed and configuration, at most twice. All attempts are logged.
- **Training failure** (divergence or non-finite loss under the registered configuration, or still failing after two restarts): the run is not replaced. It is evaluated at its last checkpoint with finite parameters. If there is none, it is evaluated as an untrained agent with the same protocol.
- No run is excluded or replaced on the basis of its score. Restarts and training failures are reported per arm and game.
- If a bug that affects an arm's registered configuration is found after runs have started, all runs of every affected arm are repeated, and this is reported as a deviation.

## Confirmatory follow-up

Any superiority claim requires a separate, future confirmatory pre-registration with a powered N. Its sample size is computed from the pilot's seed-level variance (the upper one-sided 80 % confidence limit of the pooled SD). The pilot data are not reused in its confirmatory test.

## Filing (two stages, PI decision 2026-10-23)

- **Stage 1, filed now:** this amendment text, that is the research hypothesis, the estimands, the common protocol, the analysis, failure and exclusion rules and the superseded provisions.
- **Stage 2, filed as a second locked stage before any Atari-100k run:** the configuration manifest below, which fixes the implementation details that stage 1 leaves open (architectures, adapters and objectives, including the GTM–RSSM interface).

Stage 2 cannot change any stage-1 provision. If the manifest cannot be written within the stage-1 rules, stage 1 is amended and re-filed before any run. Stage 1 cannot be filed while the open compute item below is unresolved.

## Configuration manifest (stage 2, required, immutable, part of this registration)

Before the first Atari-100k run, the PI files a dated manifest on OSF as an update to this registration. It may contain only:

1. the commit of the DreamerV3 port with the GTM transition predictor and of the MLP control, including the GTM–RSSM interface (how (z_t, h_t, a_t) is projected to the GTM code and decoded to ẑ_{t+1}, and how the discrete stochastic state is handled) and the integer codes for arms and games ;
2. the comparator outcome under the fallback rule (ITC or IRIS), its repository URL and commit, and whether the common protocol can be applied to it ;
3. the per-game random and human reference scores ;
4. the measured per-run cost of each arm on the target hardware, and the compute option adopted under the open PI confirmation item.

Development that informs the manifest may use only the CartPole dev kickoff, the 10k-step Pong smoke of the comparator, and unit tests. It never uses full-budget runs on the 6 registered games. The manifest cannot be changed after filing. Runs started before it is filed are excluded from all analyses and reported.

## Effect on hypotheses

The research hypothesis (GTM-dynamics DreamerV3 exceeds baseline DreamerV3 mean normalised human score by ≥ 10 % over 6 games) is kept as the **primary estimand and its reference value**. It is no longer tested for a superiority verdict at 5 seeds. It compares against the MLP control, not against IRIS or ITC.

**Flags for the PI:** (a) the reframing changes how every N11-C outcome may be interpreted and worded in Paper 2 §11, and must be filed as a change of interpretation ; (b) swapping IRIS for the ITC full agent changes which comparator enters 6 of the 12 secondary contrasts, and must be filed as a change to the comparison set.

## Compute

- **Registered design:** 3 arms × 6 games × 5 seeds = **90 runs**. At the registered estimate of about 24 h per run, this is **2,160 GPU-h, about 90 days sequential** on one RTX 4090. The locked file's "4 parallel runs, ~22 days" assumes four-fold throughput on one GPU, which has not been measured. The ITC full agent's per-run cost may differ from DreamerV3's and is measured for the manifest.
- **Feasibility (open PI confirmation item).** The registered design does not fit a single RTX 4090 in a reasonable time. Options, by whole arms only:
  - A. Fund the full 90 runs: about 2,160 GPU-h at the registered estimate, plus the comparator's cost difference.
  - B. Drop the comparator arm: 60 runs, about 1,440 GPU-h. The primary estimand is unaffected, and the 6 comparator contrasts are not reported.
  - C. Postpone N11-C until A is fundable.

  Reducing games or seeds is not an option, because it would change the primary estimand.
- **Status: pending (PI decision 2026-10-23).** Before choosing A, B or C, the PI requires a throughput measurement on the MacStudio (Apple M3 Ultra, 512 GB). Installing the N11-C dependencies on the MacStudio is not yet approved, so this measurement has not been made. **Option A is kept drafted but is pending, not approved.** The option is chosen, and this section is edited accordingly, before stage 1 is filed.

## PI decisions (2026-10-23)

- The estimation reframing as worded above.
- Two-stage filing: this text now (stage 1), the configuration manifest as a second locked stage before any run (stage 2), as described under Filing.
- "≥ 10 %" is read as a relative gain on the aggregate (not 10 human-normalised points).
- Comparator = the published ITC agent (Simulus+ITC) as a full-agent comparator, with the official IRIS agent only as the availability fallback.
- Compute: no option is chosen yet. The choice waits for a throughput measurement on the MacStudio (M3 Ultra, 512 GB), whose dependency installs are not yet approved (item below).

## PI confirmation item (open, to settle before filing stage 1)

- Compute option (A drafted and pending, or B / C), after the MacStudio throughput measurement.

If the PI changes any item, the text above is edited before filing, never after data are seen.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N11-C metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for the GTM-dynamics arm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no contrast.

## References

- Kim, Y., Sun, R., Kim, I., Park, B., Song, H. O. (2026). Identifiable Token Correspondence for World Models. arXiv:2605.16457.
- Hafner, D., Yan, W., Lillicrap, T. (2025). Training Agents Inside of Scalable World Models. arXiv:2509.24527.
- Micheli, V., Alonso, E., Fleuret, F. (2023). Transformers are Sample-Efficient World Models (IRIS). ICLR 2023.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
