# N11-C — Pre-registration amendment 2026-10-23: ITC comparator and DreamerV3 rationale

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Amends:** `docs/milestones/n11c-dreamer-gtm-dynamics-atari100k-2026-05-11.md` (pre-registered 2026-05-11, commit `3340440`). The dev kickoff `docs/milestones/n11c-dev-kickoff-dreamerv3-port-2026-05-11.md` is not affected.
**Date drafted:** 2026-10-23
**OSF registration:** no N11-C-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N11-C code or run exists in this repository (no `experiments/dreamer_gtm_dynamics_atari100k/`). The CartPole dev-kickoff smoke has not been committed.

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed.

## Summary of the change

1. **Replace the IRIS alternative-SOTA arm with ITC.** ITC is Kim, Sun, Kim, Park, Song, "Identifiable Token Correspondence for World Models", [arXiv:2605.16457](https://arxiv.org/abs/2605.16457).
2. **State explicitly why DreamerV3**, and not Dreamer 4, is the target architecture.

The registered H0, the control arm, the games, seeds, metric, comparison count, Bonferroni α and decision thresholds are unchanged.

## Rationale

- **Comparator.** The locked pre-registration uses IRIS (Micheli et al. 2023) as the token-based "alternative SOTA". ITC (2026-05-15) treats token-based transformer world-model decoding as a structured assignment problem. It reports gains on Atari-100k over the prior token-based state of the art. Keeping IRIS would compare GTM against a comparator that is no longer state of the art for this design family, and the "matches SOTA" wording of C'-tied-stable would then be misleading.
- **Target architecture.** Hafner, Yan, Lillicrap, "Training Agents Inside of Scalable World Models" (Dreamer 4), [arXiv:2509.24527](https://arxiv.org/abs/2509.24527) (2025-09-29), was public before the 2026-05-11 lock but is not discussed in it. It uses a transformer world model trained with shortcut forcing. The choice of DreamerV3 should be justified on the record.

## Methodology delta

### Comparator arm

- **Removed:** IRIS-transformer-dynamics (Micheli et al. 2023). Published IRIS scores may be quoted as context but are not part of any test.
- **Added:** **ITC-style token-correspondence dynamics**, following arXiv:2605.16457. It is a transformer over tokenised latents with ITC's structured-assignment decoding, used in place of the transition predictor, as the IRIS arm was.
- **Fallback, decided before any Atari run:** a faithful module-level port of ITC into the DreamerV3 RSSM may not be feasible. In that case, the ITC arm is the full ITC agent run under the same 100k-step budget, the same 6 games and the same 5 seeds, and it is labelled "full-agent comparator" in all reporting. The choice (module-level or full-agent) is recorded in a dated note before any Atari-100k run.

### Why DreamerV3 and not Dreamer 4 (rationale, no design change)

- **Interface:** H0 concerns replacing the RSSM transition predictor `(h_t) → ẑ_{t+1}`. Dreamer 4 does not use the RSSM transition-predictor MLP that H0 targets, so the registered intervention has no direct counterpart there.
- **Benchmark fit:** Atari-100k is a standard sample-efficiency benchmark for DreamerV3, with published reference scores. The control arm is therefore anchored to an established baseline.
- **Compute and tooling:** the registered budget is a single RTX 4090 with an open PyTorch port (`dreamerv3-torch`, per the dev kickoff). Dreamer 4 is designed for a much larger compute scale.
- **Consequence for claims:** any N11-C result is a statement about RSSM-based world models (DreamerV3 class). It must not be extrapolated to transformer world models such as Dreamer 4 or ITC beyond what the ITC comparison itself shows.

### Unchanged

Control arm (standard DreamerV3 MLP transition), 6 games (Pong, Breakout, Asterix, MsPacman, Seaquest, BeamRider), seeds {0, 17, 42, 73, 101}, normalised human score, official DreamerV3 hyperparameters.

## Analysis-plan delta

- Per-game Welch's t-test, GTM vs MLP and GTM vs **ITC** (was IRIS): 6 games × 2 baselines × 1 metric = **12** comparisons (unchanged count), Bonferroni α = 0.05/12 ≈ **0.00417** (unchanged).
- Decision thresholds unchanged (C'-headline ≥ 9/12, C'-tied-stable 5–8, C'-loses ≤ 4). In the C'-tied-stable wording, "Dreamer/IRIS" reads "Dreamer/ITC".
- If the full-agent fallback is used, the GTM-vs-ITC comparisons are reported as against a full agent, not against a matched dynamics module.

## Effect on hypotheses

H0 (GTM-dynamics DreamerV3 exceeds baseline DreamerV3 mean normalised human score by ≥ 10 % over 6 games) is **unchanged**. H0 compares against the MLP control, not against IRIS.

**Flag for the PI:** swapping IRIS for ITC changes *which* comparator enters 6 of the 12 comparisons in the decision family. The count, α and thresholds do not change, but the decision labels now depend on a different arm. This should be filed explicitly as a change to the comparison set.

## Compute delta

The arm count is unchanged (3 archs × 6 games × 5 seeds = 90 runs). Per-run cost of the ITC arm may differ from that of IRIS. The estimate is recorded in the dated note before the sweep.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N11-C metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for the GTM-dynamics arm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no hypothesis test.

## References

- Kim, Y., Sun, R., Kim, I., Park, B., Song, H. O. (2026). Identifiable Token Correspondence for World Models. arXiv:2605.16457.
- Hafner, D., Yan, W., Lillicrap, T. (2025). Training Agents Inside of Scalable World Models. arXiv:2509.24527.
- Micheli, V., Alonso, E., Fleuret, F. (2023). Transformers are Sample-Efficient World Models (IRIS). ICLR 2023.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
