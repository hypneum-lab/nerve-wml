# N11-A — Pre-registration amendment 2026-10-23: estimation reframing and frozen design options

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI in two stages, see Filing)
**Amends:** `docs/milestones/n11a-ijepa-gtm-predictor-2026-05-11.md` (pre-registered 2026-05-11, commit `3340440`)
**Date drafted:** 2026-10-23
**OSF registration:** no N11-A-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N11-A code or run exists in this repository (no `experiments/ijepa_gtm_predictor/`).

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed. Where it conflicts with the locked file, this amendment governs. The locked file labels its research hypothesis "H0" ; this amendment calls it the **research hypothesis**.

## Summary of the change

1. **Estimation / pilot reframing (PI decision 2026-10-23). This changes the interpretation of the study.** N11-A becomes an estimation (pilot) study. It reports effect estimates with confidence intervals and makes **no superiority verdict**. The A'-headline / A'-tied-stable / A'-loses labels and their win-count thresholds are **not applied**. Seeds, arms, metrics and compute are unchanged.
2. The registered study's open options are fixed: ImageNet-100 pretraining, one backbone scale (ViT-B/16), and specified evaluation protocols. A required immutable configuration manifest covers the remaining implementation choices.
3. A hierarchical-JEPA sub-study (H-JEPA) with a physical-state probe is **not** registered here. It is deferred to a separate amendment (see the end of this document).

## Rationale

- **Power.** With 5 seeds per arm and Bonferroni correction the registered win-count tests have little power. Fixed-df (df = 8) two-sided t calculations for two independent normal groups of 5 with equal variances:

  | Family | Comparisons | Per-test α | Power at d = 1 | d for 80 % power |
  |---|---:|---:|---:|---:|
  | Registered | 8 | 0.00625 | 6.8 % | 2.97 |
  | Single contrast, unadjusted | 1 | 0.05 | 28.6 % | 2.02 |

  d is the standardized mean difference. No seed-level variance estimate exists, so power at the registered 1.0 pp threshold cannot be computed. The PI decided on 2026-10-23 to keep 5 seeds and to report estimates instead of verdicts.

## Options fixed by this amendment

- **Pretraining data:** ImageNet-100, the same 100-class subset as N10-A. The registered compute budget (15 runs × about 12 h) already assumes it. ImageNet-1k pretraining is not run.
- **Backbone scale:** ViT-B/16 only. The optional ViT-S/16 second scale (30 runs, 16 comparisons) is not run.
- **Representation evaluated:** the average-pooled final-layer output of the target encoder, frozen.
- **Linear probe (ImageNet-100):** a linear classifier on the frozen representation, AdamW, lr 1e-3, weight decay 0, batch 1024, 50 epochs, cosine schedule, no augmentation beyond center crop ; top-1 on the val split.
- **k-NN:** k = 20, cosine similarity, training set as the memory bank, top-1 on the val split.
- **CIFAR-100 transfer:** a linear probe with the same recipe on the frozen representation of CIFAR-100 images resized to 224. The registered text said "fine-tune" without a recipe. *(Decided by the PI on 2026-10-23.)*
- **MI(z_predicted, label):** predictor outputs, average-pooled over target positions, on a fixed held-out set of 10,000 ImageNet-100 training images (selected with `numpy.random.default_rng(20261023)` and excluded from pretraining). They are discretised by k-means with k = 64 clusters (fit on a disjoint 10,000-image subset, seed 20261023). MI is the plug-in estimate, in bits, between cluster index and label.
- **Cross-attention predictor:** V-JEPA-2-style cross-attention predictor, at the same depth and width as the I-JEPA default predictor.
- **MLP predictor:** 2-layer MLP whose parameter count is within ±5 % of the GTM predictor's (adapters included). The locked file calls this arm the "vanilla MLP predictor", but the I-JEPA reference predictor is a narrow transformer. The MLP arm is therefore a control architecture, not the I-JEPA default.
- **Evaluation codes and masks:** all metrics use one deterministic evaluation pass. The GTM predictor uses argmax symbols (no Gumbel noise). For MI, context and target masks for each image are drawn once with `numpy.random.default_rng([20261023, image_index])` and shared by all arms.

## Estimation plan (replaces the registered statistical analysis and decision criteria)

### Interval method

Each contrast is a difference in seed means (GTM − comparator). Its two-sided Welch CI is: difference ± t(1 − a/2, ν) · √(s₁²/5 + s₂²/5), with ν the Welch–Satterthwaite df and a the stated per-interval error rate. Intervals assume approximately normal seed-level values. With 5 seeds this cannot be checked well, and intervals are read as pilot estimates.

### Primary estimand

θ = mean linear-probe top-1 (GTM predictor) − mean linear-probe top-1 (MLP predictor), ViT-B/16, ImageNet-100 val, in percentage points. This is the contrast named in the registered research hypothesis.

- **Interval:** 95 % Welch CI.
- **Reference value (SESOI):** Δ = 1.0 pp, the threshold in the registered research hypothesis. It was set on 2026-05-11 as the smallest linear-probe gain worth a Paper 2 section. It was not derived from a power analysis.

### Pilot categories (mutually exclusive, exhaustive, applied in this order)

For an interval [L, U] and reference Δ > 0:

0. **pilot-interval-unavailable** if the interval cannot be computed: both arms have zero seed variance, or a required df is undefined. The point estimate is reported, and nothing beyond it is stated. Identical seed outcomes are not read as certainty about the population.
1. **pilot-excludes-SESOI** if U < Δ. Wording: "a gain of Δ or more is not supported by this pilot."
2. **pilot-positive** if L ≥ Δ. Wording: "the pilot estimate is consistent only with gains of at least Δ ; superiority is not established, because this is not a confirmatory test."
3. **pilot-inconclusive** otherwise (L < Δ ≤ U).

Categories 1 and 2 cannot both hold because L ≤ U. None is a superiority, equivalence or "matches" verdict. Whether the interval lies above 0, below 0 or includes 0 is reported in addition and does not change the category.

### Secondary contrasts (descriptive)

- The registered 8 contrasts (4 metrics × {MLP, cross-attention}) are each reported with a Welch CI at level 1 − 0.05/8 (99.375 %).
- For all four metrics (linear-probe top-1, k-NN top-1, CIFAR-100 transfer top-1, MI), a positive difference favours GTM.
- The number of intervals entirely above 0, entirely below 0 and including 0 is reported **without a label**. An interval including 0 is never described as a tie or "matches".

## Randomness

The registered seeds {0, 17, 42, 73, 101} are seed labels. Run (arm a, seed s) draws all of its randomness (initialisation, data order, masking, augmentation) from `numpy.random.SeedSequence([s, a])`, with fixed integer arm codes listed in the manifest. Randomness is therefore independent across arms, and every contrast is analysed as two independent groups of 5 (Welch). All metrics are computed on fixed evaluation sets, and estimands are defined conditional on these sets.

## Failure, exclusion and replacement

- **Infrastructure failure** (crash, out-of-memory, hardware fault, preemption, or a non-finite value traced to a software bug fixed without changing the registered configuration): the run is restarted from scratch with the same seed stream and configuration, at most twice. All attempts are logged.
- **Training failure** (divergence or non-finite loss under the registered configuration, or still failing after two restarts): the run is not replaced. It is evaluated at its last checkpoint with finite parameters. If there is none, it is evaluated at initialisation with the same protocol.
- No run is excluded or replaced on the basis of its outcome. Restarts and training failures are reported per arm.
- If a bug that affects an arm's registered configuration is found after runs have started, all runs of every affected arm are repeated, and this is reported as a deviation.

## Superseded provisions of the locked file

- **Decision criteria** (A'-headline ≥ 6/8 ; A'-tied-stable 3–5 ; A'-loses ≤ 2): not applied. They are replaced by the categories above. The registered A'-tied-stable wording "matches I-JEPA SOTA" is not used for any outcome.
- **Welch's t-test with Bonferroni α = 0.00625 as a verdict:** replaced by the adjusted intervals. p-values are not used to label outcomes.
- **"Optionally 2 backbone scales" and "(or ImageNet-100 subset …, document substitution)":** resolved above.

## Confirmatory follow-up

Any superiority claim requires a separate, future confirmatory pre-registration with a powered N. Its sample size is computed from the pilot's seed-level variance (the upper one-sided 80 % confidence limit of the pooled SD). The pilot data are not reused in its confirmatory test.

## Filing (two stages, PI decision 2026-10-23)

- **Stage 1, filed now:** this amendment text, that is the research hypothesis, the estimands, the options fixed above, the analysis, failure and exclusion rules and the superseded provisions.
- **Stage 2, filed as a second locked stage before any ImageNet-100 pretraining run:** the configuration manifest below, which fixes the implementation details that stage 1 leaves open (architectures, adapters and objectives, including the GTM predictor's adapters).

Stage 2 cannot change any stage-1 provision. If the manifest cannot be written within the stage-1 rules, stage 1 is amended and re-filed before any run.

## Configuration manifest (stage 2, required, immutable, part of this registration)

Before the first ImageNet-100 pretraining run, the PI files a dated manifest on OSF as an update to this registration. It may contain only:

1. the commit of the I-JEPA training and evaluation code used for every run, and the integer arm codes ;
2. the GTM predictor's encode/decode adapter and the cross-attention and MLP predictor sizes, as implemented at that commit, with their parameter counts ;
3. the I-JEPA hyperparameters not fixed in the locked file (mask sampling, EMA momentum schedule, warm-up), taken from the I-JEPA reference configuration for ViT-B/16 ;
4. the ImageNet-100 class list and the SHA-256 of the image index files.

Development that informs the manifest may use only the ViT-T/16 stand-in named in the locked file's risk factors, on data other than ImageNet-100, and unit tests. The manifest cannot be changed after filing. Runs started before it is filed are excluded from all analyses and reported.

## Effect on hypotheses

The research hypothesis (GTM-predictor I-JEPA exceeds MLP-predictor I-JEPA linear-probe top-1 by ≥ 1.0 pp at ViT-B/16) is kept as the **primary estimand and its reference value**. It is no longer tested for a superiority verdict at 5 seeds. **Flag for the PI:** this changes how every N11-A outcome may be interpreted and worded in Paper 2 §10, and must be filed as a change of interpretation.

## Compute

Unchanged: 15 runs × about 12 h ≈ **180 GPU-h** on ImageNet-100. The optional 2-scale variant is not run.

## PI decisions (2026-10-23)

1. The estimation reframing as worded above.
2. CIFAR-100 transfer by linear probe (registered text said "fine-tune").
3. Deferral of the H-JEPA sub-study to a separate amendment.
4. Two-stage filing: this text now (stage 1), the configuration manifest as a second locked stage before any run (stage 2), as described under Filing.

No PI confirmation item remains open for N11-A. If the PI changes any of these decisions, the text above is edited before filing, never after data are seen.

## Descriptive additions (literature check, 2026-10-23)

Added before filing, as descriptive secondary reporting items. They add **no contrast, no test and no exclusion or failure rule**, and do not change any pilot category.

- **Code-usage manipulation check (GTM predictor).** For the GTM arm and each seed, report from the deterministic evaluation pass the number of distinct symbols used and the perplexity exp(H) of the empirical symbol distribution of the predictor's Gumbel code (alphabet 64), per code position, on the 10,000-image MI set. A hard Gumbel-softmax code can collapse onto few symbols (Fang, arXiv:2607.08312, single-author preprint, different setting) ; a collapsed code would make a GTM-vs-MLP difference uninterpretable as a code effect. It is a manipulation check, not an exclusion criterion. The MLP and cross-attention arms have no discrete code : not applicable.
- **Effective rank (descriptive secondary).** For every arm and seed, report the effective rank of the frozen context-encoder representation (average-pooled, same evaluation set as the MI measurement) and, optionally, of the predictor outputs. Predictor architecture shifts the effective rank that emerges in JEPA pretraining (De Melo Costa et al., "Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures", [arXiv:2610.02344](https://arxiv.org/abs/2610.02344), NeurIPS 2026 main track), so a GTM-vs-MLP difference may act through collapse dynamics rather than through the code. Reported beside the registered metrics, never as a fifth registered metric. The stage-2 manifest also records whether the cross-attention arm follows the I-JEPA default initialisation. Adding a ResidualPred arm is not done : it would change the registered arm set.

## Deferred: hierarchical JEPA sub-study (not registered here)

The literature review proposed a secondary sub-study placing GTM inside a hierarchical, action-conditioned JEPA world model:

- Zhang, Terver, Rabbat, LeCun, Balestriero, "H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning", [arXiv:2610.06805](https://arxiv.org/abs/2610.06805) (2026-10-05), stacks action-conditioned JEPAs with a distinct timescale per level. GTM's slow (θ) carrier / fast (γ) content design maps naturally onto such a hierarchy.
- Yan et al., "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models", [arXiv:2608.06799](https://arxiv.org/abs/2608.06799), and Zhu, Penachio, Mukherjee, Jonelagadda, "Spectral-Target Physical Latent Structuring for JEPA-Style World Models", [arXiv:2609.04264](https://arxiv.org/abs/2609.04264), report that JEPA world-model latents can predict well while remaining poorly grounded in the physical state, so such a sub-study would need a physical-state probe.

H-JEPA's predictors operate within each level's latent space, so "GTM between levels" does not name an existing module. A registrable design must name the replaced module, its inputs, targets and action conditioning, the hierarchy depth and strides, the planner and its budget, the environment and state variables, and a scalar probe endpoint. These cannot be fixed without reading the reference implementation. The sub-study is therefore **not** part of this filing. If pursued, it needs its own amendment, filed before any run. A physical-state probe does not apply to the registered ImageNet study, because ImageNet images carry no ground-truth simulator state.

## Not adopted

LeWorldModel ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026-03-13) would be a cheaper single-GPU JEPA world-model substrate. It is not adopted here, so that the registered ImageNet study stays as registered.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N11-A metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for a GTM arm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no contrast.

**Added 2026-10-23 (literature check, all four N10/N11 amendments).** Scalar PAC indices are many-to-one : different coupling geometries and phase polarities can give the same value. Besides the Tort modulation index and the preferred phase, any exploratory PAC diagnostic therefore also reports the **circular moments** of the coupling or the **phase-binned amplitude histogram** (Keshavarzi & Goswami, bioRxiv follow-up posted 2026-09-21, [doi:10.64898/2026.09.15.751789](https://doi.org/10.64898/2026.09.15.751789)). Exploratory only, no contrast, no test.

## References

- Zhang, W., Terver, B., Rabbat, M., LeCun, Y., Balestriero, R. (2026). H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning. arXiv:2610.06805.
- Yan et al. (2026). Is Forward Prediction Enough? Physical State Grounding for JEPA World Models. arXiv:2608.06799.
- Zhu, Penachio, Mukherjee, Jonelagadda (2026). Spectral-Target Physical Latent Structuring for JEPA-Style World Models. arXiv:2609.04264.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
- Keshavarzi, M., Goswami, S. (2026). Computing the geometry of phase–amplitude coupling: A new moment-based framework. bioRxiv. doi:10.64898/2026.09.15.751789 (posted 2026-09-21).
- De Melo Costa, Ahn, Popineau, Rimmel, Doan (2026). Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive Architectures. arXiv:2610.02344 (NeurIPS 2026).
- Fang (2026). Write-Protected Discrete Bottlenecks for Language-Grounded World Models: A Structural Limitation and Sufficient Fix. arXiv:2607.08312.
