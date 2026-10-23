# N10-A — Pre-registration amendment 2026-10-23: estimation reframing and oscillator-family comparator

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI in two stages, see Filing)
**Amends:** `docs/milestones/n10a-imagenet100-gtm-bottleneck-2026-05-11.md` (pre-registered 2026-05-11, commit `1640c5c` ; in-file amendment 2026-05-11, commit `c74f6be`)
**Date drafted:** 2026-10-23
**OSF registration:** no N10-A-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** the ImageNet-100 sweep has not started. The only N10-A run so far is a pipeline smoke test on a CIFAR-10 / frozen ResNet18 stand-in, GTM arm only, 5 epochs (`experiments/imagenet100_gtm_bottleneck/smoke_results.json`). No baseline arm has been trained and no ImageNet-100 result exists.

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed. Where it conflicts with the locked file, this amendment governs. The locked file labels its research hypothesis "H0" ; this amendment calls it the **research hypothesis**.

## Summary of the change

1. **Estimation / pilot reframing (PI decision 2026-10-23). This changes the interpretation of the study.** N10-A becomes an estimation (pilot) study. It reports effect estimates with confidence intervals and makes **no superiority verdict**. The A-headline / A-tied-stable / A-loses labels and their win-count thresholds are **not applied**. Seeds, arms, metrics and compute are unchanged.
2. **Oscillator-family comparator** added as a fourth bottleneck, reported in a separate family, with the claim it can support restricted to "GTM versus this adapted comparator".
3. Independent per-arm random streams, failure / exclusion / replacement rules, frozen architecture and metric settings, and a required immutable configuration manifest are specified. The registered third baseline is fixed.

## Rationale

- **Power.** With 5 seeds per arm and Bonferroni correction the registered win-count tests have little power. Fixed-df (df = 8) two-sided t calculations for two independent normal groups of 5 with equal variances:

  | Family | Comparisons | Per-test α | Power at d = 1 | d for 80 % power |
  |---|---:|---:|---:|---:|
  | Registered headline | 45 | 0.001111 | 1.7 % | 3.87 |
  | Oscillator (new) | 15 | 0.003333 | 4.2 % | 3.28 |
  | Single primary contrast, unadjusted | 1 | 0.05 | 28.6 % | 2.02 |

  d is the standardized mean difference. No seed-level variance estimate exists for any N10-A arm, so power at the 1.5 pp threshold cannot be computed. A win count at this power mainly reflects noise, and a non-significant comparison cannot be read as a tie. The PI decided on 2026-10-23 to keep 5 seeds and the compute budget, and to report estimates instead of verdicts. With 5 seeds per arm the expected half-width of a 95 % CI for a difference in means is about 1.46 σ (σ the common seed-level SD).
- **Oscillator comparator.** Two oscillator architectures show that phase-based inductive biases can be competitive at ImageNet scale. Neither is considered in the locked registration:
  - Xiao, Wang, Han, Shan, Li, "Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency" (KoPE), [arXiv:2604.07904](https://arxiv.org/abs/2604.07904). v1 was submitted on 2026-04-09, before the 2026-05-11 lock ; the ICML 2026 version appeared after it. It adds a Kuramoto phase state to Vision Transformers and reports data and parameter efficiency gains.
  - Dai & Song, "Winfree Oscillatory Neural Network" (WONN), [arXiv:2605.20922](https://arxiv.org/abs/2605.20922), submitted 2026-05-20, after the lock. It reports a synchronization-based oscillator network that is competitive on ImageNet-1K.

  The registered baselines contain no phase or oscillator component. Within the repo, AKOrN was the only oscillator comparator so far (toy tasks, `docs/superpowers/research/2026-05-20-akorn-*.md`).

## Estimation plan (replaces the registered statistical analysis and decision criteria)

### Primary estimand

θ = mean top-1 (GTM, code_dim = 32) − mean top-1 (VQ-VAE, code_dim = 32), in percentage points on the ImageNet-100 val split, means over the 5 seeds. This is the contrast named in the registered research hypothesis.

- **Interval:** two-sided 95 % Welch CI: difference in means ± t(0.975, ν) · √(s₁²/5 + s₂²/5), ν the Welch–Satterthwaite df.
- **Reference value (SESOI):** Δ = 1.5 pp, the threshold in the registered research hypothesis. It was set on 2026-05-11 as the smallest top-1 gain that would justify a bottleneck claim in Paper 2. It was not derived from a power analysis (no variance data existed). In this pilot it is a reference line for reading the interval, not a decision threshold for superiority.

### Pilot categories (mutually exclusive, exhaustive, applied in this order)

For an interval [L, U] and reference Δ > 0:

0. **pilot-interval-unavailable** if the interval cannot be computed: both arms have zero seed variance, or a required df is undefined. The point estimate is reported, and nothing beyond it is stated. Identical seed outcomes are not read as certainty about the population.
1. **pilot-excludes-SESOI** if U < Δ: no value in the interval reaches the SESOI. Wording: "a gain of Δ or more is not supported by this pilot."
2. **pilot-positive** if L ≥ Δ: every value in the interval is at least Δ. Wording: "the pilot estimate is consistent only with gains of at least Δ ; superiority is not established, because this is not a confirmatory test."
3. **pilot-inconclusive** otherwise (L < Δ ≤ U). Wording: "the pilot cannot distinguish gains below Δ from gains of at least Δ."

Because L ≤ U, categories 1 and 2 cannot both hold. None of the three is a superiority, equivalence or "matches" verdict. Whether the interval lies above 0, below 0 or includes 0 is reported in addition and does not change the category.

### Secondary contrasts (descriptive)

- **Registered headline set:** GTM vs {VQ-VAE, linear projection, β-VAE β = 1.0} × 5 metrics × 3 code_dims = 45 contrasts.
- **Oscillator set (new):** GTM vs oscillator bottleneck × 5 metrics × 3 code_dims = 15 contrasts.
- Each contrast is reported as the difference in means (GTM − comparator) with a two-sided Welch CI at level 1 − 0.05/k, where k is the size of its set (k = 45 → 99.889 % ; k = 15 → 99.667 %). These Bonferroni-adjusted intervals have approximate simultaneous coverage of at least 95 % under the normal seed-level model within each set.
- **Direction:** for top-1, top-5, MI(code, label) and OOD top-1, a positive difference favours GTM. For bandwidth efficiency (effective rank) no favourable direction was registered. It is reported as a signed difference and never counted as favouring either arm.
- For each set, the number of intervals entirely above 0, entirely below 0 and including 0 is reported. These counts carry **no label**. An interval that includes 0 is never described as a tie, equivalence or "matches".

### Superseded provisions of the locked file

- **Decision criteria** (A-headline ≥ 30/45 wins ; A-tied-stable 15–29 ; A-loses ≤ 10 wins or ≥ 30 losses): not applied. They are replaced by the pilot categories above. This also removes the gaps (11–14 wins had no label) and the overlap (≥ 30 losses with 15–29 wins) in the registered rules.
- **Welch's t-test per comparison with Bonferroni α = 0.00111 as a verdict:** replaced by the adjusted intervals above. p-values are not used to label outcomes.
- **Risk-factor analyses** (random-codebook GTM ablation, MoCo v3 backbone sensitivity, severity dependence): not part of this registration. If run, they are labelled exploratory and kept outside both sets.

### Confirmatory follow-up

Any superiority claim requires a separate, future confirmatory pre-registration with a powered N. Its sample size is computed from the pilot's seed-level variance (the upper one-sided 80 % confidence limit of the pooled SD). The pilot data are not reused in its confirmatory test. No pilot category obliges or rules out such a study.

## Methodology delta

### Third registered baseline

The original methodology lists **linear projection** (`nn.Linear(2048, code_dim)` + `nn.LayerNorm`) as the third baseline. The 2026-05-11 in-file amendment lists "dense autoencoder" in that place while stating that the baseline count is unchanged. This amendment registers **linear projection**, the only third baseline with a specified architecture, and treats "dense autoencoder" as a drafting inconsistency. *(Decided by the PI on 2026-10-23.)*

### New arm: oscillator bottleneck (KoPE-style, primary)

Fully specified here ; nothing is taken from reference defaults:

1. u = `nn.Linear(2048, code_dim)`(x). Split u into M = code_dim / 2 pairs (a_k, b_k) and set θ_k(0) = atan2(b_k, a_k).
2. Learned natural frequencies ω ∈ ℝ^M (initialised to 0) and coupling K ∈ ℝ^{M×M} (initialised to 0).
3. T = 8 explicit Euler steps of size h = 0.1: θ_k ← θ_k + h · [ω_k + (1/M) Σ_j K_kj sin(θ_j − θ_k)].
4. Code = [cos θ(T), sin θ(T)] (dimension code_dim), then `nn.LayerNorm(code_dim)` and the shared `nn.Linear(code_dim, 100)` head (head parity kept).
5. Same optimizer, lr, weight decay, epochs, batch size and schedule as the registered arms. No hyperparameter search for any arm. The extra M² + M parameters (≤ 4,160 at code_dim = 128) are reported.

This is a KoPE-*style adaptation* to a feature bottleneck on frozen ResNet50 features. It is **not** a reproduction of the published KoPE ViT model, and results must not be reported as such.

**Fallback (WONN-style):** identical except for step 3, which becomes the Winfree update θ_k ← θ_k + h · [ω_k − sin θ_k · (1/M) Σ_j K_kj (1 + cos θ_j)].

### Objective smoke criterion (decides primary vs fallback before the sweep)

- **Smoke data:** ImageNet-100 *training* features only. A fixed 10 % of training images, drawn with `numpy.random.default_rng(20261023)`, is held out as smoke-val. The official val split is never used in the smoke.
- **Smoke run:** code_dim = 32, seed 0, 10 epochs, registered optimizer.
- **Failure:** any non-finite loss or gradient ; or final smoke-val top-1 < 10 % (chance = 1 %) ; or final-epoch training loss not lower than epoch-1 training loss.
- If the KoPE-style module fails, the WONN-style module is smoked with the same criterion. If both fail, the oscillator set is not run, and this is reported as a deviation. The primary estimand and the headline set are unaffected. Smoke runs are excluded from all analyses, and their pass/fail outcome is recorded in the configuration manifest. Only one oscillator arm enters the analysis.

### Claim restriction for the oscillator set

Beating one adapted oscillator bottleneck cannot separate phase–amplitude coupling from capacity, optimisation or other architectural differences. Whatever the oscillator-set estimates are, N10-A makes **no statement that an advantage is specific to PAC multiplexing** rather than to a generic phase bias. PAC specificity is reported as **not addressed** by this design. Oscillator-set results are described only as GTM versus this particular adapted comparator. A matched PAC-disabled GTM ablation would be needed to address PAC specificity. It is not added here *(decided by the PI on 2026-10-23)*.

### Unchanged

Backbone (frozen ResNet50), dataset (ImageNet-100), preprocessing, code_dim sweep {32, 64, 128}, seeds {0, 17, 42, 73, 101}, the 5 metrics, optimiser settings, and the three registered baselines (with the third fixed and their settings frozen as below).

## Pairing and randomness

The registered seeds {0, 17, 42, 73, 101} are seed labels. Run (arm a, code_dim c, seed s) draws all of its randomness (initialisation, data order, Gumbel noise) from `numpy.random.SeedSequence([s, a, c])`, with fixed integer arm codes listed in the manifest. Randomness is therefore independent across arms, and every contrast is analysed as two independent groups of 5 (Welch). All metrics are computed on fixed evaluation sets (the ImageNet-100 val split and its ImageNet-C counterparts), and estimands are defined conditional on these sets.

## Frozen architecture and metric settings

These close choices that the locked file left open:

- **Evaluation codes:** every metric is computed in a single deterministic evaluation pass. GTM uses argmax symbols (no Gumbel noise), VQ-VAE the nearest codebook vector, β-VAE the posterior mean, the oscillator arm its deterministic integration. These codes feed classification, MI clustering, OOD and bandwidth metrics.

- **VQ-VAE:** one codebook of 512 vectors of dimension code_dim, commitment weight 0.25, EMA codebook update with decay 0.99 (van den Oord et al. 2017 defaults).
- **β-VAE:** Gaussian code of dimension code_dim, β = 1.0, the classifier reads the posterior mean at evaluation.
- **GTM:** `GTMBridge` with alphabet_size = 64 and the code_dim of the sweep, as in N8 T13.
- **MI(code, label):** every arm's val codes are discretised by k-means with k = 64 clusters, fit on the training codes with seed 20261023. MI is the plug-in estimate, in bits, between cluster index and label on the val split. The same pipeline is used for discrete and continuous codes.
- **OOD top-1:** the unweighted mean of top-1 over the 5 registered ImageNet-C corruptions at severity 3, on the val images of the 100 classes.
- **Bandwidth efficiency:** the number of principal components needed to reach 95 % of the variance of the val codes.

## Failure, exclusion and replacement

- **Infrastructure failure** (crash, out-of-memory, hardware fault, preemption, or a non-finite value traced to a software bug fixed without changing the registered configuration): the run is restarted from scratch with the same seed and configuration, at most twice. All attempts are logged.
- **Training failure** (divergence or non-finite loss under the registered configuration, or still failing after two restarts): the run is not replaced. It is evaluated at its last checkpoint with finite parameters. If there is none, it receives the metric values of the same architecture at initialisation, evaluated with the same protocol.
- No run is excluded or replaced on the basis of its metric values. Restarts and training failures are reported per arm.
- If a bug that affects an arm's registered configuration is found after runs have started, all runs of every affected arm are repeated with the fixed code, and this is reported as a deviation.

## Filing (two stages, PI decision 2026-10-23)

- **Stage 1, filed now:** this amendment text, that is the research hypothesis, the estimands, the analysis, failure and exclusion rules and the superseded provisions.
- **Stage 2, filed as a second locked stage before any ImageNet-100 sweep run:** the configuration manifest below, which fixes the implementation details that stage 1 leaves open (architectures, adapters and objectives, including the GTM symbol count and adapter and the baselines' encoder/decoder and loss weighting).

Stage 2 cannot change any stage-1 provision. If the manifest cannot be written within the stage-1 rules, stage 1 is amended and re-filed before any run.

## Configuration manifest (stage 2, required, immutable, part of this registration)

Before the first ImageNet-100 sweep run, the PI files a dated manifest on OSF as an update to this registration. It may contain only:

1. the commit of the training / evaluation harness used for every run, and the integer arm codes ;
2. the architecture, adapter and objective of every arm as implemented at that commit, within the settings frozen above: the GTM symbol count and encode/decode adapter, the encoder/decoder and loss weighting of the VQ-VAE and β-VAE baselines, and the parameter count of every arm ;
3. the ImageNet-100 class list (the CMC 100-class subset) and the SHA-256 of the cached feature files ;
4. the smoke outcome (pass/fail and the three smoke quantities) and the resulting oscillator choice (KoPE-style, WONN-style or none).

Development that informs the manifest may use only the CIFAR-10 stand-in, the smoke procedure above, and unit tests. It never uses the ImageNet-100 val split. The manifest cannot be changed after filing. Runs started before it is filed are excluded from all analyses and reported.

## Effect on hypotheses

The research hypothesis (GTM top-1 at code_dim = 32 exceeds VQ-VAE by ≥ 1.5 pp) is kept as the **primary estimand and its reference value**. It is no longer tested for a superiority verdict at 5 seeds. **Flag for the PI:** this changes how every N10-A outcome may be interpreted and worded in Paper 2 §8, and must be filed as a change of interpretation, not as a clarification.

## Compute delta

Oscillator arm: 3 code_dims × 5 seeds = **15 runs** on cached features, plus at most 2 smoke runs and one OOD evaluation. Oscillator integration makes each run slower than a linear bottleneck. The estimate is about 5–8 h on the RTX 4090, against the registered total of about 22 h. The reframing itself adds no compute.

## PI decisions (2026-10-23)

1. Third baseline = linear projection (not dense autoencoder).
2. No PAC-disabled GTM ablation. PAC specificity is reported as not addressed.
3. The estimation reframing as worded above.
4. Two-stage filing: this text now (stage 1), the configuration manifest as a second locked stage before any run (stage 2), as described under Filing.

No PI confirmation item remains open for N10-A. If the PI changes any of these decisions, the text above is edited before filing, never after data are seen.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N10-A metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for the GTM arm, the **preferred phase** is reported alongside it. Keshavarzi (bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)) shows that the modulation index does not distinguish a 180° inversion of the slow phase. This rule adds no contrast to either set.

## References

- Xiao, M., Wang, Y., Han, D., Shan, C., Li, D. (2026). Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency. ICML 2026. arXiv:2604.07904 (v1 submitted 2026-04-09).
- Dai, J., Song, Y. (2026). Winfree Oscillatory Neural Network. arXiv:2605.20922.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
