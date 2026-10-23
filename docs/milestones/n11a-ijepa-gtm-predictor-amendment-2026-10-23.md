# N11-A — Pre-registration amendment 2026-10-23: estimation reframing, hierarchical JEPA sub-study and physical-state probe

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Amends:** `docs/milestones/n11a-ijepa-gtm-predictor-2026-05-11.md` (pre-registered 2026-05-11, commit `3340440`)
**Date drafted:** 2026-10-23
**OSF registration:** no N11-A-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N11-A code or run exists in this repository (no `experiments/ijepa_gtm_predictor/`).

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed. Where it conflicts with the locked file, this amendment governs.

## Summary of the change

1. **Estimation / pilot reframing (PI decision 2026-10-23). This changes the interpretation of the study.** The registered ImageNet I-JEPA study becomes an estimation (pilot) study. It reports effect estimates with confidence intervals and makes **no superiority verdict**. The A'-headline / A'-tied-stable / A'-loses labels and their win-count thresholds are **not applied**. Seeds, arms, metrics and compute are unchanged.
2. The registered study's open options are fixed: ImageNet-100 pretraining, one backbone scale (ViT-B/16), and a specified CIFAR-100 transfer protocol.
3. Add a **secondary sub-study, N11-A-H**: GTM as the inter-level predictor of a hierarchical JEPA world model (H-JEPA), with a **frozen scalar physical-state probe** and a planning endpoint. It is also an estimation study.

## Rationale

- **Power.** With 5 seeds per arm and Bonferroni correction the registered win-count tests have little power. Recomputed for independent normal groups, equal variances, 5 seeds per arm, two-sided Welch-type t-tests (df = 8):

  | Family | Comparisons | Per-test α | Power at d = 1 | d for 80 % power |
  |---|---:|---:|---:|---:|
  | Registered | 8 | 0.00625 | 6.8 % | 2.97 |
  | N11-A-H (new) | 4 | 0.0125 | 11.4 % | 2.64 |
  | Single contrast, unadjusted | 1 | 0.05 | 28.6 % | 2.02 |

  d is the standardized mean difference. No seed-level variance estimate exists, so power at the registered 1.0 pp threshold cannot be computed. The PI decided on 2026-10-23 to keep 5 seeds and to report estimates instead of verdicts.
- **H-JEPA.** Zhang, Terver, Rabbat, LeCun, Balestriero, "H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning", [arXiv:2610.06805](https://arxiv.org/abs/2610.06805) (2026-10-05), stacks action-conditioned JEPAs with a distinct timescale per level. GTM is designed around a slow (θ) carrier and fast (γ) content. A hierarchy with explicit per-level timescales tests that design more directly than replacing a single-timescale predictor in image-level I-JEPA.
- **Grounding.** Yan et al., "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models", [arXiv:2608.06799](https://arxiv.org/abs/2608.06799), and Zhu, Penachio, Mukherjee, Jonelagadda, "Spectral-Target Physical Latent Structuring for JEPA-Style World Models", [arXiv:2609.04264](https://arxiv.org/abs/2609.04264), report that JEPA world-model latents can predict well while remaining poorly grounded in the physical state. A probe against ground-truth state is therefore needed in addition to task performance.

## Shared estimation rules (both studies)

### Intervals

Each contrast is a difference in means over the 5 seeds per arm (GTM − comparator). Its two-sided Welch CI is: difference ± t(1 − a/2, ν) · √(s₁²/5 + s₂²/5), with ν the Welch–Satterthwaite df and a the stated per-interval error rate.

### Pilot categories (mutually exclusive, exhaustive, applied in this order)

For an interval [L, U] and a reference value Δ > 0:

1. **pilot-excludes-SESOI** if U < Δ. Wording: "a gain of Δ or more is not supported by this pilot."
2. **pilot-positive** if L ≥ Δ. Wording: "the pilot estimate is consistent only with gains of at least Δ ; superiority is not established, because this is not a confirmatory test."
3. **pilot-inconclusive** otherwise (L < Δ ≤ U).

Categories 1 and 2 cannot both hold because L ≤ U. None is a superiority, equivalence or "matches" verdict. An interval that includes 0 is never described as a tie.

### Pairing and randomness

Seeds are the same integers across arms. They fix data order and masking draws through dedicated generators shared across arms. Parameter initialisation differs because the predictor architectures differ. Arms are therefore analysed as independent groups (Welch). In N11-A-H, the probe and planning episodes are the same for every arm (see below), but the trained models are still unpaired.

### Failure, exclusion and replacement

- **Infrastructure failure** (crash, out-of-memory, hardware fault, preemption, or a non-finite value traced to a software bug fixed without changing the registered configuration): the run is restarted from scratch with the same seed and configuration, at most twice. All attempts are logged.
- **Training failure** (divergence or non-finite loss under the registered configuration, or still failing after two restarts): the run is not replaced. It is evaluated at its last checkpoint with finite parameters. If there is none, it is evaluated at initialisation with the same protocol.
- No run is excluded or replaced on the basis of its outcome. Restarts and training failures are reported per arm.
- If a bug that affects an arm's registered configuration is found after runs have started, all runs of every affected arm are repeated, and this is reported as a deviation.

### Confirmatory follow-up

Any superiority claim requires a separate, future confirmatory pre-registration with a powered N. Its sample size is computed from the pilot's seed-level variance (the upper one-sided 80 % confidence limit of the pooled SD). The pilot data are not reused in its confirmatory test.

## Registered ImageNet I-JEPA study

### Options fixed by this amendment

- **Pretraining data:** ImageNet-100, the same 100-class subset as N10-A. The registered compute budget (15 runs × about 12 h) already assumes it. ImageNet-1k pretraining is not run.
- **Backbone scale:** ViT-B/16 only. The optional ViT-S/16 second scale (30 runs, 16 comparisons) is not run.
- **CIFAR-100 transfer:** a linear probe on frozen features, with the same recipe as the ImageNet-100 linear probe. The registered text said "fine-tune" without a recipe. *(PI confirmation item 2.)*
- **k-NN:** k = 20, cosine similarity, on the ImageNet-100 val split.

### Primary estimand

θ = mean linear-probe top-1 (GTM predictor) − mean linear-probe top-1 (MLP predictor), ViT-B/16, ImageNet-100 val, in percentage points. This is the contrast named in the registered H0.

- **Interval:** 95 % Welch CI.
- **Reference value (SESOI):** Δ = 1.0 pp, the threshold in the registered H0. It was set on 2026-05-11 as the smallest linear-probe gain worth a Paper 2 section. It was not derived from a power analysis.
- **Category:** by the pilot categories above.

### Secondary contrasts (descriptive)

- The registered 8 contrasts (4 metrics × {MLP, cross-attention}) are each reported with a Welch CI at level 1 − 0.05/8 (99.375 %).
- For all four metrics (linear-probe top-1, k-NN top-1, CIFAR-100 transfer top-1, MI), a positive difference favours GTM.
- The number of intervals entirely above 0, entirely below 0 and including 0 is reported **without a label**.

### Superseded provisions of the locked file

- **Decision criteria** (A'-headline ≥ 6/8 ; A'-tied-stable 3–5 ; A'-loses ≤ 2): not applied. They are replaced by the categories above. The registered A'-tied-stable wording "matches I-JEPA SOTA" is not used for any outcome.
- **Welch's t-test with Bonferroni α = 0.00625 as a verdict:** replaced by the adjusted intervals. p-values are not used to label outcomes.
- **"Optionally 2 backbone scales" and "(or ImageNet-100 subset …, document substitution)":** resolved above.

## N11-A-H sub-study (new, secondary)

### Design

- **Substrate:** the official H-JEPA reference implementation of arXiv:2610.06805, at its latest commit on the date the pre-run supplement is filed.
- **Environment (selection rule, frozen):** the first environment listed in the planning-results table of arXiv:2610.06805 whose reference code exposes ground-truth simulator state at every step. The rule is applied without running any arm.
- **Arms.** Only the predictor between the slow (higher) level and the fast (lower) level is replaced. Everything else follows the reference configuration.
  - **H-JEPA reference:** the predictor as in the reference implementation.
  - **GTM inter-level predictor (treatment):** GTM (Gumbel-softmax hard, alphabet_size = 64, as in the registered N11-A treatment). The slow level is the θ-rate context and the fast level is the γ-rate content. A linear decode maps the GTM code to the target latent, as in the registered N11-A treatment.
  - **MLP inter-level predictor (control):** 2-layer MLP whose parameter count is within ±5 % of the GTM predictor's (adapter included).
- **Seeds:** 0, 17, 42, 73, 101 → 3 arms × 5 seeds = **15 runs**.

### Endpoints (frozen ; one scalar each)

- **P — planning success rate:** the success rate over 200 evaluation episodes under the reference's planning protocol. Start and goal configurations are drawn once with `numpy.random.default_rng(20261023)` and are shared by all arms and seeds.
- **G — physical-state grounding:** one scalar per trained model:
  1. Probe data: 1,000 training and 200 test episodes, collected once with the reference implementation's data-collection policy using `numpy.random.default_rng(20261023)`, and shared by all arms and seeds. The split is by episode.
  2. State variables: every ground-truth state variable the simulator exposes at each step, excluding variables that are constant over the probe data. Each is standardized on the training episodes.
  3. Probe input: the concatenation of all hierarchy levels' frozen latents at step t. For a slower level, the most recent latent at or before t is used. Inputs are standardized on the training episodes.
  4. Probe: ridge regression with α = 1.0, fit on the training episodes. Score: held-out R² per state variable.
  5. **G = the unweighted mean of the held-out R² over the state variables.**

Because each endpoint is a single scalar, the family has 2 endpoints × 2 comparators = **4 contrasts**. Testing levels or variables separately would enlarge the family, and that is not done.

### Estimation

- Each of the 4 contrasts (GTM − reference and GTM − MLP, for P and for G) is reported with a Welch CI at level 1 − 0.05/4 (98.75 %). For both endpoints, a positive difference favours GTM.
- **Reference values (SESOI):** Δ_P = 5 percentage points of success rate ; Δ_G = 0.05 R². These are new values set by this amendment as the smallest differences worth reporting as a hierarchical-timescale result. They are not derived from data. *(PI confirmation item 3.)*
- **Per-endpoint category (mutually exclusive, exhaustive, applied in this order), using the 98.75 % intervals:**
  1. *excludes-SESOI* if the interval against at least one comparator has U < Δ ;
  2. *positive* if the intervals against both comparators have L ≥ Δ ;
  3. *inconclusive* otherwise.

  If L ≥ Δ against both comparators, then U ≥ Δ against both, so 1 and 2 cannot both hold.
- **Conclusions (exhaustive):**

  | P category | G category | Pre-stated conclusion |
  |---|---|---|
  | positive | positive | pilot evidence for both planning and grounding gains over both comparators ; not confirmatory |
  | positive | not positive | planning-only: the planning estimate is not accompanied by grounding evidence, and no grounding claim is made |
  | not positive | positive | grounding-only: latents are better grounded in the pilot, and no planning claim is made |
  | not positive | not positive | no pilot evidence for either gain ; any *excludes-SESOI* endpoint is stated as such |

N11-A-H results never change the reading of the registered ImageNet study. They are reported as a secondary result on hierarchical timescales.

### Pre-run supplement for N11-A-H (required, immutable, part of this registration)

Before the first N11-A-H run, the PI files a dated supplement on OSF as an update to this registration. It may contain only:

1. the H-JEPA reference repository URL and commit ;
2. the environment selected by the rule above, and its list of state variables ;
3. the commit of the GTM / MLP predictor integration ;
4. a compute estimate for the 15 runs and the probe/planning evaluation.

The supplement is filed without running any arm on the selected environment. It cannot be changed after filing, and runs started before it is filed are excluded. If no runnable reference implementation exists, or the estimate does not fit the available hardware, N11-A-H is not run **as a whole** (no partial arms or seeds). This is reported as a deviation and not replaced by an ad-hoc re-implementation.

## Effect on hypotheses

The H0 statement (GTM-predictor I-JEPA exceeds MLP-predictor I-JEPA linear-probe top-1 by ≥ 1.0 pp at ViT-B/16) is kept as the **primary estimand and its reference value**. It is no longer tested for a superiority verdict at 5 seeds. N11-A-H is a new, secondary question, registered before any N11-A-H run. **Flag for the PI:** the reframing changes how every N11-A outcome may be interpreted and worded in Paper 2 §10, and must be filed as a change of interpretation.

## Compute

- Registered study: unchanged, 15 runs × about 12 h ≈ **180 GPU-h** on ImageNet-100. The optional 2-scale variant is not run.
- N11-A-H: estimated in the pre-run supplement (see above). It is run in full or not at all.

## PI confirmation items (to settle before filing)

1. The estimation reframing as worded above (PI decision 2026-10-23).
2. CIFAR-100 transfer by linear probe (registered text said "fine-tune").
3. N11-A-H reference values Δ_P = 5 pp and Δ_G = 0.05 R².

If the PI changes any item, the text above is edited before filing, never after data are seen.

## Not adopted

LeWorldModel ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026-03-13) would be a cheaper single-GPU JEPA world-model substrate. It is not adopted here, so that the registered ImageNet study stays as registered.

## Cross-cutting reporting rule (PAC diagnostics)

No registered or added N11-A metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for a GTM arm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no contrast.

## References

- Zhang, W., Terver, B., Rabbat, M., LeCun, Y., Balestriero, R. (2026). H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning. arXiv:2610.06805.
- Yan et al. (2026). Is Forward Prediction Enough? Physical State Grounding for JEPA World Models. arXiv:2608.06799.
- Zhu, Penachio, Mukherjee, Jonelagadda (2026). Spectral-Target Physical Latent Structuring for JEPA-Style World Models. arXiv:2609.04264.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
