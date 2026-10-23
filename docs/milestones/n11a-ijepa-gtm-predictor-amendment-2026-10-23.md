# N11-A — Pre-registration amendment 2026-10-23: hierarchical JEPA arm and physical-state probe

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Amends:** `docs/milestones/n11a-ijepa-gtm-predictor-2026-05-11.md` (pre-registered 2026-05-11, commit `3340440`)
**Date drafted:** 2026-10-23
**OSF registration:** no N11-A-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** no N11-A code or run exists in this repository (no `experiments/ijepa_gtm_predictor/`).

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed.

## Summary of the change

Add a **secondary sub-study, N11-A-H**. It places GTM inside a hierarchical, action-conditioned JEPA world model (H-JEPA) and measures it with a **physical-state probe**. The registered ImageNet I-JEPA study (H0, 3 arms, 8-comparison family, decision labels) is unchanged.

## Rationale

- Zhang, Terver, Rabbat, LeCun, Balestriero, "H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning", [arXiv:2610.06805](https://arxiv.org/abs/2610.06805) (2026-10-05), stacks action-conditioned JEPAs with a distinct timescale per level. GTM is designed around a slow (θ) carrier and fast (γ) content. A hierarchy with explicit per-level timescales is a more direct test of that design than replacing a single-timescale predictor in image-level I-JEPA.
- Yan et al., "Is Forward Prediction Enough? Physical State Grounding for JEPA World Models", [arXiv:2608.06799](https://arxiv.org/abs/2608.06799), and Zhu, Penachio, Mukherjee, Jonelagadda, "Spectral-Target Physical Latent Structuring for JEPA-Style World Models", [arXiv:2609.04264](https://arxiv.org/abs/2609.04264), report that JEPA world-model latents can predict well while remaining poorly grounded in the physical state. Prediction or downstream accuracy alone therefore does not show that a predictor module improves the latent. A probe against ground-truth state is needed.

## Methodology delta

### N11-A-H sub-study (new, secondary)

- **Substrate:** H-JEPA as described in arXiv:2610.06805. One environment with simulator ground-truth state is taken from the environments used there. It is chosen and recorded in a dated note **before any run**, together with the reference implementation commit used.
- **Arms** (only the predictor between the slow, higher level and the fast, lower level is replaced; everything else follows the reference configuration):
  - **H-JEPA reference:** the predictor as in the reference implementation.
  - **GTM inter-level predictor (treatment):** GTM (Gumbel-softmax hard, alphabet_size=64, as in the registered N11-A treatment), with the slow level as the θ-rate context and the fast level as the γ-rate content.
  - **MLP inter-level predictor (control):** 2-layer MLP at the same parameter budget as the GTM predictor.
- **Seeds:** 0, 17, 42, 73, 101 (Hypneum-canonical) → 3 arms × 5 seeds = **15 runs**.
- **Metrics:**
  1. **Planning success rate** on the chosen environment, as defined in arXiv:2610.06805.
  2. **Physical-state probe:** a linear regression from frozen latents (at each hierarchy level) to the simulator's ground-truth state variables, fit on training episodes and scored by R² on held-out episodes. The state variables are listed in the dated note before any run.

### Registered ImageNet study: unchanged

The architecture, pretraining, the 4 evaluation metrics, seeds and the 8-comparison family are as registered. A physical-state probe does **not** apply to the ImageNet study, because ImageNet images carry no ground-truth simulator state. It is not added there.

## Analysis-plan delta

- **Registered family (unchanged):** 4 metrics × 2 baselines = **8** comparisons, Bonferroni α = 0.05/8 ≈ **0.00625**. The A'-headline / A'-tied-stable / A'-loses labels apply to this family only.
- **N11-A-H family (new, separate):** GTM vs {reference, MLP} × 2 metrics = **4** comparisons, Welch's t-test, Bonferroni α = 0.05/4 = **0.0125**. All results are reported in full. N11-A-H outcomes do not change the A' labels. They are reported as a secondary result on hierarchical timescales.

## Effect on hypotheses

H0 (GTM-predictor I-JEPA exceeds MLP-predictor I-JEPA linear-probe top-1 by ≥ 1.0 pp at ViT-B/16) is **unchanged**. No decision threshold changes.

N11-A-H tests a **new, secondary** question: does GTM as an inter-level predictor improve planning success or physical-state grounding over the reference and MLP predictors? Because it is a new question, it is registered here *before* any N11-A-H run. It is not a modification of H0.

## Compute delta

This cannot be estimated reliably before the reference implementation has been read. A compute estimate for 15 runs is recorded in the dated note before the sweep. If the estimate does not fit the available hardware, N11-A-H is dropped or narrowed **before** any run, and the change is reported as a deviation.

## Risk factors (new)

- **Reference availability:** if no runnable H-JEPA reference implementation is available, N11-A-H is not run. This is reported as a deviation, not replaced by an ad-hoc re-implementation.
- **Interface mismatch:** GTM produces discrete PSK codes, and H-JEPA's inter-level predictor operates on continuous latents. The adapter (linear decode from the GTM code to the target latent) is the same pattern as the registered N11-A treatment.

## Not adopted

LeWorldModel ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026-03-13) would be a cheaper single-GPU JEPA world-model substrate. It is not adopted here, so that the registered ImageNet study stays as registered.

## Cross-cutting reporting rule (PAC diagnostics)

No registered or added N11-A metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for a GTM arm, the **preferred phase** is reported alongside it (Keshavarzi, bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)). This rule adds no hypothesis test.

## References

- Zhang, W., Terver, B., Rabbat, M., LeCun, Y., Balestriero, R. (2026). H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning. arXiv:2610.06805.
- Yan et al. (2026). Is Forward Prediction Enough? Physical State Grounding for JEPA World Models. arXiv:2608.06799.
- Zhu, Penachio, Mukherjee, Jonelagadda (2026). Spectral-Target Physical Latent Structuring for JEPA-Style World Models. arXiv:2609.04264.
- Maes, Le Lidec, Scieur, LeCun, Balestriero (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
