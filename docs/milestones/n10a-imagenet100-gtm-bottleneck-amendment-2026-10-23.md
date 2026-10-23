# N10-A — Pre-registration amendment 2026-10-23: oscillator-family baseline

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI)
**Amends:** `docs/milestones/n10a-imagenet100-gtm-bottleneck-2026-05-11.md` (pre-registered 2026-05-11, commit `1640c5c` ; in-file amendment 2026-05-11, commit `c74f6be`)
**Date drafted:** 2026-10-23
**OSF registration:** no N10-A-specific OSF identifier is recorded in this repository. The programme-level OSF pre-registration is [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN) (dreamOfkiki, locked 2026-04-19).
**Data status at drafting:** the ImageNet-100 sweep has not started. The only N10-A run so far is a pipeline smoke test on a CIFAR-10 / frozen ResNet18 stand-in, GTM arm only, 5 epochs (`experiments/imagenet100_gtm_bottleneck/smoke_results.json`). No baseline arm has been trained and no ImageNet-100 result exists.

The locked pre-registration file is not modified. This amendment is append-only and takes effect only once filed.

## Summary of the change

Add one **oscillator-family bottleneck** as a fourth comparator, reported in a **separate comparison family**. The registered H0, the 45-comparison headline family, its Bonferroni α and the A-headline / A-tied-stable / A-loses thresholds are unchanged.

## Rationale

Two oscillator architectures published after the 2026-05-11 lock show that phase-based inductive biases can be competitive at ImageNet scale:

- Xiao, Wang, Han, Shan, Li, "Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency" (KoPE), ICML 2026, [arXiv:2604.07904](https://arxiv.org/abs/2604.07904). It adds a Kuramoto phase state to Vision Transformers and reports data and parameter efficiency gains.
- Dai & Song, "Winfree Oscillatory Neural Network" (WONN), [arXiv:2605.20922](https://arxiv.org/abs/2605.20922). It reports a synchronization-based oscillator network that is competitive on ImageNet-1K.

The registered baselines (VQ-VAE, linear projection, β-VAE β=1.0) contain no phase or oscillator component. If GTM beats them, the result cannot tell "phase–amplitude-coupled multiplexing helps" apart from "any phase inductive bias helps". An oscillator-family comparator is needed to make that distinction. Within the repo, AKOrN was the only oscillator comparator so far (toy tasks, `docs/superpowers/research/2026-05-20-akorn-*.md`).

## Methodology delta

### New arm

- **Oscillator bottleneck (KoPE-style, primary choice):** `nn.Linear(2048, code_dim)` drives `code_dim / 2` Kuramoto phase oscillators with learned coupling. The phases are integrated for a fixed number of steps and read out as `[cos θ, sin θ]`, which gives a `code_dim`-dimensional code. This is followed by `nn.LayerNorm` and the same `nn.Linear(code_dim, 100)` classifier head as all other arms (head parity kept).
- This arm is a KoPE-*style adaptation* to a feature bottleneck on frozen ResNet50 features. It is **not** a reproduction of the published KoPE ViT model, and results must not be reported as such.
- **Fallback (WONN-style Winfree coupling):** used only if the KoPE-style module cannot be trained stably in the smoke stage, which runs before any ImageNet-100 run. The choice between KoPE-style and WONN-style is made and recorded in a dated note before the sweep. It is never made after seeing ImageNet-100 results, and only one oscillator arm enters the analysis.
- The number of integration steps, step size and coupling initialisation are taken from the reference implementation defaults where they exist. Otherwise they are fixed in the same dated note before the sweep and are not tuned on the validation split. The oscillator arm gets no more hyperparameter search than the registered baselines: the same optimizer, lr, weight decay, epochs, batch size and schedule.

### Unchanged

Backbone (frozen ResNet50), dataset (ImageNet-100), preprocessing, code_dim sweep {32, 64, 128}, seeds {0, 17, 42, 73, 101}, the 5 metrics, and the three registered baselines.

## Analysis-plan delta

- **Headline family (unchanged):** GTM vs {VQ-VAE, linear projection, β-VAE β=1.0} × 5 metrics × 3 code_dims = **45** comparisons, Bonferroni α = 0.05/45 ≈ **0.00111**. The A-headline / A-tied-stable / A-loses rules apply to this family only.
- **Oscillator family (new, separate):** GTM vs oscillator bottleneck × 5 metrics × 3 code_dims = **15** comparisons, Welch's t-test, Bonferroni α = 0.05/15 ≈ **0.00333**. Wins, losses and ties are reported in full.
- **Pre-stated reading of the oscillator family:**
  - GTM wins ≥ 10/15: any headline-family advantage may be described as specific to PAC multiplexing relative to a generic phase bias, at this scale.
  - GTM wins 5–9/15: mixed. The PAC-specific reading is reported as not established.
  - GTM wins ≤ 4/15: any A-headline or A-tied-stable advantage is described as consistent with a generic phase/oscillator inductive bias, **not** as PAC-specific.

## Effect on hypotheses

H0 (GTM top-1 at code_dim=32 exceeds VQ-VAE by ≥ 1.5 pp) is **unchanged**. The decision thresholds are unchanged.

**Flag for the PI:** the "pre-stated reading" above constrains how the A-headline and A-tied-stable outcomes may be *worded* in Paper 2 §8. It does not change the labels or their thresholds, but it is an addition to the interpretation rules and should be filed as such.

## Compute delta

+1 architecture × 3 code_dims × 5 seeds = **15 runs** on cached features, plus one OOD evaluation. Oscillator integration makes each run slower than a linear bottleneck. The estimate is about 5–8 h on the RTX 4090, against the registered total of about 22 h.

## Cross-cutting reporting rule (PAC diagnostics)

No registered N10-A metric is a phase–amplitude coupling index. If an exploratory PAC diagnostic based on the Tort modulation index is reported for the GTM arm, the **preferred phase** is reported alongside it. Keshavarzi (bioRxiv 2026, [doi:10.64898/2026.06.05.730524](https://doi.org/10.64898/2026.06.05.730524)) shows that the modulation index does not distinguish a 180° inversion of the slow phase. This rule adds no comparison to either family.

## Noted, not resolved here

The 2026-05-11 in-file amendment lists the three baselines as "VQ-VAE, β-VAE β=1.0, dense autoencoder", while the original methodology lists "linear projection" (`nn.Linear(2048, code_dim)` + `nn.LayerNorm`) as the third baseline. This amendment refers to the third baseline as in the original methodology. The PI should state which one is registered when filing.

## References

- Xiao, M., Wang, Y., Han, D., Shan, C., Li, D. (2026). Kuramoto Oscillatory Phase Encoding: Neuro-inspired Synchronization for Improved Learning Efficiency. ICML 2026. arXiv:2604.07904.
- Dai, J., Song, Y. (2026). Winfree Oscillatory Neural Network. arXiv:2605.20922.
- Keshavarzi, M. (2026). Modulation index-based phase–amplitude coupling does not encode temporal polarity. bioRxiv. doi:10.64898/2026.06.05.730524.
