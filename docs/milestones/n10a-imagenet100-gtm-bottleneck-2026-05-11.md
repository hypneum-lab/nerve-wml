# N10-A — ImageNet-100 with ResNet50 + GTM bottleneck

**Date pre-registered:** 2026-05-11 (BEFORE any N10-A implementation)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau10-scaling-up.md
**N9 cross-reference:** Q1/Q1+/Q1++ all `tied` on toy tasks (XOR/FlowProxy/N=2,16). N10-A escalates to ImageNet subset to test community-respectability.
**Status:** Pre-registered, implementation NOT yet started.

## H0 (to refute)

GTM as a feature-bottleneck between a frozen ResNet50 backbone and a learned linear classifier on ImageNet-100 (100-class subset) provides superior representation quality at low bottleneck dimensionality (code_dim ≤ 64) compared to vanilla VQ-VAE, linear projection, and β-VAE bottlenecks. Specifically: GTM top-1 accuracy at code_dim=32 exceeds VQ-VAE top-1 by ≥1.5 percentage points (Bonferroni-corrected, 5 seeds).

## Methodology

### Architecture

- **Backbone**: torchvision ResNet50 pretrained on ImageNet-1k, FROZEN (no fine-tuning of backbone)
- **Bottleneck**: 1 of 4 architectures
  - **GTMBridge** (this work): same Gumbel-softmax hard wire-through as N8 T13 ; PSK alphabet_size=64, code_dim sweep
  - **VQ-VAE** baseline: standard codebook + commitment loss + EMA codebook update (van den Oord 2017)
  - **Linear projection** baseline: nn.Linear(2048, code_dim) + nn.LayerNorm
  - **β-VAE** baseline: continuous Gaussian bottleneck with KL penalty β ∈ {0.1, 1.0}
- **Classifier head**: nn.Linear(code_dim, 100) — same architecture for all bottlenecks (parity)

### Dataset

- **ImageNet-100**: 100 classes randomly drawn from ImageNet-1k via standard split (e.g., the 100-class subset used in CMC, Tian et al. 2020). Train ~130k images, val ~5k.
- Standard ImageNet preprocessing: resize 256, center crop 224, normalization.

### Hyperparameters

- code_dim sweep: {32, 64, 128} (3 settings)
- 5 seeds per setting per architecture: 0, 17, 42, 73, 101
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Training: 100 epochs, batch=128, cosine LR schedule
- Backbone forward computed once + cached as features (saves ~95% compute)

### Metrics

1. **top-1 accuracy** on ImageNet-100 val
2. **top-5 accuracy**
3. **MI(code, label)** estimated via empirical histogram (extension of N8 T13 method to 100-class)
4. **OOD robustness**: top-1 on 5 ImageNet-C subset perturbations (gaussian_noise, glass_blur, contrast, fog, jpeg_compression) at severity 3
5. **bandwidth efficiency**: effective rank at 95% var of code distribution (mirrors N8/N9 bw_eff)

### Statistical analysis

- Welch's t-test pairwise GTM vs each of 3 baselines, per metric, per code_dim
- Total comparisons: 3 baselines × 5 metrics × 3 code_dims = **45**
- Bonferroni α = 0.05 / 45 ≈ **0.00111**

## Decision criteria (pre-stated)

- **A-headline:** GTM wins ≥30/45 comparisons (≥2/3 metrics across all baselines and code_dims, Bonferroni-corrected) → Paper 2 §8 (new) leads with ImageNet-100 + GTM-bottleneck SOTA claim.
- **A-tied-stable:** 15-29 wins → §8 frames as "GTM matches VQ-VAE/β-VAE as bottleneck while preserving phase-coupled semantics ; advantage emerges at low code_dim".
- **A-loses:** ≤10 wins OR ≥30 losses → §8 reframes as honest scope limitation : "GTM as feature-bottleneck does not generalize from toy tasks to natural images at this scale ; PAC advantage is regime-specific".

## Compute budget

- Backbone feature extraction (one-time): ~30 min on RTX 4090
- Cached features bottleneck training: 4 archs × 3 code_dims × 5 seeds = **60 runs** × ~20 min/run = **20 h**
- OOD eval: ~30 min per arch × 4 = 2 h
- **Total**: ~22 h on kxkm-ai RTX 4090. Fits in ~1 day.

## Risk factors

- **VQ-VAE adjacency**: reviewer will demand explicit comparison + ablation "does GTM differ from VQ-VAE in more than PSK init?". Need ablation: GTM with random codebook init (no PSK) to isolate the PSK contribution.
- **Pretrained backbone bias**: ResNet50 features are already discriminative ; bottleneck advantage may be small. Mitigation: also test with a less-discriminative backbone (e.g., ImageNet-1k MoCo v3 self-supervised) as sensitivity analysis.
- **OOD severity choice**: severity 3 is the standard in literature ; if results vary wildly with severity, document the dependency.

## Cross-reference

Reproduction artefacts will live at `nerve-wml/experiments/imagenet100_gtm_bottleneck/`. Implementation in N10-A sprint (post-N9 closeout). Paper 2 §8 (new) will cite this milestone.

## Pre-registration amendment 2026-05-11

The original pre-registration above lists "β-VAE baseline: continuous
Gaussian bottleneck with KL penalty β ∈ {0.1, 1.0}" as a single
baseline, but the Bonferroni count (3 baselines × 5 metrics × 3
code_dims = 45 comparisons, α_corrected = 0.05/45 ≈ 0.00111) treats
β-VAE as one entry. With two β values it would actually be 4
baselines / 60 comparisons / α ≈ 0.000833, which is inconsistent
with the registered count.

**Resolution (locked before any sweep starts):**

- Fix β-VAE at **β = 1.0** (standard ELBO weighting, one baseline
  entry).
- Total baselines remain **3** (VQ-VAE, β-VAE β=1.0, dense
  autoencoder), comparisons remain **45**, corrected α remains
  **0.00111**.
- The β=0.1 variant is dropped from N10-A pre-registration scope ;
  it may reappear later as an exploratory sensitivity analysis,
  reported separately and not credited toward the headline verdict.

This amendment is append-only ; the original pre-registration above
is preserved verbatim.
