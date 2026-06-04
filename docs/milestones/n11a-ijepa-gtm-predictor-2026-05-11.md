# N11-A — I-JEPA + GTM predictor on ImageNet self-supervised

**Date pre-registered:** 2026-05-11 (BEFORE any N11-A implementation)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau11-jepa-wml-stack.md
**Related:** complements N10-A (supervised ImageNet bottleneck) with self-supervised representation learning.
**Status:** Pre-registered, implementation NOT yet started.

## H0 (to refute)

GTM as the **predictor module g_φ** in I-JEPA (Assran et al. 2023) replacing the vanilla MLP predictor produces superior downstream representation quality measured by ImageNet linear probe top-1 accuracy after self-supervised pretraining. Specifically: GTM-predictor I-JEPA at standard ViT-B/16 scale exceeds vanilla MLP-predictor I-JEPA top-1 by ≥1.0 percentage point (Bonferroni-corrected, 5 seeds).

## Methodology

### Architecture

I-JEPA has 3 components per the original paper:
- **Context encoder f_θ**: ViT-B/16, processes context patches of an image
- **Target encoder f_ξ**: EMA of f_θ, processes target patches (non-overlapping)
- **Predictor g_φ**: small transformer (default in I-JEPA paper) that predicts target embeddings from context embeddings + positional masks

We replace **g_φ** with one of 3 architectures (control + treatment + alternative) :
- **GTM-predictor (treatment)**: encode context embedding to PSK-multiplexed code via GTM (Gumbel-softmax hard, alphabet_size=64), decode to target embedding shape
- **MLP-predictor (control)**: standard 2-layer MLP, same parameter budget
- **Cross-attention-predictor (alternative SOTA)**: V-JEPA-2 style cross-attention predictor

### Pretraining

- ViT-B/16 backbone, frozen patch embeddings
- ImageNet-1k pretraining 100 epochs (or ImageNet-100 subset for compute fit, document substitution)
- Batch=512, AdamW lr=1.5e-4 cosine schedule, weight_decay=0.05
- Standard I-JEPA augmentations + masking strategy

### Evaluation

- Linear probe on ImageNet val (1000 classes if pretrained on -1k, 100 classes if -100)
- k-NN classification (k=20) on val
- Transfer to CIFAR-100 fine-tune
- Compute MI(z_predicted, ground_truth_label) on a held-out subset (mirrors N8-Q1 mi_h metric)

### Hyperparameters

- 5 seeds: 0, 17, 42, 73, 101 (Hypneum-canonical)
- 3 architectures × 5 seeds = **15 runs**
- Optionally 2 backbone scales (ViT-S/16, ViT-B/16) × 5 seeds × 3 archs = 30 runs

### Statistical analysis

- Welch's t-test pairwise GTM vs MLP, GTM vs cross-attention
- 4 metrics × 2 baselines = **8 comparisons** at 1 backbone scale, **16** at 2 scales
- Bonferroni α = 0.05 / 8 ≈ **0.00625** (or 0.003125 for 2-scale)

## Decision criteria (pre-stated)

- **A'-headline:** GTM-predictor wins ≥6/8 comparisons (≥3/4 metrics across both baselines, Bonferroni-corrected) → Paper 2 §10 (new) leads with "phase-coupled prediction in self-supervised representation learning ; GTM as JEPA predictor module".
- **A'-tied-stable:** 3-5 wins → §10 frames as "convergent evidence ; GTM matches I-JEPA SOTA predictor architectures while preserving phase-coupled biological plausibility".
- **A'-loses:** ≤2 wins → §10 honest scope : "PAC predictor does not transfer from toy/supervised to self-supervised regime ; future work on JEPA-specific GTM variant".

## Compute budget

- ImageNet-1k pretraining 100 epochs ViT-B/16 with I-JEPA = ~80h on A100 per published baseline ; ~50-80h on RTX 4090 (Granite 30B must yield VRAM).
- ImageNet-100 subset (10×reduced data) = ~10-15h per run.
- 15 runs × 12h average = **180h ≈ 7.5d** dedicated 4090.
- 30 runs (2-scale) = ~15d.
- Fits Q1 2026 timeline if Granite yields full VRAM during sprint.

## Risk factors

- **JEPA hyperparameter sensitivity**: I-JEPA is finicky (mask sampling, EMA momentum schedule). Dev iteration overhead is real ; budget 1 sem dev before sweep.
- **GTM as JEPA predictor**: novel architectural composition, no prior literature. Risk of training instability ; mitigate with extensive smoke at smaller scales (ViT-T/16 stand-in).
- **VRAM contention**: Granite 30B occupies 19GB, leaves 5GB. JEPA pretraining at ViT-B/16 needs 16-20GB VRAM. **Must evict Granite during sprint.**

## Cross-reference

Reproduction artefacts at `nerve-wml/experiments/ijepa_gtm_predictor/`. Implementation in N11-A sprint. Paper 2 §10 (new) will cite this milestone.
