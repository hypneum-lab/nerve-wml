# N11-C — Dreamer V3 with GTM dynamics on Atari-100k

**Date pre-registered:** 2026-05-11 (BEFORE any N11-C implementation)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau11-jepa-wml-stack.md
**Related:** complements N10-C (Procgen multi-agent comm) with single-agent model-based RL leveraging the native nerve-WML world-model framing.
**Status:** Pre-registered, implementation NOT yet started.

## H0 (to refute)

GTM as the **dynamics function** within Dreamer V3's RSSM (Recurrent State-Space Model), replacing the standard MLP transition `f(z_t, h_t, a_t) → (z_{t+1}, h_{t+1})`, improves sample efficiency on the Atari-100k benchmark. Specifically: GTM-dynamics Dreamer V3 normalized human score (mean across 6 representative Atari-100k games) exceeds baseline Dreamer V3 score by ≥10% (Bonferroni-corrected, 5 seeds per game).

## Methodology

### Architecture

Dreamer V3 RSSM (Hafner 2023):
- **Encoder** : convolutional encoder of obs frames → posterior z_t
- **Recurrent dynamics** : `(z_t, h_t, a_t) → h_{t+1}` via GRU
- **Transition predictor** : `(h_t) → ẑ_{t+1}` via MLP (vanilla) — **THIS IS WHAT WE REPLACE**
- **Decoder + reward predictor** : standard Dreamer heads

Replace the **transition predictor** with one of 3 architectures :
- **GTM-dynamics (treatment)** : project (z_t, h_t, a_t) to PSK-multiplexed code via GTM, decode to ẑ_{t+1} via linear adapter
- **MLP-dynamics (control)** : standard Dreamer V3 MLP transition (baseline)
- **IRIS-transformer-dynamics (alternative SOTA)** : Micheli et al. 2023 transformer dynamics for tokenized latent prediction

### Atari-100k benchmark

- Standard Atari-100k protocol : 100k env steps train, evaluate at the end
- 6 representative games (mirrors most papers' subset selection): Pong, Breakout, Asterix, MsPacman, Seaquest, BeamRider
- Per-game normalized score = (agent - random) / (human - random) × 100

### Hyperparameters

- 5 seeds: 0, 17, 42, 73, 101 per game
- 3 architectures × 6 games × 5 seeds = **90 runs**
- Standard Dreamer V3 hyperparams from official codebase

### Statistical analysis

- Per-game Welch's t-test pairwise GTM vs MLP, GTM vs IRIS
- 6 games × 2 baselines × 1 metric (normalized score) = **12 comparisons**
- Bonferroni α = 0.05 / 12 ≈ **0.00417**

## Decision criteria (pre-stated)

- **C'-headline:** GTM-dynamics wins ≥9/12 game-baseline pairs (≥3/4 games against both baselines, Bonferroni-corrected) → Paper 2 §11 (new) leads with "phase-coupled world-model dynamics ; GTM as Dreamer transition function".
- **C'-tied-stable:** 5-8 wins → §11 frames as "convergent evidence ; GTM-dynamics matches SOTA Dreamer/IRIS at fixed sample budget while preserving phase-coupled bio-plausible structure".
- **C'-loses:** ≤4 wins → §11 honest scope : "PAC dynamics does not improve sample efficiency on Atari-100k ; future work on PAC-RL specific architecture (e.g., MuZero-style discrete latent + GTM dynamics)".

## Compute budget

- Per game per seed : Dreamer V3 100k steps ≈ 24h on RTX 4090
- 90 runs × 24h = **2160h ≈ 90 days** sequential. **Not feasible single-GPU.**
- With 4 parallel runs (~6GB VRAM each on a 4090) : ~22 days dedicated.
- Reduced subset : 6 games × 3 archs × 3 seeds = 54 runs × 24h / 4-parallel = ~13 days. **Tight but feasible.**
- Realistic minimum : 6 games × 3 archs × 1 seed = 18 runs = ~5d. Then add seeds for top-2 promising games.

## Risk factors

- **Dreamer V3 is finicky**: official codebase in JAX, porting to PyTorch (per nerve-wml convention) = risk. Mitigate by using `dreamerv3-torch` open-source port (Wang et al.).
- **Compute scale mismatch**: Atari-100k baseline papers often run 5+ seeds per game ; reducing to 1-3 seeds = lower statistical power. Document trade-off.
- **GTM-RSSM interface**: GTM operates on continuous tensors, RSSM has stochastic discrete + deterministic continuous components. Need careful interface design.

## Cross-reference

Reproduction artefacts at `nerve-wml/experiments/dreamer_gtm_dynamics_atari100k/`. Implementation in N11-C sprint. Architecture details in HYPNEUM-PLANS/2026-05-11-n11-architecture-sketch.md. Paper 2 §11 (new) will cite this milestone.
