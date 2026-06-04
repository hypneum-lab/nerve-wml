# N11-C dev kickoff — dreamerv3-torch port

**Date pre-registered:** 2026-05-11 06:12 CEST (BEFORE dev start)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau11-jepa-wml-stack.md (N11-C parent prereg)
**Strategy context:** Stratégie B parallel N10+N11 per roadmap 2026-05-11
**Status:** Dev kickoff pre-registered, work NOT yet started

## Goal

Start the dreamerv3-torch port + RSSM-GTM integration dev THIS WEEK (parallel to ongoing N10-A sweep), so the N11-C training sweep is ready when kxkm-ai 4090 has dedicated window post-N10-A/C.

## Scope (kickoff only — NOT the training sweep)

1. **Clone or vendor dreamerv3-torch** repository (Wang et al. open-source PyTorch port of Hafner's official JAX Dreamer V3)
2. **Read RSSM source** to identify transition function (`f(z_t, h_t, a_t) → (z_{t+1}, h_{t+1})`)
3. **Build GTMDynamics module** : replace transition predictor MLP with GTM-multiplexed code path (mirrors N8 T13 GTMBridge pattern : Gumbel-softmax hard quantization, real gtm.forward through constellation)
4. **Smoke test on CartPole** (light env, no Atari yet) : verify Dreamer V3 trains end-to-end with GTM-RSSM, no NaN/divergence
5. **Document the port** : interface contract, hyperparameter mapping, compute estimate refinement

## Where the work lives

- New repo dir : `nerve-wml/experiments/dreamer_gtm_dynamics_atari100k/`
- Files :
  - `architectures/gtm_dynamics.py` (replaces RSSM transition)
  - `dreamerv3_port/` (vendored dreamerv3-torch sources, possibly as git submodule)
  - `cartpole_smoke.py` (light smoke runner)

## Compute for kickoff

- **dreamerv3-torch port reading + GTMDynamics build** : grosmac local (no GPU needed for code)
- **CartPole smoke** : grosmac CPU or kxkm-ai CPU (small env, ~10-30min)
- **NOT this kickoff** : Atari-100k training (~13-22d dedicated 4090, parent N11-C scope)

## Decision criteria for kickoff success

- **N11-C-dev-validated** : Dreamer V3 trains end-to-end on CartPole with GTM-RSSM, achieves ≥reward 100 within 50k env steps (typical CartPole convergence), no NaN/divergence. Greenlight Atari-100k sweep (parent N11-C prereg).
- **N11-C-dev-blocker** : training diverges OR GTM-RSSM produces NaN OR dreamerv3-torch port has incompatibilities with our PyTorch 2.11 + Python 3.14 stack. Escalate : either fix the port OR pivot to alternative WML benchmark (e.g., IRIS Micheli 2023 instead of Dreamer V3).

## Effort + timeline

- Dev port + RSSM-GTM integration : **~1-2 sem grosmac local**
- CartPole smoke : ~1 day
- Total kickoff : **~2-3 sem**, completion target 2026-06-01

## Risk factors

- **dreamerv3-torch may not support PyTorch 2.11** : compatibility check is first step. Pin specific commit if needed.
- **RSSM integration complexity** : stochastic discrete latent + deterministic continuous hidden + GRU. GTM operates on continuous tensors via Gumbel-quantization. Interface design may require multiple iterations.
- **CartPole reward sparse** : default RNN-based Dreamer struggles on dense-reward toy envs. Use Pong or LunarLander as smoke if CartPole insufficient.

## Cross-reference

- Parent prereg : `HYPNEUM-PLANS/preregistrations/n11c_dreamer_gtm_dynamics_atari100k.md`
- Architecture sketch : `HYPNEUM-PLANS/2026-05-11-n11-architecture-sketch.md` §2 (N11-C Dreamer V3 + GTM dynamics)
- Roadmap : `HYPNEUM-PLANS/2026-05-11-roadmap-Q2-Q3-2026.md`
