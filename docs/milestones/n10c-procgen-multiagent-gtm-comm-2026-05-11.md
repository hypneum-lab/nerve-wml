# N10-C — Procgen multi-agent GTM communication channel

**Date pre-registered:** 2026-05-11 (BEFORE any N10-C implementation)
**Spec source:** HYPNEUM-PLANS/2026-05-11-niveau10-scaling-up.md
**Architecture sketch:** HYPNEUM-PLANS/2026-05-11-n10c-architecture-sketch.md
**N9 cross-reference:** GTM design intent = phase-coupled multi-channel multiplexing. N9 toy tasks showed `mi_h` advantage but `bw_eff` collapse ; N10-C tests whether bandwidth-efficiency vs information-density trade-off resolves favourably in a multi-agent communication regime where bandwidth is the binding constraint.
**Status:** Pre-registered, implementation NOT yet started.

## H0 (to refute)

GTM as a multi-agent communication channel in cooperative Procgen environments achieves higher coordinated team reward at fixed bandwidth (16 bits/step/agent) than 3 baseline communication architectures (broadcast MLP, cross-attention ATOC-style, discrete DIAL). Specifically: GTM team reward (median across 5 seeds) exceeds best baseline by ≥10% of dynamic range at 16-bit budget on at least 2 of 3 cooperative Procgen tasks.

## Methodology

### Environments

- **Procgen-coop variant** (built from procgen package): cooperative versions of 3 environments
  - **CoinRun-coop** (custom): 2 agents share level, both must reach coin for reward
  - **Caveflyer-coop** (custom): 2 agents must keep distance ≤ X to share fuel
  - **Maze-coop** (custom): 4 agents must collectively visit all 4 quadrants
- Procgen difficulty: easy (16 levels train, 200 levels eval — standard generalization split)

### Agents

- **PPO multi-agent** (PPO-MA): independent learners with parameter-sharing across agents
- Per-agent network: small CNN trunk (3 conv layers, 32 chan) → 64-dim hidden → comm encoder → comm decoder → policy head (action) + value head
- Comm channel = 1 of 4 architectures
  - **GTM-comm** (this work): GTMBridge encode 64-dim hidden → 16-bit code → broadcast to peers → decode to 64-dim ; concat to peer's hidden ; pre-PPO step
  - **broadcast MLP** baseline: 2-layer MLP encoder + 2-layer MLP decoder ; 16-bit continuous-quantized code
  - **cross-attention** baseline: ATOC-style (Jiang & Lu 2018) ; agents attend to peer hidden states with attention budget
  - **DIAL** baseline (Foerster et al. 2016): 16-bit discrete differentiable communication

### Hyperparameters

- 5 seeds: 0, 17, 42, 73, 101 (Hypneum-canonical)
- 4 archs × 3 envs × 5 seeds = **60 runs**
- Training: 50M Procgen steps per agent (standard for procgen IMPALA / PPO baselines)
- PPO hyperparams: lr=5e-4, clip=0.2, GAE λ=0.95, γ=0.999, batch_size=2048

### Metrics

1. **Team reward** (mean episode return on 200 held-out levels)
2. **Sample efficiency** : steps to reach 50% of max reward
3. **Bandwidth utilization** : actual bits/step actively used (entropy of comm code distribution)

### Statistical analysis

- Welch's t-test pairwise GTM vs each of 3 baselines, per metric, per env
- Total comparisons: 3 baselines × 3 metrics × 3 envs = **27**
- Bonferroni α = 0.05 / 27 ≈ **0.00185**

## Decision criteria (pre-stated)

- **C-headline:** GTM wins ≥18/27 comparisons (≥2/3 metrics across all baselines and ≥2/3 envs) AND team reward gain ≥10% on ≥2 envs → Paper 2 §9 (new) leads with multi-agent comm SOTA claim. NeurIPS oral candidate territory.
- **C-tied-stable:** 9-17 wins → §9 frames as "GTM matches ATOC/DIAL at fixed bandwidth ; PAC multiplexing offers tunable bandwidth-fidelity trade-off" — workshop-level publishable, mainline borderline.
- **C-loses:** ≤6 wins → §9 honest scope : "PAC multiplexing does not transfer to RL multi-agent comm at this bandwidth budget ; design requires task-property analysis".

## Compute budget

- Each Procgen-coop run: 50M steps × ~1ms/step on RTX 4090 = ~14 h ; 60 runs = **840 h** = **35 days** sequential
- With 4 parallel runs (each using ~6GB VRAM, fits 4×6=24GB on 4090) on kxkm-ai dedicated: ~9 days
- Realistic: needs **2-3 weeks dedicated 4090** OR a multi-GPU cluster (kx6tm-23 has no GPU, electron-server has no GPU). Granite 30B blocks during this period.

## Risk factors

- **Procgen-coop variant DOES NOT EXIST off-the-shelf**: 1-2 weeks dev required to implement cooperative variants. Pre-registration assumes this dev shipping pre-sweep.
- **RL training instability**: 5 seeds may be insufficient for high-variance Procgen. Mitigation: pre-allocate compute budget for 10-seed re-run if 5-seed shows wide IQR.
- **Bandwidth measurement subtle**: 16-bit budget enforced via Gumbel-softmax temperature schedule for GTM ; for ATOC/DIAL via attention dropout / discrete sampling. Document the measurement method per arch.

## Cross-reference

Reproduction artefacts at `nerve-wml/experiments/procgen_multiagent_gtm_comm/`. Architecture details in HYPNEUM-PLANS/2026-05-11-n10c-architecture-sketch.md. Paper 2 §9 (new) will cite this milestone.
