# N10-C stage-2 configuration manifest

**Status: DRAFT — not frozen (see freeze rule: committed by a frozen git tag before any run).**

Registration: N10-C amendment of 2026-10-23, configuration manifest (stage 2). Immutable once frozen. Runs started before the freeze tag are excluded from all analyses and reported.

## 1. Commit, integer codes and bus

- Harness commit at manifest time: `70a104a981da5145bb018b8ada7b801a2c4df2e8` (source: N10C_COMMIT env). **The freeze commit must be re-recorded after merge**: this draft is committed in a later commit than the code it describes.
- Code identity independent of git (SHA-256 of each file at manifest time):

| file | SHA-256 |
|---|---|
| `experiments/procgen_multiagent_gtm_comm/coop_env.py` | `9fcf95ea0736f7e7821e4872d587ce1c98f1114506095f1eaa11639bba253873` |
| `experiments/procgen_multiagent_gtm_comm/harness/__init__.py` | `82d6bfc42ee49ac2c985f1463ca97a7cc7f36cb74218ce296702f05be4701e56` |
| `experiments/procgen_multiagent_gtm_comm/harness/bus.py` | `1bb3bc23aec588da8bfb73864215e00ee5a4510324ff3d1045bb0b9af21703c2` |
| `experiments/procgen_multiagent_gtm_comm/harness/codes.py` | `be5208212e135a17315e7f51e7837aacd7be8a1ca5bcce185f49f1b3cccbe1da` |
| `experiments/procgen_multiagent_gtm_comm/harness/comm.py` | `88bc176ab933cde3dfb2849c66b08c8b8dc5244566758a84f6255c3e7904a98d` |
| `experiments/procgen_multiagent_gtm_comm/harness/evaluate.py` | `44a407f539ac212fda7cdb6adfc72a505b9c2a281329322bbbb517de1080b82f` |
| `experiments/procgen_multiagent_gtm_comm/harness/gtm_mlx.py` | `daba415dcf627927c9545501f4c8592448b8e1f4667dc50b0de0459385f92faf` |
| `experiments/procgen_multiagent_gtm_comm/harness/manifest.py` | `dee82e9398966566e27f72b2a29eb400f9b96233aedf77eb7cfb14ad3a7ba388` |
| `experiments/procgen_multiagent_gtm_comm/harness/policy.py` | `b1fc49bcb2b0648ba74aaef2b8d2d6aec1e6a6935ff19651a97251bc756b9826` |
| `experiments/procgen_multiagent_gtm_comm/harness/rollout.py` | `5f291b8c99eed91e1bd62320b842ed2b51af349297da7340b11901e962aa65a5` |
| `experiments/procgen_multiagent_gtm_comm/harness/throughput.py` | `f0c03a4eb62184a114b83c3ba6a839b6542dc067a5728a64580477f234de616f` |
| `experiments/procgen_multiagent_gtm_comm/harness/train.py` | `579d8c473930b9049f719e7c2c553bd16a648a99bf4997f9eea49acbf762042d` |
| `experiments/procgen_multiagent_gtm_comm/harness/vec_env.py` | `81b8b8446916bd7eb5015be472fae5b1bc55aea3c01cfd2f4ac1b6d8a69e2d2a` |
| `experiments/procgen_multiagent_gtm_comm/harness/requirements.lock` | `8609ef54bb4d5879bcd4159fa45624048da05b4e077e15ca2bee748fa09724e8` |

- Files covered: `coop_env.py` (environment), the MLX PPO-MA harness (`harness/`: vec_env, comm, policy, rollout, train, evaluate) and the message bus (`harness/bus.py`).
- `ARM_CODES = {'gtm': 0, 'mlp': 1, 'atoc': 2, 'dial': 3, 'eng': 4, 'nocomm': 5}`
- `CONDITION_CODES = {'registered': 0, 'closed': 1}`
- Run randomness: `numpy.random.SeedSequence([seed, arm_code, cond_code])`, seeds {0, 17, 42, 73, 101, 131} (the five registered seeds and 131, fixed by the amendment's outcome-blind rule), 6 seeds x 10 arm-condition cells = 60 runs. Seed-to-host mapping: macM1: seeds {0, 17, 42}; KXKM-AI: seeds {73, 101, 131} (host fixed per seed; see section 4).
- Bus: all arms send m in {0,1}^16 per agent per step; shape and binary values asserted at evaluation for all arms and in training for all arms except DIAL (soft DRU outputs in training).

## 2. Network, PPO and GTM-adapter sizes (as implemented)

Per agent (parameters shared across the two agents of a pair): IMPALA CNN (16, 32, 32; 64x64x3 input scaled by 1/255) -> 256-d h; Linear(256 -> 64) -> comm.send (16 bits); partner swap; comm.receive(msg, h) -> 64; concat [h 256, recv 64, team summary 2] = 322 -> Linear(322 -> 256) -> ReLU -> policy Linear(256 -> 15) and value Linear(256 -> 1). Inputs: own image, received message, team summary (registered condition only; zeros in the closed condition). Never rewards, done flags or partner actions.

| arm | total params | comm params | IMPALA trunk | to_msg 256->64 | fuse+pi+v |
|---|---:|---:|---:|---:|---:|
| gtm | 734,256 | 8,864 | 622,144 | 16,448 | 86,800 |
| mlp | 727,520 | 2,128 | 622,144 | 16,448 | 86,800 |
| atoc | 737,344 | 11,952 | 622,144 | 16,448 | 86,800 |
| dial | 727,520 | 2,128 | 622,144 | 16,448 | 86,800 |
| eng | 726,480 | 1,088 | 622,144 | 16,448 | 86,800 |
| nocomm | 726,480 | 1,088 | 622,144 | 16,448 | 86,800 |

Comm modules (encoder/decoder sizes):
- mlp: encode Linear(64 -> 16), hard bit = (m > 0), straight-through (sigmoid) in training; decode Linear(16 -> 64).
- atoc: same sender; receiver cross-attends over the 16 received bit tokens: bit embedding 2 x 32, positional embedding 16 x 32, query Linear(256 -> 32), output Linear(32 -> 64).
- dial: encode Linear(64 -> 16); training message sigmoid(m + 2 * N(0, 1)) (noise sigma 2), evaluation hard threshold at 0; decode Linear(16 -> 64).
- eng: no sender parameters; engineered 16-bit message (in-window flag, 5-bit remaining steps, 10-bit clipped right-minus-left count); decode Linear(16 -> 64).
- nocomm: zero 16-bit message; decode Linear(16 -> 64) (floor).
- gtm: see the adapter below; decode Linear(64 + 2 x 4 -> 64).

### GTM symbol adapter

- K = 4 symbols per message from an alphabet A = 16; sender head Linear(64 -> 64) reshaped to [4, 16] logits.
- Training: Gumbel-softmax sample, tau = 1.0, hard one-hot with straight-through gradient; evaluation: argmax symbols (deterministic).
- Bus mapping (ruling R2): each symbol index -> 4-bit binary code, MSB first, giving 4 x 4 = 16 bits.
- Receiver: rebuilds the one-hot from bits, multiplexes the carrier with the GTM (gamma 40.0 Hz, theta 6.0 Hz, sample rate 1000.0 Hz, 4 symbols per theta), adaptive mean-pool of the carrier to 64 samples (pool 64), concatenated with the 8 IQ coordinates of the 4 symbols, then Linear(72 -> 64).
- Constellation: 16-PSK initialisation + 0.01 N(0, 1), trainable; cosine plasticity schedule 1.0 -> 0.1 over the run's total updates (multiplies the constellation gradient).

### PPO and training settings

- lr 0.0005 (lr 5e-4), clip 0.2, GAE lambda 0.95, gamma 0.999, Adam, value coefficient 0.5, entropy coefficient 0.01, max gradient norm 0.5.
- 3 epochs per update, minibatch 1024 pair-steps (= 2,048 agent-samples; on KXKM-AI computed by gradient accumulation over micro-batches, same update), per-minibatch advantage normalisation, 64 parallel coop envs, 128-step rollouts (16,384 agent-samples per update).
- Budget 50,000,000 pair-steps per run (about 6,104 updates); training levels start_level 0, num_levels 16; distribution_mode easy; Delta t = 20. Success measured every 1M pair-steps as the mean over the last 100 completed episodes.
- Non-finite loss or parameter: the run stops and the last finite checkpoint is kept (registered failure rule).

## 3. Scope

Caveflyer-coop and Maze-coop are **not run** under this registration (scope option D). N10-C makes no statement about them. 60 runs on two hosts (30 per host), see section 4.

## 4. Measured throughput, compute estimate and Procgen build record

### Measured training throughput (macM1)

Measured 2026-10-23 20:33:35 CEST; one job at a time, real Procgen CoinRun-coop (64 pairs, 128-step rollouts), policy, message bus and PPO update, untrained start, training levels only. Steady state discards the first 2 update cycles (mx.compile warm-up).

| arm | pair-steps/s (steady) | agent-samples/s | s per update cycle | pair-steps/s incl. warm-up | eval s/model (proxy 2x200) |
|---|---:|---:|---:|---:|---:|
| gtm | 775.8 | 1551.6 | 10.56 | 774.3 | 402 |
| nocomm | 805.6 | 1611.3 | 10.17 | 800.8 | 219 |

Host load (`uptime`) recorded in throughput.json (macM1 is shared with the second-brain VM under OrbStack, so contention is possible):
- gtm: before `20:33  up 6 days, 1 min, 3 users, load averages: 3,36 3,47 5,10`; after `20:37  up 6 days, 5 mins, 3 users, load averages: 10,86 7,61 6,48`
- nocomm: before `20:37  up 6 days, 5 mins, 3 users, load averages: 11,68 8,16 6,72`; after `20:41  up 6 days, 9 mins, 3 users, load averages: 3,34 6,12 6,20`

Evaluation cost is a **proxy**: an untrained policy, intact + shuffled passes over the 16 training levels (levels 0-15), scaled linearly to 2 x 200 episodes. Trained policies finish episodes earlier, and the 200 evaluation levels are different levels, so the real cost differs. Concurrent jobs were not measured.

### Compute estimate including evaluation

Option D on two hosts: 60 runs (36 registered + 24 closed), each 50,000,000 pair-steps, 30 per host. macM1 uses the gtm rate for its 6 gtm runs and the nocomm rate as a proxy for its 24 other runs (mlp, atoc, dial, eng were not throughput-measured). KXKM-AI uses 1290 pair-steps/s (micro_batch_pairs = 512 + clear_cache; bench/kxkm/README.md, PR #53). ALL evaluations run on KXKM-AI (macM1 checkpoints are copied there), about 6.7 min per model.

- macM1: 30 runs, gtm 17.9 h and others (nocomm proxy) 17.2 h per run, total **521 h = 21.7 days**
- KXKM-AI: 30 runs x 10.8 h = **323 h** training, plus 6.7 h for the 60 evaluations
- hosts run in parallel; wall time **521 h = 21.7 days**, set by macm1 (the bottleneck)
- amendment planning figure: macM1 about 17.6 h per run, about 528 h (22 days); KXKM about 324 h

The measured macM1 estimate is within the amendment's planning figure.

### Procgen arm64 build record

- Build patch: `experiments/procgen_multiagent_gtm_comm/bench/procgen-0.10.7-arm64-build.patch`, **SHA-256 `df9bbfa1d454ee51cbea1b8af47fbb1966a79f4a075357ba2a4c95acca834e7f`**.
- Procgen 0.10.7, source commit `5e1dbf3`; installed package version: 0.10.7+5e1dbf3.
- Qt 5.15.19 (Qt 5.15.19 via Homebrew `qt@5`, found through `PROCGEN_CMAKE_PREFIX_PATH`); observed: `unavailable (FileNotFoundError)`.
- clang: `Apple clang version 21.0.0 (clang-2100.1.1.101) | Target: arm64-apple-darwin27.0.0`
- Xcode: `Xcode 26.6 | Build version 17F113`
- SDKROOT: `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk` (documented build value (SDKROOT not set in this shell)); `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`; `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
- Python 3.10.22 (macOS-27.0.1-arm64-arm-64bit); MLX 0.31.1 (runtime 0.31.1); numpy 1.26.4; gym3 0.3.3; gym 0.26.2.
- The patch changes the build configuration only, not game code. This build is used for the macM1 training runs only (seeds 0, 17, 42).

### Hosts, seed-to-host mapping and KXKM-AI record

Host is a block factor fixed per seed; all arms and both conditions of a seed run on that seed's host.

- macM1: seeds {0, 17, 42} (Apple M1 Max, 32 GB, MLX Metal); micro_batch_pairs = none (full minibatch), clear_cache = off, n_workers = 1 (registered per-host implementation setting, `harness/codes.py` HOST_SETTINGS, applied by run_campaign to every run of this host and recorded in each run.json; n_workers 1 is the default, not measured).
- KXKM-AI: seeds {73, 101, 131} (RTX 4090 24 GB, MLX CUDA 12); micro_batch_pairs = 512, clear_cache = on, n_workers = 8 (registered per-host implementation setting, `harness/codes.py` HOST_SETTINGS, applied by run_campaign to every run of this host and recorded in each run.json; n_workers 8 measured +8 % end to end vs 1, bench/kxkm/README.md).
- KXKM-AI software (venv310, recorded 2026-10-23, re-verify at deploy): python 3.10.18, mlx 0.31.1, mlx-cuda-12 0.31.1, numpy 1.26.4, gym3 0.3.3, gym 0.26.2, procgen 0.10.7; `CUDA_HOME` must point at the CUDA headers for the JIT kernels. Version pins: mlx 0.31.1 and mlx-cuda-12 0.31.1 on KXKM-AI; MLX 0.31.1 (Metal) on macM1.
- KXKM-AI Procgen build: the official Procgen 0.10.7 x86-64 wheel `procgen-0.10.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`, PyPI SHA-256 `4b594d14e42f0f2166e59a9e294477b906eb99c90c3343b570b7124cbc865f53`.
- ALL evaluations (200 held-out levels, intact and shuffled passes, cross-seed probe) run on KXKM-AI with the x86-64 build; macM1 checkpoints are copied there. **The two Procgen builds are NOT bit-identical**: on 20 of 20 CoinRun levels tested the frame / reward streams differ between macM1 (arm64) and KXKM-AI (x86-64), although each machine is deterministic across reruns; frames agree early in an episode on some levels and diverge late. Evidence, procedure and result archived in `experiments/procgen_multiagent_gtm_comm/bench/procgen-builds/README.md`; SHA-256 of the archived files:

| file | SHA-256 |
|---|---|
| `README.md` | `15d5eff766f0417b9e8db81b6efb893e70dab6a80aec21714725149de642235c` |
| `frames.py` | `38e945bff44bd1df9474c7a6f77aa5a0a27c241526c483feea7ff0059bb7088d` |
| `frames2.py` | `492e717831ed45b57fe2e707b8e2933120604d8ba96a81dfa47e2e0f1a100370` |
| `frames_kxkm.json` | `2534fcc0df757b2eba397aecb6a1195d6898db195fc7e89cc176d21c80e30df2` |
| `frames_kxkmb.json` | `2534fcc0df757b2eba397aecb6a1195d6898db195fc7e89cc176d21c80e30df2` |
| `frames_macm1.json` | `d3c4bd15db59ac8f049b2b8c10b2c8881fbd60c1a3751a894da3c19206445493` |
| `frames_macm1b.json` | `d3c4bd15db59ac8f049b2b8c10b2c8881fbd60c1a3751a894da3c19206445493` |
| `h_kxkm.txt` | `b038143f2b80649ac856e4eedafc8f78439a6c37c835c7fbc0ea0ac96e501a3b` |
| `h_macm1.txt` | `cc41042fa1e1eb27b62f66ea3a6c73a95a1ea9bd53dbf0384f634c31f488d5a0` |

macM1-trained models are therefore evaluated on the x86 rendering.

### Reproducibility (nondeterminism)

On both Metal and CUDA, MLX 0.31.1 convolution backward in the IMPALA trunk is nondeterministic (Metal about 1e-7 relative, CUDA about 1.4e-5 to 4.5e-5 relative per gradient; initialisation, forward pass and the first rollout are bit-identical; no deterministic switch exists). Runs are statistically, not bitwise, reproducible from the seed; an infrastructure-failure restart with the same seed stream will not retrace the same trajectory. Evidence: `experiments/procgen_multiagent_gtm_comm/bench/kxkm/repeat/` and `bench/kxkm/README.md` (branch feat/n10c-grad-accum, PR #53).

## Smoke runs and development disclosure

7 smoke runs of 200,000 pair-steps each (cap: 3 per arm, 1M steps), training levels only (start_level 0, num_levels 16), smoke seed 9001 (not a registered seed), 64 pairs, 128-step rollouts. Their results informed no registered quantity; outcomes are listed unfiltered.

| arm | condition | outcome | pair-steps done | wall s | pair-steps/s (incl. warm-up) | episodes | success (last 100) | losses finite |
|---|---|---|---:|---:|---:|---:|---:|---|
| gtm | registered | completed | 200000 | 403 | 497 | 329 | 0.02 | True |
| mlp | registered | completed | 200000 | 248 | 808 | 214 | 0.00 | True |
| atoc | registered | completed | 200000 | 250 | 800 | 266 | 0.00 | True |
| dial | registered | completed | 200000 | 248 | 806 | 244 | 0.00 | True |
| eng | registered | completed | 200000 | 249 | 804 | 466 | 0.00 | True |
| nocomm | registered | completed | 200000 | 256 | 782 | 217 | 0.00 | True |
| gtm | closed | completed | 200000 | 255 | 784 | 1304 | 0.13 | True |

Development used only the stub (unit tests with scripted environments), these smoke runs and throughput measurements on untrained policies. Levels 1000-1199 were never touched and no full-length run result was used.

## Artefact archiving (PI decision 2026-10-23)

After each run and each evaluation, the run directory (`run.json`, checkpoints, EvalRecord JSON, cross-play records) is stored on electron-server (Tailscale 100.78.191.52) at `electron@100.78.191.52:/home/electron/archives/n10c/`. KXKM-AI cannot reach electron-server, so electron-server PULLS the completed runs from KXKM-AI (`campaign/pull_archive.sh`, rsync over ssh to `electron@10.2.0.237:/ai/saisail/n10c-run/campaign`, completed runs only, idempotent, logged); macM1 can push with `--archive-to`. The copy is post-hoc storage only and changes no registered quantity. macM1 checkpoints are additionally copied to KXKM-AI for evaluation (campaign `sync` step), and are archived from that KXKM copy. (This item is outside items 1-4 of the amendment's manifest list.)

## Implementation rulings disclosed

- R1: the harness venv (Python 3.10) has no scipy (uv.lock pins scipy 1.17, which needs Python >= 3.11); the analysis code runs in the repo environment.
- R2: GTM message over the 16-bit bus = 4-bit binary code (MSB first) of each of the 4 symbol indices; in training the straight-through one-hot is mapped to bits by a fixed table (value = hard bits, gradient through the softmax), and the receiver rebuilds a differentiable one-hot per symbol as prod_j (b_j if CODE_TABLE[a,j] = 1 else 1 - b_j), exact for hard bits. This gradient path differs from the GTMBridge soft-IQ path of the earlier code.
- R3: train_run takes keyword arguments log_every_pair_steps and ckpt_every_pair_steps (defaults 1,000,000).
- Task 4: the IMPALA trunk is duplicated between policy.py (live) and bench/bench_ppo_mlx.py (frozen archived evidence); never run together.
- Task 6: advantage normalisation is applied per minibatch (as in common Procgen PPO baselines); this is a PPO detail not in the registered list.
- Task 6: time_to_half requires a full window of 100 completed episodes.
- Task 6: the budget is exactly 50,000,000 pair-steps (final rollout truncated and updated on the partial rollout); the training success rate is measured at the exact boundaries k x 1M pair-steps; checkpoints are written after the first update ending at or after each 1M boundary (labelled with the actual step, < 8,192 pair-steps from the boundary) and once at the exact end.
- Task 6: the engineered right-minus-left counter counts only actions the environment applies (an agent that already finished is frozen; its actions are ignored).
- Task 6: GTM and DIAL message noise is stored per pair-step in the rollout and replayed in the PPO update (so the PPO ratio is 1 for an unchanged policy).
- Task 7: evaluation action sampling = one uniform per agent per step (agent 0 then agent 1) from default_rng([20261023, j]), inverse CDF on the float64 softmax; identical across passes and models. Evaluation episodes are capped at 5,000 steps (an error, not an outcome). Evaluation messages are deterministic (GTM argmax symbols).
- R4: gradient accumulation (micro_batch_pairs, clear_cache; PR #53) and n_workers (PR #54) are merged into the harness branch; the freeze commit must contain both. The KXKM-AI settings above refer to them.
- Task 9: smoke runs = 1 per arm in the registered condition plus gtm in the closed condition, 200,000 pair-steps each (7 runs), training levels only. The amendment allows up to 3 per arm of up to 1M steps; more may be added before the freeze within that cap.
