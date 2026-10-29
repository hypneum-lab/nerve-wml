# N10-C stage-2 configuration manifest

**Status: FROZEN (tag n10c-freeze-20261029).**

Registration: N10-C amendment of 2026-10-23 as amended on 2026-10-25 (training fix C, section 5), configuration manifest (stage 2). Immutable once frozen. Supersedes the manifest of tag n10c-freeze-20261024, whose campaign was stopped and is excluded. Runs started before the freeze tag are excluded from all analyses and reported.

## 1. Commit, integer codes and bus

- Harness commit at manifest time: `b9aabefb3362d134818b3a06507c6d8189781fd2` (source: git). **The freeze commit must be re-recorded after merge**: this draft is committed in a later commit than the code it describes.
- Code identity independent of git (SHA-256 of each file at manifest time):

| file | SHA-256 |
|---|---|
| `experiments/procgen_multiagent_gtm_comm/coop_env.py` | `9fcf95ea0736f7e7821e4872d587ce1c98f1114506095f1eaa11639bba253873` |
| `experiments/procgen_multiagent_gtm_comm/harness/__init__.py` | `82d6bfc42ee49ac2c985f1463ca97a7cc7f36cb74218ce296702f05be4701e56` |
| `experiments/procgen_multiagent_gtm_comm/harness/bus.py` | `7f474f51af0583497f76f5d6f00cbe5b8ee778abb22c2b49b2b51f6cf06dc75d` |
| `experiments/procgen_multiagent_gtm_comm/harness/codes.py` | `0105fb7529d2de0c7f64775cc82dd176cf1a5fd3632e427cc4fa850a48c233d0` |
| `experiments/procgen_multiagent_gtm_comm/harness/comm.py` | `bee074c91c7c4176c38cd9557b330ce287d68a72ef76bf56e2ded9a7da654328` |
| `experiments/procgen_multiagent_gtm_comm/harness/devledger.py` | `118a19687fb8e2cc8b7a235c5b1c69c0b407e4b103e55c3a255d2f14ddd570f6` |
| `experiments/procgen_multiagent_gtm_comm/harness/evaluate.py` | `ee688cbcbc980cdf822ae81c62965a006180db0035e8cea56afff6de6b520f51` |
| `experiments/procgen_multiagent_gtm_comm/harness/gate_report.py` | `953de7122f1945575a2b0045140005d93171f3a102f7a4989e509f04f67be2e1` |
| `experiments/procgen_multiagent_gtm_comm/harness/gtm_mlx.py` | `daba415dcf627927c9545501f4c8592448b8e1f4667dc50b0de0459385f92faf` |
| `experiments/procgen_multiagent_gtm_comm/harness/manifest.py` | `ff4f3260ce8153895054d01b5bd68c896b4b505088fbd997a24fa5b18cf3e57d` |
| `experiments/procgen_multiagent_gtm_comm/harness/options.py` | `e668fe7cae6e84ff4f9ef63391f610e8055270f16419578f60b5ebfdc504763f` |
| `experiments/procgen_multiagent_gtm_comm/harness/policy.py` | `924eba38deadad2353a79b7447329b7eea4db9790e4a30b0292db94b1c8e4099` |
| `experiments/procgen_multiagent_gtm_comm/harness/pretrain.py` | `7e95be33e98bcbdbd8187d291f1a3aa81fd2654a3a98d1fbcc13c33f350e38ad` |
| `experiments/procgen_multiagent_gtm_comm/harness/rollout.py` | `fc7e60ac4922c25418af14daacc8877cbcb5b42065a194cdb4f46eb00de26671` |
| `experiments/procgen_multiagent_gtm_comm/harness/throughput.py` | `6df48d3625eff6dcfcef8c14924c2a674981f326f32221c5d916c86ea5cd49af` |
| `experiments/procgen_multiagent_gtm_comm/harness/train.py` | `7e67d04c61f3988f2709db59ee8fde617712e15b29eae0111c7b81d53d02eba5` |
| `experiments/procgen_multiagent_gtm_comm/harness/vec_env.py` | `ce8c9a0903ce82410c01f8a69a280d6459c3287e3f24e31a04376a1cf340b28b` |
| `experiments/procgen_multiagent_gtm_comm/harness/requirements.lock` | `8609ef54bb4d5879bcd4159fa45624048da05b4e077e15ca2bee748fa09724e8` |

- Files covered: `coop_env.py` (environment), the MLX PPO-MA harness (`harness/`: vec_env, comm, policy, rollout, train, evaluate) and the message bus (`harness/bus.py`).
- `ARM_CODES = {'gtm': 0, 'mlp': 1, 'atoc': 2, 'dial': 3, 'eng': 4, 'nocomm': 5, 'gtm_mis': 6, 'eng_gtmrx': 7, 'gtm_warm': 8, 'vq': 9}`
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
- Bus mapping (ruling R2): each symbol index -> 4-bit binary code, MSB first, giving 4 x 4 = 16 bits (validated on the bus).
- Receiver: takes the straight-through one-hot directly in training (rebuilt from the 16 bits at evaluation), multiplexes the carrier with the GTM (gamma 40.0 Hz, theta 6.0 Hz, sample rate 1000.0 Hz, 4 symbols per theta), adaptive mean-pool of the carrier to 64 samples (pool 64), concatenated with the 8 IQ coordinates of the 4 symbols, then Linear(72 -> 64).
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

Option D on two hosts: 60 runs (36 registered + 24 closed), each 50,000,000 pair-steps, 30 per host. macM1 uses the gtm rate for its 6 gtm runs and the nocomm rate as a proxy for its 24 other runs (mlp, atoc, dial, eng were not throughput-measured). KXKM-AI uses 1290 pair-steps/s (micro_batch_pairs = 512 + clear_cache; bench/kxkm/README.md, PR #53). ALL evaluations run on KXKM-AI (macM1 checkpoints are copied there), about 6.7 min per model, plus the cross-seed cross-play probe: 10 cells x 6 fixed pairs = 60 passes at 0.5 of one model evaluation each (amendment figure; one intact pass instead of the intact + shuffled pair).

- macM1: 30 runs, gtm 17.9 h and others (nocomm proxy) 17.2 h per run, total **521 h = 21.7 days**
- KXKM-AI: 30 runs x 10.8 h = **323 h** training, plus 6.7 h for the 60 evaluations and 3.4 h for the 60 cross-play passes (KXKM total **333 h**)
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

## 5. Amendment 2026-10-25: registered training configuration (fix C)

Amendments (SHA-256 at manifest time):

| file | SHA-256 |
|---|---|
| `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md` | `0114dac7e2777b2d02c12bcb5ff4c73cdd9df08e8df516d60497eefa54eed59b` |
| `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` | `dfb3b96f15376a5bbfd2a7f5cc1ea7d8d9e29afb0dde1492f94e2f4a6ec618e7` |

- Registered preset `fix-c-coin-pretrain` (`campaign.guards.REGISTERED_TRAIN_OPTIONS`): coin_reward = 0.5, adam_eps = 1e-05, mask_finished = True, liveness = abort, liveness_every = 100000, pretrained_trunk = True, adam_bias_correction = False. Adam bias correction off (PI decision, amendment section 4.3); the bias-corrected preset is development-only.
- Liveness guard: probe on 128 pair-steps at step 0 and after the first update ending at or after every 100,000 pair-steps; flagged if the block-3 positive fraction or the live-h fraction is below 0.01 or a std is <= 1e-06; abort mode. A first abort restarts the run once from scratch (`MAX_LIVENESS_RESTARTS = 1`, counted apart from the infrastructure restarts); a second abort is a training failure, evaluated at its last checkpoint.
- Trunk pre-training: single-agent CoinRun, training levels 0-15, 1,000,000 steps; shared trunk of seed s from `SeedSequence([s, 99])`, own trunk of a control run from `SeedSequence([101, 99, arm_code, condition_code])`; pre-freeze registered procedure on the seed's fixed host with its settings (ledger kind `pretrain_registered`; an unusable trunk gets one restart, then the PI decides). The trunk files are committed under `manifest/trunks/` (PI decision 2026-10-28).

### Pinned trunks

| trunk | host | procedure commit | SeedSequence | trunk SHA-256 | pretrain.json SHA-256 | coin rate first → 1M | wall s |
|---|---|---|---|---|---|---|---:|
| `seed-0` | macm1 | `1381a389cff6` | `[0, 99]` | `9d2a2acf83c065bdb10e003eb284429b75587d18aa0a27b149ff02eb8f4b4389` | `e48c1a2c2fb80c66ece656ffebe8498b76fb65dd1d5bd8905a84d351cc0d30f6` | 0.44 → 0.52 | 656 |
| `seed-17` | macm1 | `1381a389cff6` | `[17, 99]` | `807e9fff9c1f62854a8ab2663eda01b58a188b437c4b6271b37abd911d8769b6` | `e5e3c920f544c770192b8b918361ed13d8c90f707623438a982884442488aa91` | 0.45 → 0.89 | 652 |
| `seed-42` | macm1 | `1381a389cff6` | `[42, 99]` | `41700e7c67c87c74f7f05b9e54d3ac84e424a5e4d9f8e7d863d5e878f8069594` | `dc84f83ad0518cf0ef0c6a164fe3542a3f505ee995f779870050848b04e0bc2b` | 0.43 → 0.82 | 652 |
| `seed-73` | kxkm | `1381a389cff6` | `[73, 99]` | `2f6daec72c44c46b9a985515f8d53626542517bbf29f34a69731b93956edb013` | `71ab4dcfc231ee1020b8fb44794abfbddd56bc505719a8da32e8c57d0114defa` | 0.45 → 0.85 | 375 |
| `seed-101` | kxkm | `1381a389cff6` | `[101, 99]` | `a7ed1a66ed823bfa66a0d0f3a5f292d71f80e548a1438d67a42e37c7dd6e014f` | `c842b754e2bf8ea8eb8fb583fa257ba6bbca202a571739e8d692b253ee8c2172` | 0.56 → 0.95 | 374 |
| `seed-131` | kxkm | `1381a389cff6` | `[131, 99]` | `6f6998da8bff141ab27c3c69dfefcbcb294b6289147391335150456ec9a9b77d` | `09f05e2640855e9957c177e3a2dffcae388c11aab0d3b385b56698d03b2cfdf4` | 0.36 → 0.9 | 374 |
| `seed-101-registered-mlp` | kxkm | `1381a389cff6` | `[101, 99, 1, 0]` | `c52a0936d97180a250d7c4d28cafaec4a3895bf6c0e58b842196b808a79bd72c` | `3d8c1056b7c2a1987be7f80cb61209f9c86344e8ee5bdb0143cd94431ca21226` | 0.55 → 0.82 | 374 |
| `seed-101-registered-atoc` | kxkm | `1381a389cff6` | `[101, 99, 2, 0]` | `0a08208a7ee104f53e9f9e92858a07e7ea6e4bef0192bc3677541d7e6e106125` | `b1554551590450c2fee76668c91010664d2dea97da41a3cb24f7f98c021bd595` | 0.43 → 0.91 | 374 |
| `seed-101-registered-gtm` | kxkm | `1381a389cff6` | `[101, 99, 0, 0]` | `60959f0784a89609fe2ab40836cac5bca03d2265f09a07d0a57737ad39bf925c` | `499d994c53f55adf790955fe6abfb024ed05afacc7346b0a0f1ce7aa778764aa` | 0.56 → 0.93 | 373 |
| `seed-101-closed-gtm` | kxkm | `1381a389cff6` | `[101, 99, 0, 1]` | `ea4c865765eee8710222022da97a1db62094336cb8f6cc1f158689ec9f3f828e` | `98fb0e23141c4c90b2e6e00e752f864d5b7eb14b26e32ef5313eadc22391a714` | 0.37 → 0.62 | 375 |
| `seed-101-registered-nocomm` | kxkm | `1381a389cff6` | `[101, 99, 5, 0]` | `cee1e3b8b438adc5f67b9b7f0509ae6be0496ac7dc410c6d0ff26e915a7fe7f7` | `a8e76c8422f23acf0a6e58a9410d7c99eb00dca706bd6af40d9f1de7e60c1473` | 0.43 → 0.84 | 375 |
| `seed-101-closed-nocomm` | kxkm | `1381a389cff6` | `[101, 99, 5, 1]` | `1eb97f49ab20b717a4e2b7b214d09b4eef83edee4ec9d113af0a747612aec6d5` | `69467719489b79716ba19b24fead879ac910aa368481443727b7eb3eac79ce88` | 0.6 → 0.91 | 374 |
| `seed-101-registered-dial` | kxkm | `1381a389cff6` | `[101, 99, 3, 0]` | `de86cbea9126f4acf5edb0916e70663c07cdaed260cc443fbece0b5ee959ea81` | `29ca246a749af5976f7f66e331717dd39e3a53ead6daae782d510422f869773f` | 0.41 → 0.81 | 374 |
| `seed-101-closed-dial` | kxkm | `1381a389cff6` | `[101, 99, 3, 1]` | `3e6cd13d2edd0bfa3f8f96abd82a6f421cff02af82565349af4a505b777bacc4` | `cfab8b28248577bc5c78a2b75094471d17c6723e94e220f364eee055cfab1801` | 0.33 → 0.57 | 375 |
| `seed-101-registered-eng` | kxkm | `1381a389cff6` | `[101, 99, 4, 0]` | `a45919d08413c107305e3f3a09923de24fa8fa128f1d40f02230b9d03a64af8b` | `0d3c6898f9b4dcd546ba40cbb473587ab4ea01a3e2d33d96d1d89861ee8783a2` | 0.49 → 0.92 | 375 |
| `seed-101-closed-eng` | kxkm | `1381a389cff6` | `[101, 99, 4, 1]` | `8589fbbb507612fabaf48be2eb227398b7655bc91ccdc918a6a4059ea799b2c7` | `05bebefc7315cf3eea7cb8884b82624bc88daa4d9dddd568d78f345170e39c4d` | 0.5 → 0.87 | 374 |

### Per-run-pretraining control (seed 101, KXKM-AI)

10 runs, each trained right after its shared-trunk registered run of seed 101; descriptive only, never pooled into the estimands or the cross-play cells:

- `prc-registered-mlp-s101` after `registered-mlp-s101`
- `prc-registered-atoc-s101` after `registered-atoc-s101`
- `prc-registered-gtm-s101` after `registered-gtm-s101`
- `prc-closed-gtm-s101` after `closed-gtm-s101`
- `prc-registered-nocomm-s101` after `registered-nocomm-s101`
- `prc-closed-nocomm-s101` after `closed-nocomm-s101`
- `prc-registered-dial-s101` after `registered-dial-s101`
- `prc-closed-dial-s101` after `closed-dial-s101`
- `prc-registered-eng-s101` after `registered-eng-s101`
- `prc-closed-eng-s101` after `closed-eng-s101`

### Throughput and compute under fix C (planning figures)

| host | pair-steps/s | runs | training hours |
|---|---:|---:|---:|
| macm1 | 775 | 30 | 538 |
| kxkm | 1,351 | 40 | 411 |

Held-out evaluations: 70 models on KXKM-AI. The section-4 throughput table above was measured under the frozen configuration of 2026-10-24 and is superseded for planning by this table.

### Learnability gate and Adam study records

- `docs/superpowers/research/2026-10-25-n10c-gate/index.json`, SHA-256 `e1ac984835db831780450b33ebdeb4511be5fea3eb2bf3501f93129e56b8cb03` (run records, ledger ids, archive locations; amendment section 4).

## Smoke runs and development disclosure

9 smoke runs of 200,000 pair-steps each (cap: 1M steps per run; 3 per arm, GTM-comm 4), training levels only (start_level 0, num_levels 16), smoke seed 9001 (not a registered seed), 64 pairs, 128-step rollouts. Their results informed no registered quantity; outcomes are listed unfiltered.

| arm | condition | outcome | pair-steps done | wall s | pair-steps/s (incl. warm-up) | episodes | success (last 100) | losses finite |
|---|---|---|---:|---:|---:|---:|---:|---|
| gtm | registered | completed | 200000 | 403 | 497 | 329 | 0.02 | True |
| mlp | registered | completed | 200000 | 248 | 808 | 214 | 0.00 | True |
| atoc | registered | completed | 200000 | 250 | 800 | 266 | 0.00 | True |
| dial | registered | completed | 200000 | 248 | 806 | 244 | 0.00 | True |
| eng | registered | completed | 200000 | 249 | 804 | 466 | 0.00 | True |
| nocomm | registered | completed | 200000 | 256 | 782 | 217 | 0.00 | True |
| gtm | closed | completed | 200000 | 255 | 784 | 1304 | 0.13 | True |
| gtm | registered | crashed | - | 0 | 0 | - | - | - |
| gtm | registered | completed | 200000 | 270 | 742 | 312 | 0.00 | True |

Failure detail (gtm/registered): FileNotFoundError(2, 'No such file or directory')

Development ledger (persistent, one per-arm ledger for every real-environment development run, written before each run starts; `manifest/smoke_ledger.json`): 38 gate_dev, 8 pretrain_dev, 16 pretrain_registered, 9 smoke, 6 throughput_learning; counted learning runs per arm atoc 1, dial 1, eng 1, gtm 4, mlp 1, nocomm 1; cap 3 per arm (gtm 4, PI decision 2026-10-24) and 1,000,000 pair-steps per run, training levels only. Further throughput measurements use either a frozen random policy (learning rate 0; kind `measurement`, ledgered, not counted) or, when learning is enabled for throughput / profiling, the separate `throughput_learning` category (each <= 1,000,000 pair-steps, training levels 0-15, ledgered, not counted against the allowance; the six runs of 2026-10-23 below are declared as a deviation).
Throughput / profiling runs with learning enabled (category `throughput_learning`, allowed by the amendment change of 2026-10-24 13:17, each <= 1,000,000 pair-steps, training levels only, separate from the smoke allowance): gtm 245,760, gtm 114,688, mlp 114,688, gtm 65,536, gtm 163,840, nocomm 163,840 pair-steps.
**Declared deviation:** 6 of them ran on 2026-10-23, before the rule was finalised, with learning updates (lr 5e-4) from an untrained start on training levels; the PI accepted them as declared engineering measurements (amendment, change of 2026-10-24 13:17). Recorded individually in the ledger and in the amendment; nothing touched held-out data.

Development used only the stub (unit tests with scripted environments), these smoke runs, throughput measurements on untrained or frozen random policies, and the `throughput_learning` throughput / profiling runs with learning enabled listed above (each <= 1,000,000 pair-steps, training levels 0-15; the six runs of 2026-10-23 are a declared deviation). Levels 1000-1199 were never touched and no full-length run result was used.

## Artefact archiving (PI decision 2026-10-23)

After each run and each evaluation, the run directory (`run.json`, checkpoints, EvalRecord JSON, cross-play records) is stored on electron-server (Tailscale 100.78.191.52) at `electron@100.78.191.52:/home/electron/archives/n10c-fixc/` (the archive of the stopped campaign of 2026-10-24 stays at `archives/n10c/`). KXKM-AI cannot reach electron-server, so electron-server PULLS the completed runs from KXKM-AI (`campaign/pull_archive.sh`, rsync over ssh to `electron@10.2.0.237:/ai/saisail/n10c-run2/campaign`, completed runs only, idempotent, logged); macM1 can push with `--archive-to`. The copy is post-hoc storage only and changes no registered quantity. macM1 checkpoints are additionally copied to KXKM-AI for evaluation (campaign `sync` step), and are archived from that KXKM copy. (This item is outside items 1-4 of the amendment's manifest list.)

## Implementation rulings disclosed

- R1: the harness venv (Python 3.10) has no scipy (uv.lock pins scipy 1.17, which needs Python >= 3.11); the analysis code runs in the repo environment.
- R2: GTM message over the 16-bit bus = 4-bit binary code (MSB first) of each of the 4 symbol indices (the bus carries and validates these 16 hard bits). In training the receiver gets the GTMBridge straight-through one-hot [K=4, A=16] directly (value identical to the hard symbols, gradient through the Gumbel-softmax); at evaluation and in the cross-play and shuffle probes it rebuilds the same one-hot from the 16 bits as prod_j (b_j if CODE_TABLE[a,j] = 1 else 1 - b_j), exact for hard bits (identical forward).
- R3: train_run takes keyword arguments log_every_pair_steps and ckpt_every_pair_steps (defaults 1,000,000).
- Task 4: the IMPALA trunk is duplicated between policy.py (live) and bench/bench_ppo_mlx.py (frozen archived evidence); never run together.
- Task 6: advantage normalisation is applied per minibatch (as in common Procgen PPO baselines); this is a PPO detail not in the registered list.
- Task 6: time_to_half requires a full window of 100 completed episodes.
- Task 6: the budget is exactly 50,000,000 pair-steps (final rollout truncated and updated on the partial rollout); the training success rate is measured at the exact boundaries k x 1M pair-steps; checkpoints are written after the first update ending at or after each 1M boundary (labelled with the actual step, < 8,192 pair-steps from the boundary) and once at the exact end.
- Task 6: the engineered right-minus-left counter counts only actions the environment applies (an agent that already finished is frozen; its actions are ignored).
- Task 6: GTM and DIAL message noise is stored per pair-step in the rollout and replayed in the PPO update (so the PPO ratio is 1 for an unchanged policy).
- Task 7: evaluation action sampling = one uniform per agent per step (agent 0 then agent 1) from default_rng([20261023, j]), inverse CDF on the float64 softmax; identical across passes and models. Evaluation episodes are capped at 5,000 steps (an error, not an outcome). Evaluation messages are deterministic (GTM argmax symbols).
- R4: gradient accumulation (micro_batch_pairs, clear_cache; PR #53) and n_workers (PR #54) are merged into the harness branch; the freeze commit must contain both. The KXKM-AI settings above refer to them.
- Task 9: smoke runs = 1 per arm in the registered condition plus gtm in the closed condition, 200,000 pair-steps each, training levels only; the gtm registered smoke was run twice more after its gradient path changed (one crashed at 0 steps): 9 smoke runs in all. The amendment allows counted learning runs (smoke / dev_train) of up to 1M pair-steps each: 3 per arm, GTM-comm 4 (PI decision 2026-10-24). Throughput / profiling runs with learning enabled form a separate category, `throughput_learning` (each <= 1M pair-steps, training levels 0-15, not counted against that allowance). The caps are enforced in code (harness/devledger.py, used by harness/throughput.py) with a persistent ledger (manifest/smoke_ledger.json) written before each run starts; train_run itself refuses any budget above 1M pair-steps without the registered execution context.
