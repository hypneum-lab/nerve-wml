# N10-C MLX Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything N10-C needs before its stage-2 manifest can be filed: an MLX PPO multi-agent harness for CoinRun-coop on macM1 with the six message arms, the 16-bit message bus, training/evaluation/shuffle-probe code, the frozen estimation analysis, the end-to-end throughput measurement and the manifest generator.

**Architecture:** A self-contained package `experiments/procgen_multiagent_gtm_comm/harness/` runs in its own Python 3.10 venv (Procgen 0.10.7 arm64 + MLX 0.31.1); it never imports `torch`, `track_p` or `nerve_wml`. The GTM channel is an MLX port of `GammaThetaMultiplexer`, pinned to the PyTorch reference by a golden-file parity test. The PPO sample unit is the **pair-step** (both agents' observations and messages at one step), so a minibatch carries both ends of every message. Analysis code (numpy/scipy only) lives beside it and also runs in the repo's main Python 3.12 env.

**Tech Stack:** Python 3.10, MLX 0.31.1, numpy < 2, scipy, Procgen 0.10.7 (arm64 source build), gym3 0.3.3, pytest; reference golden generated once with the repo's PyTorch env.

**Spec:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md` (stage 1) + locked `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md`. Evidence: `experiments/procgen_multiagent_gtm_comm/bench/README.md`.

## Global Constraints

- All compute runs on macM1 via `ssh macm1` (never locally on GrosMac). Harness venv: `~/scratch/n10c/.venv` (Python 3.10), Procgen built with `experiments/procgen_multiagent_gtm_comm/bench/procgen-0.10.7-arm64-build.patch` and `SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk`, `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, `PROCGEN_CMAKE_PREFIX_PATH=$(brew --prefix qt@5)/lib/cmake`, `CMAKE_POLICY_VERSION_MINIMUM=3.5`. Run Python outside the procgen source dir.
- Pinned deps only (HITL supply-chain rule): `mlx==0.31.1`, `numpy<2`, `gym3==0.3.3`, `gym==0.26.2`, `setuptools<70`, `scipy` and `pytest` at the versions in the repo `uv.lock`. Record them in `experiments/procgen_multiagent_gtm_comm/harness/requirements.lock`.
- Env: CoinRun-coop only, `distribution_mode="easy"`, Δt = 20, success-only team reward, train `num_levels=16, start_level=0`, held-out levels 1000–1199 (`num_levels=1, start_level=1000+j`). Held-out levels are never touched before the manifest is filed (smoke runs evaluate on training levels only).
- Message: per agent per step m ∈ {0,1}¹⁶. GTM: K = 4 symbols × A = 16. DIAL: σ(m + 𝒩(0, 2²)) in training, hard threshold at 0 at evaluation.
- PPO (registered): lr 5e-4, clip 0.2, GAE λ 0.95, γ 0.999, minibatch 2048 agent-samples (1024 pair-steps); plus fixed here: 64 parallel coop envs, 128-step rollouts (16,384 agent-samples per update), 3 epochs, Adam, value coef 0.5, entropy coef 0.01, max grad norm 0.5. Budget: 50M steps per agent (= 50M pair-steps, ≈ 6,104 updates).
- Integer codes (manifest): `ARM_CODES = {"gtm": 0, "mlp": 1, "atoc": 2, "dial": 3, "eng": 4, "nocomm": 5}`, `CONDITION_CODES = {"registered": 0, "closed": 1}`. Seeds {0, 17, 42, 73, 101}; run randomness from `numpy.random.SeedSequence([seed, arm_code, cond_code])`.
- Policy inputs: own image, received message embedding, team summary (registered condition only). Never rewards, done flags or partner actions.
- Commits: conventional, subject ≤ 50, body ≤ 72, no AI attribution, never `--no-verify`. MLX/procgen tests use `pytest.importorskip` so the repo's Linux CI (Python 3.12, no MLX) skips them.

## Review Focus

1. **Episode boundaries inside a rollout** — a coop episode that ends mid-rollout must reset GAE (no bootstrap across `done`) and the next pair-step must start from the reset observation; tested in Task 6.
2. **Frozen agent after its coin** — its actions are ignored but it still sends messages from its frozen observation; the bus must not drop them; tested in Task 3 and Task 7.
3. **Partner swap** — agent 0 must receive agent 1's message and vice versa, never its own; tested in Task 3.
4. **Shuffle probe on short donor episodes** — when the donor episode π(j) is shorter than the current one, the last donor message is repeated (`min(t, T−1)`); tested in Task 7.
5. **Non-finite parameters** — a NaN after an update must stop training at the last finite checkpoint (training-failure rule); tested in Task 6.

---

### Task 1: Harness environment, GTM golden reference and MLX multiplexer

**Files:**
- Create: `experiments/procgen_multiagent_gtm_comm/harness/__init__.py`, `harness/requirements.lock`, `harness/codes.py`, `harness/gtm_mlx.py`
- Create: `scripts/n10c_make_gtm_golden.py` (runs in the repo PyTorch env)
- Create: `tests/golden/n10c_gtm_k4_a16.npz`
- Test: `tests/unit/n10c/test_gtm_mlx.py`, `tests/unit/n10c/__init__.py`

**Interfaces:**
- Produces: `codes.ARM_CODES`, `codes.CONDITION_CODES`, `codes.run_seed_sequence(seed: int, arm: str, condition: str) -> np.random.SeedSequence`; `gtm_mlx.GTMConfigMLX` (frozen dataclass: `gamma_hz=40.0, theta_hz=6.0, sample_rate_hz=1000.0, alphabet_size=16, symbols_per_theta=4`); `gtm_mlx.GammaThetaMultiplexerMLX(cfg, constellation: mx.array)` with `__call__(codes: mx.array[int32, B,K]) -> mx.array[float32, B,T]` (1-channel, no noise, no role) and attribute `constellation [A,2]`.

- [ ] **Step 1:** Write `scripts/n10c_make_gtm_golden.py`: builds `GammaThetaMultiplexer(GammaThetaConfig(alphabet_size=16, symbols_per_theta=4), seed=0)`, encodes `codes = torch.randint(0, 16, (32, 4), generator=seed 0)`, saves `constellation`, `codes`, `carrier` to `tests/golden/n10c_gtm_k4_a16.npz`. Run it with `uv run python scripts/n10c_make_gtm_golden.py` (repo env, on macM1).
- [ ] **Step 2: failing test** `test_carrier_matches_torch_golden`: load the npz, build `GammaThetaMultiplexerMLX(GTMConfigMLX(), mx.array(constellation))`, assert `np.allclose(np.array(mlx_carrier), carrier, atol=1e-5)`; `test_k_above_capacity_raises` (K=5 → `ValueError`); `test_run_seed_sequence_distinct_per_arm` (gtm vs mlp entropy differs, same inputs → same entropy).
- [ ] **Step 3:** Run `PYTHONPATH=. ~/scratch/n10c/.venv/bin/python -m pytest tests/unit/n10c/test_gtm_mlx.py -q` → FAIL (import).
- [ ] **Step 4:** Implement `gtm_mlx.py` as a line-by-line port of `GammaThetaMultiplexer.forward` (time grid `n_per_gamma × K_cap`, unit-norm symbols, γ quadratures, θ envelope depth 0.45, Gaussian packets — copy constants from `track_p/multiplexer.py`), and `codes.py`.
- [ ] **Step 5:** Re-run → PASS. Write `requirements.lock` from `uv pip freeze` of the harness venv.
- [ ] **Step 6:** Commit `feat(n10c): add MLX GTM multiplexer port`.

### Task 2: Coop env window view and vectorised pairs

**Files:**
- Modify: `experiments/procgen_multiagent_gtm_comm/coop_env.py` (add method)
- Create: `harness/vec_env.py`
- Test: `tests/unit/test_procgen_coop_team_summary.py` (extend), `tests/unit/n10c/test_vec_env.py`

**Interfaces:**
- Produces: `ProcgenCoopEnv.window_view(i: int) -> tuple[bool, int]` = (agent i reached its coin inside the open window, remaining scoring steps 0–20) — computed whether or not `team_summary` is open, never delivered as an observation; `VecCoopEnv(n_pairs: int, base_seed_seq: np.random.SeedSequence, team_summary: bool, num_levels: int = 16, start_level: int = 0, env_factory=None)` with `reset() -> PairObs` and `step(actions: np.ndarray[int32, n_pairs, 2]) -> PairStep`; `PairObs = (obs: uint8[n,2,64,64,3], summary: float32[n,2,2])` (zeros when closed); `PairStep` dataclass: `obs, summary, team_reward: float32[n], done: bool[n], finished: bool[n,2], window: (in_window: bool[n,2], remaining: int32[n,2])`. Auto-resets a pair whose `done` is True and returns the new episode's first observation in `obs` (the `done` flag refers to the finished episode).

- [ ] **Step 1: failing tests** (scripted envs as in the existing team-summary tests): `test_window_view_matches_summary` (when open, `window_view(partner)` equals `(summary[0]==1, round(summary[1]*20))`); `test_window_view_available_when_closed`; `test_vec_env_autoreset_returns_new_episode_obs`; `test_vec_env_pairs_share_level_seed`; `test_closed_condition_summary_is_zero`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `window_view` (reuse the `_first_coin_agent` / `_steps_since_first_coin` state; remaining = `delta_t - elapsed + 1` as in `team_summaries`) and `VecCoopEnv` as a list of `ProcgenCoopEnv`, pair k seeded from `base_seed_seq.spawn(n_pairs)[k]`.
- [ ] **Step 4:** Run → PASS (repo env for the coop test, harness venv for vec_env).
- [ ] **Step 5:** Commit `feat(n10c): add window view and vectorised pairs`.

### Task 3: Message arms and the 16-bit bus

**Files:**
- Create: `harness/comm.py`, `harness/bus.py`
- Test: `tests/unit/n10c/test_comm.py`, `tests/unit/n10c/test_bus.py`

**Interfaces:**
- Consumes: `GammaThetaMultiplexerMLX` (Task 1).
- Produces: base class `Comm(nn.Module)` with `send(x: mx.array[B,64], *, train: bool, key: mx.array | None, eng_bits: mx.array | None) -> tuple[mx.array[B,16] float, mx.array[B,16] float]` returning `(msg_for_receiver, hard_bits)` — `msg_for_receiver` carries the straight-through gradient in training and equals `hard_bits` at evaluation (DIAL: noisy sigmoid in training); and `receive(msg: mx.array[B,16], h: mx.array[B,256]) -> mx.array[B,64]`. Subclasses: `GTMComm` (Linear 64→4·16 logits; Gumbel-softmax hard, τ = 1.0, ST in training, argmax at eval; bits = 4-bit binary of each index, MSB first; receiver rebuilds one-hot from bits → `GammaThetaMultiplexerMLX` carrier → adaptive mean-pool to 64 + soft IQ (one-hot @ constellation, K·2) → Linear → 64; constellation init PSK + 0.01 noise from the run stream; constellation gradient scaled by cosine schedule 1.0 → 0.1 over the run's total updates), `MLPComm` (Linear 64→16, hard > 0, ST through sigmoid; receiver Linear 16→64), `ATOCComm` (sender as MLP; receiver: 16 bit tokens = Embedding(2, 32) + learned position(16, 32), query Linear 256→32 from own hidden, single-head attention, output Linear 32→64), `DIALComm` (Linear 64→16; train σ(m + 2·ε), ε ~ 𝒩(0,1); eval hard m > 0; receiver Linear 16→64), `EngComm` (sender returns `eng_bits` unchanged; receiver Linear 16→64), `NoComm` (zeros; receiver Linear 16→64 so parameter counts stay comparable). `make_comm(arm: str, key) -> Comm`. `eng_bits_from_window(in_window: bool, remaining: int, right_minus_left: int) -> np.ndarray[16]` (bit 0 in_window; bits 1–5 remaining unsigned; bits 6–15 clip(n, −512, 511) 10-bit two's complement; right = actions {6,7,8}, left = {0,1,2}).
- Produces: `MessageBus(arm: str)` with `deliver(bits: np.ndarray[n,2,16], *, phase: str) -> np.ndarray[n,2,16]` (partner swap: out[:,0]=bits[:,1], out[:,1]=bits[:,0]); asserts shape (n,2,16) and values in {0,1} in eval for all arms and in train for all arms except DIAL; raises `MessageBusViolation`; `bits_logged: int` counter (16 per delivered message).

- [ ] **Step 1: failing tests:** `test_every_arm_eval_bits_are_binary_16`; `test_gtm_bits_roundtrip_indices` (bits → indices → bits identity over all 16⁴ edge values sampled); `test_gtm_receiver_gradient_reaches_sender` (nonzero grad on `encode_logits` through ST); `test_dial_train_messages_not_binary_eval_binary`; `test_eng_bits_layout` (in_window=True, remaining=20, n=−3 → bit0=1, bits1–5 = 10100, bits 6–15 = 1111111101; n=600 clipped to 511); `test_nocomm_sends_zeros`; `test_bus_swaps_partners`; `test_bus_rejects_nonbinary_eval` and `test_bus_allows_dial_train_soft`; `test_frozen_agent_message_still_delivered` (bus never inspects `finished`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `comm.py`, `bus.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(n10c): add message arms and 16-bit bus`.

### Task 4: Pair policy network

**Files:**
- Create: `harness/policy.py`
- Test: `tests/unit/n10c/test_policy.py`

**Interfaces:**
- Consumes: `make_comm` (Task 3), IMPALA CNN definition from `bench/bench_ppo_mlx.py` (copy the module classes into `policy.py`; do not import the bench script).
- Produces: `PairPolicy(arm: str, key)` with `__call__(obs: mx.array[float32, B,2,64,64,3], summary: mx.array[B,2,2], eng_bits: mx.array[B,2,16] | None, *, train: bool, key) -> PolicyOut`; `PolicyOut` = `logits [B,2,15]`, `value [B,2]`, `hard_bits [B,2,16]`. Per agent: IMPALA (16,32,32) → 256 hidden h; `Linear(256→64)` → `comm.send`; partner swap inside the module (agent i receives partner's `msg_for_receiver`); `comm.receive(msg, h)` → 64; concat `[h, recv, summary]` (258+64 = 322) → `Linear(322→256)` → ReLU → policy `Linear(256→15)` and value `Linear(256→1)`. Parameters shared across both agents. Observations scaled /255.

- [ ] **Step 1: failing tests:** `test_output_shapes_all_arms`; `test_partner_swap_in_policy` (perturb agent 1's image → agent 0's logits change for every arm except nocomm; for nocomm they do not); `test_summary_ignored_when_zero_closed` (closed condition passes zeros; output deterministic in eval); `test_param_count_reported` (`param_count(policy) -> int` > 0, recorded for the manifest).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(n10c): add pair policy network`.

### Task 5: Rollout buffer and GAE on pair-steps

**Files:**
- Create: `harness/rollout.py`
- Test: `tests/unit/n10c/test_rollout.py`

**Interfaces:**
- Produces: `RolloutBuffer(n_pairs: int, horizon: int)` storing per pair-step `obs, summary, eng_bits, actions[2], logp[2], value[2], team_reward, done`; `compute_gae(rewards: np.ndarray[T,n], values: np.ndarray[T,n,2], dones: np.ndarray[T,n], last_value: np.ndarray[n,2], gamma=0.999, lam=0.95) -> tuple[adv[T,n,2], ret[T,n,2]]` — the team reward is credited to both agents; `done` stops bootstrapping; `minibatches(rng, size_pairs=1024) -> Iterator[dict[str, mx.array]]` over flattened pair-steps.

- [ ] **Step 1: failing tests:** `test_gae_no_bootstrap_across_done` (reward at t with done=True: adv equals r − V(t)); `test_gae_matches_reference_loop` (vectorised vs naive loop on random data, atol 1e-6); `test_minibatches_cover_all_pairs_once`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(n10c): add pair-step rollout and GAE`.

### Task 6: PPO trainer with checkpoints and training-success log

**Files:**
- Create: `harness/train.py` (library + CLI `python -m harness.train --arm gtm --seed 0 --condition registered --steps 50000000 --out runs/...`)
- Test: `tests/unit/n10c/test_train.py`

**Interfaces:**
- Consumes: `VecCoopEnv` (Task 2), `PairPolicy` (Task 4), `RolloutBuffer`, `compute_gae` (Task 5), `MessageBus` (Task 3), `run_seed_sequence` (Task 1).
- Produces: `train_run(arm: str, seed: int, condition: str, total_pair_steps: int, out_dir: Path, *, env_factory=None, n_pairs=64, horizon=128) -> RunRecord`; `RunRecord` JSON at `out_dir/run.json`: arm, seed, condition, codes, total steps, wall time, samples/s, restarts, `using_stub: bool` (a scientific run with `using_stub=True` raises before training starts), `training_failure: bool`, `success_curve: list[(step, rate)]` measured every 1,000,000 pair-steps as the mean over the last 100 completed coop episodes, `time_to_half: int | None` (first logged step with rate ≥ 0.5, else None → cap 50M in analysis), `bits_logged`. Checkpoint `out_dir/ckpt_<step>.safetensors` every 1M steps if all parameters are finite; on a non-finite loss or parameter, stop, mark `training_failure=True`, keep the last finite checkpoint. PPO loss compiled with `mx.compile` (as in `bench_ppo_mlx.py`), clipped surrogate + 0.5 value + 0.01 entropy, gradient norm clipped at 0.5. Engineered arm: harness tracks per-agent right−left counters and feeds `eng_bits_from_window` from `window_view`.

- [ ] **Step 1: failing tests** (scripted env factory, tiny budgets): `test_train_run_writes_record_and_checkpoint` (budget 2 updates); `test_success_curve_logged_every_interval` (interval overridden to 2 updates for the test via parameter `log_every_pair_steps`); `test_nan_stops_at_last_finite_checkpoint` (inject NaN into a parameter after update 1 → `training_failure` True, last checkpoint = update-1 checkpoint); `test_same_seed_same_first_update` (determinism of the run stream); `test_engineered_counters_reset_per_episode`.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(n10c): add MLX PPO-MA trainer`.

### Task 7: Evaluation and message-shuffle probe

**Files:**
- Create: `harness/evaluate.py` (library + CLI)
- Test: `tests/unit/n10c/test_evaluate.py`

**Interfaces:**
- Consumes: `PairPolicy`, `ProcgenCoopEnv`, checkpoint format (Task 6).
- Produces: `evaluate_model(ckpt: Path, arm: str, condition: str, levels: range, *, env_factory=None) -> EvalRecord`; `EvalRecord` JSON: `R_intact`, `R_shuffled`, `S = R_intact − R_shuffled`, `completion_time` (mean steps to success over successes, `None` if none), `bandwidth_bits` (plug-in entropy in bits of the 16-bit message distribution over all intact steps), per-level returns and lengths. Action sampling on level j uses `numpy.random.default_rng([20261023, j])` (categorical from softmax, same generator for both passes); messages are argmax/hard (no noise). Derangement: `derangement(n: int, rng) -> np.ndarray` drawn once with `default_rng(20261023)` (resample permutations until no fixed point). Shuffled pass: at step t of level j, agent i receives the message the same sender role emitted at step `min(t, T_π(j) − 1)` of intact level π(j); messages computed during the shuffled pass are discarded. Levels default `range(1000, 1200)`; smoke evaluation passes training levels `range(0, 16)`.

- [ ] **Step 1: failing tests:** `test_derangement_has_no_fixed_points_and_is_deterministic`; `test_shuffle_uses_last_donor_message_when_donor_shorter`; `test_nocomm_S_is_zero` (deterministic: S == 0.0 exactly); `test_eval_rng_identical_across_passes`; `test_completion_time_none_without_success`; `test_bandwidth_entropy_bounds` (0 for constant messages, ≤ 16).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(n10c): add evaluation and shuffle probe`.

### Task 8: Frozen estimation analysis

**Files:**
- Create: `experiments/procgen_multiagent_gtm_comm/analysis.py` (numpy/scipy only; runs in both envs)
- Test: `tests/unit/test_n10c_analysis.py` (repo env, no importorskip)

**Interfaces:**
- Produces: `welch_ci(x: np.ndarray, y: np.ndarray, level: float) -> tuple[float, float, float] | None` (diff, low, high; `None` when both variances are zero or df undefined); `one_sample_ci(x, level) -> tuple[float, float, float] | None`; `fieller_ci(a_arm, nocomm, eng, level=0.95) -> tuple[float, float, float] | None` (None unless the denominator interval lies entirely above 0; Cov(a,b)=v_NOCOMM; t at the smaller Welch df); `pilot_category(intervals: list[tuple|None], delta=0.10) -> str` returning one of `"pilot-interval-unavailable"`, `"pilot-excludes-SESOI"`, `"pilot-positive"`, `"pilot-inconclusive"` in that precedence; `primary_report(results: dict) -> dict` building the 3 primary contrasts at 1 − 0.05/3, the k = 9 secondary contrasts at 1 − 0.05/9 (sample efficiency as comparator − GTM with cap 50M; bandwidth signed, never counted as favourable), NI per arm/condition, S intervals at 95 % with the positive-control rule (engineered closed-condition S interval must lie above 0, else "insensitive"/"uninterpretable").

- [ ] **Step 1: failing tests** (hand-computed values): `test_welch_ci_matches_scipy` (against `scipy.stats.ttest_ind(..., equal_var=False).confidence_interval`); `test_zero_variance_returns_none`; `test_pilot_category_precedence` (one unavailable → unavailable; U<Δ for one baseline → excludes; all L≥Δ → positive; else inconclusive); `test_fieller_none_when_denominator_crosses_zero`; `test_positive_control_insensitive_blocks_S_interpretation`.
- [ ] **Step 2–4:** FAIL → implement → PASS (`uv run python -m pytest tests/unit/test_n10c_analysis.py` on macM1).
- [ ] **Step 5:** Commit `feat(n10c): add frozen estimation analysis`.

### Task 9: End-to-end throughput, smoke runs and stage-2 manifest

**Files:**
- Create: `harness/throughput.py`, `harness/manifest.py`, `experiments/procgen_multiagent_gtm_comm/runs/.gitignore` (ignore `ckpt_*`)
- Create (generated, committed): `experiments/procgen_multiagent_gtm_comm/manifest/throughput.json`, `manifest/smoke_runs.json`, `manifest/stage2-manifest-DRAFT.md`
- Test: `tests/unit/n10c/test_manifest.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `measure_throughput(arm: str, n_updates: int = 20) -> dict` (real procgen, training levels only: pair-steps/s and agent-samples/s including env stepping, policy, bus and update; eval seconds per model for 2 × 200 episodes measured on training levels as a proxy, labelled as such); `write_manifest(out: Path) -> None` writing: git commit of `coop_env.py`/harness, `ARM_CODES`/`CONDITION_CODES`, per-arm parameter counts and layer sizes, GTM adapter (K=4, A=16, τ=1.0, cosine plasticity 1.0→0.1 over total updates, pool 64), PPO settings, throughput JSON, compute estimate for 50 runs (training + evaluation), Procgen build record (patch SHA-256, procgen commit 5e1dbf3, Qt 5.15.19, clang/Xcode SDK versions, Python, MLX versions), the smoke-run list with outcomes, and the statement that Caveflyer-coop and Maze-coop are not run. Status line: "DRAFT — to be filed on OSF by the PI as stage 2 before any run".

- [ ] **Step 1: failing tests:** `test_manifest_contains_required_items` (every item 1–4 of the spec's manifest list present as a heading); `test_manifest_never_mentions_heldout_results` (no field reads levels ≥ 1000).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** On macM1 (outside second-brain peak hours), run `measure_throughput` for `gtm` and `nocomm`, then at most 3 smoke runs per arm of ≤ 1M pair-steps each (training levels only), recording outcomes in `smoke_runs.json`; then `write_manifest`.
- [ ] **Step 6:** Commit `chore(n10c): add throughput, smoke runs, manifest`.

### Task 10: Parallel environment stepping (multi-process VecCoopEnv)

**Files:**
- Modify: `harness/vec_env.py`, `harness/train.py`, `harness/throughput.py`
- Regenerate: `manifest/throughput.json`, `manifest/stage2-manifest-DRAFT.md`
- Test: `tests/unit/n10c/test_vec_env.py`, `tests/unit/n10c/test_train.py`

**Interfaces:**
- Consumes: `VecCoopEnv` (Task 2), `train_run` (Task 6), `measure_throughput`, `write_manifest` (Task 9).
- Produces: `VecCoopEnv(..., n_workers: int = 1)`: pairs split into `n_workers` contiguous blocks, each stepped in its own `multiprocessing` process (spawn context) over pipes; `n_workers=1` keeps the current in-process path. Pair k's seed stream, level seeds, auto-reset and returned arrays are identical for every `n_workers` (the worker count is an implementation detail, never a registered quantity). `train_run(..., n_workers: int = 1)` and the CLI flag `--n-workers`. Workers are closed on `close()` and on exceptions.

- [ ] **Step 1: failing tests:** `test_vec_env_identical_across_worker_counts` (scripted, picklable module-level env factory; 50 random steps; every PairStep field equal for n_workers in {1, 2, 4}); `test_vec_env_workers_closed_on_error`; `test_train_run_identical_first_rollout_across_worker_counts` (same seed, same rollout actions and first-update params within the Task 6 tolerance).
- [ ] **Step 2:** Run, FAIL. **Step 3:** Implement. **Step 4:** Run, PASS.
- [ ] **Step 5:** On macM1 (and macM3 for comparison), measure end-to-end throughput for n_workers in {1, 4, 8} (gtm arm, training levels only), pick the fastest as the registered setting, regenerate `throughput.json` and the manifest draft (worker count recorded as an implementation setting).
- [ ] **Step 6:** Commit `perf(n10c): step coop envs in worker processes`.

---

## Execution note

Tasks 1 → 9 are sequential through their Interfaces (each consumes the previous task's names). Task 8 is independent and may run in parallel with Tasks 2–7. No full-length run and no held-out evaluation happen in this plan: they start only after the PI files the stage-2 manifest on OSF.
