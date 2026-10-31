# N10-C-bis necessity gate — G7 annex: indicative pilot on a DGX Station GB300 (L0 draft)

**Status:** L0 DRAFT on branch `prereg/n10c-bis-gate-gb300`, not merged, not tagged; PI
decisions of 2026-10-31 recorded (section 0); the freeze waits for the PI's review of the
hashed requirement files (`freeze-manifest.md`). It binds nothing until it is merged on
green CI **before the first pilot run**.
**Annex to:** section « Necessity gate (PI decision 2026-10-30) », items G1–G6, of
`docs/milestones/2026-10-25-n10c-bis-control-arms.md` (merged by PR #91, commit `1d08ecf`).
That addendum is still a DRAFT and append-only; its item G7 points to this file.
**Scope:** an **indicative pilot** of the necessity gate (`eng − nocomm ≥ 0.20`, condition
`closed`, 10M pair-steps) on the GB300. The decisional gate stays on KXKM-AI as G2–G5
say. N10-C (tags `n10c-freeze-20261024`, `n10c-freeze-20261029`), its running campaign,
its hosts and every N10-C-bis provision other than G5 are **not modified**.
**Date drafted:** 2026-10-31, before any gate run on any host (no `gate_dev` entry exists
for `nocomm`; the 13 `eng` entries are earlier development runs of other designs).
**Decision behind it:** PI multiple-choice decision of 2026-10-31, 21:20 CET: prepare the
pre-registrations for a 24 h access to one DGX Station GB300 on 2026-11-01.

## 0. PI decisions 2026-10-31 (QCM, 22:20–22:26 CET; every earlier TBD is closed)

Recorded before any pilot run. Where the text below conflicts with them, they govern.

| id | decision |
|---|---|
| D1 | Three seeds, 9101, 9102, 9103, in this order; arm order alternates per seed (section 3). |
| D2 | **Indicative only.** The GB300 result is a pilot. No bis-A or N10-D decision is taken from it; G3 does not apply to it; the decisional gate stays on KXKM-AI (G5 unchanged). |
| D3 | One rerun per run (G2): `GATE_DEV_ALLOWANCE` `eng` = 13 + 2k, `nocomm` = 2k, k = 3: `eng` 19, `nocomm` 6 (section 7, companion commit and test). The decisional KXKM-AI gate gets its own allowance later, by a separate dated change. |
| D4 | Python **3.12** (CPython 3.12.15) for the pilot venvs, not 3.10: a declared difference from N10-C (KXKM-AI `venv310`, Python 3.10.18). |
| D5 | Procgen 0.10.7 is built on the host from the kit of nerve-wml PR #101 (prepared in an arm64 container); build and CoinRun smoke are mandatory A0 steps under the 3 h cap (section 4). |
| D6 | `filelock`: Procgen's `filelock<4` constraint is **lifted** (its `setup.py` patched to accept `filelock` 4.1.0); the N10-C lock is unchanged. Declared deviation of the build, not of N10-C. |
| D7 | Both MLX backends, `mlx-cuda-12` and `mlx-cuda-13` 0.31.1, are installed (one venv each) and smoke-tested (S1–S5, « A0 »). If `mlx-cuda-12` passes, the pilot runs under it; otherwise under `mlx-cuda-13`. Both are declared now; the backend kept is recorded. |
| D8 | Hardening of G2 adopted: a second non-OK attempt of any run makes the session INCONCLUSIVE. |

One run at a time, as G5 says (concurrent arms would be two GPU jobs of ours).

PI actions (not decisions): copy the trunk files of the three seeds to the GB300 host
(they live on macM1 / KXKM-AI, outside the repository) as `trunk-<seed>.safetensors`;
review the hashed requirement files (`requirements-gb300-*.txt`, summary in
`requirements-review.md`) before access.

## 1. Change of substrate (declared)

G5 fixes the gate host as **KXKM-AI**, and it stays so for the decisional gate. This annex
adds an indicative pilot on a **different substrate**:

| | G5 (KXKM-AI) | G7 (GB300) |
|---|---|---|
| CPU / ISA | x86-64 | 72-core Arm Neoverse V2 (Grace), aarch64, 64 KiB pages |
| GPU | RTX 4090, 24 GB, compute capability 8.9 | 1 Blackwell Ultra, 252 GB HBM3e, compute capability 10.3 |
| OS | Ubuntu (x86-64) | Ubuntu 24.04 aarch64, Linux 6.17, CUDA 13.1/13.2 stack |
| Python | 3.10.18 | 3.12.15 (D4) |
| MLX backend | `mlx-cuda-12` 0.31.1 | `mlx-cuda-12` or `mlx-cuda-13` 0.31.1, chosen by the A0 smoke tests (D7, section 6) |
| Procgen | official 0.10.7 x86-64 wheel (SHA-256 `4b594d14…`) | 0.10.7 built from source (commit `5e1dbf3`) on aarch64 |

Hardware facts: NVIDIA DGX Station product page and DGX Station Development Guide, read
on 2026-10-31 (sources listed in the PR body). Consequences, stated before any result:

1. **Comparability is intra-host only.** Both arms of every seed run on the GB300, from
   the same checkout, venv, Procgen build, MLX backend and per-run settings. The gate
   statistic `eng − nocomm` is computed only from GB300 runs. No GB300 gate run is
   compared with, or pooled with, a KXKM-AI or macM1 run (including the earlier
   `gate_dev` runs of section 4 of the 2026-10-25 amendment and B3).
2. **Procgen is not bit-identical across builds** (0 of 20 CoinRun levels identical
   between the macM1 arm64 and KXKM-AI x86-64 builds,
   `experiments/procgen_multiagent_gtm_comm/bench/procgen-builds/README.md`). The GB300
   build is a third build. The gate reads training success on levels 0–15 only, so it is
   self-consistent on its host; it never touches held-out levels 1000–1199.
3. **Indicative only (D2).** The pilot measures a training-success difference after a fixed
   optimisation budget, on one build, backend and set of trunks; it is not a proof that
   the message is necessary. It informs the planning of the KXKM-AI gate and nothing
   else: no bis-A or N10-D decision, no G3 consequence, no pooling with the KXKM-AI gate.
4. MLX convolution backward is nondeterministic on CUDA (stage-2 manifest fix C,
   « Reproducibility »); runs are statistically, not bitwise, reproducible. Nothing here
   depends on bit identity.

## 2. Host rules

1. **Host:** one DGX Station GB300 (hostname, serial or asset id, driver version and
   `nvidia-smi` output recorded at start, section 5.1). The GPU used is the GB300 GPU,
   selected by PCI bus id (the CUDA and NVML orders can differ when a display card is
   fitted); the device must report compute capability 10.3.
2. **Never in parallel with another GPU job of ours.** No other process of ours uses
   the GB300 GPU while a gate run is alive (no NH-81, no model serving, no benchmark, no
   N10-C job). The lane (section 8) checks the GPU's compute processes before each run and
   every 60 s during it; another compute process before a run blocks it, one during a run
   marks the run contaminated and stops the session. Graphics contexts are not seen by
   this check: no other workload of ours is started on the host during the gate. CPU-only
   jobs of ours (the rescience-nh01 ARM reproduction) may run on a disjoint set of cores
   (`taskset`); core sets, start and end times of every job are logged. `taskset`
   separates cores, not memory bandwidth or the shared CPU/GPU power budget; this is
   recorded, not controlled.
3. **One run at a time** (G5). The E1-small clause of G5 is specific to KXKM-AI and does
   not apply on the GB300.
4. **Per-run settings:** the KXKM-AI registered implementation settings named by G2
   (`micro_batch_pairs = 512`, `clear_cache = on`, `n_workers = 8`), passed explicitly
   on the command line (section 8) and recorded in each `run.json`. The larger GB300 memory
   is not used to change them.

## 3. Seeds (D1)

Development seeds of the pilot namespace 91xx (G2), disjoint from the registered seeds
{0, 17, 42, 73, 101, 131} and from the smoke seed 9001. Set fixed by the PI (D1) before any
pilot run, in this order:

| order | seed | trunk (`init_trunk_sha256`) | trunk record |
|---|---|---|---|
| 1 | 9101 | `0c27f2daa0a976e6407a85d6dfa6eafe20a6285c6324795b2194366b1a9aa704` | `docs/superpowers/research/2026-10-25-n10c-gate/macm1/gate/pretrain/pretrain.json` |
| 2 | 9102 | `b6805dee76c2f62be352383899d3b3b199458babbac1d260acdfffb78aa62bf1` | `…/macm1/gate-s9102/pretrain/pretrain.json` (the trunk of rows 20–21) |
| 3 | 9103 | `8c0b72ee4f362e133771f6e0de93e6bac96ced0edb077b8b63f2c974cfd24a7a` | `…/kxkm/adam-study/pretrain-s9103/pretrain.json` |

Arm order within a seed alternates, fixed now: 9101 `eng` then `nocomm`; 9102 `nocomm`
then `eng`; 9103 `eng` then `nocomm` (so that a drift of host
load over the session does not line up with one arm).

- **Why these seeds.** They are the first three development seeds whose trunks are
  recorded in the repository, so no new pre-training is needed (`pretrain_dev` has used
  8 of 8); 9104 and 9105 also have trunks and are left out (D1); three pairs of 10M runs
  take about 12.3 h at the KXKM-AI rate (about 4.1 h per pair, G5). They are not new or
  blind: their earlier development runs (registered condition, 1M pair-steps, other
  designs) are published. The gate runs use condition `closed`, whose seed streams
  (`SeedSequence([seed, arm, condition])` with integer arm and condition codes) differ from
  every earlier run, but the trunks are the same. The gate is exploratory (G4), so reuse
  is allowed; the dependence is disclosed, not removed.
- **Trunks are checked before the first gate run.** All three trunk files are hashed by
  the lane before it starts any run; a missing file or a mismatch stops the session
  before any gate run, and the PI decides (re-ordering or dropping a seed is then a
  dated decision taken before any gate result). No trunk is re-trained on the GB300.
  No seed is substituted or added once the first gate run has started.
- **Analysed set and verdict.** Seeds run in the order above. A seed's pair starts only
  under the time rule of section 9; no further pair starts after the first pair that
  the rule refuses. The outcome is one of:

  | outcome | condition |
  |---|---|
  | **indicative PASS** / **FAIL** (D2) | at least one complete pair, the session ended by the time rule or after the last seed, and no stop condition of section 9 other than the time rule; statistic = the difference of the one pair, or the mean over complete pairs of the per-seed difference (G2); PASS if ≥ 0.20 |
  | **INCONCLUSIVE** | no complete pair; or the session stopped on a second non-OK attempt of a run (outcome-related: no verdict is read from the other pairs, D8), on contamination or on a hardware error |

  A pair killed by the deadline (section 9) is time-driven, not outcome-driven: it is
  excluded and reported, and the verdict uses the complete pairs before it. The time
  rule does not budget reruns, so a rerun in the last admitted pair can end that way.

  A complete pair has two runs whose status is OK (section 8). An INCONCLUSIVE session is
  reported with every run; any further gate run is a new, dated, non-blind PI decision,
  not part of this pre-registered gate.

## 4. Procgen build on Linux aarch64 (D5, D6)

- **Kit:** nerve-wml PR #101 (branch `prereg/procgen-aarch64-build`, head
  `8ba0d8a53797e946cac256773a87d2dd8783f29b`), prepared in an arm64 container. Source
  `openai/procgen` 0.10.7, commit `5e1dbf341d291eff40d1f9e0c0a0d5003643aebf` (SHA-256 of
  the `git archive` recorded in the kit, `881b1f4a…0794`). Patches 0001–0003: remove
  `filelock<4` from `setup.py` (D6); restrict `-march` to x86 targets; add
  `-fsigned-char -ffp-contract=off` on aarch64. No SSE/AVX intrinsics are used, so no
  `sse2neon`. The macM1 patch (`bench/procgen-0.10.7-arm64-build.patch`, SHA-256
  `df9bbfa1…e7f`) only touches `if(APPLE)` and is inert on Linux. No game code changes.
- **On the host:** the build and a CoinRun smoke test (S3, S4) are mandatory A0 steps,
  inside the **3 h cap of A0**; if A0 is not done within 3 h, the session stops before any
  pilot run. The SHA-256 of the built wheel, of `libenv.so`, of the installed package tree
  and of the build log are recorded in the build record on the host, before the first
  pilot run (nothing is compiled before access).
- **Not bit-identical:** CoinRun levels from this build are not guaranteed identical to
  the x86-64 or macOS builds. This is acceptable: the pilot is intra-host and indicative
  (D2, section 1).

## 5. Software, hashed requirements and smoke tests

### 5.1 Hashed requirements (to be HITL-reviewed by the PI before access)

Generated 2026-10-31 on electron-server with `uv pip compile --python-platform
aarch64-manylinux_2_35 --python-version 3.12 --generate-hashes --no-build --exclude-newer
2026-10-17` (metadata and wheels only, nothing installed or built; `manylinux_2_35`
because the `mlx` 0.31.1 wheels are `manylinux_2_35_aarch64`; Ubuntu 24.04 has glibc
2.39). The sdist-only entries were hashed from the PyPI JSON digests. No file is newer
than 2026-10-17 (two weeks before the decision).

| file | SHA-256 | use |
|---|---|---|
| `requirements-gb300-n10c-bis-gate-cuda12.txt` | `4a10e6c6417bf6a9445cf82f53dc6f8e927ca26584ff4539895d865ad8444038` | venv A0-12 |
| `requirements-gb300-n10c-bis-gate-cuda13.txt` | `929339cd815b49244640f31c09944d45d10d89ab826a83dcab36613ee7db841a` | venv A0-13 |
| `requirements-gb300-build-tools.txt` | `70705ec0dbb8897aa502d421767cdc02909ccd673857018d93d838003b03b358` | `setuptools`, `wheel` for the sdist builds (no build isolation) |

Package-by-package summary with source and the explicit list of packages without an
aarch64 wheel: `requirements-review.md`. Without an aarch64 wheel (sdist, built on the
host under review): `gym` 0.26.2, `glcontext` 3.1.0, `glfw` 1.12.0, `moderngl` 5.13.0,
`imageio-ffmpeg` 0.3.0; Procgen 0.10.7 (source build, section 4). Versions match the
N10-C locks; the CUDA runtime wheels are resolved by uv for aarch64 (`nvidia-cublas-cu12`
12.9.2.10, `nvidia-cuda-nvrtc-cu12` 12.9.86, `nvidia-cudnn-cu12` 9.27.0.42,
`nvidia-nccl-cu12` 2.32.3; for CUDA 13: `nvidia-cublas` 13.8.1.7, `nvidia-cuda-nvrtc`
13.4.92, `nvidia-cudnn-cu13` 9.27.0.42, `nvidia-nccl-cu13` 2.32.3). Install, per venv:
`pip install --require-hashes --no-deps -r requirements-gb300-build-tools.txt`, then
`pip install --require-hashes --no-deps --no-build-isolation -r <file>`, then the
Procgen build. CPython 3.12.15 (D4): python-build-standalone aarch64 archive, its URL and
SHA-256 recorded in the build record.

### 5.2 Mandatory smoke tests (before the first gate run; none is a gate run)

| id | test | pass condition |
|---|---|---|
| S0 | host record: `nvidia-smi`, `nvidia-smi -q`, `uname -a`, `getconf PAGESIZE`, `lscpu`, `free -g`, `python -VV`, `pip freeze`, SHA-256 of the lock and of the checkout (`git rev-parse HEAD`, `git status --porcelain` empty) | recorded; HEAD is the merge commit of this annex or a descendant that changed only `docs/`, and `docs/prereg/n10c-bis-gate-gb300/` (lane included) is byte-identical to the merge commit; checked before S5, which writes a `measurement` entry into the ledger |
| S1 | `nvidia-smi --id=<UUID> --query-gpu=compute_cap` gives 10.3 (the lane checks it again); MLX on the GPU: `python -c "import mlx.core as mx; print(mx.default_device()); a=mx.ones((4,)); mx.eval(a@a)"`, then one forward and backward pass of a 3×3 `nn.Conv2d` on a 64×64×3 batch | runs on the GPU device without NVRTC / kernel errors; result finite |
| S2 | `python experiments/procgen_multiagent_gtm_comm/bench/bench_ppo_mlx.py` (NN-only, synthetic data, no environment) | completes; timings recorded |
| S3 | Procgen: `python -c "import procgen"`, then `bench/procgen-builds/frames.py` twice | both runs print 20 level hashes and the two outputs are byte-identical (host determinism); the hashes are recorded and are not expected to equal macM1 or KXKM-AI |
| S4 | real wrapper on training levels: `python -c "from experiments.procgen_multiagent_gtm_comm.coop_env import ProcgenCoopEnv as E; e=E(seed=9001, num_levels=16, start_level=0, allow_stub=False); o=e.reset(); assert not e.using_stub; print(type(o))"` (the `smoke_test` module is not used: it allows the stub and does not restrict levels) | exits 0, `using_stub` false |
| S5 | throughput on a frozen random policy, ledger kind `measurement` (not counted): `python -m experiments.procgen_multiagent_gtm_comm.harness.throughput throughput --arms eng nocomm --n-updates 10 --n-workers 8 --preset fix-c-coin-pretrain --init-trunk <trunk 9101> --host-settings kxkm --out <session>/smoke/throughput.json` | for each arm `training_failure` false and `pair_steps_per_s_incl_warmup` finite and > 0; these two values are `RATE_ENG` / `RATE_NOCOMM` of the time rule (section 9); during S5, `nvidia-smi --query-compute-apps=pid` is recorded and every listed PID must belong to the S5 process tree (this checks the lane's contamination filter, which excludes the trainer and its spawned env workers) |

`python` and `pip` in S0–S5 are those of the run venv (`$PY`). S1 is run from a small
script committed with the build record before access (fixed batch, loss, assertion of
finite values and gradients). S3 compares the JSON objects after dropping the
`building procgen...done` line that may precede them. A failed smoke test stops the
session before any gate run, except the backend rule of section 6 (D7). A smoke test may be repeated after an environment fix (each attempt logged);
no smoke test reads team success of a gate run.

## 6. MLX backend (D7)

- Both backends are installed, `mlx` 0.31.1 + `mlx-cuda-12` 0.31.1 (venv A0-12) and
  `mlx` 0.31.1 + `mlx-cuda-13` 0.31.1 (venv A0-13), and S1–S5 run in each (« A0 »).
- **Rule, fixed now:** if every A0 smoke test passes under `mlx-cuda-12`, the pilot runs
  under `mlx-cuda-12`; otherwise, if they pass under `mlx-cuda-13`, under `mlx-cuda-13`;
  otherwise the session stops. Speed and numerical values play no part. The backend kept
  (and the failing test, if any) is recorded; it is the same for every pilot run and
  never changes once the first pilot run has started.

## 7. Development-ledger allowance (G6, D3)

`eng` has used 13 of its 13 `gate_dev` slots and `nocomm` has none (default cap 4,
`harness/devledger.py`). G6 asks for « one run per arm and seed plus one rerun », raised
by a separate, dated code change before the first gate run. G2 allows one rerun **per
run** that crashes or aborts, and a crashed attempt still uses a slot, so with k seeds
each arm needs up to 2k slots: `eng` = 13 + 2k, `nocomm` = 2k. With k = 3 (D1, D3), the
companion commit sets `eng` 19 and `nocomm` 6; the test `test_trunk_fix.py` checks those
values. The ledger kind stays `gate_dev`; the
10M cap (`GATE_MAX_STEPS`) is unchanged. The ledger file of the GB300 checkout is copied
at the start of the session (`smoke_ledger.initial.json`); if any `gate_dev` attempt of
`eng` or `nocomm` was made elsewhere after the merge, the two ledgers are reconciled
before the session.

## 8. Commands

The gate runs are launched only by `docs/prereg/n10c-bis-gate-gb300/gate_g7_lane.sh`
(committed with this annex; its SHA-256 goes into the build record). From a clean clone
of `hypneum-lab/nerve-wml` at the merge commit (`REPO`), with the run venv (`PY`),
`CUDA_HOME` set, and absolute paths:

```bash
REPO=... PY=... OUT_ROOT=<new session dir> TRUNK_DIR=<dir with trunk-<seed>.safetensors> \
END_EPOCH=<end of access, Unix s> RATE_ENG=<S5> RATE_NOCOMM=<S5> \
GPU_UUID=<nvidia-smi -L> CORES=<taskset list> \
  nohup bash docs/prereg/n10c-bis-gate-gb300/gate_g7_lane.sh </dev/null >/dev/null 2>&1 &
```

The lane, in order: refuses an existing `OUT_ROOT`; copies the initial ledger; checks the
four trunk SHA-256 (section 3); then for each seed in order applies the time rule, and
for each arm in the seed's order checks the GPU, refuses an existing run directory and
runs

```bash
taskset -c "$CORES" nice -n 10 "$PY" -m experiments.procgen_multiagent_gtm_comm.harness.train \
  --arm <arm> --seed <seed> --condition closed --steps 10000000 --out <run dir> \
  --log-every 200000 --ckpt-every 1000000 --preset fix-c-coin-pretrain --dev-kind gate_dev \
  --micro-batch-pairs 512 --clear-cache --n-workers 8 --init-trunk <trunk>
```

with `CUDA_VISIBLE_DEVICES=<GPU UUID>`. `--log-every 200000` is the cadence of the 10M
`gate_dev` runs of section 4.3 part B of the 2026-10-25 amendment (50 logs; the measure of
G2 is the mean of the last 5, at 9.2M–10M). The liveness guard in `abort` mode (G2) comes
from the preset. The trainer's stdout, which carries the success curve, goes to a
**sealed** file (`<run dir>.result.sealed.json`); stderr to `<run dir>.stderr.log`.

**Run status.** A zero exit code does not establish success (a liveness abort returns
normally). A run is OK only if its `run.json` shows: `using_stub` false,
`training_failure` false, `liveness_abort` false, `pair_steps_done` = 10,000,000,
`init_trunk_sha256` = the table of section 3, `train_options.liveness` = `abort`,
`micro_batch_pairs` 512, `clear_cache` true, `n_workers` 8, and five final success logs
at 9.2, 9.4, 9.6, 9.8 and 10M pair-steps, finite, each over a window of 100 completed
episodes. The lane prints only OK or the names of the failed conditions, never a success
value. A run that is not OK gets its one rerun (G2) in a new directory; a second non-OK
attempt stops the session (section 3, INCONCLUSIVE).

## 9. Stopping rules and repatriation

**Stop the session (no further gate run)** on: a smoke failure not covered by the
backend rule of section 6; a trunk mismatch before the first run; a second non-OK attempt of a run; another
GPU compute process outside the trainer's process tree, before or during a run, or a
failed `nvidia-smi` query (contamination, fail closed; the run is killed); a GPU Xid /
ECC error or a host reset (the operator sends TERM to the lane, which kills the trainer
and writes `SESSION_STOP`; `nvidia-smi -q` is saved before and after each run so that an
ECC or Xid event can be found afterwards, and a run with such an event counts as a
hardware error); the deadline kill below (TERM, then KILL after 60 s). Every stop is written to
`SESSION_STOP` with its reason; section 3 gives the outcome.

**Blinding within the session.** Nobody reads the sealed result files, `run.json`
success fields or checkpoints before the session has ended (lane done or stopped).
Monitoring uses `driver.log` (start, end, exit code, OK / failed conditions), the
stderr logs, `nvidia-smi` and the process table only. The session ends irrevocably when
the lane exits; no seed, pair or rerun is added after the sealed files are opened.

**Time rule** (in the lane). With `RATE_ENG`, `RATE_NOCOMM` the S5 values
`pair_steps_per_s_incl_warmup`, a seed's pair starts only if
now + 1.5 × (10M / `RATE_ENG` + 10M / `RATE_NOCOMM`) ≤ end of access − 2 h. The rule is
not re-estimated from the gate runs during the session. A run still alive at end of
access − 1 h is killed: deadline stop (INCONCLUSIVE if it belonged to a started pair).

**Repatriation (before the end of access):**

1. `SHA256SUMS` with relative paths, sorted, over every file of the session directory
   except `SHA256SUMS` itself: run directories (`run.json`, checkpoints), sealed result
   files and logs, `driver.log`, `SESSION_STOP` if any, the initial and final ledger
   (`manifest/smoke_ledger.json` of the GB300 checkout, copied in), the four trunk files,
   the smoke, build and host records, the hashed lock, `pip freeze`, the exact
   environment (`env`), the commands run, the GPU UUID and PCI bus id, the core sets and
   timestamps of every job on the host;
2. one archive `n10c-bis-gate-gb300-20261101.tar` of that directory, and the archive's
   own SHA-256 in a separate file;
3. copy to electron-server `~/archives/hypneum/n10c-bis-gate-gb300/` and verify there
   (`sha256sum -c` on the archive hash and, after extraction, on `SHA256SUMS`);
4. nothing is deleted on the GB300 before step 3 succeeds.

`run.json.runtime` and `registration` are null for development runs, hence the separate
host record. The records (`run.json`, ledger entries merged by run id, build and smoke
records) are added later to `docs/superpowers/research/2026-10-25-n10c-gate/gb300/` and
appended to the addendum by a results PR, as for B3.

## 10. What this annex does not change

The gate itself (arms, condition, preset, 10M pair-steps, training levels, measure,
threshold 0.20, exploratory status G4; G3 does not apply to the pilot (D2); the failure handling of G2 is hardened as stated in D8); every N10-C
file, tag, manifest, run or host; the N10-C-bis hypotheses, arms, cells, seeds and host
of the confirmatory campaign (KXKM-AI). The freeze of this annex is the merge on green
CI before the first gate run; an annotated tag is optional for an exploratory gate and is
the PI's choice (if one is made, a gate-specific name, never `n10c-bis-freeze-*`). The
N10-C-bis freeze itself still depends on this gate's verdict, its entry in the stage-2
manifest, the PI decisions and the Filing procedure; merging this annex does not
authorise it. Deployed N10-C clones stay at their freeze tag: the annex's code change
makes this HEAD unusable for registered N10-C launches, as intended by the guard.
