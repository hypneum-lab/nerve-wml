# N10-C-bis necessity gate — G7 host annex: DGX Station GB300 (L0 draft)

**Status:** L0 DRAFT on branch `prereg/n10c-bis-gate-gb300`, not merged, not tagged. It
binds nothing until it is merged on green CI **before the first gate run**.
**Annex to:** section « Necessity gate (PI decision 2026-10-30) », items G1–G6, of
`docs/milestones/2026-10-25-n10c-bis-control-arms.md` (merged by PR #91, commit `1d08ecf`).
That addendum is still a DRAFT and append-only; its item G7 points to this file.
**Scope:** the necessity gate (`eng − nocomm ≥ 0.20`, condition `closed`, 10M pair-steps)
only. N10-C (tags `n10c-freeze-20261024`, `n10c-freeze-20261029`), its running campaign,
its hosts and every N10-C-bis provision other than G5 are **not modified**.
**Date drafted:** 2026-10-31, before any gate run on any host (no `gate_dev` entry exists
for `nocomm`; the 13 `eng` entries are earlier development runs of other designs).
**Decision behind it:** PI multiple-choice decision of 2026-10-31, 21:20 CET: prepare the
pre-registrations for a 24 h access to one DGX Station GB300 on 2026-11-01.

## 0. Open PI decisions (each one blocks the merge)

| id | decision | proposal in this draft |
|---|---|---|
| TBD-1 | Seed set and order (G2: only the PI adds seeds, before any gate run) | 9101, 9102, 9103, 9104, in this order (section 3) |
| TBD-2 | Run order: one run at a time (G5 as written) or the two arms of a seed concurrently on the one GPU | one run at a time (section 2.3) |
| TBD-3 | `GATE_DEV_ALLOWANCE` values; reading of G6 « plus one rerun » (per arm, or per run) | per arm: `eng` 13 + k + 1, `nocomm` k + 1; with k = 4: `eng` 18, `nocomm` 5 (section 7) |
| TBD-4 | Python 3.10 interpreter for the run venv (Ubuntu 24.04 ships 3.12 only) | CPython 3.10.18 (the KXKM-AI patch level), python-build-standalone aarch64, hashed |
| TBD-5 | Procgen Linux aarch64 build mode (the macM1 patch is inert on Linux, section 4) | in-place development build (`-march=native`), no source change |
| TBD-6 | Lock entries without an aarch64 wheel (`glcontext`, `glfw`, `moderngl`, `imageio-ffmpeg`) | build from sdist under HITL review; omission only as a declared deviation |
| TBD-7 | Exact patch versions of the CUDA 12 runtime wheels pulled by `mlx-cuda-12` 0.31.1 | the versions installed in the KXKM-AI `venv310`, read by the PI (not recorded in the repository) |

PI actions (not decisions): copy the trunk files of the chosen seeds to the GB300 host
(they live on macM1 / KXKM-AI, outside the repository) and review the hashed requirement
files before access (section 5).

## 1. Change of substrate (declared)

G5 fixes the gate host as **KXKM-AI**. This annex moves the gate, and only the gate, to a
**different substrate**:

| | G5 (KXKM-AI) | G7 (GB300) |
|---|---|---|
| CPU / ISA | x86-64 | 72-core Arm Neoverse V2 (Grace), aarch64, 64 KiB pages |
| GPU | RTX 4090, 24 GB, compute capability 8.9 | 1 Blackwell Ultra, 252 GB HBM3e, compute capability 10.3 |
| OS | Ubuntu (x86-64) | Ubuntu 24.04 aarch64, Linux 6.17, CUDA 13.1/13.2 stack |
| MLX backend | `mlx-cuda-12` 0.31.1 | `mlx-cuda-12` 0.31.1 (fallback `mlx-cuda-13` 0.31.1, section 6) |
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
3. **Limitation.** The gate decides whether the task needs the message, a property of the
   task that is assumed not to depend on the build. If the gate passes, N10-C-bis
   (bis-A) still trains on KXKM-AI with the x86-64 build; the gate result is evidence
   about the task on the GB300 build, carried over to KXKM-AI by that assumption. This is
   disclosed; it is not tested by this annex.
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
   N10-C job). Before each run, `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
   is logged; a process of ours other than the run blocks the start. CPU-only jobs of
   ours (the rescience-nh01 ARM reproduction) may run concurrently on a disjoint set of
   cores (`taskset`), never on the cores given to the gate run, and the core sets are
   logged.
3. **One run at a time** (G5), unless the PI decides otherwise under TBD-2. If the PI
   chooses concurrent arms, the two arms of one seed start together and no third run
   overlaps them.
4. **Per-run settings:** the KXKM-AI registered implementation settings named by G2
   (`micro_batch_pairs = 512`, `clear_cache = on`, `n_workers = 8`), passed explicitly
   on the command line (section 8) and recorded in each `run.json`. The larger GB300 memory
   is not used to change them.

## 3. Seeds (TBD-1)

Development seeds of the pilot namespace 91xx (G2), disjoint from the registered seeds
{0, 17, 42, 73, 101, 131} and from the smoke seed 9001. Proposed set, fixed before any
gate run, in this order:

| order | seed | trunk (`init_trunk_sha256`) | trunk record |
|---|---|---|---|
| 1 | 9101 | `0c27f2daa0a976e6407a85d6dfa6eafe20a6285c6324795b2194366b1a9aa704` | `docs/superpowers/research/2026-10-25-n10c-gate/macm1/gate/pretrain/pretrain.json` |
| 2 | 9102 | `b6805dee76c2f62be352383899d3b3b199458babbac1d260acdfffb78aa62bf1` | `…/macm1/gate-s9102/pretrain/pretrain.json` (the trunk of rows 20–21) |
| 3 | 9103 | `8c0b72ee4f362e133771f6e0de93e6bac96ced0edb077b8b63f2c974cfd24a7a` | `…/kxkm/adam-study/pretrain-s9103/pretrain.json` |
| 4 | 9104 | `894688cd4a62b868ea1f18d7021be57a64acb52699a8742048c9e2508da4f16f` | `…/kxkm/adam-study/pretrain-s9104/pretrain.json` |

- These seeds already carry development runs (registered condition, 1M pair-steps) of
  other designs; the gate runs use condition `closed`, whose seed streams
  (`SeedSequence([seed, arm, condition])`) differ from every earlier run. The gate is
  exploratory (G4), so reuse of development seeds is allowed; the four trunks avoid any
  new pre-training. No trunk is re-trained on the GB300: a trunk whose SHA-256 does not
  match the table is not used, and that seed is dropped (recorded) unless the PI decides.
- Seeds 9105 and above are not in the set. If the PI wants seeds never used by any
  development run, 9111–9114 occur nowhere in the repository, but they need new
  pre-trainings (`pretrain_dev` has used 8 of 8) and their trunks would be built on the
  GB300; that alternative is not drafted here.
- **Stopping by time, never by result.** Seeds run in the order above. A seed's pair is
  started only if both of its runs can end before the end of access minus the
  repatriation margin (section 9). The verdict uses the seeds whose two runs completed,
  which is a prefix of the order above. With one seed the statistic is that seed's
  difference; with more, the mean over seeds of the per-seed difference (G2).

## 4. Procgen build on Linux aarch64 (TBD-5)

- Source: `openai/procgen` tag `0.10.7`, commit `5e1dbf341d291eff40d1f9e0c0a0d5003643aebf`.
- The macM1 build patch `bench/procgen-0.10.7-arm64-build.patch`
  (SHA-256 `df9bbfa1d454ee51cbea1b8af47fbb1966a79f4a075357ba2a4c95acca834e7f`) removes
  `-Werror` inside `if(APPLE)` of `procgen/CMakeLists.txt`; **on Linux it changes
  nothing**. It is applied anyway, so that the source tree is the macM1 tree.
- Linux-specific risk: with `PROCGEN_PACKAGE=ON` (wheel build) the CMake file adds
  `-march=ivybridge`, an x86 target that GCC on aarch64 rejects; the in-place
  development build adds `-march=native` instead. Proposal: the in-place development
  build, the mode of the macM1 build (its import printed `building procgen...done`), with
  no source change beyond the patch. Any further change (a flag, a header) is a
  deviation: its diff and SHA-256 go into the build record before the first gate run.
- Toolchain: Ubuntu 24.04 `gcc`/`g++`, `cmake`, Qt 5 (`qtbase5-dev`), versions recorded.
- **Build record, written at build time** (this annex cannot know the values):
  procgen version string, source commit, patch SHA-256, compiler and CMake versions,
  Qt version, exact build command and environment, SHA-256 of the built `libenv.so`, of
  the installed package tree (sorted list of file hashes) and of the full build log.
  Stored with the run artefacts (section 9).

## 5. Software, hashed requirements and smoke tests

### 5.1 Required packages (to be hashed and HITL-reviewed before access)

No lockfile is downloaded or generated by this draft. The run venv needs the exact
versions below; for each, the aarch64 file is resolved, its SHA-256 recorded in
`requirements-gb300-aarch64.lock` (`--require-hashes`) and the file reviewed by a human
before access. The x86-64 / macOS hashes of `bench/requirements-*.lock` do not apply.

| package | version | aarch64 availability on PyPI (2026-10-31, metadata only) |
|---|---|---|
| CPython | 3.10.x (TBD-4) | python-build-standalone aarch64 build |
| `mlx` | 0.31.1 | `cp310 manylinux_2_35_aarch64` wheel |
| `mlx-cuda-12` | 0.31.1 | `py3-none-manylinux_2_35_aarch64` wheel |
| `nvidia-cublas-cu12` | 12.9.* (exact: TBD-7) | dependency of `mlx-cuda-12` |
| `nvidia-cuda-nvrtc-cu12` | 12.9.* (exact: TBD-7) | dependency of `mlx-cuda-12`; CUDA 12.9 is the first toolkit with `sm_103` |
| `nvidia-cudnn-cu12` | 9.* (exact: TBD-7) | dependency of `mlx-cuda-12` |
| `nvidia-nccl-cu12` | exact: TBD-7 | dependency of `mlx-cuda-12` |
| `numpy` | 1.26.4 | `cp310 manylinux_2_17_aarch64` wheel |
| `gym3` | 0.3.3 | pure-Python wheel |
| `gym` | 0.26.2 | sdist only |
| `gym-notices` | 0.1.0 | pure-Python wheel |
| `cffi` | 1.17.1 | `cp310` aarch64 wheel |
| `pycparser` | 3.11 | pure-Python wheel |
| `cloudpickle` | 3.1.2 | pure-Python wheel |
| `filelock` | 4.1.0 | pure-Python wheel |
| `imageio` | 2.38.0 | pure-Python wheel |
| `pillow` | 12.3.0 | `cp310` aarch64 wheel |
| `glcontext` | 3.1.0 | sdist only (TBD-6) |
| `glfw` | 1.12.0 | sdist only (TBD-6) |
| `moderngl` | 5.13.0 | sdist only (TBD-6) |
| `imageio-ffmpeg` | 0.3.0 | sdist only (TBD-6) |
| `procgen` | 0.10.7 | no aarch64 wheel, no sdist: source build (section 4) |
| fallback only: `mlx-cuda-13` | 0.31.1 | `py3-none-manylinux_2_35_aarch64` wheel; deps `nvidia-cublas`, `nvidia-cuda-nvrtc`, `nvidia-cudnn-cu13` 9.*, `nvidia-nccl-cu13` (exact versions hashed with the rest) |

Versions are those of `bench/requirements-mlx.lock`, `bench/requirements-procgen-py310.lock`
and the KXKM-AI `venv310` record of the stage-2 manifest fix C (python 3.10.18, mlx 0.31.1,
mlx-cuda-12 0.31.1, numpy 1.26.4, gym3 0.3.3, gym 0.26.2, procgen 0.10.7). The fallback
wheels are hashed and reviewed with the rest so that a fallback needs no unreviewed
download during access. `CUDA_HOME` points at CUDA headers for the MLX JIT kernels; the
headers used are recorded.

### 5.2 Mandatory smoke tests (before the first gate run; none is a gate run)

| id | test | pass condition |
|---|---|---|
| S0 | host record: `nvidia-smi`, `nvidia-smi -q`, `uname -a`, `getconf PAGESIZE`, `lscpu`, `free -g`, `python -VV`, `pip freeze`, SHA-256 of the lock and of the checkout (`git rev-parse HEAD`, `git status --porcelain` empty) | recorded; HEAD is the merge commit of this annex or a descendant that changed only `docs/` |
| S1 | MLX on the GPU: `python -c "import mlx.core as mx; print(mx.default_device()); a=mx.ones((4,)); mx.eval(a@a)"`, then one forward and backward pass of a 3×3 `nn.Conv2d` on a 64×64×3 batch | runs on the GPU device without NVRTC / kernel errors; result finite |
| S2 | `python experiments/procgen_multiagent_gtm_comm/bench/bench_ppo_mlx.py` (NN-only, synthetic data, no environment) | completes; timings recorded |
| S3 | Procgen: `python -c "import procgen"`, then `bench/procgen-builds/frames.py` twice | both runs print 20 level hashes and the two outputs are byte-identical (host determinism); the hashes are recorded and are not expected to equal macM1 or KXKM-AI |
| S4 | `python -m experiments.procgen_multiagent_gtm_comm.smoke_test --steps 100` | `using_stub` is false (no stub warning); shapes as on KXKM-AI |
| S5 | throughput on a frozen random policy, ledger kind `measurement` (not counted): `python -m experiments.procgen_multiagent_gtm_comm.harness.throughput throughput --arms eng nocomm --n-updates 5 --n-workers 8 --preset fix-c-coin-pretrain --init-trunk <trunk 9101> --host-settings kxkm --out <dir>` | completes; pair-steps/s recorded; used only for the time rule of section 3 |

A failed smoke test stops the session before any gate run, except S1/S2 failures that
meet the fallback trigger of section 6. A smoke test may be repeated after an
environment fix (each attempt logged); no smoke test reads team success of a gate run.

## 6. MLX backend and the declared fallback

- **Primary:** `mlx` 0.31.1 with `mlx-cuda-12` 0.31.1, as on KXKM-AI.
- **Fallback (a declared, possible deviation):** `mlx` 0.31.1 with `mlx-cuda-13` 0.31.1.
- **Trigger, and only trigger:** S1 or S2 fails under `mlx-cuda-12` because its kernels do
  not compile or do not run on compute capability 10.3 (NVRTC or JIT compile error,
  « no kernel image », unsupported architecture, illegal instruction in a CUDA kernel),
  reproduced once after a clean venv rebuild. Slowness, a different numerical result,
  memory pressure or a failure of Procgen are **not** triggers.
- The backend is fixed before the first gate run and is the same for every gate run.
  A backend switch after the first gate run started is not allowed: the session stops
  and the PI decides. A triggered fallback is recorded (error output, both pip freezes)
  in the run artefacts and in the results note; it changes a pinned version and is
  reported as a deviation.

## 7. Development-ledger allowance (G6, TBD-3)

`eng` has used 13 of its 13 `gate_dev` slots and `nocomm` has none (default cap 4,
`harness/devledger.py`). G6 asks for « one run per arm and seed plus one rerun », raised
by a separate, dated code change before the first gate run. Reading per arm, with k seeds:
`eng` = 13 + k + 1 and `nocomm` = k + 1. The companion commit of this branch sets the
values for the proposed k = 4 (`eng` 18, `nocomm` 5) and is marked TBD by PI: if the PI
keeps another k, or reads the rerun as one per run (`eng` 13 + 2k, `nocomm` 2k), the
values are edited before the merge. The ledger kind stays `gate_dev`; the 10M cap
(`GATE_MAX_STEPS`) is unchanged.

## 8. Commands

From a clean clone of `hypneum-lab/nerve-wml` at the merge commit (`REPO`), with the run
venv (`PY`), `CUDA_HOME` set and `PYTHONPATH=.`; for seed `S` with trunk `T` and output
root `OUT` (a new directory per attempt, never reused):

```bash
for ARM in eng nocomm; do   # order within a seed: eng, then nocomm
  (cd "$REPO" && nice -n 10 "$PY" -m experiments.procgen_multiagent_gtm_comm.harness.train \
    --arm "$ARM" --seed "$S" --condition closed --steps 10000000 \
    --out "$OUT/gb300-closed-$ARM-s$S" --log-every 200000 --ckpt-every 1000000 \
    --preset fix-c-coin-pretrain --dev-kind gate_dev \
    --micro-batch-pairs 512 --clear-cache --n-workers 8 --init-trunk "$T" \
    > "$OUT/gb300-closed-$ARM-s$S.log" 2>&1)
done
```

`--log-every 200000` is the cadence of the 10M `gate_dev` runs of section 4.3 part B of
the 2026-10-25 amendment (50 logs; the measure of G2 is the mean of the last 5, i.e.
9.2M–10M). The liveness guard in `abort` mode (G2) comes from the preset
(`train_options.liveness = "abort"` in `run.json`, as in the part-B records). Each run's
`run.json` must show `using_stub = false`, the preset, `train_options.liveness = "abort"`,
the trunk hash of section 3, the three per-run settings and `pair_steps_done = 10000000`;
a run that does not is reported and not used.

## 9. Stopping rules and repatriation

**Stop the session (no further gate run)** on: a smoke failure outside the fallback
trigger; a GPU Xid / ECC error or a host reset; a second failure of the same run
(G2: one ledgered rerun after a crash or a liveness abort, then report and the PI
decides); any process of ours other than the run found on the GPU during a run (that run
is reported as contaminated, not rerun without a PI decision); a trunk hash mismatch for
every remaining seed.

**Blinding within the session:** the team-success curve of a gate run is not read before
the last gate run of the session has ended; liveness probes, abort status, throughput
and crash logs may be read.

**Time rule:** a seed's pair starts only if, at the S5 rate, its two runs end at least
**2 h** before the end of access (repatriation margin).

**Repatriation (before the end of access):**

1. `sha256sum` of every file under the session directory (run directories with
   `run.json`, logs and checkpoints, the ledger file `manifest/smoke_ledger.json` of the
   GB300 checkout, the smoke and build records, pip freeze, host record) into
   `SHA256SUMS`;
2. one archive `n10c-bis-gate-gb300-20261101.tar` of that directory plus `SHA256SUMS`,
   and the archive's own SHA-256;
3. copy to electron-server `~/archives/hypneum/n10c-bis-gate-gb300/` and verify there
   (`sha256sum -c` on the archive hash and, after extraction, on `SHA256SUMS`);
4. nothing is deleted on the GB300 before step 3 succeeds.

The records (`run.json`, ledger entries merged by run id, build and smoke records) are
added later to `docs/superpowers/research/2026-10-25-n10c-gate/gb300/` and appended to
the addendum by a results PR, as for B3.

## 10. What this annex does not change

The gate itself (arms, condition, preset, 10M pair-steps, training levels, measure,
threshold 0.20, failure handling, consequences G3, exploratory status G4); every N10-C
file, tag, manifest, run or host; the N10-C-bis hypotheses, arms, cells, seeds and host
of the confirmatory campaign (KXKM-AI). The freeze of this annex is the merge on green
CI before the first gate run; an annotated tag is optional for an exploratory gate and is
the PI's choice.
