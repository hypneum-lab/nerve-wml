# Changelog

All notable changes to `nerve-wml` follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **N10-C-bis pre-registration addendum (DRAFT, 2026-10-25), documentation
  only.** Three control arms the running N10-C campaign lacks: a
  mismatched-message arm (content vs presence), a warm-started receiver
  (optimisation vs representation, two stages) and an IB-VQ baseline with
  GTM's K = 4 x A = 16 symbol structure (modern baseline, symbol-structure
  control); directional hypotheses H-bis-1..3 with SESOI 0.10, estimation
  plan in the N10-C format (6 seeds, Welch, adjusted intervals, pilot
  categories), compute plan on KXKM-AI after the N10-C KXKM seeds, and the
  binding rule "no positive reading of N10-C before N10-C-bis". N10-C
  (tag `n10c-freeze-20261024`, 60 runs, started 2026-10-24) is not
  modified. `docs/milestones/2026-10-25-n10c-bis-control-arms.md`.

### Corrections

- **L1a gate did not pass as specified at seed 0, n=1024.** The spec
  (section 5.2) requires untrained accuracy <= 10 %; the recorded run has
  13.1 %, and `scripts/lif_l1a_train.py` never evaluated that condition,
  so the "closes the L1a gate" claim in v1.9.0 (and `gate_passed: true`
  in the 2026-05-30 JSON) holds only for the other two conditions. The
  learned gain is unchanged (89.1 % final vs 13.1 % untrained, chance
  8.3 %). Gate script now checks all three conditions. Details, all sizes
  and affected claims:
  `docs/superpowers/research/2026-10-24-lif-l1a-gate-erratum.md`.
  No Zenodo record exists for v1.9.0 (no GitHub release was published,
  so the GitHub-Zenodo webhook never fired); the L1a claim was never
  archived on Zenodo. The correction must be carried in the
  description of the next published release.
- **Erratum 2026-10-25 — N-3 ablations do not establish "type-checker,
  not differential mechanism" (issue #71).** The three ablations behind
  that reading ran on an inactive mechanism: `n_received: 0` in all 30
  cells of `papers/paper1/figures/ablation_n3_guard.json` and
  `apply_consolidation: false` (AttributeError on `MockNerve`) in all
  six cells of `ablation_n3_predictive.json`, so Δ = 0 is
  uninformative. Requalified in Paper 1 §Method and §Threats to
  Validity, `docs/research-notes/n3-gate-role.md` and the README:
  typing function established (the guard is exercised at send time,
  `n_caught: 70` at v = 0.05, seed 0); causal utility for communication
  and learning not established by these ablations. A differential test
  needs `n_received > 0`, active consolidation and a pre-registered
  matched null. JSONs unchanged.
- **Erratum 2026-10-25 — stale facts in Paper 1 and metadata (issue
  #64).** Residual "15/15" mentions in `main.tex` (figure caption,
  Sleep-EDF paragraph) and in `scripts/render_scaling_figure.py` aligned
  on the 19/20 recount (4/5 at N=2, 5/5 at N=16/32/64); the scaling
  table's N=2 row now reports the 5-seed rerun
  (`w2_hard_n2_multiseed.json`: median 10.71 %, max 16.91 %, 4/5)
  instead of the 1-seed value; "all experiments use N=4" (§Experiments,
  §Limitations) and "pool-scale multi-seed for TRF is future work"
  replaced by the measured N ∈ {16, 32, 64} and triple-substrate
  N ∈ {15, 30, 60} results. Intel Loihi 2 is no longer presented as a
  deployment target: the `lava-nc` SDK is archived upstream (GitHub
  `archived: true`, last push 2026-05-13); `neuromorphic/loihi_stub.py`,
  `docs/neuromorphic/deployment-guide.md`, README and Paper 1 §Gate
  Neuro requalified (the stub still raises `NotImplementedError`).
  Bylines reduced to the single PI (Clément Saillant) in
  `.zenodo.json`, `CITATION.cff`, `CONTRIBUTORS.md` and the design spec.
  The committed `w2_hard_scaling.pdf/.png` still carry the old title and
  must be re-rendered. Details:
  `docs/superpowers/research/2026-10-25-phase0-truth-errata.md`.

## [1.9.4] — 2026-06-20

Paper 1 §Information Transmission gains Test (12): end-to-end validation
of Claim B through the *full deployable discrete protocol* (learned
per-edge codebook transducer + γ/θ PAC channel), closing the
idealised-path gap of Tests (2)–(3). Framed honestly as a consolidation
of the existing cross-substrate alignment (Platonic effect, Test 6), not
a new phenomenon.

### Added

- **Paper 1 Test (12), §Information Transmission** — fidelity ratio
  acc_roundtrip/acc_direct over 20 seeds on HardFlowProxyTask through
  the real discrete protocol: (a) idealised continuous-logit path 1.007
  CI95 [0.975, 1.037]; (b) untrained transducer + γ/θ channel 0.168
  [0.158, 0.179] (round-trip accuracy 0.092 ≈ chance 1/12); (c) trained
  transducer + γ/θ channel **0.924** [0.892, 0.954]. γ/θ channel
  code-survival 1.00 at every seed (PSK constellation analytically
  max-min-distance optimal → never the bottleneck); the untrained
  per-edge transducer is the bottleneck, task-supervised Gumbel-softmax
  training closes the gap; ~8% residual is the discretisation cost.
  Numbers trace to `papers/paper1/figures/transmission_full_protocol.json`.

### Changed

- **Abstract Claim B** now reports deployable-protocol round-trip
  fidelity (0.924) alongside the idealised-path 99%/97% figures.

## [1.9.3] — 2026-06-20

BioFieldWML Phase 2 (SNN-PC belief substrate) transitions from unit-test-only
scaffolding to a measured pilot result on HardFlowProxyTask (12-class
XOR-on-noise, n=50, kxkm x86 CPU). Ablation discloses that the |error| feature
is a task-aligned shortcut; the honest substrate-intrinsic edge is μ-driven.

### Added

- **`papers/paper1/figures/biofield_edge_null_n50.json`** — full [μ,σ,|error|]
  probe result (n=50, HardFlowProxyTask, kxkm x86 CPU, ~212 s). BioField acc
  0.8502 vs MLP 0.5549; mean edge +0.2953 CI95 [0.2745,0.3150] (50/50
  positive, CI excludes 0, exceeds symmetric init-variance null p95=0.1036).
- **`papers/paper1/figures/biofield_ablation_n50.json`** — feature-mask
  ablation: three variants (full [μ,σ,|error|], no_error [μ,σ], mu_only [μ])
  each run n=50 against the same MLP baseline and MLP-vs-MLP null (~380 s).

### Result (from `biofield_edge_null_n50.json` + `biofield_ablation_n50.json`)

**Honest headline — substrate-intrinsic edge is +0.14, not +0.30.**

| Variant | Mean edge | CI95 | Sign rate | > null p95 |
|---|---|---|---|---|
| Full [μ,σ,\|error\|] | **+0.2953** | [0.2745, 0.3150] | 50/50 | yes |
| No error [μ,σ] | **+0.1391** | [0.1161, 0.1616] | 48/50 | yes |
| μ only | **+0.1413** | [0.1178, 0.1651] | 48/50 | yes |
| MLP-vs-MLP null | −0.0041 (mean abs 0.0227) | [−0.0168, 0.0077] | ~coin-flip | — |

The full +0.2953 edge must be reported alongside the explicit caveat that
~0.16 of it is the |error|=|input_proj(x)| feature — a task-aligned shortcut,
not a belief-state property. Removing |error| isolates the genuine
belief-substrate expressivity edge: **+0.1391 (μ,σ) / +0.1413 (μ only),
CI95 excludes 0, 48/50 positive, above symmetric init-variance null p95=0.1036**.
σ adds nothing beyond μ alone (Δ ≈ 0.002). **μ-driven.** Honest scope:
substrate-INTRINSIC on HardFlowProxyTask; NOT agnostic/SOTA.

### Changed

- `pyproject.toml` version bump `1.9.2` → `1.9.3`.

## [1.9.2] — 2026-06-20

Null-control for the two surviving findings: scaling-plateau real floor and
pool-scale vs architecture-scale orthogonality. The committed
`run_bigger_arch.py` reported a ~26 % relative gap at d=128 on 3 seeds
with no null; this release closes the methodological gap with n=50 seeds
and a symmetric MLP-vs-MLP init-variance null for both.

### Added

- **`papers/paper1/figures/orthogonality_null_n50.json`** — arch-scale
  null-control artifact (n=50, d ∈ {16,128}, N=16, kxkm-ai x86 Linux CPU).
  d=128: real edge 0.1204 CI95 [0.1075,0.1326] excl.0, 49/50 positive,
  4.83× symmetric init-variance null; d=16: 0.0099 CI95 [0.0073,0.0125],
  43/50, 3.14× null.
- **`papers/paper1/figures/ortho_corner_check_n10.json`** — N×d corner check
  (10 seeds × {N=2,N=64} × {d=16,d=128}): d=128 gap ~0.20 at N=2 and N=64
  (pool-scale ~unchanged); d=16 gap 0.036→0.020 as N grows — axes confirmed
  independent.
- **`scripts/orthogonality_null_replication.py`** — committed driver importing
  the existing `run_bigger_arch` / `track_w_pilot` drivers and the
  `nerve_wml.methodology` helpers (no inline duplication of null logic).
  Header note: Run on kxkm-ai per compute policy, CPU cross-arch.

### Result (from `orthogonality_null_n50.json` + `lif_edge_null_n50.json`)

**Orthogonality (finding #6):** d=128 LIF–MLP edge 0.1204, CI95 [0.1075,
0.1326] excludes 0, 49/50 seeds positive, **4.83× symmetric init-variance
null**. d=16: 0.0099, CI95 [0.0073,0.0125], 43/50, 3.14× null. N×d corner
confirms pool-scale and arch-scale move the gap independently.

**Scaling plateau (finding #1 / #3):** at N=64 edge 0.0092, CI95
[0.0066,0.0118] excludes 0, 40/50 positive, ~2× above null p95 → real
non-zero ~1 % floor confirmed (from committed `lif_edge_null_n50.json`).

Honest scope: substrate-INTRINSIC effects on HardFlowProxyTask, NOT
agnostic/SOTA.

### Documentation

- README architecture-scale row updated to cite null+CI (d=128 4.83× null,
  d=16 3.14× null, N×d corner, both JSON artifacts).
- README findings #6 and "probably novel" paragraph updated with null
  evidence and N=64 plateau CI.
- README summary sentence updated with N=64 ~1 % floor CI.

## [1.9.1] — 2026-06-20

Null-control and bootstrap CI for the LIF spike-expressivity edge (finding #5).
The previously missing null closes the methodological gap flagged at review:
the directional LIF advantage was reported in 19/20 pairwise seeds but had no
same-substrate null to exclude init-variance as an alternative explanation.

### Added

- **`nerve_wml.methodology.accuracy_edge_null`** — two reusable primitives:
  `accuracy_edge_bootstrap_ci(edges)` returns `AccuracyEdgeCiResult` (bootstrap
  95 % CI on per-seed signed accuracy differences, sign-rate, zero-exclusion
  flag); `same_substrate_null_result(null_edges)` returns
  `SameSubstrateNullResult` (init-variance band: mean |edge|, abs_p95,
  symmetry CI). Both exported from `nerve_wml.methodology`.
- **12 unit tests** (`tests/unit/test_accuracy_edge_null.py`): symmetric null
  includes zero; constant positive edge CI excludes zero; sign-rate near 50/50
  for symmetric null; abs_p95 correct; empty-input raises; reproducibility.
- **`scripts/lif_edge_null_replication.py`** — committed driver that imports
  the real `run_w2_hard` / `_run_w2_hard_scale` from `scripts/track_w_pilot.py`
  and the new methodology helpers. Header note: Run on kxkm-ai per compute policy.
- **`papers/paper1/figures/lif_edge_null_n50.json`** — result artifact from the
  kxkm-ai x86 Linux CPU run (n=50 seeds, 2026-06-19, ~1708 s).

### Result (from `lif_edge_null_n50.json`)

Pooled sign-rate: **173/200** favour LIF. Bootstrap CI95 **excludes 0 at every
N ∈ {2, 16, 32, 64}** (CI95: N=2 [+0.050, +0.084]; N=16 [+0.019, +0.029];
N=32 [+0.007, +0.014]; N=64 [+0.007, +0.012]). MLP-vs-MLP same-substrate null
is symmetric around 0 at every N (sign-rate ≈ coin-flip; null CI includes 0).
Real edge is **2.7×–5.4×** the null mean |edge| at pool scale (N=16: 5.4×;
N=32: 2.7×; N=64: 4.4×). **N=2 caveat:** fat right tail in the null means a
single seed's edge can be matched by init luck (`real_edge_exceeds_null_p95=False`);
effect is cleanest at pool scale (N≥16) where the null collapses (~0.002–0.004)
and `real_ci_lo_above_null_p95` holds. Compute: x86 Linux CPU (cross-arch
replication of macM1 origin). Not a GPU result.

### Documentation

- README finding #5 and direction-stability row updated to cite the null+CI
  result (n=50, cross-arch, CI95 excludes 0 all N, 2.7×–5.4× null, caveat N=2).
- README "probably novel" paragraph reframes the edge magnitude honestly
  (~1–7% at pool scale) and adds the null-control evidence.

## [1.9.0] — 2026-06-15

Two external-review items answered with merged work, two ADRs, and a lockstep
paper requalification of the substrate-agnosticism claim. No headline number was
fabricated or inflated; the interpretive scope was narrowed and the bias/
representation caveat is now mandatory on every citation of the MI/H figure.

### REVIEW ② — MI/entropy estimator bias (ADR-0001)

`docs/adr/0001-mi-bias-correction-and-null-model.md` (accepted). Answers the
"strongest attack" (Paninski 2003 / Panzeri 2007): a plug-in MI carries a
positive finite-sample bias, so an inflated MI/H could be the null expectation
rather than evidence of alignment.

- **Added** a **Panzeri–Treves** analytic bias-corrected discrete estimator
  (`mi_panzeri_treves_discrete`) and a **permutation null for every estimator**
  (shuffle the `c_MLP ↔ c_LIF` pairing). Re-measurement on the real saved codes
  (`tests/golden/codes_mlp_lif.npz`, N=5000 eval samples, 3 seeds, 800-step
  training) via `scripts/remeasure_mi_review_item2.py`.
- **Two-stage honest verdict:**
  1. *Not* a sampling-bias / null artefact. At N=5000 the plug-in bias is
     negligible (0.007 bit): plug-in MI/H **0.932**, Panzeri–Treves **0.931**,
     Miller-Madow 0.939, permutation null ~0.005, **p < 0.01** across all seeds.
     The figure is held up by large N, **not** by seed-to-seed stability.
  2. *But* ~**10× over-stated** as "shared information". The 0.93 is post-VQ
     (argmax over a ~7–27-symbol alphabet). Debiased **Kraskov–KSG on the
     pre-VQ continuous embeddings** of the same activations gives MI/H ≈
     **0.095** — the argmax quantisation manufactures most of the apparent
     alignment (H(a) collapses alongside it). VQ amplification ratio ≈ **9.8×**.
- **Sampling-bias control** (two independent 8-symbol streams, true MI = 0)
  reproduces the Panzeri mechanism quantitatively: plug-in null MI = 1.395 bit
  at N=20 → 0.007 bit at N=5000; Panzeri–Treves drives it to ~0.

### REVIEW ① — novelty vs prior art (ADR-0002 + paper requalification)

`docs/adr/0002-novelty-vs-prior-art-relative-representations.md` (accepted).
Answers the prior-art attack (Relative Representations / zero-shot stitching,
Moschella et al. 2022 arXiv:2209.15430; Platonic-vs-Aristotelian, arXiv:2602.14486).

- **Conceded as prior art:** cross-architecture representational convergence is
  **not novel to this work** — it was established by Moschella et al. and the
  zero-shot stitching literature in 2022–2023 (PRH, Huh et al. 2024). No
  experiment re-run; the existing controls (Test 10 frozen-encoder, ADR-0001)
  already bound the claim.
- **Contribution delimited** (lockstep edit across three sites in
  `papers/paper1/main.tex`): §Related Work explicit concession; new
  "Delimiting the contribution" paragraph (`sec:contribution-delim`) splitting
  *established (not claimed)* from *new here* = measurement at the **discrete
  protocol-output** level on structurally heterogeneous MLP / spiking-LIF /
  Transformer substrates; §Threats Test (10) "Platonic-vs-Aristotelian caveat"
  noting a low gap / high MI is bounded by shared task+data+objective and does
  *not* demonstrate a task-independent universal code.

### Paper requalified (lockstep, ADR-0001)

`papers/paper1/main.tex` — the MI/H figure is now stated as a **post-VQ**
measurement at the §Claim B measurement site, in the contributions list, and in
a new "Bias and representation caveat" passage: survives Panzeri–Treves
debiasing (0.931) and the permutation null (~0.005, p < 0.01), **but** KSG on
the pre-VQ continuous embeddings gives MI/H ≈ 0.095, so the figure must be cited
alongside the continuous value. Paper-integration tests pass (13/13).

### Documentation

- `README.md` — Claim B and the headline MI table now cite the post-VQ figure
  with the N=5000 / debiased / p<0.01 qualifier **and** the pre-VQ continuous
  value ≈0.10; the bare "0.91–0.96 / 0.96" wording no longer stands without the
  VQ caveat.

### Added (substrates & track-w/track-p, since v1.8.1)

- **L1a Full-LIF sim training + scaling sweep** (#48) — `feat(lif)`: end-to-end
  LIF simulation training with alphabet scaling sweep; closes the L1a gate.
- **EWC hardening — honest rehearsal vs EWC** (#46) — `feat(track-w)`:
  diagonal Fisher EWC (`track_w/continual/ewc.py`) alongside fixed-replay
  `RehearsalBuffer` (`rehearsal.py`); honest head-to-head comparison shipped.
- **EWC hardening + track_w status docs** (#47) — `docs(track-w)`: documents
  the EWC / rehearsal verdict and updated substrate status.
- **SpikingKikiWML substrate scaffold** (#38) — `feat(track_w)`: new conformant
  substrate `SpikingKikiWML` under the WML Protocol.
- **MmapNpzProvider for SpikingKiki bundle** (#39) — `feat(track_w)`:
  memory-mapped NPZ provider for zero-copy spike bundle access.
- **Spiking-kiki bridge design notes** (#40) — `docs`: bridge #5/#6 design
  notes formalising the SpikingKiki ↔ nerve-wml interface.
- **Refresh track_w substrate list in README** (#41) — `docs(readme)`.
- **SpikingKikiBioClient bio coupling** (#42) — `feat(track_w)`: bio coupling
  client for the SpikingKiki bridge.
- **SpikingKikiBlockProvider Protocol** (#43) — `feat(track_w)`: typed Protocol
  for block-level SpikingKiki providers.
- **WML substrate paradigms map** (#37) — `docs`: rate-LIF / PC-VMP / wet-bio
  bridge paradigm map added to docs.
- **track-p role out-of-band 2-channel carrier** (#45) — `feat(track-p)`:
  role tag delivered over a separate out-of-band channel alongside the
  primary nerve carrier; closes the role-carrier gate.

## [Unreleased] — 2026-05-20 (gap-analysis remediation sprint)

### Summary

41 commits across 8 PRs merged into master. 458 fast tests passing (baseline 385), +73 net new assertions. Six empirical claims revised through systematic statistical reinforcement; one critical 4-arm ordering claim (spectral_entropy) further revised post-merge via batch-size sensitivity discovery. Fact-check audit protocol adopted and shipped with CI enforcement. Four GPU backends benchmarked across 4 hosts; MLX issue #3568 filed documenting M1 vs M3+/M5 bit-divergence in erf.h.

### Claims revised under scrutiny (6 total)

1. **spectral_entropy 4-arm strict ordering** (`null < akorn_best < gtm < simple_gating`) — batch-size dependent. Only B=256 reproduces canonical ordering; B∈{64,128,512} show akorn_best below null. **Revised claim:** gtm > null (Δ ≥ 0.05 bits, p < 1e-15) and gtm < simple_gating (Δ ≥ 1.0 bits, p < 1e-15) hold at all B; akorn_best reported as high-variance intermediate (σ ≈ 0.6).
2. **CKNNA scale artefact** (N-dependent performance drop 0.92 → 0.87) — confirmed 0.92 at N=256, 0.87 at N=16384 (σ=0.05), cross-host reproducible (M5 / macM1 / M3 Pro / M3 Ultra). Theoretical interpretation added (Gröger, Wen & Brbić 2026 null-calibration framework).
3. **MLP vs MPS speedup ratio** at CKNNA N=16384 — measured as 1.05× (unified-memory M5 GPU memory-bound, matches CPU behaviour). Cross-hardware MLX bench revealed M1 vs M3+/M5 bit-divergence.
4. **Transducer training wall-clock** (500 steps, 64-alphabet, CPU M5) — confirmed 0.83 s; MPS 1.3× speedup, CUDA 3.9× speedup (on contested RTX 4090 with ~4 GB peak Gram overhead).
5. **GPU necessity for nerve-wml** — verdict revised to "not needed". Transducer training <1 s on CPU; CKNNA breakeven at N≥4096 with MPS yielding modest 2–3× (not load-critical). SSH latency to CUDA host exceeds any realistic in-process batch amortization.
6. **AKOrN minimal flavour specification** — n_oscillators=32 in Plan A.4 differs from Miyato et al. (ICLR 2025) full AKOrN (n=64). Documented as limited-capacity variant; strict equivalence claim removed.

### Plans completed

- **Plan A** — Validation suite (11 tasks, Claims 1–7 Renf 1–6, backend bench)
- **Plan A.2** — Statistical hardening (5 tasks, null-model calibration, paired Wilcoxon p-values, bootstrap CI95)
- **Plan A.3** — Hyperparameter wiring (3 tasks, n_steps=32, lr=0.05, n_osc ∈ {32,64})
- **Plan A.4** — AKOrN comparison (3 tasks, minimal vs full topology, separation metrics)
- **Plan B** — Paper revisions (8 tasks, §Method, §Results, new citations, abstract figures)
- **Plan C** — Biological substrate (11 tasks, BioFieldWML Phase 1–2, CL1 sleep-every validation, Lee et al. 2024 SNN-PC scheduler)

### Reinforcement campaigns (9 confirmed, 1 audit)

- **Renf 1** — AKOrN sweep 50 seeds: top cell synchrony 0.4542 ± 0.1624 (robust across hp variants).
- **Renf 4 v3** — Extended eval: transducer 67.8 s, gtm 359.6 s, scale factor 2.4 s (cpu multiproc).
- **Renf 5** — 3-way backend bench (CPU M5 vs MPS M5 vs CUDA RTX 4090): CKNNA wins CPU for n≤1024, CUDA 40× at n=4096–8192, memory-bound parity at n=16384.
- **Renf 6** — 1750 cells = 50 seeds × 5 σ × 7 N (CKNNA scale sensitivity macm1 CPU multiproc, 52 min wall-clock).
- **Renf 7** — Spectral entropy means and arm ordering across hosts (macm1 CPU 254.7 s, MPS 253.3 s, tied memory-bound).
- **Renf 8** — Transducer 50 seeds, tie p=0.627 (M5 + macm1, statistically flat).
- **Renf 9** — MLX 2.07× speedup M5 MPS vs CPU at N=16384 (informational; confirmed memory-bound saturation).
- **Renf 10** — Batch-size sweep B∈{64,128,256,512}: spectral_entropy arm ordering B-dependent, 2-arm claims robust.
- **Audit** — Fact-check protocol: 27 OK / 0 DIVERGENT / 5 ORPHAN on master at HEAD acbdc34. ORPHANs are Renf 11–13 branches not yet merged.

### Cross-hardware MLX investigation

- **Issue filed**: ml-explore/mlx#3568 — Apple GPU bit-divergence in `mx.random.normal` (M1 applegpu_g13s vs M3 Pro/M3 Ultra/M5 g15+).
- **Triangulation**: 4-host survey (M1 Max, M3 Pro, M3 Ultra, M5), 32 hash comparisons, MLX 0.31.2 identical version across all.
- **Localization**: `mlx/backend/metal/kernels/erf.h:42–69` erfinv branch (documented, not a bug per se).
- **Classification**: documentation-request scope (no bit-exact guarantee across Apple GPU architectures in MLX 0.31.2).

### Fact-check audit protocol adopted

- **Rule**: every numerical claim in paper / research note must trace to a JSON cell or executed log line in the same session.
- **Tool**: `scripts/factcheck_audit.py --ci` enforces mechanically; CI workflow `.github/workflows/factcheck.yml` runs on push/PR to `docs/superpowers/research/**`.
- **Coverage**: 13 headline claims enumerated; 8 source JSONs on master (Renf 1, 4, 5, 6, 7, 8, 9, 10).
- **State at acbdc34**: 27 checks OK (no divergence), 0 broken, 5 orphaned (awaiting unmerged branch data).
- **Tolerance**: per-claim override support; defaults 0.005, string-match for boolean verdicts.

### Backend benchmarks (GPU recommendation updated)

| Workload | CPU M5 | MPS M5 | CUDA RTX 4090 | Recommendation |
|---|---:|---:|---:|---|
| CKNNA N=256 | 0.61 ms | 4.38 ms | 0.15 ms | **CPU** (launch overhead) |
| CKNNA N=1024 | 6.31 ms | 7.72 ms | 0.25 ms | **CPU** (parity) |
| CKNNA N=4096 | 113.14 ms | 51.14 ms | 2.61 ms | **CUDA** 43× (MPS 2.2×) |
| CKNNA N=8192 | 454.10 ms | 190.04 ms | 11.65 ms | **CUDA** 39× (MPS 2.4×) |
| CKNNA N=16384 | 3195.67 ms | 3058.64 ms | **OOM** | **CPU/MPS** (memory-bound parity) |
| Transducer 500 steps | 0.827 s | 0.616 s | 0.213 s | **CUDA** 3.9× (MPS 1.34×) |

**Verdict for nerve-wml**: Keep all code paths on CPU. No GPU path justified for current workload sizes. If future config bumps CKNNA n→4096+, route locally via MPS for modest 2–3× (no SSH cost). CUDA amortization threshold exceeds realistic batch scales for this simulator.

### Biological substrate integration

- **BioFieldWML Phase 1**: MockBioCultureClient scheduler with sleep-every ≥ 1 validation, episode-bound Add(layer) accumulator.
- **BioFieldWML Phase 2**: Lee et al. 2024 SNN-PC (predictive coding via spiking neurons, Frontiers Comput. Neurosci., DOI 10.3389/fncom.2024.1338280) integrated as optional substitution-compatible estimator for Kuramoto-style cell populations. **Phase 2 is no longer unit-test-only scaffolding**: see v1.9.3 for the measured pilot result on HardFlowProxyTask (genuine belief-state edge +0.14, μ-driven, CI95 excludes 0, > null).
- **Testing**: 458 fast tests + 35+ slow statistical tests covering unit (L1), info-theoretic (L2), integration (L3), golden (L4) strata. No real CL1/FinalSpark API key used (MockBioCultureClient only).

### Citations added to refs.bib

- Roy & Vetterli 2007 (effective rank, spectral analysis)
- Gröger, Wen & Brbić 2026 (Aristotelian view, null-calibration framework)
- Wei et al. 2024 (Diff-eRank, LLM evaluator robustness)
- Liu et al. 2024 (HNN review, NSR journal)
- Pedersen et al. 2024 (NIR, neural integrator review)
- Mayr et al. 2025 (AER biohybrid systems)
- Bastos et al. 2020 (predictive routing, hierarchical cortex)
- Gabhart et al. 2025 (predictive coding, unified framework)
- Ruffini et al. 2025 (Comparator framework, brain-inspired)

### Master state at sprint close

- **HEAD**: `acbdc34c6fd…`
- **Fast suite**: 458 passed, 1 skipped
- **Slow suite**: 35+ statistical integration tests
- **Lint**: ruff + mypy clean on Plan C / A.x scope
- **Factcheck**: `python scripts/factcheck_audit.py --ci` → 27 OK / 0 DIVERGENT

### Honest limitations (carry-forward to v0.x.x)

- **spectral_entropy 4-arm ordering is B-dependent** — only B=256 gives canonical. 2-arm robust claims (gtm > null, gtm < simple_gating) survive across B∈{64,128,256,512}.
- **AKOrN minimal flavour** (n_oscillators=32) differs from Miyato et al. full topology (n=64). Not equivalent; documented as limited-capacity variant.
- **CKNNA is N-dependent** (Gröger et al. 2026 null-calibration framework formalizes it). Cross-paper comparisons require normalisation.
- **HSIC standalone non-informative** — use linear-CKA for metric-learning tasks.
- **MLX bit-divergence across Apple GPU architectures** (M1 vs M3+/M5) confirmed in erfinv branch; no bit-exact guarantee in MLX 0.31.2.
- **No real bioculture API keys used** — Phase 1/2 tested against MockBioCultureClient only.

### Acknowledgements

Multi-host empirical work: macm1 (M1 Max 8-core), macM3 (M3 Pro 12-core), studio (M3 Ultra 20-core), grosmac (M5 12-core), kxkm-ai (RTX 4090 CUDA). Fact-check audit adapted from bouba_sens Sprint 10. MLX issue triangulation contributed to ml-explore/mlx upstream.

## [Unreleased] (pre-sprint)

### Fixed

- `tests/unit/test_pyproject_axioms_extras.py` — replaced
  `test_version_is_1_8_0` (hardcoded literal that drifted on every
  patch bump) with `test_version_matches_changelog_top_entry`, which
  derives the assertion from `pyproject.toml` ↔ `CHANGELOG.md`
  consistency. Tautological release-version literals are
  release-process-fragile by design.
- `tests/unit/test_readme_gates.py` — replaced
  `test_readme_lists_paper_drafts` (asserted presence of obsolete
  `paper-v0.3-draft` label) with `test_readme_links_to_paper_sources`,
  which asserts structural file-path anchors that survive paper
  reorganisation.

## [1.8.1] — 2026-05-10

Patch release. No code change. Triggers Zenodo version-DOI mintage
for the v1.8.0 axioms-axis integration via the GitHub-Zenodo webhook
that previously fired with stale v1.2.0 metadata.

### Changed

- `.zenodo.json` description and keywords refreshed to match the
  actual v1.8.0 release content (axioms-axis integration,
  GammaThetaMultiplexer, DR-2 weakened predicate consumption).
  Adds `version` and `publication_date` fields.

### Added

- Nested `CLAUDE.md` files under `scripts/` and `tests/` for
  Claude Code progressive disclosure (improves agent context
  efficiency when working in subtrees).
- `tests/golden/README.md` documenting the >100 MB fixture pattern
  (regenerated locally or pulled from Zenodo, never committed —
  GitHub rejects files above 100 MB).
- Append-only superpowers planning notes for the v1.7.0 paper
  review and the v1.8.0 axioms integration.

### Notes

- Companion to dream-of-kiki Paper 2 outline re-spec
  (commit `55ca274` on `hypneum-lab/dream-of-kiki`, 2026-05-10).
- Defensive antecedence: this version-DOI provides the immutable
  timestamp for the GammaThetaMultiplexer (master commit
  `77efb4d`, merged 2026-04-20) relative to RecursiveMAS
  (arXiv:2604.25917, 2026-04-28).

## [1.8.0] — 2026-04-24

Axioms-axis integration with `dream-of-kiki`. `kiki_oniric.axioms.DR0..DR4`
are now first-class inputs to `from_dream_of_kiki`; DR-2 weakened
predicate (upstream amendment 2026-04-21) consumed automatically.
Strictly additive — dict-based spec contract from v1.7.0 preserved.
See `docs/changelog/v1.8.0.md` for the full rationale.

### Added

- `nerve_core.axioms_compat.check_upstream_axioms_version` with
  pinned `C-v0.8.0+PARTIAL` target and `UpstreamAxiomsVersionWarning`.
  Runs once at `nerve_core` import.
- DR-2 predicate consumption: `_check_dr2_predicate_if_present`
  invokes the upstream `_dr2_precondition` predicate when DR-2 is an
  `Axiom` instance and the spec carries an `operation_order` hint.
- Real `kiki_oniric.axioms.DR0..DR4` instances accepted as first-class
  spec values by `from_dream_of_kiki` (side-install only — PyPI rejects
  VCS URLs in published metadata; see `docs/changelog/v1.8.0.md`).

### Changed

- `_validate_spec` now calls `_check_dr2_predicate_if_present` at the
  end of its checks.
- `docs/integration-dream-of-kiki.md` Status line updated to "LIVE
  (axioms axis, v1.8.0+)".

### Unchanged

- Dict-based spec contract from v1.7.0 — fully backward-compatible.
- `bridge/dream_*.py` consolidate scaffold — runtime integration is
  v1.9.0 (blocked on upstream `kiki_oniric.consolidate()` publication).

## [1.7.0] — 2026-04-21

Review-response release to a 2026-04-21 TMLR-style external
review of the v1.6.0 paper. 5 major (F1–F5) + 7 minor (m1–m10,
W2) review concerns closed with measured evidence on branch
`review-v1.7.0`. No change to the v1.2.3 scientific baseline or
to any v1.6.0 headline number; two new paper tests (10 + 11)
broaden Claim B and reframe it on frozen-encoder evidence.
Fully backward-compatible.

### Added

- Paper Test (10) "Frozen-encoder baseline" — shared-encoder
  MI/H = 0.9486 (3 seeds), distinct-encoders control MI/H =
  0.7622. Reframes Claim B as "VQ protocol supplies shared
  frontend through codebook" (review F3, commits `98c248b` +
  `d0de1c4`).
- Paper Test (11) "Matched-capacity scale sensitivity" on
  Sleep-EDF. Sweet spot at d=128: MI/H = 0.72, MLP 0.82, LIF
  0.83, gap 0.006. Scale-invariant polymorphy at d ∈
  {32, 64, 128}. d=16 under-specifies LIF on real EEG; d=256
  MLP overfits while LIF holds (review F1, commits `261cad9` +
  `10b249b`).
- `scripts/baseline_frozen_encoder.py` — frozen-encoder pipeline
  + distinct-encoders control with null-model z.
- `scripts/hyperparam_sensitivity.py` — architecture vs pool
  scale orthogonality sweep on HardFlowProxyTask N=2
  matched-capacity (review m3, commits `8da3488` + `16634b8` +
  `dcdb55d`).
- `scripts/track_w_pilot.py::run_w2_hard_multiseed` honours the
  5-seed contract at N=2 with triple-pinned seeding (`random`,
  `numpy`, `torch`); completes the scaling-law seed symmetry
  (review F2 + m9, commit `0d591fa`).
- `tests/test_determinism_seed0.py` — bit-for-bit seed=0
  invariant (code-review Minor #3, commit `3fdbba1`).
- §Method γ/θ type-checker framing — γ/θ recast as a discrete
  type-checker on Neuroletter multiplexing, consistent with
  v1.5.3's N-3 gate investigation (review F5, commit `7a0c597`).
- §Method matched-capacity design rationale — explicit defence
  of `d_hidden=128` against smaller/larger alternatives
  (commit `261cad9`).
- §Related Work: PRH rhetoric softened to "biologically-inspired
  alignment" + `aristotelianprh2026` citation (review F4, commit
  `aef9e7d`).
- §Related Work: `peng2025gridlikevq` + `zhao2025channelavq`
  citations (review m6 + m7, commit `2faf585`).
- Abstract: version tag `v1.3.0` → `v1.7.0`, MI/H + WML glossary
  entries, "15/15 → 19/20 seed claim" fix (review m1 + m2 + W2,
  commit `d487735`).
- `docs/changelog/v1.7.0.md` — full scientific rationale.
- `docs/research-notes/paper-v1.7.0-review-response.md` —
  point-by-point response with concern ↔ task ↔ commit table.

### Reproducibility

- `papers/paper1/figures/baseline_frozen_encoder.json`
  (frozen-encoder shared + distinct, 3 seeds each).
- `papers/paper1/figures/eeg_matched_scale_sweep.json` (Sleep-EDF
  d_hidden ∈ {16, 32, 64, 128, 256}).
- `papers/paper1/figures/hyperparam_sensitivity.json` (HardFlow
  d_hidden + lr sweep at N=2).
- `papers/paper1/figures/w2_hard_n2_multiseed.json` (5-seed N=2
  scaling-law anchor; median gap 10.71 % reproduces v1.6.0
  bit-for-bit).

### Scientific findings

- **Frozen-encoder spread = 0.19 MI/H.** Shared-encoder MI/H =
  0.95 reproduces nerve-wml Test (1) range 0.91–0.96;
  distinct-encoders MI/H = 0.76 localises the alignment source
  to the shared frontend. Claim B is reframed: the VQ protocol
  supplies the shared frontend through the codebook.
- **Sleep-EDF sweet spot at d=128.** MLP 0.82 / LIF 0.83 /
  gap 0.006 on matched-capacity scale sweep; scale-invariant
  polymorphy at d ∈ {32, 64, 128}.
- **Direction stability strengthened to 19/20.** N=2 rerun at
  5 seeds preserves LIF ≥ MLP in 4/5 seeds (with the failing
  seed at a 4 % gap, below contract); combined with N=16/32/64
  at 5 seeds each (5/5 each), the abstract's direction-stability
  claim is now 19/20 pairwise measurements.

See [`docs/changelog/v1.7.0.md`](docs/changelog/v1.7.0.md) for
the full scientific rationale and
[`docs/research-notes/paper-v1.7.0-review-response.md`](docs/research-notes/paper-v1.7.0-review-response.md)
for the review concern ↔ commit table.

## [1.6.0] — 2026-04-21

Broadens Claim A/B from synthetic benchmarks + MNIST to a
canonical real neural recording: Sleep-EDF Expanded EEG,
5-class sleep-stage classification via the v1.5.0
`MlpWML.from_spectrogram` factory. No API change, no regression
on v1.2.3 baseline.

### Added

- Paper Test (9) "Real neural data (Sleep-EDF)" in section
  Information Transmission. Cross-domain MI/H(a) table across
  HardFlowProxyTask / MoonsTask / MNIST / Sleep-EDF / DVNC.
- `scripts/eeg_preprocess_sleep_edf.py` full wiring
  (bandpass + resample + segment + per-subject split).
- `scripts/save_codes_eeg.py` with `--spectrogram` and
  `--d-hidden` flags; default now lr=1e-3 steps=2000 with
  class-balanced sampling + inverse-frequency weighted CE.
- `docs/research-notes/sleep-edf-pipeline-protocol.md`
  already present from v1.5.x cycle; now reflects the
  delivered configuration.

### Reproducibility

- `tests/golden/codes_mlp_lif_eeg_n10.npz` (12.9 MB,
  10 subjects, 3 seeds, 128-dim spectrogram embeddings).
- `papers/paper1/figures/mi_eeg_n10.json` (plug-in 0.66,
  Miller-Madow 0.66, KSG 1.94 nats, MINE 3.83 nats,
  null-model z 1263-1351, bootstrap CI95 [0.63, 0.70]).
- MLP acc 0.76, LIF acc 0.80, pairwise gap 0.036.

### Dependencies

- `mne>=1.12.1` added (transitive: pooch, requests, tqdm);
  required by the Sleep-EDF fetch and preprocessing path.

See [`docs/changelog/v1.6.0.md`](docs/changelog/v1.6.0.md) for
the full scientific rationale.

## [1.5.3] — 2026-04-21

Methodology release honouring the v1.5.2 cross-lab methodology
commitment. Adds the `nerve_wml.methodology` submodule shared with
`bouba_sens` (section 6.3 pre-registered methodology). The v1.2.3
scientific baseline is unchanged; the MI/H headline is now reported
with null-model significance, bootstrap CI, and four-estimator
robustness.

### Added

- `nerve_wml.methodology.mi_null_model` — permutation significance
  test (z > 1000, p < 10⁻³ on the 3-seed MLP↔LIF codes).
- `nerve_wml.methodology.bootstrap_ci_mi` — non-parametric bootstrap
  confidence interval (CI95 [0.82, 0.99] across seeds).
- `nerve_wml.methodology.mi_estimators` — `mi_plugin_discrete`,
  `mi_miller_madow_discrete`, `mi_kraskov_ksg_continuous`,
  `entropy_discrete`.
- `nerve_wml.methodology.mi_mine_estimator` — MINE (Belghazi 2018
  Donsker-Varadhan bound, 128-hidden critic, tail-averaged).
- `scripts/save_codes_for_checks.py` — produces the
  `tests/golden/codes_mlp_lif.npz` reproducibility artefact
  containing 3-seed argmax codes + pre-VQ continuous embeddings.
- `scripts/measure_mi_null_model.py`, `measure_mi_bootstrap_ci.py`,
  `measure_mi_multi_estimator.py`, `measure_mi_mine.py` — four
  light-weight measurement scripts consuming the NPZ.
- `scripts/ablation_n3_guard.py` + `scripts/ablation_n3_predictive.py`
  — N-3 gate investigation closure (three convergent ablations).
- `docs/research-notes/n3-gate-role.md` — full reasoning trace.
- `papers/paper1/main.tex` — new Test (7) "Multi-estimator
  robustness" with Table 3 (plug-in / Miller-Madow / KSG / MINE
  side-by-side) and an honest interpretation flagging the pre-VQ
  continuous-estimator divergence as an open methodological
  question.
- `scipy` added as dependency (required by KSG digamma).

### Changed

- `README.md` — Status header bumped to v1.5.3; Cross-lab methodology
  commitment section updated to reflect the three delivered checks
  plus the continuous-estimator divergence between KSG and MINE.

### Reproducibility

- `tests/golden/codes_mlp_lif.npz` — 3-seed MLP+LIF codes (shape
  `(3, 5000)` int64) plus pre-VQ embeddings (shape `(3, 5000, 16)`
  float32).
- `papers/paper1/figures/mi_{null_model,bootstrap_ci,multi_estimator,mine}.json`
  — primary result JSONs.

See [`docs/changelog/v1.5.3.md`](docs/changelog/v1.5.3.md) for the
full scientific rationale.

## [1.5.1] — 2026-04-21

First PyPI release (`pip install nerve-wml`). Patch bump that syncs the
package metadata: v1.5.0 shipped with `pyproject.toml` `[project].version`
still at `"1.4.0"` (the v1.4.0 release commit bumped it, but the three
subsequent PRs merged on top without a second bump). Per-version Zenodo
DOI dropped from `CITATION.cff` — only the concept DOI
`10.5281/zenodo.19656342` remains, resolving to the latest record.

### Fixed

- `pyproject.toml` version now `"1.5.1"`. Wheels built from the v1.5 line
  report the correct version in `pip list` / `__version__`.

### Changed

- `CITATION.cff` identifier block: concept DOI only, no per-release churn.

See [`docs/changelog/v1.5.1.md`](docs/changelog/v1.5.1.md) for the full
rationale.

## [1.5.0] — 2026-04-21

Bundle of three features requested by downstream consumers (`bouba_sens`
and `dream-of-kiki`). No regression on the v1.2.3 scientific baseline —
all new behaviour is opt-in and off by default.

### Added

- `track_p.transducer.TransducerGating` enum (`HARD` | `GUMBEL_SOFTMAX`)
  plus `gumbel_tau` kwarg on `Transducer.__init__`, per-call `hard` /
  `tau` overrides on `forward`. Default stays `HARD` so v1.2.3 runs
  reproduce bit-identically. Opt-in `GUMBEL_SOFTMAX` returns the
  `(B, alphabet_size)` differentiable soft distribution instead of the
  argmax long codes — keeps gradients alive through the code axis.
  Motivated by [#5](https://github.com/hypneum-lab/nerve-wml/issues/5)
  (bouba_sens B-2 Me3-delta under-threshold in 5/5 worlds).
- `track_w/spectrogram.py` — `SpectrogramEncoder` wrapping
  `torch.stft → magnitude → top-N bins → temporal mean → linear
  projection`. Shipped with `MlpWML.from_spectrogram(sample_rate,
  window_sec, hop_sec, n_bins, target_carrier_dim)` classmethod factory.
  Callable as `encoder(waveform)` for both `(B, T)` and `(T,)` inputs;
  output shape `(B, target_carrier_dim)`. Motivated by
  [#7](https://github.com/hypneum-lab/nerve-wml/issues/7) (DRY for
  bouba_sens MIT-BIH ECG + Studyforrest audio consumers).
- `nerve_core/from_dream_of_kiki.py` — `from_dream_of_kiki` + dual
  `to_dream_of_kiki`, `DreamOfKikiAxiomError`, `REQUIRED_AXIOMS`
  (`DR-0..DR-4`). **Scaffold only**: spec validation live, runtime
  wiring gated on `dream-of-kiki` publishing a versioned `axioms` public
  API. Design doc [`docs/integration-dream-of-kiki.md`](docs/integration-dream-of-kiki.md)
  gives the DR-X → nerve-wml mapping table. Motivated by
  [#6](https://github.com/hypneum-lab/nerve-wml/issues/6).

### Tests

- +35 new unit tests (14 transducer gating + 11 spectrogram encoder +
  10 dream-bridge scaffold). Existing 21 multiplexer tests unchanged.

### Known issue

- `pyproject.toml` `[project].version` stayed at `"1.4.0"` — fixed in
  v1.5.1. No functional impact.

## [1.4.0] — 2026-04-21

Exposes opt-in plasticity gating on `GammaThetaMultiplexer`. Motivated by
[#4](https://github.com/hypneum-lab/nerve-wml/issues/4) — bouba_sens B-1
Amedi-2007 congenital-blindness gap directionally falsified across 4/5
worlds in ADR-0005 + ADR-0009; the only architectural difference between
T1 (congenital) and T2 (late-acquired) was whether Phase 1 ran, with
identical multiplexer plasticity. This release lets `AdaptationLoop` give
T1 / T2 biologically distinct plasticity profiles.

### Added

- `GammaThetaMultiplexer.__init__` accepts `plasticity_schedule:
  Callable[[int], float] | None` and `constellation_lock_after: int | None`.
- `GammaThetaMultiplexer.step()` advances an internal `plasticity_step`
  long buffer. When `constellation_lock_after` is set and the counter
  crosses it, `constellation.requires_grad` is permanently set to
  `False` (biological critical-period lock-in).
- `plasticity_schedule` callback multiplies the gradient flowing into
  `constellation` on every `.backward()`. A constant-1.0 schedule is
  exactly equivalent to no hook (identity).
- `state_dict()` / `load_state_dict()` round-trip preserves
  `plasticity_step`; the lock is re-applied on load if the saved counter
  already crossed the threshold.

### Unchanged

- Default construction reproduces v1.3.0 behaviour byte-for-byte.
  The 21 pinned multiplexer contract tests still pass.

### Packaging

- `pyproject.toml` version bumped from the drifted `"0.1.0"` to `"1.4.0"`
  to re-sync with the git tag trajectory (`v1.3.0` → `v1.4.0`).

See [`docs/changelog/v1.4.0.md`](docs/changelog/v1.4.0.md) for the full
rationale + downstream validation plan.

## [1.2.0] — 2026-04-20

Closes the three remaining scientific debts identified in the v1.1.1 audit: real-data validation (MNIST), bigger-architecture sensitivity (d_hidden=128), and temporal streaming (sequential tokens). Three new figures published.

### Added

- `track_w/tasks/mnist.py` — MNISTTask seed-stable flattened loader (torchvision, optional `mnist` extra).
- `track_w/tasks/sequential.py` — SequentialFlowProxyTask (16-token sequence, label at a supervised timestep).
- `track_w/configs/wml_config.py` — WmlConfig with `.mnist()` and `.large()` presets.
- `track_w/streaming_hooks.py` — per-timestep rollout helpers.
- `input_dim` parameter on MlpWML / LifWML / TransformerWML (backward compatible).
- `track_w.pool_factory.build_pool_cfg(cfg)` — config-driven pool.
- `scripts/run_mnist_pilots.py`, `run_bigger_arch.py`, `run_temporal_pilots.py` + three figure renderers.

### Scientific findings (v1.2)

- **MNIST (real data):** MLP 0.942, LIF 0.941, median gap **1.03 %**, `MI/H = 0.882` over 3 seeds.
- **Bigger arch (d_hidden=128):** substrate asymmetry AMPLIFIES (median gap **26 %**) — spike expressivity scales with `n_neurons`. Architecture scale and pool scale are orthogonal dimensions. Claim B survives: `MI/H > 0.50` even when accuracies diverge.
- **Temporal streaming:** `MI/H = 0.72` at trained step, `0.71` at filler step — alignment is structural, not task-pressure-gated.

### Paper

- §Information Transmission extended with subsections (4a) MNIST, (4b) architecture scale, (4c) temporal streaming, each with figure.
- Three figures: `mnist_scaling.pdf`, `bigger_arch_scaling.pdf`, `temporal_info_tx.pdf`.

## [1.1.0] — 2026-04-20

A single intensive session upgraded four scientific claims from architectural postulates to empirical measurements. Paper drafts v0.4 through v0.8 track the iterations.

### Added

- **LifWML.emit\_head\_pi** — learned `nn.Linear(n_neurons, alphabet_size)` symmetric to `MlpWML.emit_head_pi`. The protocol `step()` preserves the cosine-similarity pattern-match decoder (N-1 invariant); classification pilots read out the learned head for apples-to-apples comparison. Resolved §13.1 debt #1.
- **TransformerWML** (`track_w/transformer_wml.py`) — third substrate: tokenized input + `nn.TransformerEncoder(n_layers × n_heads)` + `emit_head_pi` / `emit_head_eps`. Obeys WML Protocol and invariants W-1, W-2, W-5. 7 unit tests pin the Protocol compliance surface.
- **W2-hard scaling pilots** — `run_w2_hard_n16`, `run_w2_hard_n32`, `run_w2_hard_n64` plus their multi-seed wrappers (`_multiseed`). RNG-isolated per cohort (MLP / LIF / task-eval) using explicit seed parameter.
- **Triple-substrate polymorphism pilot** — `run_w_triple_substrate(hard=False|True)`. Trains MLP + LIF + TRF on the same task with RNG isolation; reports `triple_gap = (max − min) / max`.
- **Inter-substrate information-transmission pilots** — `scripts/measure_info_transmission.py`: mutual-information between emitted codes, round-trip fidelity MLP→LIF→MLP through learned transducers, and cross-substrate merge where a frozen LIF recovers task accuracy from MLP-emitted codes only.
- **Four-point scaling-law figure** — `scripts/render_scaling_figure.py` produces `papers/paper1/figures/w2_hard_scaling.{pdf,png}` with median ± IQR error bars and a 5 % contract band.

### Scientific findings (honest)

- **Polymorphism scaling law (4 points, 5 seeds each except N=2)** — median gap:
  - $N=2 \to 10.71\%$
  - $N=16 \to 6.71\%$ (max $10.35\%$)
  - $N=32 \to 2.39\%$ (max $4.75\%$ — every seed satisfies the 5 % contract)
  - $N=64 \to 2.73\%$ (plateau; max $3.71\%$)
  Monotonic decay between $N=2$ and $N=32$, plateau at $\sim 2\text{--}3\%$ for $N \geq 32$. Direction stable: LIF $\geq$ MLP in **15/15 multi-seed measurements**.
- **Information transmission measured** — on HardFlowProxyTask, for independently trained MLP and LIF on the same input: $\mathrm{MI}(c_{\text{MLP}}, c_{\text{LIF}}) / H(c_{\text{MLP}}) \approx 0.91$ (substrates share $\sim 91\%$ of their code information), round-trip fidelity $\approx 0.99$, cross-merge ratio $\approx 0.97$. Claim B (substrate-agnostic information transmission) is empirical, not just architectural.
- **Triple-substrate saturation** — on FlowProxyTask, MLP / LIF / TRF all converge to $1.000$ (triple-gap $0\%$). On HardFlowProxyTask at $N=1$: $0.547 / 0.605 / 0.529$ (triple-gap $12.6\%$). Pool scaling not yet measured for TRF.

### Paper

- Drafts v0.4 through v0.8 push substantive §Threats rewrites:
  - v0.4 — decoder-asymmetry artefact documented
  - v0.5 — N=16 multi-seed distribution
  - v0.6 — scaling-law table (N=16 / N=32)
  - v0.7 — N=64 plateau + scaling-law figure
  - v0.8 — §Information Transmission (new section)
- Eight paper tags shipped: `paper-v0.2-draft`, `paper-v0.3-draft`, `paper-v0.4-draft`, `paper-v0.5-draft`, `paper-v0.6-draft`, `paper-v0.7-draft`, `paper-v0.8-draft`.

### Infrastructure

- **240+ tests passing** across unit, integration, golden, and info-transmission layers.
- Commits split across feature branches `feat/w2-hard-multiseed`, `feat/transformer-wml`, `feat/info-transmission`; all merged into `master` at v1.1.0 tag.



## [1.0.0] — 2026-04-19

First stable release. All eleven gates pass on commodity Apple Silicon; the paper v0.3 draft consolidates every gate's measurements.

### Added

- **Gate P** — Track-P protocol simulator (`track_p/sim_nerve.py`, `track_p/vq_codebook.py`, `track_p/transducer.py`, `track_p/router.py`). Pilots P1–P4 pass on toy signals.
- **Gate W** — Track-W WML lab (`track_w/mock_nerve.py`, `track_w/mlp_wml.py`, `track_w/lif_wml.py`). MLP ↔ LIF polymorphism gap 0 % on FlowProxyTask 4-class.
- **Gate M** — merge pipeline (`bridge/sim_nerve_adapter.py`, `bridge/merge_trainer.py`) retaining 100 % of mock baseline.
- **Gate M2** — four §13.1 scientific shortcuts resolved: P3 γ-priority ablation (26 % collision without rule), W2 true-LIF polymorphie on HardFlowProxyTask (12.1 % gap — honest), W4 rehearsal CL (forgetting 100 % → 0 %), P1 random-init VQ + codebook rotation (dead codes 39 % → 0 %).
- **Paper v0.2** — ablation table, figures 2–4 (W4 forgetting, P1 dead-code curves, W2 histogram), §Threats, §Reproducibility.
- **Gate Scale** — W1/W2/W4 pilots at N=16 plus W2 stress at N=32; router strongly connected for all N ∈ {4, 8, 16, 32}.
- **Gate Interp** — `interpret/` package: semantics extractor (`build_semantics_table`), torch k-means (`cluster_codes_by_activation`), plain-HTML report renderer (`render_html_report`). Cluster entropy > 2 bits on toy data.
- **Gate Neuro** — `neuromorphic/` package: INT8 symmetric quantization (`quantize_lif_wml`), pure-numpy mock runner (`MockNeuromorphicRunner`), software-vs-mock delta check, Loihi 2 / Akida stubs with informative `NotImplementedError`.
- **Gate Dream** (partial) — `bridge/dream_bridge.py` ε-trace collect/encode/apply pipeline, env-gated by `DREAM_CONSOLIDATION_ENABLED`, with `MockConsolidator` for CI. Full resolution awaits `kiki_oniric` v0.5+ public `consolidate()` surface.
- **Gate Adaptive** — `track_p/adaptive_codebook.py` with `active_mask`-based shrink/grow, `bridge/transducer_resize.py` reshaping transducers while preserving argmax on kept rows. Multi-cycle stability tested.
- **Gate LLM Advisor** — `bridge/kiki_nerve_advisor.py` with env-gated, never-raising `advise(query_tokens, current_route) -> dict | None`. Warm-path latency < 50 ms; disabled-path overhead < 5 ms. Self-contained wiring recipe at `docs/integration/micro-kiki-wiring.md`.
- **Paper v0.3** — abstract names all 11 gates; new `§Integrations` section covering Adaptive / Neuromorphic / Dream / LLM Advisor.
- **Harness** — `harness/run_registry.py` produces bit-stable `run_id` from `(c_version, topology, seed, commit_sha)`.
- **227 tests passing**, coverage ≥ 95 % on every package, `ruff` + `mypy` clean on 49 source files.

### Scientific findings (honest)

- **FlowProxyTask 4-class saturates** both MLP and LIF substrates at 1.000 — the 0 % polymorphie gap is a degenerate best case. Documented in paper §Threats.
- **HardFlowProxyTask (12-class XOR on noise)** exposes real variance: `acc_mlp = 0.547`, `acc_lif = 0.480`, **gap = 12.1 %** — violates < 5 % on non-linear tasks. LIF's cosine-similarity decoder lags the MLP π head. Paper claim is now narrowed to linearly-separable regimes; closing the gap on harder tasks is explicit future work.
- **Untrained-LIF INT8 mock-runner delta ≈ 19 %** on random inputs — INT8 quantization of binary-like codebooks is coarse. Trained LIFs are expected to tighten.

### Infrastructure

- Eleven gate tags on origin, all `git push`-able and linked from README: `gate-p-passed`, `gate-w-passed`, `gate-m-passed`, `gate-m2-passed`, `gate-scale-passed`, `gate-interp-passed`, `gate-neuro-passed`, `gate-dream-passed`, `gate-adaptive-passed`, `gate-llm-advisor-passed`, plus `paper-v0.2-draft` and `paper-v0.3-draft`.
- No vendor SDK runtime deps: Loihi, Akida, `dream-of-kiki`, `sentence-transformers` are all opt-in.
- `MIT` for code, `CC-BY-4.0` for docs.

### Cited in

- `dreamOfkiki` Paper 1 v0.2 §7.4 cross-substrate portability (DR-3 Conformance Criterion). OSF pre-registration: [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN).

[1.0.0]: https://github.com/hypneum-lab/nerve-wml/releases/tag/v1.0.0
