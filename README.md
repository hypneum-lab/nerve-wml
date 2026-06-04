# nerve-wml

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19656342.svg)](https://doi.org/10.5281/zenodo.19656342)
[![OSF](https://img.shields.io/badge/OSF-10.17605%2FOSF.IO%2FQ6JYN-lightgrey)](https://doi.org/10.17605/OSF.IO/Q6JYN)

**Substrate-agnostic nerve protocol for inter-module communication in hybrid neural systems.**

Citation : each release is archived on Zenodo (concept DOI [10.5281/zenodo.19656342](https://doi.org/10.5281/zenodo.19656342) resolves to the latest version) and linked to the parent programme's OSF pre-registration ([10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN)).

Research engine that validates a discrete-code communication layer between heterogeneous neural modules (World Model Languages, or WMLs). Modules exchange **neuroletters** over a sparse learned topology, multiplexed on gamma/theta rhythms, and converted between local codebooks by per-edge transducers. The paper draft is at [`papers/paper1/main.tex`](papers/paper1/main.tex); the full spec is at [`docs/superpowers/specs/2026-04-18-nerve-wml-design.md`](docs/superpowers/specs/2026-04-18-nerve-wml-design.md).

## 2026-05-20 — Gap-analysis remediation sprint

Reinforcement sprint that hardened the empirical claims via systematic
statistical scrutiny. **97 commits, 13 PRs merged**, +458 fast-suite
tests passing (from 385 baseline). Detailed entry in [`CHANGELOG.md`](CHANGELOG.md).

**6 initial claims revised** under scrutiny:

1. *learned strictly dominates relative_rep* → tie at `n_anchors=64`
   (Δ=0.0001, p=0.627, n=50, bit-exact M5 ↔ macM1).
2. *GTM doesn't collapse (synchrony 0.20 > 0.08)* → metric anti-monotone
   (null=0.32 > GTM=0.20 > simple_gating=0.08); replaced with
   `spectral_entropy` (Roy & Vetterli 2007 effective rank).
3. *AKOrN minimal clusters with simple_gating* → sub-parametrised; at
   `n_oscillators=64, n_steps=32, lr=0.05` AKOrN reaches synchrony 0.45.
4. *CKNNA grows with N (signal)* → mechanical decay artefact (Gröger,
   Wen & Brbić 2026 Prop 4.2). Signal/null d_z grows monotonically.
5. *HSIC golden=8.5320 is stable scientific number* → standalone HSIC
   non-informative (`|r|≤0.015` with 7 other alignment metrics on 1750
   cells); always normalise to linear-CKA.
6. *spectral_entropy 4-arm strict ordering* → B-dependent (Renf 10);
   robust 2-arm: gtm > null, gtm < simple_gating at all B.

**7 findings now publishable:**

- AKOrN top-cell at n=50 = phase-aligner not coder: synchrony 0.53 ± 0.20
  (cross-host: macM3 0.49 ± 0.19), accuracy bimodal 0.38 ± 0.31.
- 3/4 substrates (gtm, simple_gating, null) **bit-exact cross-arch**
  (M1 g13s ↔ M3 Pro g15s ↔ M5 g17g) under PyTorch deterministic path;
  Kuramoto Euler integrator is the only cross-arch-divergent arm.
- MLX `random.normal` cross-arch non-bit-exact (M1 vs M3+/M5), opened
  upstream as [ml-explore/mlx#3568](https://github.com/ml-explore/mlx/issues/3568).
- macM3 ~2-3× faster than macm1 on Python+torch multiproc CPU.
- MLX 2× faster than `torch.MPS` at N=16384 for CKNNA Gram (`scripts/cknna_mlx_bench.py`).
- Task scope (alphabet=64, K=7) saturates gtm and simple_gating equally;
  spectral_entropy is the only discriminating axis.
- Empirical confirmation of Gröger 2026 Prop 4.2 `E[mKNN]=k/(n-1)` on
  synthetic Gaussian substrate (see `papers/paper1/supplementary/cknna-n-dependence-replication/`).

**Methodology institutionalised:**

The protocol *every numerical claim must trace to a JSON cell or
executed log line in the same session* is now enforced by
`scripts/factcheck_audit.py --ci` and runs as a GitHub Actions check
on every PR touching `docs/superpowers/research/` or the audit script.
**Current state: 32 OK / 0 DIVERGENT / 2 ORPHAN** (orphans = JSONs on
unmerged branches).

## 2026-05-11 milestones — N8 → N14 portfolio sprints

Tonight's six-sprint push closed the GammaThetaMultiplexer (GTM) empirical
arc and locked four scaling pre-registrations for future hardware sprints.
Headline numbers below remain the v1.2.3 baseline ; the new work
characterises the GTM contract envelope and shipped six critic-driven
fixes (ablation void, degenerate-metric handling, Cat C honest framing,
Jonckheere wrong test, β-VAE confound, paper/JSON count mismatch).

| Sprint | Verdict / artifact | Notes |
|---|---|---|
| **N8 Q1** GTM benchmark (HardFlowProxyTask N=2) | **tied** (3W / 5L / 1T vs latent-space baselines) | mi_h dominance preserved across all 8 conditions |
| **N9 Q1+** scaling N=16 | **tied-stable** (3W / 5L / 1T identical to N=2) | seed-stable across 5 seeds |
| **N9 Q1++** FlowProxyTask 4-class | **tied-stable-cross-task** (3W / 1L / 2T post-degenerate-exclusion) | confirms tie generalises beyond HardFlow |
| **N8 Q2** 15 adversarial substrates (Cat A/B/C) | **`ge_3_FP_reformulate`** | Conformance Criterion C+ now requires C2 axiom property tests in addition to structural invariants |
| **N9 Q2+** Cat D modular / E replay-loop / F asymmetric (10 more) | **25 / 25 cumulative FP** | structural layer alone insufficient at 6 categories |
| **N10-A / N10-C** | **pre-reg locked, deferred** | ImageNet-100 + ResNet50 + GTM bottleneck (β-VAE β=1.0 fixed per critic) ; Procgen multi-agent GTM communication |
| **N11-A / N11-C** | **pre-reg locked, deferred** | I-JEPA + GTM predictor on ImageNet self-supervised ; Dreamer V3 + GTM dynamics on Atari-100k |
| **N13 / N14** | **pre-reg locked, deferred** | OFAT GTM internals (5 hyperparams × 4-5 levels × 5 seeds) ; Latin Hypercube 7-dim coverage (50 points × 5 seeds) |

Cumulative on master HEAD `7528ada` : **295 commits**, **v1.8.1** released
on PyPI, 12 OSF-style pre-registrations across the Hypneum programme,
7 verdicts shipped on the GTM axis, 5 pre-regs deferred to future
scaling sprints, 1 critic-saver retract closed downstream
(`bouba_sens` Q3+, see that repo's README).

Methodology lessons promoted to portfolio-wide discipline :
multi-seed-first-class (single-seed claims are fragile, see `bouba_sens`
N9 Q3 Retract evidence), degenerate-metric handling in `analyse.py`,
Cohen's d effect sizes added to verdict tables, **honest reporting of
underpowered ablations** (the cosine plasticity schedule's effect on
GTM outcomes is below the noise floor for HardFlowProxyTask : the
constellation is 128 of 17,888 trainable parameters (0.7 %), so
scaling its gradient by a cosine schedule does not produce a measurable
behavioural difference (Δ < 0.005 RTF, less than seed standard
deviation). This is a **negative result** consistent with the
convergent-evidence framing : on HardFlowProxyTask, plasticity
scheduling is **not load-bearing** — the discrete-channel structure
(PSK alphabet + Gumbel quantization) drives the `mi_h` advantage, not
the temporal modulation of constellation learning. A larger fraction
of the model would need to participate in the schedule for the
ablation to discriminate ; we leave this as a methodological
observation for future work, e.g. per-symbol learned embeddings, or a
`constellation_lock_after` ablation with the constellation frozen
mid-training).

---

## 2026-05-19 update — related-work consolidation + BioFieldWML plan

Paper 1's §Related Work grew from 5 to 6 threads (new "Adjacent abstraction
layers" thread, NIR at model-graph IR + AER at transport — orthogonal
abstraction layers, not competitors), §Information Transmission Test (6)
added a "Global vs Local alignment" paragraph addressing the two 2026 PRH
critiques, and the GTM rationale was re-grounded on Bastos 2020 / Friston
2025 / Ruffini 2025. In parallel, a 4th conformant substrate plan
(`BioFieldWML`) was versioned for the first time, sitting alongside the
existing `MlpWML` / `LifWML` / `TransformerWML` substrates under the WML
Protocol.

| Item | Where | What |
|---|---|---|
| §Related Work extended (5 → 6 threads) | `papers/paper1/main.tex` | New "Adjacent abstraction layers" thread positions nerve-wml against NIR (model IR) and AER (transport) — orthogonal abstraction layers |
| §Information Tx Test (6) PRH refinement | `papers/paper1/main.tex` | New "Global vs Local alignment" paragraph engages `platoscave2026` + `aristotelianprh2026`; nerve-wml's mutual-kNN measurement claimed as *local* (null-calibrated 18.8× random), not global |
| GTM re-grounding | `papers/paper1/main.tex` | Replaces sole reliance on Bastos & Friston 2012 with Bastos 2020 predictive routing + Friston 2025 + Ruffini 2025 laminar Comparator |
| HNN positioning (closes #13) | `papers/paper1/main.tex` "Surrogate-gradient SNN" thread | Cites Liu et al. 2024 (NSR) and positions nerve-wml in the *weak-coupling* corner of the HNN taxonomy |
| BioFieldWML plan versioned | `docs/superpowers/plans/2026-05-19-bio-substrate-wml.md` | Plan for a 4th conformant substrate alongside MLP / LIF / Transformer; 5 (b)-classified mechanisms encapsulated inside `step()` (Tomé STDP, Pignatelli IE, Palacios SNN-PC, Bellitto WSCL, Tucker-Friston E/I); cross-refs dream-of-kiki `biophysical-stratification` spec |

Headline measurements and claims A/B remain unchanged — this update is
documentation and planning only, not new experimental findings.

**(i)+(ii) framing:** `BioFieldWML` is the **(i)** side — one conformant
substrate among N, DR-3 preserved, fully encapsulated under the existing WML
Protocol. The **(ii)** side is the new dream-of-kiki biophysical-stratification
sub-theory (`dream-of-kiki/docs/specs/2026-05-20-biophysical-stratification.md`).
The two documents are linked but independent: nerve-wml's bio-substrate plan
does not depend on the stratification theory being finalised, and vice versa.

**Open follow-up:** `nerve-wml#16` — complete `friston2025pivot` bibliographic
metadata (volume/pages/DOI) before final paper submission.

---

## Status — v1.8.0 (2026-04-24, on PyPI)

Installable via `pip install nerve-wml`. For the real
`kiki_oniric.axioms` integration (dream-of-kiki bridge), side-install
`dreamofkiki` first (PyPI rejects VCS URLs in published metadata, so
no `[axioms]` extras group is shipped):

```bash
pip install "dreamofkiki @ git+https://github.com/hypneum-lab/dream-of-kiki@v0.9.1"
pip install nerve-wml
```

Six releases landed on 2026-04-21 → 2026-04-24 (v1.4.0 → v1.8.0) on top
of the v1.2.3 scientific baseline; see
[§ Post-v1.2.3 API additions](#post-v123-api-additions-2026-04-21) below
or [`CHANGELOG.md`](CHANGELOG.md) for the per-version diff. The **scientific
claims below are the v1.2.3 baseline and remain load-bearing** — the newer
releases added opt-in knobs (plasticity schedule, Gumbel-softmax gating,
spectrogram encoder, dreamOfkiki axiom bridge scaffold) and the
`nerve_wml.methodology` submodule with the four MI robustness primitives
(null-model, bootstrap CI, Miller-Madow, Kraskov KSG, MINE) — all without
changing any headline measurement.

The project is empirically defensible across three experimental axes: real
data, architecture scale, and temporal streaming. Two claims are quantified:

**Claim A — Substrate-agnostic polymorphism (task competence converges).**
Three structurally distinct substrates (stateless MLP, spiking LIF with surrogate-gradient, attention-based Transformer) reach comparable accuracies via the shared Nerve Protocol.

**Claim B — Substrate-agnostic information transmission (codes align).**
Independent substrates share 91–96 % of their emitted code information; a frozen LIF can recover a trained MLP's task competence via a learned linear transducer.

### Headline measurements

| Axis | Finding | Reference |
|---|---|---|
| **Pool scaling law** (MLP ↔ LIF, HardFlow) | $N=2 \to 10.71\%$, $N=16 \to 6.71\%$, $N=32 \to 2.39\%$, $N=64 \to 2.73\%$ plateau. 5 % contract holds distributionally at $N \geq 32$. | `figures/w2_hard_scaling.pdf` |
| **Triple-substrate pool** (MLP + LIF + TRF) | $N=15 \to 8.16\%$, $N=30 \to 5.86\%$, $N=60 \to 4.33\%$ | v1.1.4 |
| **Mutual information** (codes MLP ↔ LIF) | $\mathrm{MI}/H = 0.91$ at $N=1$ (5 seeds), **0.96** at $N=16$ pool (192 cross-pairs) | `figures/info_transmission.pdf` |
| **Round-trip fidelity** (MLP → LIF → MLP) | **0.99** mean (3 seeds) | v0.8 |
| **Cross-substrate merge** (LIF fed by MLP codes only) | **0.97** mean (3 seeds) | v0.8 |
| **MNIST real data** | MLP 0.942, LIF 0.941, gap **1.03 %**, MI/H **0.882** | `figures/mnist_scaling.pdf` |
| **MoonsTask** (2nd distribution) | MI/H = **0.74** (3 seeds) | v1.1.4 |
| **Architecture scale** ($d_\text{hidden}=128$) | Gap AMPLIFIES to 26 % on XOR (arch vs pool scale are orthogonal); Claim B survives | `figures/bigger_arch_scaling.pdf` |
| **Temporal streaming** (16-token sequence) | MI/H = **0.72** at trained step, **0.71** at filler step — structural alignment | `figures/temporal_info_tx.pdf` |
| **Platonic RH alignment** (Huh 2024, pre-VQ mutual-kNN) | MLP ↔ LIF = **0.174** at k=10 (18.8× random, 3 seeds); stable across k∈[5,50] | `figures/platonic_rh_alignment.json` |
| **Real neural data** (Sleep-EDF EEG, v1.6.0) | See paper Test (9); Claim B confirmed on 5-class sleep-stage via `MlpWML.from_spectrogram` + d_hidden=128 | `figures/mi_eeg_d128_spectro.json` |
| **Frozen-encoder baseline** (review F3, v1.7.0) | Shared MI/H=0.95 (matches nerve-wml Test 1), Distinct MI/H=0.76 (without shared frontend); Claim B reframed as "VQ protocol supplies shared frontend through codebook" | `figures/baseline_frozen_encoder.json` |
| **Matched-capacity scale sweep** (Sleep-EDF, v1.7.0) | Sweet spot at d=128: MI/H=0.72, MLP=0.82, LIF=0.83, gap=0.006. Scale-invariant polymorphy at d ∈ {32, 64, 128}. d=16 insufficient for LIF convergence on real EEG; d=256 MLP overfits while LIF holds | `figures/eeg_matched_scale_sweep.json` |
| **Direction stability** (LIF ≥ MLP on hard task) | **19/20** pairwise seeds (4/5 at N=2; 5/5 at N=16, 32, 64) + 5/5 triple-substrate, preserved on Sleep-EDF (+0.007 LIF edge) | — |

LIF's spike dynamics give it a substrate-intrinsic $\sim 2$–$3\%$ expressivity edge on XOR-style boundaries (plateau floor). Pool averaging compresses this, architecture width amplifies it.

### Seven concrete findings

1. **The original 12.1 % gap was a decoder asymmetry bug, not a substrate limit.** LIF had a fixed cosine decoder, MLP had a learned head; symmetrizing flipped the sign (LIF now leads).
2. **Single-seed measurements lie.** Multi-seed revealed the N=16 median is 6.7 %, not the lucky 1.68 %.
3. **Scaling law is real and monotonic.** Four-point decay $10.7\% \to 6.7\% \to 2.4\% \to 2.7\%$ plateau.
4. **Claim B is empirical, not architectural.** MI 0.91–0.96, round-trip 0.99, cross-merge 0.97.
5. **Substrate-direction is stable in 19/20 seeds.** LIF's spike edge is a real property, not noise.
6. **Architecture scale and pool scale are orthogonal.** Pool compresses the gap; arch width amplifies it.
7. **Code alignment is structural, not task-gated.** MI at filler timesteps $\approx$ MI at trained timesteps (0.71 vs 0.72).

### Methodological findings (v1.2.1–v1.2.3)

- **MI/H vs CKA on the same argmax codes** (v1.2.1). Mean 0.953 (MI/H) vs 0.910 (CKA argmax one-hot) over 3 seeds. The 4.3 pp gap tracks soft many-to-one code mappings that kernel-alignment metrics miss. MI/H is not CKA renamed — it is the discrete-protocol cousin with measurably different semantics. See `scripts/measure_cka_vs_mi.py` and `docs/positioning.md`.
- **Related Work verified** (v1.2.2). Paper §Related Work cites Kornblith 2019 CKA, Morcos 2018 PWCCA, Moschella 2022 relative representations (ICLR 2023), Saxe 2024 universality, and Hinton 2015 KD — all verified via WebFetch, provenance table in `docs/positioning.md`.
- **KD match-compute ablation honest verdict** (v1.2.3). At matched compute on HardFlowProxyTask (3 seeds), cross-merge (0.508) ≈ KD-through-transducer (0.520) within noise. Vanilla Hinton KD (0.534) is best because the student can re-train its core. Cross-merge's contribution is **methodological, not performance-based**: it isolates protocol channel capacity from student learning capacity by freezing both substrates and supervising with ground-truth labels only. See `scripts/measure_kd_ablation.py`.

### What the paper genuinely claims vs not

Three findings probably novel: (1) the four-point scaling law with plateau at $\sim 2\text{–}3\%$ substrate-intrinsic floor, (2) reproducible $\sim 2\text{–}3\%$ LIF spike-expressivity edge over matched-capacity MLP on XOR-on-noise (19/20 seeds), (3) orthogonality of pool-scale (compresses gap) and architecture-scale (amplifies gap).

The paper explicitly does **not** claim: a new learning algorithm, superiority over knowledge distillation on task accuracy, or universal representations — that debate is addressed by Saxe 2024 and the Nature MI 2025 editorial (s42256-025-01139-y) cited in `docs/positioning.md`.

### Cross-lab methodology commitment

The sister project `bouba_sens` (2026-04-21, `github.com/hypneum-lab/bouba_sens` tag `v0.5.0`) demonstrated that pre-registered findings in this programme must pass three critical tests before publication: **null-model partition controls**, **bootstrap confidence intervals** on sub-threshold effects, and **multi-estimator robustness** checks for MI-based claims. As of **v1.5.3** (2026-04-21) all three checks are implemented in `nerve_wml.methodology` and applied to the MI/H headline: **null-model rejects chance** at z > 1000 (p < 10⁻³ over 3 seeds × 1000 shuffles), **bootstrap** gives CI95 [0.82, 0.99] intra-seed width ~0.005, and **discrete cross-estimator robustness** holds between plug-in and Miller-Madow (Δ = 0.007). Two continuous estimators (Kraskov KSG and MINE) were applied to the pre-VQ embeddings; they diverge by an order of magnitude (KSG 0.09, MINE 0.99), making the pre-VQ absolute MI magnitude an open methodological question — see paper §Information Transmission Test (7). The post-VQ discrete MI/H headline is unaffected by this ambiguity.

## Status — 11 gates

| Tag | What it proves |
|---|---|
| [`gate-p-passed`](../../releases/tag/gate-p-passed) | Track-P protocol simulator correct on toy signals |
| [`gate-w-passed`](../../releases/tag/gate-w-passed) | `MlpWML` and `LifWML` interoperate with < 5 % gap through the same nerve (N=4) |
| [`gate-m-passed`](../../releases/tag/gate-m-passed) | Merge fine-tunes only transducers; retains ≥ 95 % of mock-baseline accuracy |
| [`gate-m2-passed`](../../releases/tag/gate-m2-passed) | Four scientific shortcuts from §13.1 resolved with honest measurements |
| [`gate-scale-passed`](../../releases/tag/gate-scale-passed) | Polymorphie + continual learning hold at N=16 pools; router stays connected to N=32 |
| [`gate-interp-passed`](../../releases/tag/gate-interp-passed) | Per-WML `code → concept` semantics table rendered as HTML |
| [`gate-neuro-passed`](../../releases/tag/gate-neuro-passed) | LifWML → INT8 artefact → pure-numpy mock runner (Loihi / Akida stubs documented) |
| [`gate-dream-passed`](../../releases/tag/gate-dream-passed) | ε-trace consolidation bridge to dream-of-kiki (schema v0; partial — awaits kiki_oniric v0.5+) |
| [`gate-adaptive-passed`](../../releases/tag/gate-adaptive-passed) | Per-WML alphabet shrinks/grows via `active_mask` + transducer resize |
| [`gate-llm-advisor-passed`](../../releases/tag/gate-llm-advisor-passed) | Env-gated, never-raising `NerveWmlAdvisor` for micro-kiki, < 50 ms warm latency |

Paper drafts: `paper-v0.2-draft` … `paper-v0.9-draft` track the iterations that produced the v1.2 claims above. Release tags `v1.0.0`, `v1.1.0` … `v1.1.4`, `v1.2.0`, `v1.2.3`, `v1.3.0`, `v1.4.0`, `v1.5.0`, `v1.5.1` archive the code snapshots; see [`CHANGELOG.md`](CHANGELOG.md) for per-version findings.

## Track-W hardening — EWC vs rehearsal (2026-05-30, PR #46)

Continual-learning hardening of gate W on a deliberately difficult task
(`HardSplitTask` — shared 12-class head, Tasks 0 and 1 on non-overlapping
label splits). Three methods compared under matched settings:

| Method | Forgetting ↓ (lower = better) | Notes |
|---|---|---|
| **Rehearsal** (20 % replay) | **0.034** | Best by a wide margin |
| **EWC** (λ=1000) | 0.532 | Meaningful regularisation vs none |
| **None** (fine-tune only) | 0.767 | Catastrophic forgetting baseline |

Honest framing: this is a diagnostic comparison on a single hard task, not a
new gate. EWC at λ=1000 provides significant regularisation (0.532 vs 0.767),
but rehearsal (0.034) dominates when a replay buffer is available. Raw numbers
trace to `docs/superpowers/research/2026-05-30-w4-ewc-comparison.json`.

Implementation in `track_w/continual/`:
- `ewc.py` — `estimate_fisher(wml, data_loader)` (diagonal Fisher from
  squared log-likelihood gradients) + `penalty(wml, fisher, theta_star, lam)`.
  Respects W-1 (no mutation of another WML's parameters) and W-2 (codebook
  regularised alongside MLP weights).
- `rehearsal.py` — `RehearsalBuffer` dataclass (fixed-replay, batch-mixing
  for Task 1 training, same scaffold reused by `run_w4_compare`).
- `tasks/hard_split.py` — `HardSplitTask` shared-head 12-class split task.

## Post-v1.2.3 API additions (2026-04-21)

Three issues filed by downstream consumers (`bouba_sens`, `dream-of-kiki`)
landed on 2026-04-21 as opt-in knobs — **no change to v1.2.3 headline
numbers**, all new behaviour is off by default.

| Release | Issue | Feature | Motivation (downstream) |
|---------|-------|---------|-------------------------|
| **v1.4.0** | [#4](https://github.com/hypneum-lab/nerve-wml/issues/4) | `GammaThetaMultiplexer` gains `plasticity_schedule` + `constellation_lock_after` | `bouba_sens` B-1 Amedi-2007 gap directionally falsified in 4/5 worlds; biologically-distinct T1/T2 plasticity profiles are the probe. |
| **v1.5.0** | [#5](https://github.com/hypneum-lab/nerve-wml/issues/5) | `Transducer` gains `TransducerGating.GUMBEL_SOFTMAX` (opt-in soft distribution) | `bouba_sens` B-2 Me3-delta under-threshold in 5/5 worlds; hard argmax gating may be too abrupt for post-lesion MI migration. |
| **v1.5.0** | [#7](https://github.com/hypneum-lab/nerve-wml/issues/7) | `MlpWML.from_spectrogram(...)` factory + `SpectrogramEncoder` | DRY: `bouba_sens` MIT-BIH ECG fetcher + future Studyforrest audio share one canonical STFT → carrier path. |
| **v1.5.0** | [#6](https://github.com/hypneum-lab/nerve-wml/issues/6) | `nerve_core.from_dream_of_kiki(...)` scaffold (runtime gated upstream) | Pin the public axiom-bridge contract today so `bouba_sens` can plumb the call site before `dream-of-kiki` publishes its versioned `axioms` API. |
| **v1.5.1** | — | Packaging: `pyproject.toml` `version` sync (stale `1.4.0` on the v1.5.0 commit), `CITATION.cff` keeps concept DOI only. | v1.5.0 shipped with a stale version field — first PyPI release carries the correct metadata. |

Design docs: [`docs/integration-dream-of-kiki.md`](docs/integration-dream-of-kiki.md), changelog files at [`docs/changelog/v1.4.0.md`](docs/changelog/v1.4.0.md) and [`docs/changelog/v1.5.1.md`](docs/changelog/v1.5.1.md).

## Install

```bash
# From PyPI (v1.5.1+)
pip install nerve-wml

# From source, with dev extras (tests + lint)
git clone https://github.com/hypneum-lab/nerve-wml.git
cd nerve-wml
uv sync --all-extras
```

Python 3.12+, macOS arm64 (MLX-friendly) or Linux x86_64. No vendor SDK deps are pulled by default (Loihi, Akida, dream-of-kiki, sentence-transformers are all optional integrations).

## Run the suite

```bash
uv run pytest -m "not slow"    # 220+ tests under 80 s on commodity M-series
uv run pytest                  # full suite incl. paper figure rendering
uv run pytest --cov=nerve_core --cov=track_p --cov=track_w --cov=bridge --cov=harness --cov=interpret --cov=neuromorphic
```

## Reproduce the gate numbers

```bash
uv run python scripts/track_p_pilot.py       # Gate P (+ Task 6 ablation)
uv run python scripts/track_w_pilot.py       # Gate W
uv run python scripts/track_w_pilot.py scale # Gate Scale (N=16, N=32)
uv run python scripts/merge_pilot.py         # Gate M
uv run python scripts/interpret_pilot.py     # Gate Interp (emits reports/interp/*.html)
uv run python scripts/adaptive_pilot.py      # Gate Adaptive
```

## Reproduce the v1.1 / v1.2 findings

```bash
# v1.1 scaling law + information transmission + triple substrate
uv run python scripts/render_scaling_figure.py      # 4-point pool scaling (N=2..64)
uv run python scripts/render_info_tx_figure.py      # MI + round-trip + cross-merge
uv run python scripts/measure_info_transmission.py  # full info-tx battery

# v1.2 real data + bigger arch + temporal
uv sync --extra mnist                               # pull torchvision
uv run python scripts/render_mnist_figure.py        # MNIST Claims A + B
uv run python scripts/render_bigger_arch_figure.py  # d=128 gap amplification
uv run python scripts/render_temporal_figure.py     # streaming MI per timestep
```

## Build the paper

```bash
uv run python scripts/render_paper_figures.py   # regenerate figures from frozen golden NPZs
cd papers/paper1 && tectonic main.tex           # or pdflatex, bibtex, pdflatex, pdflatex
```

## Integrations (env-gated, default off)

- **Dream consolidation**: `DREAM_CONSOLIDATION_ENABLED=1` + install `dream-of-kiki` locally → `bridge.dream_bridge.DreamBridge`.
- **LLM advisor (micro-kiki)**: `NERVE_WML_ENABLED=1` + `NERVE_WML_CHECKPOINT_PATH=/path/to/checkpoint` → `bridge.kiki_nerve_advisor.NerveWmlAdvisor`. Wiring recipe: [`docs/integration/micro-kiki-wiring.md`](docs/integration/micro-kiki-wiring.md).
- **Neuromorphic hardware**: install `lava-nc` or `akida` → wire in `neuromorphic.loihi_stub` / `neuromorphic.akida_stub`. Schema v0: [`docs/neuromorphic/deployment-guide.md`](docs/neuromorphic/deployment-guide.md).

## Cited in

- **dreamOfkiki — Paper 1 v0.2 (2026-04-19), §7.4 cross-substrate portability** — [github.com/hypneum-lab/dream-of-kiki](https://github.com/hypneum-lab/dream-of-kiki). The Gate W and Gate M measurements reported here (MlpWML / LifWML polymorphism on FlowProxyTask and HardFlowProxyTask) provide the empirical corroboration cited in Paper 1 as independent evidence of the substrate-agnosticism principle (DR-3 Conformance Criterion). OSF pre-registration: [10.17605/OSF.IO/Q6JYN](https://doi.org/10.17605/OSF.IO/Q6JYN).

## Program context

This repository is part of **hypneum-lab**, which develops executable formal frameworks for cognitive AI. The programmatic parent is `dreamOfkiki` (paper 1 formal framework, paper 2 empirical); `nerve-wml` is the reference implementation for the substrate-agnostic communication principle.

Sibling repositories:

- [dream-of-kiki](https://github.com/hypneum-lab/dream-of-kiki) — formal framework (axioms DR-0..DR-4, Conformance Criterion, Paper 1)
- [kiki-flow-research](https://github.com/hypneum-lab/kiki-flow-research) — Wasserstein-gradient-flow engine (upstream)
- [micro-kiki](https://github.com/hypneum-lab/micro-kiki) — 35 domain-expert MoE-LoRA deployable instance (advisor consumer)
- **nerve-wml** (this repo) — substrate-agnostic nerve protocol + cross-substrate polymorphism

## Repository layout

```
nerve_core/        Neuroletter, Nerve + WML Protocols, invariants (N-1..N-5, W-1..W-4)
track_p/           Track-P — SimNerve, VQCodebook, Transducer, SparseRouter, AdaptiveCodebook
track_w/           Track-W — MockNerve + 6 WML substrates (MlpWML, LifWML, TransformerWML, BioWML, BioFieldWML, SpikingKikiWML), toy tasks, training loop, pool factory. See `docs/substrate-paradigms.md` for the paradigm map.
bridge/            Merge, dream, LLM advisor — SimNerveAdapter, MergeTrainer, DreamBridge, NerveWmlAdvisor
harness/           R1 reproducibility — run_registry
interpret/         Gate Interp — code_semantics, clustering, HTML renderer
neuromorphic/      Gate Neuro — spike_encoder, INT8 export, mock_runner, vendor stubs
scripts/           All gate pilots + figure renderers + freeze_golden
tests/             Unit + integration + golden NPZ regressions
docs/              specs/, integration/, neuromorphic/, dream/, interpret/
papers/paper1/     LaTeX source + bib + Makefile (figures regenerated deterministically)
```

## License

MIT (code) + CC-BY-4.0 (docs).
