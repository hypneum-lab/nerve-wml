# DVNC baseline comparison protocol

**Status:** planned (2026-04-21 evening)
**Target:** Semaine 2 of Voie 3 roadmap (post-v1.5.3).
**Scripts:** `scripts/baseline_dvnc.py` (scaffold prepared, wiring TODO).
**Output path:** `tests/golden/codes_dvnc.npz` +
`papers/paper1/figures/baseline_dvnc.json`.

## Why this baseline

Tang et al.\ 2021 "Discrete-Valued Neural Communication" (NeurIPS,
arXiv:2107.02367) is the closest prior work to nerve-wml's VQ-
bottleneck inter-module communication: two agents communicate
through a shared discrete codebook, with commitment-loss training
on the codebook vectors. Without a direct numerical comparison,
any reviewer familiar with DVNC will ask: **"how does this compare
to Tang 2021?"** and the paper's position becomes weaker than it
needs to be.

## What we compare

| Dimension | nerve-wml | DVNC (Tang 2021) |
|---|---|---|
| Substrates | 2 **heterogeneous** (MLP + LIF spiking) | 2 **homogeneous** (GRU+GRU or MLP+MLP) |
| Task regime | Supervised | Cooperative RL |
| Codebook | 64 codes per WML + transducers | Shared codebook |
| Commitment loss | Yes (VQ-VAE recipe) | Yes |
| Training | CE loss on HardFlowProxyTask | Policy gradient + comm cost |

On the same 3-seed HardFlowProxyTask(dim=16, n_classes=12) config
used by `run_w2_hard`, we train a DVNC-style architecture in
**supervised** mode (retire the RL rollout, keep the VQ shared
codebook + 2-agent communication) and measure:

1. Accuracy per agent.
2. Pairwise gap (acc_a - acc_b).
3. MI/H(a) on emitted codes (plug-in, the nerve-wml Claim B
   headline metric).
4. Effective alphabet utilisation (# codes used / 64).

## Decision criteria (ahead of execution)

The honest outcome matrix:

| Case | nerve-wml acc | DVNC acc | MI/H nerve | MI/H DVNC | Narrative |
|---|---|---|---|---|---|
| A | 0.55 | 0.55 +/- 0.03 | 0.93 | 0.90 +/- 0.05 | "Comparable on homogeneous config; nerve-wml adds heterogeneous case DVNC doesn't cover." |
| B | 0.55 | 0.70 | 0.93 | 0.98 | "DVNC outperforms on homogeneous by design (no substrate gap); our contribution is extending to ANN/SNN." |
| C | 0.55 | 0.40 | 0.93 | 0.60 | "Our transducer architecture recovers the information signal where shared-codebook alone loses it." |
| D | DVNC fails to converge on HardFlowProxyTask | n/a | n/a | "DVNC was not designed for 12-class XOR-on-noise; report as out-of-distribution baseline caveat." |

Case A is most likely. Case B is acceptable. Case C would be
surprisingly favorable. Case D triggers a narrative where we say
"the closest prior method does not transfer to our benchmark" and
note that in the paper.

**Key rule:** we do NOT claim to "beat" DVNC. We claim
**complementarity** (heterogeneous substrate case that DVNC
does not address). If we accidentally win on their turf, that's
a bonus; if we lose, the narrative still holds.

## 3-day schedule

### Day 1 — Reproduction check (compute: kxkm-ai CPU)

**AM (2h)**
```
ssh kxkm@kxkm-ai
cd ~
git clone https://github.com/dianbo-liu/DVNC.git dvnc-reference
# OR whatever the canonical DVNC repo is; check arXiv 2107.02367
cd dvnc-reference
ls -la   # inspect layout
```

Identify:
- Is `VectorQuantizer` or `VQBottleneck` module isolatable?
- What are the dependencies? Any pinned torch version?
- Is there a `train.py` + `config.yaml` entry point?

**PM (3h)**

Run their own baseline to confirm the repo still works:
```
cd dvnc-reference
python train.py --config configs/default.yaml --seed 0
# OR whatever their entry point is
```

If their repo is obsolete (torch 1.x dep, dead packages, etc.),
document the blocker and switch to plan 1B:
- Re-implement a minimal VQ-bottleneck communication architecture
  from their paper's algorithm section (~2h), skip reproduction.

**Day 1 livrable:** either (a) DVNC repo cloned + their baseline
reproduced, or (b) decision to skip reproduction + minimal
reimplementation planned.

### Day 2 — Adaptation to HardFlowProxyTask (compute: kxkm-ai CPU)

**AM (4h)**

Write `scripts/baseline_dvnc.py` based on the scaffold below.
Wire in:
1. DVNC's `VectorQuantizer` module (imported from vendored copy
   under `third_party/dvnc/` OR re-implemented minimally).
2. Two lightweight agent encoders (can be literal copies of
   `MlpWML.core`, for homogeneity with nerve-wml).
3. A shared codebook (the whole point of DVNC: shared, unlike
   nerve-wml per-WML codebooks).
4. Supervised CE loss on HardFlowProxyTask (no RL).

Hyperparameters must **match** `run_w2_hard`:
```
dim=16, n_classes=12, d_hidden=16, codebook_size=64
batch=64, lr=1e-2, steps=800, seeds=[0, 1, 2]
```

**PM (2h)**

Run the 3 seeds on kxkm-ai. Verify accuracies are non-trivial
(> 0.20, well above chance 0.083). Save `codes_dvnc.npz` with
schema compatible with `measure_*.py`:

```
agent_a_codes: int64[3, 5000]
agent_b_codes: int64[3, 5000]
agent_a_embeddings: float32[3, 5000, 16]
agent_b_embeddings: float32[3, 5000, 16]
seeds: int64[3]
n_eval: 5000
steps: 800
```

**Day 2 livrable:** `codes_dvnc.npz` in `tests/golden/` on kxkm-ai.

### Day 3 — Comparison + paper (compute: GrosMac OK, light)

**AM (3h)**

scp the NPZ back to GrosMac. Adapt `scripts/measure_*.py` to
accept custom NPZ keys (add `--codes-a-key` and `--codes-b-key`
optional args) OR duplicate scripts into
`scripts/measure_dvnc_*.py`. Light pref: add args, avoid
duplication.

Run:
```
uv run python scripts/measure_mi_null_model.py \\
    --codes tests/golden/codes_dvnc.npz \\
    --codes-a-key agent_a_codes \\
    --codes-b-key agent_b_codes
```
and similarly for bootstrap, multi_estimator, mine.

Produce comparative table:

| Metric | nerve-wml (MLP<->LIF) | DVNC (A<->B) |
|---|---|---|
| mean acc | 0.55 | Y.YY |
| median gap | 2.7% | Z.Z% |
| MI/H plug-in | 0.932 | X.XXX |
| MI/H Miller-Madow | 0.939 | Y.YYY |
| Alphabet used | 7-27/64 | Z/64 |

**PM (2h)**

Paper edit — recommended site: **new paragraph in Related Work**
(section 2, after the Moschella/Huh paragraph), titled
"Direct comparison to DVNC". Length: ~120 words + table
reference. NOT a new Test (8), to keep the experiments section
stable.

Draft:
> "We adapt DVNC~\\cite{tang2021dvnc} to the HardFlowProxyTask
> supervised regime by isolating its VQ-bottleneck module and
> training two homogeneous agents with cross-entropy loss. On
> 3 seeds at the same hyperparameters as run\\_w2\\_hard, DVNC
> yields [accuracy / MI / alphabet figures]. The principal
> distinction with nerve-wml is substrate symmetry: DVNC agents
> are structurally identical, so no substrate-intrinsic
> asymmetry appears. Our heterogeneous MLP<->LIF configuration
> exposes a different regime: the spike-dynamics substrate
> difference produces a reproducible 2-3% plateau gap (Test ~5)
> that the shared-codebook principle preserves in information
> (MI/H = 0.93) while respecting substrate biology (Test ~7)."

Commit + push. Optional: bump to v1.5.4 or bundle into v1.6.0.

**Day 3 livrable:** 1-2 commits (scripts + paper + data JSON).
Release decision deferred to after comparison content is
seen.

## Danger zones

- **DVNC repo obsolete / dead.** Estimated ~30% probability. Fix:
  reimplement the minimal VQ-bottleneck from the paper
  algorithm (1-2h additional Day 1 work).
- **DVNC does not converge on HardFlowProxyTask.** Its GRU-centric
  agent architecture may fail on tabular 16-dim data. Fix:
  document as failure mode ("DVNC was designed for spatial grid
  tasks; transfer to tabular XOR is not expected to succeed"),
  use that as the narrative.
- **measure_* scripts reject the NPZ schema.** Fix: add --key-*
  optional args OR duplicate into measure_dvnc_*.py. Prefer
  args, documented in day 3 AM block.
- **GPU contention on kxkm-ai.** Training 2 GRU agents is
  light (< 5 min per seed CPU), so no GPU actually needed.
  If DVNC's original code uses GPU hard-coded: --device cpu
  override in our adaptation.

## Cross-refs

- `scripts/baseline_dvnc.py` — scaffold prepared 2026-04-21 evening.
- `scripts/save_codes_for_checks.py` — reference shape for NPZ output.
- `scripts/measure_mi_*.py` — downstream consumers for the
  NPZ codes.
- `papers/paper1/refs.bib` — `tang2021dvnc` entry already added
  in v1.5.2 Related Work update (commit `dfbdbee`).
- `docs/research-notes/n3-gate-role.md` — sibling research note
  from Phase A, reference for format.
