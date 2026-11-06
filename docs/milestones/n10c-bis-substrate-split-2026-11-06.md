# N10-C-bis necessity gate A+ — Deviation declaration 2026-11-06: mixed host (macM1 + KXKM-AI), declared a priori

**Status:** DRAFT — to be signed by the PI with the dated go of `docs/prereg/n10c-bis/RUNBOOK-aplus.md`. Not frozen. It changes no frozen or locked provision of N10-C.
**Declares against:** G5 of the N10-C-bis addendum `docs/milestones/2026-10-25-n10c-bis-control-arms.md` (section « Necessity gate », DRAFT, not frozen; « KXKM-AI, after E1-small has ended, one run at a time »), and the KXKM-AI-only clause of G2 (« KXKM-AI per-host settings »). It also records the PI's dated exemption of the gate from the roadmap rule « no GPU campaign before S6 » (`hypneum-lab/.github`, `docs/ROADMAP.md`).
**Date drafted:** 2026-11-06. The date is set by the GitHub push time of its first commit; git dates are not evidence of time.
**Investigator:** Clément Saillant (Hypneum Lab, sole investigator).

The addendum, the frozen N10-C amendments and the locked pre-registration are not modified. This document is append-only and touches only `docs/`.

## 1. Deviation: the gate runs on two hosts, with the cell-to-host table fixed before any run

**This deviation is declared before any A+ run exists.** Its grounds are resource grounds, not outcomes. It is not outcome-blind with respect to N10-C seed 0: the third blinding breach of 2026-11-05 (`…-amendment-2026-11-04-interim.md`) precedes it, and the PI's decisions of 2026-11-06 were taken after that breach. No A+ outcome exists to be read.

### 1.1 Decision and grounds

The PI decided on 2026-11-06 at 13:58 CET, on the plan `docs/prereg/n10c-bis/three-seeds-plan-2026-11-06.md` (PR #114, section 6):

- Q1 / Q2: gate A+ runs in weeks S3–S5 (9–29 Nov) as development runs, with A+ values not read before the freeze.
- Q5: mixed host, macM1 + KXKM-AI, with the substrate deviation declared a priori.

Grounds:

1. **Time to the M1 decision (29 Nov).** On KXKM-AI alone, A+ takes about 18.6 GPU-h in series. It also competes with the E1-small controls and the N10-C seed-0 evaluation for the same card, which G5 forbids sharing. With a second host, the nominal wall time falls to about 12.5 h, and the KXKM-AI queue gets shorter.
2. **macM1 is available** since N10-C run 2 stopped (2026-11-05, declaration of 2026-11-04). It stays the second brain's host, so it takes one heavy job at a time and pauses the day model only during the block.

### 1.2 Cell-to-host table (fixed here, before any run)

Allocation rule, fixed without reference to any result:
- the new seed goes to KXKM-AI, where its trunk is pre-trained;
- of the two Adam-study seeds, the higher one goes to macM1.

| seed | host | trunk (SHA-256 prefix) | cells, in order | per-host settings |
|---|---|---|---|---|
| 9104 | KXKM-AI | `894688cd4a62` (pre-trained on KXKM-AI, 2026-10-25) | `eng`, `nocomm`, `nocomm_rep` | micro-batch 512, `clear_cache`, 8 workers |
| 9106 | KXKM-AI | to be pre-trained on KXKM-AI before the first gate run | `nocomm`, `nocomm_rep`, `eng` | idem |
| 9105 | macM1 | `d4916633ae72` (pre-trained on KXKM-AI, 2026-10-25; file copied) | `eng`, `nocomm`, `nocomm_rep` | `codes.HOST_SETTINGS["macm1"]`: no micro-batch, no `clear_cache`, 1 worker |

Fixed for every cell:
- condition `closed`;
- 10,000,000 pair-steps;
- preset `fix-c-coin-pretrain`;
- ledger kind `gate_dev`;
- liveness guard in `abort` mode;
- training levels 0–15.

`nocomm_rep` of seed S is `nocomm` with seed S+10 (9114, 9115, 9116) and the trunk of seed S, on S's host (plan, section 6.6, item 4).

The two per-host settings give the same update. Gradient accumulation over micro-batches of 512 pairs leaves the PPO minibatch unchanged (amendment of 2026-10-23, « Hardware, framework and hosts »). They are the settings of each host's existing measurements.

### 1.3 What differs between hosts

| aspect | KXKM-AI | macM1 |
|---|---|---|
| ISA / GPU | x86-64, RTX 4090 (CUDA) | arm64, M1 Max (Metal) |
| MLX backend | CUDA | Metal |
| Procgen 0.10.7 | official x86-64 wheel | arm64 source build with the archived build patch |
| bit-identity of frames | — | **not bit-identical** to x86-64 (amendment of 2026-10-23, reproducibility note); effect on results unknown |
| trunk origin | pre-trained on this host | pre-trained on KXKM-AI, used on macM1 (as was done for seed 9102, macM1 → KXKM-AI, 2026-10-25) |

### 1.4 Rules that hold the deviation in place

- **Pairing stays inside a host.** For every seed, D_s = `eng − nocomm` and D0_s = `nocomm_rep − nocomm` compare three runs of the same seed, trunk, host, build and code. No difference is ever taken across hosts.
- **Host is not a factor** in the decision rule. With k = 3 and a 2 + 1 split, a host effect cannot be estimated. The seed-level D_s are averaged as G2 says, and the host of each seed is reported next to it.
- **No cell changes host after this declaration.** A cell may be rerun once (G2) only on its declared host. Moving a seed to another host before any of its runs starts needs a new dated declaration. After one of its runs has started, the seed is reported as not measurable instead.
- **G5's no-parallel rule stays in force on each host.** On KXKM-AI no other GPU job of ours runs in parallel (`campaign/kxkm/gpu_wait.sh`). On macM1 the block holds the shared heavy-job lock (`~/scratch/.heavy.lock`) and pauses the day model (`campaign/macm1/day_model_paused.sh`). The two hosts run in parallel with each other. This is the deviation itself.
- **Measure.** The gate uses the training success curve (G2), computed in-process on each host. No held-out evaluation and no cross-host evaluation is needed. The x86-64-only evaluation rule of N10-C is therefore not touched.

## 2. Seed status declared with it (Q3, working hypothesis to be confirmed at the go)

- **9104 and 9105 are not untouched seeds.** In the 2026-10-25 Adam study each has a KXKM-AI trunk and four `gate_dev` runs (`gtm` and `eng`, condition `registered`, 1M pair-steps, presets C and C-bc), and those results were read (amendment of 2026-10-25, section 4.3).
- **None of them has a `nocomm` run or a `closed` run.** The G2 contrast has never been observed on them.
- **9106** has no trace in the repository on 2026-11-06.
- **9101–9103 (G7 pilot)** are excluded whether or not G7 results exist.
- The decision criterion (Q4) is G2 unchanged, with the controls of section 1.3 of the plan. It is also a working hypothesis to be confirmed at the go.

## 3. Status of the A+ runs

As in G4:
- the A+ runs are exploratory development runs (`gate_dev`);
- they never use held-out levels;
- they are never pooled with N10-C or N10-C-bis runs;
- they are not a pilot variance estimate.

The roadmap exemption is dated 2026-11-06 and covers these 9 runs (plus at most one rerun each) and one trunk pre-training. It covers nothing else.

## 4. PI signature

- [ ] Signed by the PI on ______ (date, time CET), with the go of `RUNBOOK-aplus.md`, confirming section 1.2 and the working hypotheses of section 2.
