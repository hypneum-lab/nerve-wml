# N10-D — causal use of message on the symmetric twin (L0 draft)

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. The open "TBD by PI" items TBD-11, TBD-15, TBD-16 and TBD-17 block
the freeze (hypneum-prereg); they are listed in section 11.
**Version:** v2 (2026-11-08), supersedes v1 (NH-96) after the blind contradictor
report of 2026-11-08 and 12 PI decisions of 2026-11-08 16:30–16:45 CET; L0 draft,
not frozen.
**Origin of v1:** `docs/milestones/2026-10-25-n10c-bis-control-arms.md`, section
"Necessity gate", item G3. NH-96 is not abandoned: it is re-read as a positive control
of the intervention machinery in section 3; it is not a primary outcome and spends no
alpha.
**Annex G7 (PR #99) is not touched.**

## 1. Research questions

V1 (primary): On the symmetric twin (`sym`), where the listener's own observation
includes the target, does the trained listener paired with the engineered sender
(`eng_sym`) choose the conflict candidate t′ — the one named by the perturbed message —
over the placebo candidate t″ at a rate that reliably exceeds the placebo baseline, with
a CI lower bound on the mean per-seed ρ ≥ δ_use (TBD-16)?

V2 (co-primary, modulator): Does the message-dependence index ρ differ between
the engineered sender (`eng_sym`) and the jointly trained sender (`learned_sym`), at the
same trunk, seed set and training budget?

The two questions are co-primaries under a fixed, pre-specified hierarchical
procedure at FWER 0.01 applied to {ρ_eng, Δρ}, where ρ_eng is the mean per-seed ρ
for `eng_sym` and Δρ is the two-sample contrast ρ(eng_sym) − ρ(learned_sym). V1 is
tested first in a fixed, pre-specified hierarchical order (not data-dependent); V2
only if V1 rejects: V1 at α₁ = 0.005, then V2 at α₂ = 0.01. This is more
conservative than classical Holm (which orders by p-value), controls FWER ≤ 0.01,
and reflects the gating: V2 (modulator) is meaningful only if V1 (message use)
rejects.

NH-96 ("channel plumbing and trunk learn the engineered code on asym") is not a
primary. `eng_asym` is the positive control of the intervention machinery (section 3,
PC3): it must drop under message shuffle by approximately S_intact − 1/C, as a
confirmation that the derangement pipeline works. No α is spent on it.

## 2. Prior work this does not redo

The following published results define what is already known and constrain the
scientific space. This study does not replicate them.

### 2.1 Kharitonov, Chaabouni, Bouchacourt, Baroni 2020 (arXiv:1905.13687)

Guess Number game, GS / SCG / REINFORCE, C = 2^k (k bits), joint training. At k = 8
(symmetric twin — listener sees all bits): "this information is entirely ignored by
Receiver" (§ 4.1); "the drop in accuracy after shuffling is tiny" (supplement § 7).
Intervention: uniform shuffle of messages in the evaluation batch (EGG
`intervention_message`, commit 54eb11c6).
Limit: seeds selected on success (REINFORCE ≈ 20 %); no fixed sender; no
conflict/placebo dissociation; no paired null by seed.

### 2.2 Villecroze & Moulin-Frier 2020 (hal-03100681)

Fixed oracle sender, single-agent RL, variable view size. With any view size the
listener "uses almost exclusively the communication channel and neglects the visual
input even when it is able to see the target" (§ V-B). Answer opposite to 2.1.
Limit: CIC metric, no conflict/placebo, 10 seeds; different architecture.

### 2.3 Chauhan & Bera 2026 (arXiv:2609.34373)

PPO, 25 paired seeds, engineered / learned / silence / oracle, three arenas. Coupling-
rigid arena (listener already holds the information — analogue of `sym`): "every
channel is within 0.001 of silence and nothing is significant" (§ 4.1, Table 2).
Limit: score difference only; no within-episode conflict/placebo dissociation; no
seed-paired null for the sym analogue.

### 2.4 OpenReview hukTsKYu8Y

OFF / transplant / restoration design, three distinct agent responses, continuous
channel, supervised training. Channel used; modules homogeneous.
Limit: not RL; no arms without channel; no paired null.

### 2.5 Gap this study addresses

The two published answers to "does the sym listener use the message?" are opposite
(2.1: ignored; 2.2: used). No published study combines the same PPO trunk, fixed vs
learned sender, within-episode conflict/placebo dissociation, paired null by seed, and
no seed selection on the same task. The modulator V2 is an inference from results 2.1
and 2.2; the gap is narrow and unconfirmed by a citation descent.

**Warning (incomplete descent):** citations from arXiv:1905.13687 were not fully
retrieved (Semantic Scholar returned 429 on 2026-11-08); a paper implementing V1
may exist. See Reading lock (section 14).

## 3. Design

### 3.1 Trained arms

| arm | channel | training condition | role |
|---|---|---|---|
| `eng_sym` | engineered sender, ideal code, fixed | `sym` | primary (V1) |
| `learned_sym` | jointly trained sender, alphabet A | `sym` | V2 comparison |
| `nocomm_sym` | zeroed | `sym` | gate G1: sufficiency of own path |
| `eng_asym` | engineered sender, ideal code, fixed | `asym` | PC3: positive control of machinery, no α |

**What v1 and spec.py already planned for `sym`:** `spec.py` v1 defines
`Condition = Literal["asym", "sym"]` and `Arm = Literal["eng", "nocomm", "learned",
"msg_shuffle"]`; all combinations are syntactically supported. `protocol.md` v1
explicitly planned `eng_sym` (sym gate: mean ≥ 1 − SYM_EPSILON) and `nocomm_sym`
(descriptive control of the trunk). `learned_sym` was **not** planned in v1 (sym
was a leak check only; v1 primary was on `asym`). Training `learned_sym` requires
new runs (see TBD-11).

Restoration and frozen-message conditions are not in this design (PI decision 6):
at L = 1 without state these are trivially equivalent to the intact condition and
carry no causal information about message use.

The observation encoding (one-hot or embedding, TBD-11) is frozen at the freeze. The
result is declared as specific to that encoding; it does not generalise to other
encodings by default.

### 3.2 Evaluation conditions

Five conditions, all evaluated on the same n_eval = 500 fresh episodes per
(backend, arm, seed). Episode seeds are disjoint from training. All five conditions
share exactly the same episode list; the five evaluations are aligned episode by
episode.

| label | listener receives | listener sees (sym) | counted for ρ |
|---|---|---|---|
| `intact` | m(t) — correct message | t + C candidates | — (descriptive) |
| `msg_shuffle` | m(t′′) for another episode (deranged) | t + C candidates | — (negative control) |
| `conflict` | m(t′), t′ ≠ t drawn uniformly from the C − 1 other candidates | t + C candidates | 1(action = t′) |
| `field_t_deranged` | m(t) — correct message | t_rand (deranged) + C candidates | — (descriptive) |
| `placebo` | m(t) — intact | t + C candidates | 1(action = t″) |

`conflict` generation: for each episode e, t′_e is drawn uniformly from the C − 1
candidates visible to the listener that are not t_e; the engineered (or learned) sender
emits the message for t′_e. This is the symmetric-twin analogue of A21 § 3.3.
Both t (intact in the listener's sym observation) and the conflict target t′ are
present in the candidate list; the listener can in principle pick either.

`placebo` generation: for each seed, before evaluation, one t″_e ≠ t_e is
designated for each episode, using a fixed random seed declared in the freeze manifest.
The message is intact (says t correctly) and the listener sees t; the question is
whether the listener accidentally or systematically picks t″.

`field_t_deranged`: the listener's direct observation of t is replaced by a different
identity t_rand drawn uniformly from K = 16 (coincidence with t permitted); the C
candidates and the message remain intact. This is the analogue of EGG
`intervention_input` restricted to the t field, candidates kept. It is descriptive;
it does not enter ρ.

`msg_shuffle` on `eng_asym` (asym condition, PC3): same uniform episode derangement
as `derange_episodes`; on asym, the message is the only channel toward t, so
S_intact − S_shuffle ≈ 1 − 1/C analytically (A21 § 2.3). This is used for PC3
only; `eng_asym` does not participate in the conflict/placebo design.

### 3.3 Environment (same as v1 section 7)

One step per episode, identical on both backends:
1. Target t uniform over K = 16 identities; C − 1 = 3 distractors drawn without
   replacement; C = 4 candidates shuffled per episode.
2. Speaker observation: t only. Listener (asym): C candidates in shuffled order.
   Listener (sym): same C candidates plus t.
3. Speaker emits one symbol (alphabet A = 16; the engineered code maps each identity
   to one symbol). Channel transforms it. Listener receives channel output with
   its observation. Listener acts.
4. Listener action: one index in {0, ..., C − 1}; always a valid candidate.
5. Reward: 1 if the chosen candidate is t, else 0; episode ends.

TBD-11: observation encoding (one-hot or embedding), training budget B, evaluation
action rule (argmax or sampled). L = 1 (one step), trunk MLP 2 × 256, 100 logs/run:
working values, not decisions.

## 4. Statistics

### 4.1 Per-seed statistic ρ_i (V1)

For each (backend, arm ∈ {eng_sym, learned_sym}, seed i), using the same 500 episodes
evaluated under `conflict` and `placebo`:

    rho_i = (#{action = t'_e : episode e in conflict set} / N_conflict)
            - (#{action = t''_e : episode e in placebo set} / N_placebo)

where N_conflict = N_placebo = n_eval = 500. Under H0 (message not used):
the listener picks the candidate matching its own field t, never the designated
wrong candidates t′ or t″, so ρ_i ≈ 0. Under H1 (message dominant): the listener
follows the conflict message (picks t′) and follows the intact placebo message (picks
t, not t″), so ρ_i ≈ 1.

### 4.2 Two-sample contrast Δρ (V2)

    Delta_rho = mean_i(rho_i(eng_sym)) - mean_i(rho_i(learned_sym))

The 25 eng_sym seeds and 25 learned_sym seeds are trained independently (distinct runs,
distinct parameter initializations even at the same seed index). There is no
within-index covariance between independently trained agents. Δρ is therefore a
two-sample estimator, declared as such. The same 25 seed indices are used for both
arms within each backend, enabling the within-index permutation null (section 5.2),
but the estimator is two-sample, not paired.

## 5. Nulls

Both nulls use the same pipeline and the same 500 paired episodes. B = 10 000 draws
each. Empirical p = (#{null ≥ observed} + 1) / (B + 1).
Report: observed statistic, null median, null P95, null P99, empirical p.

### 5.1 Null 1: conflict/placebo exchange (for ρ, V1)

For each draw b ∈ {1, ..., B}:
  For each seed i:
    For each episode e ∈ {1, ..., 500}:
      With probability 1/2 (independent coin per episode):
        swap the conflict and placebo labels of episode e
        (action_conflict_e and action_placebo_e are exchanged)
    Compute rho_i_null from the relabeled assignment.
  rho_null_b = mean_i(rho_i_null)

Under H0 (message not used): the listener's action on episode e does not depend on
which perturbation (conflict or placebo) is applied; the per-episode swap leaves the
action distribution invariant, so rho_null ≈ rho_observed in distribution.

### 5.2 Null 2: arm-label permutation (for Δρ, V2)

For each draw b ∈ {1, ..., B}:
  For each seed index j ∈ {1, ..., k}:
    With probability 1/2 (independent coin per index):
      swap rho_j(eng_sym) and rho_j(learned_sym)
  Delta_rho_null_b = mean(arm labeled eng) - mean(arm labeled learned)

P-value: two-sided, p = (#{|Delta_rho_null| >= |Delta_rho_observed|} + 1) / (B + 1),
because V2 does not predict a direction a priori (Kharitonov 2020 and Villecroze 2020
give opposite answers).

## 6. Evaluation order and gates

Checks run in this order. The first failure that applies determines the outcome.

**Gate design (corrects F1 of the contradictor report):** gates are at the arm level,
not at the seed level and not on the primary arm. No seed is excluded after the fact.
The gate on `nocomm_sym` (G1) tests whether the own path is functional; it cannot
make any ρ(eng_sym) value unreachable, because a low ρ(eng_sym) is a valid result
("message not used") that does not cause G1 to fail. The decodability gate G2
affects V2 only; V1 proceeds unaffected if G2 fails.

### 6.1 Arm-level gates

**G1 — `nocomm_sym` path sufficiency (TBD-15):**
mean over k seeds of `nocomm_sym` success (intact condition) ≥ 1 − ε (TBD-15).
If this fails: the listener cannot succeed via its own field on this arm/backend
combination; the question "which path does the listener use?" is ill-posed.
Outcome: "non mesurable" for both V1 and V2.
ε (TBD-15): working value 0.75; exact value set before the freeze after the pilot
nocomm_sym distribution and the evaluation action rule (TBD-11) are known. Note:
ε = 0.75 gives threshold 1 − 0.75 = 0.25 = 1/C for C = 4, meaning any learning seed
would pass; the final value must be strictly > 1/C + a margin established from the
pilot `nocomm_sym` distribution.

**G2 — decodability of `learned_sym` (TBD-15 linked):**
MI(message; t) or a linear probe accuracy for `learned_sym`, measured on CPU,
committed before the confirmatory run, must exceed a threshold declared before the
freeze. If this fails: the learned channel carries no decodable information about t;
ρ(learned_sym) = 0 is a trivial consequence, not a scientific result.
Outcome: "non mesurable" for V2 only; V1 on eng_sym proceeds.

### 6.2 Positive controls

Three scripted or known controls; all must pass before reading the primaries.
They confirm the intervention pipeline without spending α.

**PC1 — scripted "message only" listener (ρ ≈ 1):**
A listener that always picks the candidate named by the message. Under conflict
this listener picks t′ (rate ≈ 1); under placebo it picks t (rate picking t″ ≈ 0).
Expected ρ ≈ 1. If far from 1: the conflict/placebo generator is broken.

**PC2 — scripted "field t only" listener (ρ ≈ 0):**
A listener that always picks the candidate matching its own t-field, ignoring the
message. Under conflict it picks t (not t′); under placebo it picks t (not t″).
Expected ρ ≈ 0. If far from 0: the t-field derangement pipeline is broken, or the
conflict condition is misconfigured such that t′ ≠ t is not enforced.

**PC3 — `eng_asym` under message shuffle (drop ≈ S_intact − 1/C):**
The trained listener on `asym` with the engineered sender must drop in success by
approximately S_intact − 1/C under `msg_shuffle` (A21 § 2.3: the expected success
under any episode derangement is exactly 1/C). With S_intact ≈ 1 on `asym`, the
expected drop is ≈ 0.75. If PC3 fails: the message-shuffle pipeline is broken on
this backend.

Pass criterion: the observed drop S_intact − S_msg_shuffle on `eng_asym` must exceed
0.50 (two thirds of the oracle contrast 0.75). Working value; refined after the pilot
and committed before the freeze. If S_intact < 0.90 on `eng_asym` (learning
incomplete), the threshold becomes 0.5 × (S_intact − 1/C).

### 6.3 Primaries (read only after G1, G2, PC1–PC3 pass)

Read ρ(eng_sym) for V1 and Δρ for V2 from the outcome table (section 7).
Apply the hierarchical procedure of section 1: test V1 at α₁ = 0.005 using null 1
p-value; if V1 rejects, test V2 at α₂ = 0.01 using null 2 p-value.

## 7. Outcome table

D1 = mean_i(ρ_i(eng_sym)); L1, U1 = lower and upper bounds of its 99 % CI
(level follows α₁ = 0.005).
D2 = Δρ; L2, U2 = lower and upper bounds of its 98 % CI
(level follows α₂ = 0.01).
CI method: bootstrap percentile or paired t; method fixed before the freeze and
the Monte-Carlo power rerun with it (section 8).

**Decision rule:** the formal test is the permutation p-value. V1 rejects if
p₁ < α₁ = 0.005 (null 1); V2 rejects if p₂ < α₂ = 0.01 (null 2, only if V1
rejects). The CI does not determine whether V1 or V2 rejects; it classifies the
outcome among passing results only: a CI bound crossing δ_use or δ_eq gives the
outcome label (rows 2–3 and Δρ rows below) but the threshold that triggers the
report is the p-value gate.

Rows are checked in this order; the first that holds is the outcome.

| order | outcome | applies to | rule | reading |
|---|---|---|---|---|
| 0 | non mesurable | V1 and V2 | gate G1 fails | own path not functional; no ρ read |
| 0′ | non mesurable (V2 only) | V2 | gate G2 fails | learned channel not decodable; V1 proceeds |
| 1 | non mesurable | V1 and V2 | PC1, PC2 or PC3 fails | pipeline broken; no ρ read |
| 2 | message dominant | V1 | L1 ≥ δ_use (TBD-16) | listener prefers message over own field under conflict |
| 3 | message ignored | V1 | U1 < δ_eq (TOST, TBD-17) | listener does not use message within the equivalence margin |
| 4 | non concluant | V1 | otherwise | uncertainty; report D1 and CI |
| — | Δρ positive | V2 | L2 ≥ δ_use (TBD-16) | jointly trained sender yields less message dominance |
| — | Δρ negative | V2 | U2 ≤ −δ_use (TBD-16) | jointly trained sender yields more message dominance than the engineered sender |
| — | Δρ null | V2 | U(|D2|) < δ_eq (TOST, TBD-17) | no detectable difference between sender types |
| — | non concluant | V2 | otherwise | report D2 and CI |

δ_use (TBD-16) and δ_eq (TBD-17) are set after the pilot and before the freeze.
Justification procedure: the between-seed sd of ρ from the pilot, combined with the
maximum binomial noise at n_eval = 500 (sd ≤ sqrt(0.25 / 500) ≈ 0.022 per proportion
term), determines the achievable precision. The per-seed sd of ρ_i (difference of
two proportions) is at most √2 × 0.022 ≈ 0.032. Both "message dominant" and "message
ignored" must remain achievable for some ρ value after δ_eq and δ_use are set:
δ_eq < δ_use, with a gap of at least a few times the noise floor.

No seed is excluded after the fact. The per-seed distribution of ρ_i is reported as
a histogram or stem-and-leaf plot with every outcome (by backend). Non-rejection is
"non concluant", never "refuted".

## 8. Analysis plan

### 8.1 Seeds

k = 25 seeds per backend (PI decision 10). Same 25 seed indices for `eng_sym` and
`learned_sym` within each backend (enables null 2). Confirmatory seed list declared
in the freeze manifest, disjoint from development seeds 9600–9607 and from any pilot
seeds. Pilot: development seeds 9600–9607, `nocomm_sym` only (measure G1 rate and
between-seed sd for ε calibration; throughput TBD-11). `eng_sym` pilot: 8 seeds,
`conflict` and `placebo` conditions; measure per-seed ρ distribution for power and
δ calibration. Pilot seeds are never pooled with confirmatory runs.

### 8.2 Power

Verified by Monte-Carlo after the pilot (once δ_eq and δ_use are set), before the
freeze, on the actual decision rule and the actual CI method. Statistic for V1:
mean(ρ_i(eng_sym)) and its CI. Statistic for V2: Δρ and its CI.

If the per-backend power for V1 or V2 is < 0.80 at the pilot planning effect, k is
raised by amendment before the freeze. The amendment must be filed before any
confirmatory run.

No planning distribution for ρ_i is assumed in this draft; it will be read from the
pilot (unlike v1, where the bimodal pi = 0.80 assumption was used before any pilot).

### 8.3 CI method

Bootstrap percentile or paired t for ρ (V1, 99 % CI) and for Δρ (V2, 98 % CI).
Method fixed before the freeze; the Monte-Carlo power is rerun with the chosen method.
The choice follows the same reasoning as v1 TBD-16 (Hesterberg 2015, arXiv:1411.5279,
section 5.3: percentile under-covers at small n; paired t needs normality). CI level
follows α: 99 % for V1 (α₁ = 0.005) and 98 % for V2 (α₂ = 0.01). These are wider
than the nominal 95 % — they classify outcomes, not the rejection test.

## 9. Backends

Both backends, EGG (forked, commit 54eb11c6) and JaxMARL (@31b0a23), run the same
environment (section 3.3). Same learner (N10-C PPO trunk adapted per backend), same
seed indices, same n_eval = 500. Each backend has its own null calibration, its own
per-seed ρ report, and its own V1/V2 outcome status from section 7.

Divergence rule: the global V1 verdict requires both backends to reach the same
conclusive outcome (message dominant or message ignored); any disagreement, or any
"non concluant" or "non mesurable" on either backend, gives a global "non concluant".
Same for V2. Disagreement between backends is a separate hypothesis for an amendment;
no result is selected after the fact.

JaxMARL / Metal viability on macM1 (ARM MPS) must be established at the pilot
(TBD-11). If JAX Metal does not support the required ops at pilot time, the declared
fallback (EGG only, or JaxMARL on CPU only) is filed as a deviation before the first
confirmatory run. The fallback must be declared before the pilot run, not after.

## 10. Host

All pilot and confirmatory runs (training and evaluation for all arms and backends)
on **macM1** (Apple M1 32 GB, CPU/MPS). Rationale: on-site, available, no queue;
oMLX paused during runs to free memory; suitable for 2 × 256 MLP trunk and referential
game at L = 1.

Operational constraints: `lockf` for concurrent file access; training data and results
on the T7 volume (macM1); archives on ES + T7 (double copy, sha256 verified before
any deletion); oMLX paused at run start.

Before any pilot run: measure throughput without reading any score (TBD-11, one cell,
one arm, one backend), to bound run time and to establish JaxMARL / Metal viability.

## 11. What is not done here

No run, no install, no compute, no freeze. Missing before the pilot:
- Hashed environments for both backends; `backends.py` raises `NotImplementedError`.
- N10-C PPO trunk adaptation per backend.
- Engineered sender and scripted listeners.
- Conflict/placebo generator (analogous to A21 twin; see `null-v2.md` section 3).
- A priori leak audit script (control 4a of v1; retained in principle).
- Throughput measurement (TBD-11).

Missing before the freeze:
- TBD-11: training budget B, evaluation action rule.
- TBD-15: ε for gate G1; decodability threshold for G2.
- TBD-16: δ_use; TBD-17: δ_eq; both from the pilot ρ distribution.
- CI method (bootstrap or paired t) and power Monte-Carlo with it.
- Freeze manifest with seed list, encoding, hashes and SWHID.

## 12. TBD items

| id | status | description |
|---|---|---|
| TBD-11 | OPEN | Throughput measurement without score (macM1, one cell of one arm, one backend, no score read). Needed to bound run time and to establish JaxMARL / Metal viability. Working values: L = 1, 100 logs/run, trunk MLP 2 × 256, n_eval = 500. Training budget B not fixed. B fixed from the pilot as the smallest budget at which eng_sym and nocomm_sym success plateau (change < 0.01 over the last 20 % of steps on all pilot seeds), committed before the freeze. |
| TBD-15 | OPEN | ε for gate G1 (mean `nocomm_sym` ≥ 1 − ε). Working value 0.75; conservative. Exact value set before the freeze from the pilot `nocomm_sym` distribution and the evaluation action rule (TBD-11). Also covers the decodability threshold for gate G2. |
| TBD-16 | OPEN | δ_use: CI lower bound threshold for outcome "message dominant". Set after pilot, before freeze; justified by pilot between-seed sd and binomial noise floor (max ≈ 0.022 per proportion). |
| TBD-17 | OPEN | δ_eq: TOST upper bound for outcome "message ignored". Same justification as TBD-16; δ_eq < δ_use required so both outcomes remain achievable. |

## 13. References

- Kharitonov, Chaabouni, Bouchacourt, Baroni 2020, "Entropy Minimization In Emergent
  Languages", ICML 2020, PMLR v119, arXiv:1905.13687.
- Villecroze, Moulin-Frier 2020, "When and why emergent communication is non-trivially
  compositional", SMILES workshop @ ICDL 2020, hal-03100681.
- Chauhan, Bera 2026, "Emergence, Not Bandwidth: Physical Coupling and the Limits of
  Learned Multi-Agent Communication", arXiv:2609.34373v2.
- OpenReview id hukTsKYu8Y (cited in the reading note; full text not independently
  resolved in this document).
- Lowe, Foerster, Boureau, Pineau, Dauphin 2019, "On the Pitfalls of Measuring Emergent
  Communication", AAMAS 2019, arXiv:1903.05168.
- Eccles, Bachrach, Lever, Lazaridou, Graepel 2019, "Biases for Emergent Communication
  in Multi-agent Reinforcement Learning", NeurIPS 2019, arXiv:1912.05676.
- Hesterberg 2015, "What Teachers Should Know About the Bootstrap",
  Am. Stat. 69:371-386, arXiv:1411.5279.

The contradictor report (2026-11-08), the options document (2026-11-08) and the 12 PI
decisions (2026-11-08 16:30–16:45 CET) are on file with the PI
(`docs/prereg/n10d-reviews-2026-11-08/`; not committed in this branch).

## 14. Changes from v1 (2026-11-08)

This section records what changed from v1 (`prereg/n10d-skeleton`, e0c64a8,
2026-11-08), introduced after the blind contradictor report and 12 PI decisions.

### Primary change: axis

v1 (NH-96) asked whether the channel plumbing and PPO trunk realise the `eng − nocomm`
contrast on `asym`. The contradictor found (§ 1.1–1.3) that this question is answered
analytically and by Kharitonov 2020 and Chauhan & Bera 2026. v2 asks whether the
trained listener uses the message causally on `sym`, where the information is redundant.

### Dropped from v1

- NH-96 as primary hypothesis; `asym` as the primary condition.
- The necessity question (`eng_asym − nocomm_asym`); `nocomm_asym` arm.
- Restoration and frozen-message conditions (PI decision 6: trivial at L = 1).
- `sym` as a mere leak check; it becomes the primary condition in v2.
- Outcomes A/B/C/D and the margin 0.30 of v1; the CI rule (v1 TBD-16); k = 20.
- Host KXKM-AI / electron-server (now macM1 for all runs).
- v1 amendments A1–A22 (they informed v2 but are not carried forward as amendments).

### Added in v2

- V1 primary: ρ = P(conflict = t′) − P(placebo = t″), co-primary with V2 under
  Holm FWER 0.01.
- V2 co-primary: Δρ two-sample contrast (eng_sym vs learned_sym).
- Five evaluation conditions: intact, msg_shuffle, conflict, field_t_deranged, placebo.
- Null 1 (conflict/placebo per-episode exchange) and null 2 (arm-label permutation).
- TOST outcome "message ignored" (TBD-17 δ_eq) and "message dominant" (TBD-16 δ_use).
- `learned_sym` trained on sym (not in v1).
- `eng_asym` repurposed as PC3 positive control (no α).
- k = 25 per backend (up from 20); macM1 as host (all runs).
- Sections "Prior work this does not redo" and "Reading lock".
- TBD-16 and TBD-17 (new); TBD-11 and TBD-15 retained with new meanings.

## 15. Reading lock

This v2 draft must pass a **new blind contradictor review** before the freeze. The
review should verify three items:

1. **Literature gap** (sections 2.1–2.5): has V1 (conflict/placebo dissociation on
   `sym`, PPO, fixed sender, no seed selection, paired null) been published? The
   citation descent from arXiv:1905.13687 was incomplete (Semantic Scholar 429 on
   2026-11-08, § 2.5); a full descent is required before the freeze.

2. **F1 correction** (section 6): can all outcomes of ρ (message dominant, message
   ignored, non concluant) be reached regardless of gate results? G1 is on
   `nocomm_sym`, not on `eng_sym`; G2 affects V2 only. Verify that no gate
   configuration blocks a valid ρ(eng_sym) outcome (non-circularity check).

3. **Null validity** (section 5.1): is the per-episode conflict/placebo exchange a
   valid null for the causal use test? Specifically, verify that the exchange is
   valid under the assumption that actions are determined per episode and not across
   the evaluation batch (H-d of A21 § 2.3); and that t′_e and t″_e are assigned
   deterministically and do not depend on the action being measured.

The v0.5 reading note, the 2026-11-08 contradictor report, and the results of these
three tests must be recorded in the freeze manifest before any freeze tag is pushed.
No freeze tag while any TBD item is open.
