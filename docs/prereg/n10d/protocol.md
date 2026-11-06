# N10-D - asymmetric communication-necessity study (L0 skeleton)

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. The open "TBD by PI" items TBD-11 and the CI method of TBD-16 block
the freeze (hypneum-prereg); the other questions the literature cannot settle are listed
in section 9, "Open for the PI".
**Amended:** 2026-11-06 after a blind contradictor, every finding checked in source
(section 8), then by PI decisions of 2026-11-06 16:55 CET (section 9), then reconciled
with the parallel revision of PR #100 (section 8, "Reconciliation 2026-11-06"; PI
decision 17:12 CET), then reconciled a second time with the head of PR #100 at cb7d66b
(PI decisions of 17:05 CET on TBD-10 and TBD-12 to TBD-16, section 0; section 8,
"Second reconciliation (cb7d66b)"), then by the sourced analysis the PI chose as arbiter
(~18:00 CET, `analysis-4-points-2026-11-06.md`, section 9), then by the PI decision on
the primary outcome (21:05 CET, section 9). Nothing was frozen, so the
text is edited in place; the trail is sections 8 and 9.
**Prepared:** 2026-10-31 (night), by decision of the PI (multiple-choice, 2026-10-31 22:11 CET).
**Origin:** `docs/milestones/2026-10-25-n10c-bis-control-arms.md`, section "Necessity gate",
item G3: on failure of the gate (`eng - nocomm >= 0.20`) N10-C-bis is not frozen and the
next study is N10-D, "an asymmetric cooperative task in which each agent holds private
information that its partner needs", pre-registered in its own document before any run.
NH-96 (catalogue of new hypotheses, `.github`, `2026-10-30-etat-des-lieux/synthese/priorites-hypotheses.md`)
is the hypothesis behind it. No equivalent branch, PR or file existed at 2026-10-31.
**Annex G7 (PR #99) is not touched.**

## 0. PI decisions 2026-10-31 (multiple-choice, 22:30-22:35 CET), updated 2026-11-06

Nothing in this table is frozen. CLOSED means decided by the PI for this draft, not
registered; OPEN items block the freeze.

| id | decision (closed unless marked TBD) |
|---|---|
| TBD-1 | CLOSED. Both backends, JaxMARL and EGG. EGG is forked into the `ailiance` org at commit 54eb11c6 with code review. Articulation and divergence rule: section 7 |
| TBD-2 | CLOSED. The N10-C PPO trunk, adapted to each backend (section 7) |
| TBD-3 | CLOSED. K=16, C=4, A=16; blind ceiling 1/C = 0.25 (asym) stays exact |
| TBD-4 | CLOSED. Seeds: pilot 9600+, confirmatory 600+. k derived by the rule of section 4 (alpha 0.05). Margin: TBD-12; k: TBD-13. *The 2026-10-31 formula (k from sigma_seed, minimum 5) and its 2026-11-06 A4 amendment (sign test from a pilot sigma_D, minimum 8) are superseded by TBD-13 and TBD-14* |
| TBD-5 | CLOSED. D_sym must not exceed the P99 of its calibrated null. *Superseded 2026-11-06 (A5) by PI-1: `sym` leaves H1; it is a leak check with its own gate (section 3, control 3; section 5, case D)* |
| TBD-6 | CLOSED. D_asym < 0.10 = failed prediction; "the architecture blocks" is a new hypothesis for a separate amendment (section 5) |
| TBD-7 | CLOSED. Host: KXKM-AI after N10-C-bis; if possible a second DGX GB300 slot, declared as alternative host. Each arm and each seed stays on ONE host (no mixing inside an arm-seed cell) |
| TBD-8 | CLOSED. Grouped HITL review at lock time; flax one version back under the 2-week rule (0.12.10). Hashes still to be produced |
| TBD-9 | CLOSED. Pilot: 8 development seeds 9600-9607, reduced budget. NH-96 scores are not read; only sigma_seed is extracted, by `experiments/n10d/sigma_seed.py`, committed before the pilot. Not amended (TBD-14): the pilot reads `nocomm` only |
| PI-1 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 1). `sym` leaves H1; it stays a leak check reported next to the results |
| PI-2 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 2). The null is paired by seed (seed effect common to both arms), `nh96-null.md` section 2 |
| TBD-10 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). Divergence rule of section 7: per-backend statuses and conjunctive combination table (both backends must conclude). k is sized for the global power (section 4, TBD-13). *Superseded 2026-11-06 18:30 by sourced analysis (PI decision: follow the analysis): two B give a global B, "weaker than predicted on both backends"; identical verdicts A, B or C propagate, any disagreement is "non concluant" (section 7)* |
| TBD-11 | OPEN. Deferred until a throughput measurement without score (one cell, macM1 or KXKM). Working values, not decisions: L = 1, 100 logs per run, trunk MLP 2 x 256, pilot budget B/4; observation encoding per backend; evaluation action rule (sampled or argmax). B is not fixed. n_eval = 500 CLOSED (A8, confirmed by the PI 2026-11-06 16:55, section 9) |
| TBD-12 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). Absolute margin m = 0.30, written 0.4 x (1 - 1/C). Under the CI rule (TBD-16) a listener at 55 % no longer passes: the CI lower bound of D_asym must reach 0.30 |
| TBD-13 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). k by simulation of the actual rule (TBD-16: CI rule + sign test) under the planning assumption of section 4, power >= 0.90 for the conjunction of the two backends (TBD-10): K_CONFIRM = 18 per backend (macM1, 20 000 simulations, 2026-11-06) (section 4, `experiments/n10d/required_seeds_mc.py`). The former k = max(MIN_K, k_mean, k_sign) is obsolete. *Superseded 2026-11-06 18:30 by sourced analysis (PI decision: follow the analysis): k = 20 per backend, a design constant (global power 0.933-0.964 at pi = 0.8, 0.885 at pi = 0.75 in the analysis; Monte-Carlo rerun at k = 20: 0.934 at pi = 0.8, 0.808 at pi = 0.75); MIN_K removed (section 4)* |
| TBD-14 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). sigma_D comes from the planning assumption (section 4), not from the pilot. TBD-9 is not amended: the pilot reads `nocomm` only; sigma_seed(nocomm/asym) is a consistency check (control 4b), it does not size k. No sigma is transported to sym. *Confirmed 2026-11-06 18:30 by the sourced analysis: pi = 0.8 is written as a planning hypothesis with a sensitivity table (section 4); the paired read-out of PR #115 is not kept* |
| TBD-15 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). Sym gate on `eng_sym` only: mean over k seeds >= 1 - epsilon, epsilon = 0.05 (`SYM_EPSILON`); `nocomm_sym` not gated. Case D iff the lower bound of the 95 % CI of D_sym >= delta_sym = 0.10 (`SYM_LEAK_DELTA`). *Confirmed 2026-11-06 18:30 by the sourced analysis: `nocomm_sym` is reported with its CI (descriptive, not a gate); epsilon depends on the evaluation action rule (TBD-11)* |
| TBD-16 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). CI rule, section 5: A iff the lower bound of the 95 % paired CI of D_asym >= margin and the sign test rejects; NH-96 rejected iff the upper bound < margin; otherwise "non concluant". Percentile bootstrap for now; BCa or paired t to be fixed at the freeze. *Superseded 2026-11-06 18:30 by sourced analysis (PI decision: follow the analysis): the CI rule stays the single decision rule, and B also requires the sign test to reject; the CI method is REOPENED (OPEN): paired t or expanded percentile (Hesterberg 2015), to be fixed before the freeze and the Monte-Carlo rerun with it (section 4)* |

## 1. Question

Do the channel wiring and the N10-C PPO trunk, adapted per backend, realise a large
part of the contrast that the task makes available? The information need itself is
analytical, not empirical: the target is uniform among C candidates, so the optimal
`eng - nocomm` contrast is 1 - 1/C = 0.75 on `asym` and 0 on `sym` (`oracle_contrast()`;
K=16, C=4). This is the positive-listening quantity Pl of Eccles et al. 2019 (section
4.1), computable without training. N10-D does **not** test whether the message is
informationally necessary; that holds by construction. It tests the plumbing and the
learner on this task: outcome A reads "the channel plumbing and the trunk learn the
engineered code on this task". NH-96 is labelled VARIANT (expected result: Lowe et al.
2019; Lazaridou et al. 2017 report 99-100 % success on a referential game of this kind).
The symmetric twin is not part of the question (PI-1); it is a leak check.

## 2. Link with the gate

- Gate passes (`eng - nocomm >= 0.20` on N10-C): N10-D is not needed for necessity;
  it stays an optional follow-up. Nothing here is run.
- Gate fails: N10-C-bis is not frozen (G3); N10-D is the next study. A gate failure is
  ambiguous between "the Procgen task does not need the message" and "the N10-C channel
  or trunk does not carry it". N10-D may support the first reading only under a
  condition this protocol does not cover: an N10-D arm executed inside the N10-C harness
  with the same channel code. That arm is an extension, outside this protocol (A10);
  without it, no N10-D outcome is read back onto N10-C. A gate failure is a
  design-decision input, not an N10-D result; gate runs are never pooled with N10-D runs.

## 3. Design

Conditions: `asym` (only the speaker sees the target; the listener sees C candidates
that contain it) and `sym` (both see the target). H1 and the NH-96 null are defined on
`asym` only; `sym` is a leak check: its analytical contrast is 0, it has its own gate
(control 3) and its own read-out (case D, section 5). Arms (code:
`experiments/n10d/spec.py`; environment: section 7):

| arm | channel | role |
|---|---|---|
| `eng` | engineered sender, ideal code, fixed | upper reference, as in the gate |
| `nocomm` | zeroed | blind reference |
| `learned` | trained jointly, alphabet A | emergent-protocol arm (secondary) |
| `msg_shuffle` | evaluation only, on the trained `eng` pair: message of another episode of the same evaluation batch (uniform episode derangement, no label exclusion, `derange_episodes`) | negative control |

`msg_shuffle` permutation (`interfaces.derange_episodes`, applied per evaluation batch by
`Channel.transmit` with a seeded `random.Random(batch_seed)`): a fixed-point-free
permutation of the episodes of the batch, so each listener receives the message emitted
for another episode; it keeps the marginal of messages and their length; identical
target labels across episodes are not excluded (a 1/K coincidence is not a leak). A
fixed permutation of target identities is not allowed: the message would stay a
deterministic function of the target and a listener could learn it up to 1.0.

Metric (primary, A8): S = success of the frozen policy pair after training, evaluated
on n_eval = 500 fresh episodes (confirmed by the PI 2026-11-06), episode seeds disjoint from
the training stream, so S given the seed is Binomial(n_eval, p)/n_eval. The former
metric "mean of the last 5 logged training values" is dropped: its windows are not a
Binomial(n_eval) and n_eval was undefined for it. This departs from gate G2, whose
numbers are not compared with N10-D numbers. D = mean over seeds of per-seed
`S_eng - S_nocomm`, paired by seed. Secondary: `learned - nocomm`, `eng - msg_shuffle`
(the latter is NWML-N1, secondary exploratory, PI 2026-11-06 21:05, section 9).
Blind ceiling (exact, `blind_ceiling()`): asym 1/C, sym 1.0. Oracle contrast: section 1.

Quantiles used by the controls (`experiments/n10d/null.py`, measured sigma_seed, the
registered k and n_eval): P99_asym = P99 of the paired null of D on asym; L99 = P99 of
the level null of one blind arm mean on asym (`simulate_level_null`). Controls 2 and 4
use L99, their own level quantile, never the P99 of D (A9).

Controls (reported next to every result, same pipeline):
1. Positive: a scripted listener that reads the true target through the message must
   reach >= 0.95 on `asym`. If not, the pipeline cannot measure and no D is read.
2. Negative: `msg_shuffle` mean on `asym` must be <= L99 (blind ceiling within its level null).
3. Learning gates (floor effects are "not measurable", never "null effect"):
   - asym: `eng` above `blind ceiling + P99_asym` before D_asym is interpreted.
   - sym (TBD-15): mean of `eng_sym` over k seeds >= 1 - epsilon, epsilon = 0.05
     (`decision.sym_gate`); `nocomm_sym` is not gated, it is reported with its 95 % CI
     (descriptive control of the trunk). If the gate fails, the sym leak check is "not
     measurable". epsilon = 0.05 depends on the evaluation action rule (sampled or
     argmax, TBD-11): argmax can reach higher levels than sampling (Havrylov & Titov
     2017, section 3).
   The gates use the arm means; the training curve is never a gate.
4. Leak audit (A15). (a) A priori, once per backend, before the pilot, committed with
   the environment hash: the asym listener observation holds no field derived from the
   target (no index, no position, candidate order shuffled per episode); the target
   position is uniform over the C slots on 10 000 generated episodes (chi-square,
   alpha 0.01); scripted blind listeners (always slot 0; argmax of each single
   observation feature) stay <= L99 on the same episodes. Result in the manifest.
   EGG reads the target position from its data file, so for EGG the file generator is
   what is audited. A centralised critic that sees the target during training is not
   an execution-time leak.
   (b) At run time: `nocomm` on `asym` above L99 means a side channel; the run set is
   invalid (outcome row 0, section 5). sigma_seed(nocomm/asym) clearly above 0 points
   the same way (consistency check, TBD-14).

## 4. Analysis plan

Confirmatory (after freeze only), seeds 600+, asym: one-sided paired sign test of D,
H0 P(D_i > 0) = 1/2, so the count of positive seeds is Binomial(k, 1/2); ties count as
non-positive; reject if the count is >= m_k, the smallest m with P(Bin(k, 1/2) >= m)
<= 0.05 (`sign_test_critical`). Alongside: paired bootstrap 95 % CI of D_asym (10 000
resamples, percentile). Both are combined by the CI rule of section 5 (TBD-16,
`experiments/n10d/decision.py`).

Planning assumption (TBD-13, TBD-14). This is a planning hypothesis, not a measurement;
no source gives the fraction of learning seeds for a PPO trunk on a Lewis game with
C = 4 (`analysis-4-points-2026-11-06.md`, point 1). Seeds are bimodal: a fraction pi
learns the engineered code (D_i ~ N(0.75, 0.03)) and the rest stays blind
(D_i ~ N(0, 0.027), the binomial sd of a paired difference at n_eval = 500). Under this
mixture sigma_D = 0.75 sqrt(pi (1 - pi)), a function of the effect itself. Planning value
pi = 0.8: Delta = 0.60, sigma_D = 0.30, q = P(D_i > 0) = 0.90. Sensitivity in pi:

| pi | mean D | sigma_D | q = P(D_i > 0) |
|---|---|---|---|
| 0.70 | 0.525 | 0.344 | 0.85 |
| 0.75 | 0.563 | 0.325 | 0.875 |
| 0.80 | 0.600 | 0.300 | 0.90 |

sigma_D is not estimated from the pilot (TBD-14): the pilot runs at a reduced budget, so
it samples another mixture; on 8 bimodal seeds the sd is unstable and degenerate when all
or no seed learns; and reading it reads part of the effect. The sd of the paired
differences is computed only after unblinding of the confirmatory runs, as a sensitivity
analysis. sigma_seed is measured on `nocomm` alone (control 4b).

k (TBD-13): **k = 20 per backend**, a design constant (PI decision ~18:00, following the
sourced analysis). It is checked, not searched, by a simulation of the actual decision
rule (D_asym >= 0.10, CI lower bound >= margin, sign test rejects) under the planning
assumption; global power = p^2 (both backends must conclude, TBD-10; independence of the
backends is assumed, not shown). Script: `experiments/n10d/required_seeds_mc.py`, numpy,
seed 20261106, percentile bootstrap. First run (PR #100, macM1, 2026-11-06,
`--sims 20000 --boot 2000`, scan): per-backend power 0.812, 0.838, 0.878, 0.875, 0.891,
0.919, 0.947, 0.952 at k = 11 to 18 and 0.970, 0.966, 0.977 at k = 19, 20, 21; type I
against the margin at k = 18: 0.023. k = 18 met the global target 0.90 only just
(0.906) and fell to about 0.74 at pi = 0.75 (analysis, point 3). The power is not
monotone in k (the sign-test critical count moves in steps); the two-point approximation
of the analysis puts a local peak at k = 20, the Monte-Carlo does not (0.970, 0.966,
0.977 at k = 19, 20, 21).
Rerun at k = 20 (fact, simulation only, no run data): macM1, 2026-11-06 18:42 CET, branch
head 7af254d, `uv run python -m experiments.n10d.required_seeds_mc --sims 20000 --boot 2000`
(20 000 simulations, 2 000 bootstrap resamples, seed 20261106, 28 s):

| pi | per-backend power | global power p^2 |
|---|---|---|
| 0.70 | 0.774 | 0.599 |
| 0.75 | 0.899 | 0.808 |
| 0.80 | 0.966 | 0.934 |

Type I against the margin (true mean D = 0.30, pi = 0.4) at k = 20: 0.025; P(A) under H0
(pi = 0): 0. At pi = 0.75 the Monte-Carlo gives a global power of 0.808, below the 0.885
of the two-point approximation of the analysis (point 3) and below the 0.90 target: the
margin of k = 20 against a lower pi is smaller than the analysis stated (open item 1 of
section 9).
The CI method is OPEN (TBD-16, two options): the lower bound of a paired t interval, or
the expanded percentile interval (Hesterberg 2015, arXiv:1411.5279, section 5.3 p. 50:
the percentile interval under-covers in small samples, one-sided non-coverage 0.077,
0.048, 0.036 at n = 5, 10, 20 for 0.025 nominal). The method is fixed before the freeze
and the simulation above is rerun with it. The sign test enters the rule only as a
condition of A and B; its count and p-value are otherwise reported as secondary,
descriptive statistics, and it does not size k. The former rules (k = max(MIN_K, k_mean,
k_sign), the pilot sigma_D of PR #115 with q = Phi(0.30 / sigma_D), MIN_K) are not kept.

Pilot: 8 development seeds 9600-9607, asym; `sigma_seed.py` reads `nocomm` only
(TBD-14), never computes D, prints no k, reads no NH-96 score. The pilot seeds are
never pooled with confirmatory runs. Non-rejection is "non concluant", never "refuted".

## 5. Outcomes and consequences (one decision table, A7; case C closed by TBD-6)

D-hat = point estimate of D_asym; L, U = lower and upper bounds of its 95 % paired
bootstrap CI. The rule is TBD-16, the CI rule (`decision.decide_asym`), applied per
backend. Rows are checked in this order; the first that holds is the outcome, so every
result has exactly one.

| order | outcome | rule | reading |
|---|---|---|---|
| 0 | invalid | leak audit (control 4, a or b) fails | side channel; the run set is invalid, no D read |
| 1 | non mesurable | any other asym control (1, 2, 3-asym) fails | no D read, never "supported" |
| 2 | C | D-hat < 0.10 | failed prediction. "The architecture blocks" is a new hypothesis, reserved to a separate amendment (e.g. a channel wired into the policy input) filed before any new run |
| 3 | A | L >= 0.30 and the sign test rejects | NH-96 supported: the channel plumbing and the trunk learn the engineered code on this task; never read as "necessity confirmed" (section 1) |
| 4 | B | U < 0.30 and the sign test rejects | weaker than predicted; NH-96 rejected (`nh96-null.md` section 4): the channel transmits and the trunk learns in part; report D-hat with its CI |
| 5 | non concluant | anything else (the CI contains 0.30, or the sign test does not reject) | the prediction is neither met nor excluded; report D-hat and CI |

B is a rejection of magnitude (U < 0.30), not a non-rejection; the sign test shows
D > 0. It is not read as partial necessity (necessity is fixed by construction, section
1) nor as "the architecture blocks" (that is C). Partial learning is a documented state
(local optima, Eccles et al. 2019 Table 2; one or two symbols suffice in part, Zhang
2024), so with every A, B or C the distribution of the per-seed D_i is reported: D-hat
does not separate "some seeds learn everything" from "each seed learns in part".
Example: D-hat = 0.20 with U = 0.27 and a rejecting sign test is B; the same with a
non-rejecting sign test is "non concluant".

Case D (sym, not an H1 outcome, PI-1): sym gate passed (`eng_sym` mean >= 0.95) and the
lower bound of the 95 % paired CI of D_sym >= 0.10 (TBD-15, `decision.sym_leak_status`).
Label: "possible leak or optimisation effect" (trained policies at finite budget can sit
below 1.0 without any leak: `nocomm_sym` at 0.625 against `eng_sym` at 1.0 gives
D_sym = 0.375, which fires case D); audit required. It is reported next to the asym
outcome and does not change it, but the reading of that outcome is suspended until the
audit is reported. A gap below 0.10 does not suspend A-C. The degenerate sym null
(`nh96-null.md` section 3) is reported descriptively only.

Margin (A12, TBD-12): 0.30 is 0.40 of the oracle contrast 0.75 (success 0.55, against
0.25 blind and 1.0 optimal). It asks the trained pair to recover a clear share of the
available gain while staying more than ten times above every asym null P99 (<= 0.026,
`nh96-null.md` section 3), so the margin, not noise, binds; near-optimal learning is not
required for a plumbing test. It is a PI choice, not derived from data.

## 6. Not done here

No run, no install, no compute. Missing before the pilot: hashed environments for both
backends, the real environments (`backends.py` raises), the PPO-trunk adaptation, the
engineered sender and scripted listener, the a priori leak-audit script (control 4a),
ledger code. Before the freeze: TBD-11 (throughput measurement without score, then the
budget; evaluation action rule), sigma_seed, the CI method of TBD-16 (paired t or
expanded percentile) and the Monte-Carlo rerun with it, freeze manifest.

## 7. Two backends (TBD-1, TBD-2) and the environment (A11)

Environment, identical on both backends (one step per episode):
1. The target t is uniform over K = 16 identities; C - 1 = 3 distractors are drawn
   uniformly without replacement from the other identities; the C candidates are
   shuffled uniformly per episode.
2. Speaker observation: t (one identity). Listener observation: asym, the C candidates
   in shuffled order; sym, the same plus t.
3. Order: the speaker emits one symbol of alphabet A = 16 (A = K: the engineered code
   maps each identity to one symbol); the arm's channel transforms it; the listener
   receives the channel output with its observation; the listener acts.
4. Listener action: the index in {0, ..., C - 1} of one candidate (always a valid one).
5. Reward: common to both agents, 1 if the chosen candidate is t, else 0; episode ends.
TBD-11: observation encoding per backend (one-hot or embedding), message length of the
`learned` arm if above one symbol, evaluation action rule (sampled or argmax). JaxMARL
runs a custom environment implementing this list, not MPE `simple_reference` (3
landmarks, communication dimension 10, `local_ratio` 0.5: a different game; JaxMARL
@31b0a23, `simple_reference.py:25,26,31,37,161`). EGG runs
it as a Lewis signalling game.

Same learner (the N10-C PPO trunk, adapted per backend; same hyperparameters wherever
the interface allows, each difference listed in the manifest), same metric and same
seeds list. JaxMARL runs the environment in JAX; EGG (forked, commit 54eb11c6,
reviewed) in PyTorch; neither is pooled with the other. Each backend gets its own null
calibration and its own sigma_seed. Each (backend, arm, seed) cell stays on one host.

Divergence rule (TBD-10, closed 2026-11-06 17:05, case B amended 18:30 by the sourced
analysis). Each backend gets its own status, the outcome of section 5 (invalid, non
mesurable, C, A, B or non concluant, plus case D), reported separately; the statuses are
the same words as in section 5. Combination (`decision.combine_backends`): identical
verdicts propagate, both A -> global A; both B -> global B, "weaker than predicted on
both backends" (the channel transmits and the trunk learns in part; the magnitude
prediction of NH-96 is rejected, U < 0.30 and the sign test rejects on each backend;
neither partial necessity nor an architecture block); both C -> global C. Any
disagreement, and any pair including invalid, non mesurable or non concluant -> global
"non concluant". A concordant global A or B is a conjunction of per-backend rejections,
so it holds at level alpha (intersection-union structure); two
non-rejections do not establish a common null, and the conjunction is not a Bonferroni
correction. If the backends disagree, the cause is a separate hypothesis for an
amendment, and no result is selected after the fact.

## References

- Lowe, Foerster, Boureau, Pineau, Dauphin 2019, "On the Pitfalls of Measuring Emergent
  Communication", AAMAS 2019, arXiv:1903.05168 (section 6.2).
- Eccles, Bachrach, Lever, Lazaridou, Graepel 2019, "Biases for Emergent Communication
  in Multi-agent Reinforcement Learning", NeurIPS 2019, arXiv:1912.05676 (section 4.1,
  positive listening Pl, computed explicitly from optimal policies).
- Lazaridou, Peysakhovich, Baroni 2017, "Multi-Agent Cooperation and the Emergence of
  (Natural) Language", ICLR 2017, arXiv:1612.07182 (section 4, Table 1).
- Havrylov, Titov 2017, "Emergence of Language with Multi-agent Games: Learning to
  Communicate with Sequences of Symbols", NeurIPS 2017, arXiv:1705.11192 (section 3).
- Hesterberg 2015, "What Teachers Should Know About the Bootstrap", Am. Stat. 69:371-386,
  arXiv:1411.5279 (Table 5, section 5.3).
- Zhang 2024, "A Combinatorial Approach to Neural Emergent Communication", COLING 2025,
  arXiv:2410.18806 (abstract, section 3.1).
Every other source of the sourced analysis, with what was read and what only as an
abstract: `analysis-4-points-references.tsv`.

Both contradictor passes of 2026-11-06 read these three texts in full
(`docs/n10d/contradictor-2026-11-06/`, PR #113). References are cited by arXiv id, not
by bracket number (A20: the arXiv version of Lowe 2019 numbers Lazaridou 2017 as [25]).

## 8. Amendments 2026-11-06 (blind contradictor, verified)

Sources, all in `docs/n10d/contradictor-2026-11-06/` (PR #113, branch at 8ca3c20):
`verification-2026-11-06.md` (source verification without the reading note; "n." below
is its finding number) and `VERIFICATION.md` with `verification-{claude,codex}-report.md`
(verification with the reading note and the brainstorm). 22 findings, 0 refuted.

- A1 (n. 2): learning gate split; asym unchanged, sym gets its own gate (section 3, control 3).
- A2 (n. 1): outcome A reworded "plumbing and PPO learning validated"; analytical
  contrast 0.75 / 0 written as reference (`oracle_contrast`, section 1).
- A3 (n. 10): `sigma_seed.py` CLI reads n_eval from the filtered rows, error if they differ; CLI test.
- A4 (n. 3, 4, 14, 16, 17): k sized for the sign test on a sigma_D bound from the eng
  and nocomm spreads; MIN_K 5 -> 8; table in section 4. *The bound was replaced by the
  paired sd of D (PI decision 2026-11-06 16:55, section 9), itself superseded by the
  second reconciliation (TBD-13, TBD-14).*
- A5 (n. 5, 11): `sym` out of H1 and paired null, both per PI decisions of
  `brainstorm-2026-11-01.md` section 0 (items 1-2); `nh96-null.md` sections 1-2.
- A6 (n. 13): latent blind success capped at B0 in `null.py`; `nh96-null.md` section 2.
- A7 (n. 7): one decision table tying sign test, CI and margin to the outcomes (section 5).
- A8 (n. 6): metric = separate evaluation of the frozen policy, n_eval = 500 proposed;
  last-5-logs metric dropped (section 3, `interfaces.Trainer`).
- A9 (n. 8, 18): `msg_shuffle` evaluation only; controls 2 and 4 on the level quantile L99.
- A10 (n. 19): reading of a gate failure back onto N10-C made conditional on an
  N10-D arm in the N10-C harness, declared outside this protocol (section 2).
- A11 (n. 12, 15): explicit environment replaces "simple_reference-like" (section 7, `backends.py`).
- A12 (n. 20): margin 0.30 stated as 0.40 of the oracle, with a justification (section 5).
- A13 (n. 9): Eccles 2019 and Lazaridou 2017 added (References).
- A14 (`VERIFICATION.md`): `msg_shuffle` deranges episodes, never target identities (`derange_episodes`).
- A15 (`VERIFICATION.md`): control 4 gains an a priori audit of the generator.
- A16 (`VERIFICATION.md`): case D labelled "possible leak or optimisation effect".
- A17 (`VERIFICATION.md`): TBD-10 proposal gives one status per backend.
- A18 (n. 11, `VERIFICATION.md`): "0 by construction" on sym holds for optimal policies only (`nh96-null.md` section 5).
- A19 (n. 10, 22): tests for the degenerate sym null and the CLI n_eval path.
- A20 (`VERIFICATION.md`): Lowe 2019 numbering; no bracket number is used here, nothing else to change.

Not applied: none. Freeze lock (verrou de lecture): not passed; no tag.

### Reconciliation 2026-11-06

Two branches revised the 6e682cc draft in parallel on 2026-11-06: PR #100
(`prereg/n10d-skeleton`, 1ccb78b and e5fbe88, 13:48, its revisions R1-R13, from the 19
points of `VERIFICATION.md`) and PR #115 (this branch, A1-A20 above and section 9). The
PI decided at 17:12 CET to bring the head of #100 into #115 by one merge commit.
**PR #115 is the single branch to merge into #100; PI decision 17:12 CET.**
Resolution rule, in order: (a) the PI decisions of 16:55 (section 9) always win;
(b) where both branches fixed the same point differently, the stricter text for the
freeze lock is kept (declares more, requires more, leaves less implicit) and the other
is noted below in one line; (c) what only #100 adds is kept; (d) no duplication: one
amendments section, one decision table, one sigma_D, one function per role; (e) the tests
of both branches are kept and must pass.

Where #100 won or was kept:
- Section 0: rows PI-1 and PI-2.
- Status "invalid" kept apart from "non mesurable" (row 0 of section 5, per-backend
  statuses); explicit backend combination (both A, both C, otherwise "non concluant";
  two non-rejections give no common null; not a Bonferroni correction). Alternative
  (#115): "global status only if both are the same".
- Case D suspends the reading of the asym outcome until the audit is reported; the
  0.625 counterexample is written in section 5.
- `Channel.transmit(messages, *, batch_seed)` works per batch with a seeded derangement;
  the reason a fixed permutation of target identities is forbidden is written in section 3.
- `Trainer.evaluate` returns integer counts (successes, n_eval). Alternative (#115): a
  float success rate. One method, on `Trainer` (#100's separate `Evaluator` dropped).
- Control 4a: for EGG the data-file generator is audited; a centralised critic that sees
  the target during training is not an execution-time leak.
- JaxMARL `simple_reference` source lines (section 7, `backends.py`); in `nh96-null.md`:
  no H0 on sym and its degenerate null stated, the latent effect is not required by the
  game, a sigma measured on asym is never transported to sym.

Where #115 won:
- n_eval = 500 (PI, 16:55) over #100's TBD-11 working value.
- `msg_shuffle` is evaluation only, on the trained eng pair (A9). Alternative (#100): an
  arm trained and evaluated under the derangement; not kept, left to the PI. Closed by
  the PI decision of 2026-11-06 21:05 CET (section 9): evaluation only.
- The last-5-logs training summary is dropped (A8, confirmed by the PI); #100 kept it as
  a secondary summary.
- Latent blind success capped at B0 (A6); #100 clipped to [0, 1]. The table of
  `nh96-null.md` section 3 is the capped one; #100's table (uncapped) is superseded.
- One level null, `simulate_level_null` (L99 on the level); #100's `simulate_arm_null`
  (same draws minus B0) is removed and its test rewritten on the level null.
- One decision table (section 5, A7) over #100's prose cases A-D.
- Name `derange_episodes` (not `episode_derangement`).

Points of this first reconciliation that the second one (below) replaced: the pilot
sigma_D with declared breach no. 1, MIN_K 8 and the paired-mean floor, eps_sym =
1/n_eval on both sym arms with failure as flag D, the point-plus-sign-test table with
L > 0, TBD-10/12/16 left open, and `sign_test_threshold` raising.

Tests: all tests of both branches were kept at the first reconciliation; #100's tests
were adapted to the merged code (`episode_derangement` -> `derange_episodes`;
`simulate_arm_null` -> the level null minus B0). The two
`test_sym_null_is_degenerate_at_zero_sigma` are one test holding both sets of assertions.

### Second reconciliation (cb7d66b)

After the first reconciliation, PR #100 moved on with three commits: 31bc66e (TBD-10
and TBD-12 to TBD-16 closed by the PI, QCM in session, 17:05), d719c8e (sym gate on
`eng_sym` only, CI decision rule, `decision.py`) and cb7d66b (k by Monte-Carlo of that
rule, `required_seeds_mc.py`, K_CONFIRM = 18). Its owner stopped there. The two branches
then disagreed on four design points, and the PI (~18:00 CET) chose to follow a sourced
analysis (`analysis-4-points-2026-11-06.md`, references `analysis-4-points-references.tsv`)
rather than the resolution rule of the first reconciliation. The head cb7d66b was merged
into this branch by one merge commit.

Taken from #100: the 17:05 closures of section 0; sigma_D from the planning assumption
and a pilot that reads `nocomm` only (no breach; `sigma_seed.py` of #100); the sym gate
on `eng_sym` alone with case D through the CI lower bound of D_sym; the CI rule as the
single decision rule (`decision.py`, `tests/test_n10d_decision.py`); k by simulation of
that rule (`required_seeds_mc.py`, `tests/test_n10d_seeds_mc.py`).

Decided by the analysis (one line each; detail in section 9):
1. sigma_D: #100 kept; pi = 0.8 written as a hypothesis with a sensitivity table; the
   pilot read-out of #115 (breach no. 1, paired sd, k from the pilot) is withdrawn and
   the paired sd becomes a post-unblinding sensitivity analysis.
2. Sym gate: #100 kept, `nocomm_sym` reported with its CI; eps_sym = 1/n_eval on both
   arms and "gate failure = flag D" withdrawn. On the argument "if both arms pass
   >= 1 - epsilon then |D_sym| <= epsilon": the arithmetic is right and aimed at the
   e5fbe88 text, but in #115 flag D was the gate failure itself, so case D was not
   empty there; the double gate is withdrawn because 1/n_eval sits below the binomial
   noise of published levels (0.92-0.99) and depends on the evaluation action rule.
3. k and rule: the CI rule of #100 with k raised from 18 to 20; the point-plus-sign-test
   table of #115 (P(A | D = margin) = 0.17-0.32) and MIN_K are withdrawn; the CI method
   is reopened (TBD-16).
4. Case B: two B propagate as a global B, B requires the sign test, and the internal
   inconsistency of #100 (section 5 "NH-96 rejected" against section 7 "non concluant"
   for the same case) is removed: section 7 now uses the statuses of section 5.

Tests: the tests of #100 are kept; #115's paired-read-out tests (`sigma_d`,
`required_seeds` with MIN_K 8, the "Pilot read-out" CLI tests) go with the code they
tested; its sign-test table is kept on `sign_test_critical`. New: the sign-test rows of
the analysis, B without the sign test is "non concluant", backend combination,
K_CONFIRM = 20 with the pi sensitivity.

## 9. PI decisions 2026-11-06 (16:55, 17:05, ~18:00 and 21:05 CET)

### 16:55 CET

Taken by the PI on the choices listed in the draft PR (#115) after section 8.

| item | decision |
|---|---|
| sigma_D (A4) | CHANGED. sigma_D = sd (ddof = 1) of the paired per-seed differences eng - nocomm on the asym pilot, fed directly to `required_seeds`; the Cauchy-Schwarz bound is dropped. The pairing is a declared blinding breach (breach no. 1). MIN_K 8, sign-test table and MAX_K 200 unchanged. *Superseded 2026-11-06 18:30 by sourced analysis (PI decision: follow the analysis): sigma_D from the planning assumption, pilot reads `nocomm` only, no breach, MIN_K removed (point 1 below)* |
| n_eval (A8) | CONFIRMED without change: n_eval = 500 (still valid), eps_sym = 1/n_eval. *eps_sym = 1/n_eval superseded 2026-11-06 18:30 by sourced analysis (PI decision: follow the analysis): epsilon = 0.05 on `eng_sym` only (point 2 below)* |
| A10 | CONFIRMED without change: reading "Procgen does not need the message" stays conditional on an N10-D arm in the N10-C harness, outside this protocol (section 2). Still valid |
| Q8 (N10-C-bis vs N10-D) | DEFERRED to the session "point de reprise roadmap v2"; nothing decided here |

### 17:05 CET (PR #100)

TBD-10 and TBD-12 to TBD-16 closed, PI decision, QCM in session, 2026-11-06 17:05; the
rows are in section 0. Where the analysis below modifies them (k 18 -> 20 in TBD-13; CI
method reopened in TBD-16; case B in TBD-10) they are marked "superseded 2026-11-06
18:30 by sourced analysis (PI decision: follow the analysis)". TBD-12 (margin 0.30),
TBD-14 and TBD-15 stand, with the additions written in section 0.

### ~18:00 CET

| item | decision |
|---|---|
| arbiter | 2026-11-06 ~18:00: follow the sourced analysis `docs/prereg/n10d/analysis-4-points-2026-11-06.md` ("selon l'analyse scientifique et des papiers"); PR #115 then merges into #100 (PI decisions 17:12 and ~18:00) |
| 1. sigma_D (TBD-14) | « σ_D (TBD-14) : hypothèse de planification bimodale π = 0,8 (Δ = 0,60, σ_D = 0,30), écrite comme hypothèse avec table de sensibilité π ∈ {0,70 ; 0,75 ; 0,80} ; pilote nocomm seul, aucune brèche ; la lecture appariée de PR #115 n'est pas retenue (pilote à budget réduit, σ̂ instable sur 8 graines bimodales, variance fonction de l'effet). » Applied in section 4 |
| 2. sym gate (TBD-15) | « Porte sym (TBD-15) : `eng_sym` ≥ 0,95 seul ; cas D si la borne basse de l'IC 95 % de D_sym ≥ 0,10 ; `nocomm_sym` rapporté avec son IC (contrôle descriptif du tronc) ; ε = 1/n_eval sur les deux bras non retenu (seuil inférieur au bruit binomial, dépend de la règle d'évaluation TBD-11). » Applied in sections 3 and 5 |
| 3. k and rule (TBD-13, TBD-16) | « Règle unique (TBD-16) : règle IC, A ssi L ≥ 0,30 et test des signes ; k (TBD-13) = 20 par backend par simulation de cette règle (global ≈ 0,93-0,96 à π = 0,8 ; ≈ 0,885 à π = 0,75), méthode d'IC (t apparié ou percentile élargi) fixée au gel et MC rejoué avec elle ; MIN_K = 8 et la table point + signes de PR #115 non retenus (P(A \| D = marge) = 0,17-0,32). » Applied in sections 4 and 5; Monte-Carlo at k = 20 in section 4 |
| 4. combination (TBD-10) | « Combinaison (TBD-10) : deux A → A ; deux B → B global « plus faible que prédit sur les deux backends » (intersection-union, niveau α), B par backend exigeant U < 0,30, D̂ ≥ 0,10 et le rejet du test des signes ; deux C → C ; autre couple → « non concluant » ; la zone U < 0,30 n'est pas classée C ; §5 et §7 alignés. » Applied in sections 5 and 7 |

### 2026-11-06 21:05 CET (primary outcome, Q3)

PI decision, QCM 21:05-21:10 CET, question Q3 option (a) of the hypotheses refresh of
2026-11-06 (hypneum-lab/.github PR #30, report section 7).

| item | decision |
|---|---|
| H1 (primary) | KEPT AS REGISTERED: D = per-seed paired `S_eng - S_nocomm` on `asym` (section 3), read by the rule of section 5. H1 tests the channel plumbing and the PPO learning trunk on this task; the information need holds by construction and NH-96 is labelled VARIANT with an expected result, and the protocol already states that N10-D does not test whether the message is informationally necessary (section 1). No parameter, threshold, k, margin or rule changes |
| NWML-N1 (causal use of the channel) | DECLARED SECONDARY EXPLORATORY. Question: does the trained listener use the channel causally? Test: episode derangement of the channel, `msg_shuffle`, evaluation only, on the trained `eng` pair as defined in section 3 (contrast `eng - msg_shuffle`, already listed as secondary there); applying it to the `learned` arm is not decided here. It is not a primary outcome, spends no alpha, and no verdict propagates from it: it changes no outcome A-D of section 5 and is reported as exploratory next to them. Control 2 (section 3) stays a control with its own L99 gate, unchanged |
| identifier | `NWML-N1` is provisional; the definitive NH- number is assigned in catalogue v3 (hypneum-lab/.github PR #30) |
| not covered | No new arm (the trained-under-derangement alternative of section 8 stays out); no code, test or numerical parameter changes; no freeze, no tag |

### Open for the PI (not decidable from the literature)

1. pi, the fraction of seeds in which the PPO trunk learns the engineered code on this
   game: no source gives it for PPO at C = 4; 0.8 and the accepted sensitivity are a PI
   choice. The Monte-Carlo at k = 20 gives a global power of 0.934 at pi = 0.8 but 0.808
   at pi = 0.75 (the analysis approximated 0.885) and 0.599 at pi = 0.70 (section 4).
2. The value of the absolute margin 0.30 (TBD-12): a PI choice, no literature fixes it.
3. The independence of the two backends, assumed and not shown; if false, the global
   power changes in a direction the literature does not predict.
4. The cost per seed (TBD-11): no figure yet; the step from k = 18 to 20 (24 more runs)
   can only be weighed after the throughput measurement without score.
5. The evaluation action rule (argmax or sampling, TBD-11): it sets the reachable level
   of `eng_sym` and `nocomm_sym`, hence the relevance of epsilon = 0.05.
6. The CI method (paired t, expanded percentile; BCa not read): the literature read
   leans against the percentile at small n; choice and rerun of the simulation open.
7. Whether a declared blinding breach is acceptable as a lab policy (hypneum-prereg): a
   rule of the lab, not a statistical question.

No freeze, no tag; the freeze lock (verrou de lecture) is not passed.
