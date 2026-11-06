# N10-D - asymmetric communication-necessity study (L0 skeleton)

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. Open "TBD by PI" items (TBD-10, TBD-11) block the freeze (hypneum-prereg).
**Amended:** 2026-11-06 after a blind contradictor, every finding checked in source
(section 8). Nothing was frozen, so the text is edited in place; the trail is section 8.
**Prepared:** 2026-10-31 (night), by decision of the PI (multiple-choice, 2026-10-31 22:11 CET).
**Origin:** `docs/milestones/2026-10-25-n10c-bis-control-arms.md`, section "Necessity gate",
item G3: on failure of the gate (`eng - nocomm >= 0.20`) N10-C-bis is not frozen and the
next study is N10-D, "an asymmetric cooperative task in which each agent holds private
information that its partner needs", pre-registered in its own document before any run.
NH-96 (catalogue of new hypotheses, `.github`, `2026-10-30-etat-des-lieux/synthese/priorites-hypotheses.md`)
is the hypothesis behind it. No equivalent branch, PR or file existed at 2026-10-31.
**Annex G7 (PR #99) is not touched.**

## 0. PI decisions 2026-10-31 (multiple-choice, 22:30-22:35 CET)

| id | decision (closed unless marked TBD) |
|---|---|
| TBD-1 | CLOSED. Both backends, JaxMARL and EGG. EGG is forked into the `ailiance` org at commit 54eb11c6 with code review. Articulation and divergence rule: section 7 |
| TBD-2 | CLOSED. The N10-C PPO trunk, adapted to each backend (section 7) |
| TBD-3 | CLOSED. K=16, C=4, A=16; blind ceiling 1/C = 0.25 (asym) stays exact |
| TBD-4 | CLOSED. Seeds: pilot 9600+, confirmatory 600+. k derived from sigma_seed by the formula of section 4 (margin 0.30, alpha 0.05, power 0.9, minimum 5). *Amended 2026-11-06 (A4): k sized for the sign test, on a sigma_D bound from the eng and nocomm spreads, minimum 8* |
| TBD-5 | CLOSED. D_sym must not exceed the P99 of its calibrated null, with sigma_seed measured on development seeds (no fixed 0.05). *Superseded 2026-11-06 (A5): `sym` leaves H1 (PI decision, `brainstorm-2026-11-01.md` section 0, item 1); sym is a descriptive control with its own gate (section 3)* |
| TBD-6 | CLOSED. D_asym < 0.10 = failed prediction; "the architecture blocks" is a new hypothesis for a separate amendment (section 5) |
| TBD-7 | CLOSED. Host: KXKM-AI after N10-C-bis; if possible a second DGX GB300 slot, declared as alternative host. Each arm and each seed stays on ONE host (no mixing inside an arm-seed cell) |
| TBD-8 | CLOSED. Grouped HITL review at lock time; flax one version back under the 2-week rule (0.12.10). Hashes still to be produced |
| TBD-9 | CLOSED. Pilot: 8 development seeds 9600-9607, reduced budget. NH-96 scores are not read; only sigma_seed is extracted, by `experiments/n10d/sigma_seed.py`, committed before the pilot. *Amended 2026-11-06 (A4): the script also reads the eng/asym spread; it never computes a mean or a difference* |
| TBD-10 | OPEN. Backend-divergence rule (section 7): proposal "status per backend, global conclusion only if both agree" (A17) |
| TBD-11 | OPEN. Reduced pilot step budget; exact trunk adaptation per backend; environment details marked TBD-11 in section 7. n_eval proposed 500 (A8, to validate) |

## 1. Question

Do the channel wiring and the N10-C PPO trunk, adapted per backend, realise a large
part of the contrast that the task makes available? The information need itself is
analytical, not empirical: the target is uniform among C candidates, so the optimal
`eng - nocomm` contrast is 1 - 1/C = 0.75 on `asym` and 0 on `sym` (`oracle_contrast()`;
K=16, C=4). This is the positive-listening quantity Pl of Eccles et al. 2019 (section
4.1), computable without training. N10-D does **not** test whether the message is
informationally necessary; that holds by construction. It tests the plumbing and the
learner on this task: outcome A reads "channel plumbing and PPO learning validated on
this task". NH-96 is labelled VARIANT (expected result: Lowe et al. 2019; Lazaridou et
al. 2017 report 99-100 % success on a referential game of this kind).

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
`asym` only; `sym` is a descriptive control arm: its analytical contrast is 0, D_sym is
reported without a decision, and it has its own plumbing gate (control 3). Arms (code:
`experiments/n10d/spec.py`; environment: section 7):

| arm | channel | role |
|---|---|---|
| `eng` | engineered sender, ideal code, fixed | upper reference, as in the gate |
| `nocomm` | zeroed | blind reference |
| `learned` | trained jointly, alphabet A | emergent-protocol arm (secondary) |
| `msg_shuffle` | evaluation only, on the trained `eng` pair: message of another episode of the same evaluation batch (uniform episode derangement, no label exclusion, `derange_episodes`) | negative control |

Metric (primary, A8): S = success of the frozen policy pair after training, evaluated
on n_eval = 500 fresh episodes (value to validate, TBD-11), episode seeds disjoint from
the training stream, so S given the seed is Binomial(n_eval, p)/n_eval. The former
metric "mean of the last 5 logged training values" is dropped: its windows are not a
Binomial(n_eval) and n_eval was undefined for it. This departs from gate G2, whose
numbers are not compared with N10-D numbers. D = mean over seeds of per-seed
`S_eng - S_nocomm`, paired by seed. Secondary: `learned - nocomm`, `eng - msg_shuffle`.
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
   - sym (plumbing control, not a hypothesis): `eng_sym` and `nocomm_sym` means both
     >= 1 - eps_sym, eps_sym = max(P99 of the sym null at sigma_seed = 0, 1/n_eval) =
     1/n_eval, because that null is degenerate (every draw is exactly 1.0, P99 = 0):
     "1.0 to within one evaluation episode". Failure gives flag D (section 5).
4. Leak audit (A15). (a) A priori, once per backend, before the pilot, committed with
   the environment hash: the asym listener observation holds no field derived from the
   target (no index, no position, candidate order shuffled per episode); the target
   position is uniform over the C slots on 10 000 generated episodes (chi-square,
   alpha 0.01); scripted blind listeners (always slot 0; argmax of each single
   observation feature) stay <= L99 on the same episodes. Result in the manifest.
   (b) At run time: `nocomm` on `asym` above L99 means a side channel; the run set is invalid.

## 4. Analysis plan

Confirmatory (after freeze only), seeds 600+, asym: one-sided paired sign test of D,
H0 P(D_i > 0) = 1/2, so the count of positive seeds is Binomial(k, 1/2); ties count as
non-positive; reject if the count is >= m_k, the smallest m with P(Bin(k, 1/2) >= m)
<= 0.05. Alongside: paired bootstrap 95 % CI of D_asym (10 000 resamples, percentile).

k is fixed in the manifest by a formula written in advance (`required_seeds`,
`experiments/n10d/sigma_seed.py`, A4): sigma_D = sd(S_eng) + sd(S_nocomm), total
sample sd over the asym pilot seeds (binomial term included). This bounds
sd(S_eng - S_nocomm) for any covariance (Cauchy-Schwarz), so no pairing is read and no
factor 2 is applied. Under the alternative D_i ~ N(0.30, sigma_D), q = P(D_i > 0) =
Phi(0.30 / sigma_D); k = the smallest k in [8, 200] with power P(Bin(k, q) >= m_k) >= 0.9;
above 200 the design needs an amendment. Power is not monotone in k (discrete m_k):

| k | m_k | size P(Bin(k, 1/2) >= m_k) | power q = 0.8 | power q = 0.9 |
|---|---|---|---|---|
| 5 (former minimum) | 5 | 0.031 | 0.328 | 0.590 |
| 8 | 7 | 0.035 | 0.503 | 0.813 |
| 10 | 9 | 0.011 | 0.376 | 0.736 |
| 12 | 10 | 0.019 | 0.558 | 0.889 |

Resulting k: 8 for sigma_D <= 0.20, 13 at 0.25, 16 at 0.30 (tests in
`tests/test_n10d_sigma.py`). Pilot: 8 development seeds 9600-9607, asym, arms `nocomm`
and `eng`; the script returns n_eval, sigma_seed (nocomm, binomial-corrected, for the
null), sigma_D and k, and computes no mean, no difference and no NH-96 score. The pilot
seeds are never pooled with confirmatory runs. Non-rejection is "non concluant", never "refuted".

## 5. Outcomes and consequences (one decision table, A7; case C closed by TBD-6)

D-hat = point estimate of D_asym; U = upper bound of its 95 % bootstrap CI. Rows are
checked in this order; the first that holds is the outcome, so every result has one.

| order | outcome | rule | reading |
|---|---|---|---|
| 1 | non mesurable | any asym control (1, 2, 3-asym, 4) fails | no D read, never "supported" |
| 2 | C | U < 0.30 or D-hat < 0.10 | failed prediction, NH-96 rejected. "The architecture blocks" is a new hypothesis, reserved to a separate amendment (e.g. a channel wired into the policy input) filed before any new run |
| 3 | A | sign test rejects and D-hat >= 0.30 | NH-96 supported: channel plumbing and PPO learning validated on this task |
| 4 | B | anything else | non concluant: the prediction is neither met nor excluded; report D-hat and CI |

Example: D-hat = 0.25 with U = 0.40 is not C (U >= 0.30, D-hat >= 0.10) and not A
(D-hat < 0.30), so it is B. D-hat >= 0.30 with a non-significant sign test is B.

Flag D (sym, A16): the sym gate of control 3 fails. Label: "possible leak or
optimisation effect". It is reported next to the asym outcome, does not change it, and
D_sym gets no test (its null is degenerate, `nh96-null.md` section 3).

Margin (A12): 0.30 is 0.40 of the oracle contrast 0.75 (success 0.55, against 0.25 blind
and 1.0 optimal). It asks the trained pair to recover a clear share of the available
gain while staying more than ten times above every asym null P99 (<= 0.026,
`nh96-null.md` section 3), so the margin, not noise, binds; near-optimal learning is not
required for a plumbing test. It is a PI choice (TBD-4), not derived from data.

## 6. Not done here

No run, no install, no compute. Missing before the pilot: hashed environments for both
backends, the real environments (`backends.py` raises), the PPO-trunk adaptation, the
engineered sender and scripted listener, the a priori leak-audit script (control 4a),
ledger code. Before the freeze: TBD-10/11, sigma_seed, sigma_D, k, freeze manifest.

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
landmarks, communication dimension 10, `local_ratio` 0.5: a different game). EGG runs
it as a Lewis signalling game.

Same learner (the N10-C PPO trunk, adapted per backend; same hyperparameters wherever
the interface allows, each difference listed in the manifest), same metric and same
seeds list. JaxMARL runs the environment in JAX; EGG (forked, commit 54eb11c6,
reviewed) in PyTorch; neither is pooled with the other. Each backend gets its own null
calibration, its own sigma_seed and sigma_D. Each (backend, arm, seed) cell stays on one
host. Divergence rule (TBD-10, proposal, A17): each backend gets its own status (non
mesurable, A, B or C, plus flag D), reported separately; a global conclusion is drawn
only if both statuses are the same. Otherwise the global status is "non concluant", the
cause is a separate hypothesis for an amendment, and no result is selected after the fact.

## References

- Lowe, Foerster, Boureau, Pineau, Dauphin 2019, "On the Pitfalls of Measuring Emergent
  Communication", AAMAS 2019, arXiv:1903.05168 (section 6.2).
- Eccles, Bachrach, Lever, Lazaridou, Graepel 2019, "Biases for Emergent Communication
  in Multi-agent Reinforcement Learning", NeurIPS 2019, arXiv:1912.05676 (section 4.1,
  positive listening Pl, computed explicitly from optimal policies).
- Lazaridou, Peysakhovich, Baroni 2017, "Multi-Agent Cooperation and the Emergence of
  (Natural) Language", ICLR 2017, arXiv:1612.07182 (section 4, Table 1).

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
  and nocomm spreads; MIN_K 5 -> 8; table in section 4.
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
- A16 (`VERIFICATION.md`): flag D labelled "possible leak or optimisation effect".
- A17 (`VERIFICATION.md`): TBD-10 proposal gives one status per backend.
- A18 (n. 11, `VERIFICATION.md`): "0 by construction" on sym holds for optimal policies only (`nh96-null.md` section 5).
- A19 (n. 10, 22): tests for the degenerate sym null and the CLI n_eval path.
- A20 (`VERIFICATION.md`): Lowe 2019 numbering; no bracket number is used here, nothing else to change.

Not applied: none. Freeze lock (verrou de lecture): not passed; no tag.
