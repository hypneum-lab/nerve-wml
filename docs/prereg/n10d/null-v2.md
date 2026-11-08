# N10-D v2 — null specifications and TOST (L0 draft)

**Status:** L0 DRAFT, not frozen. Companion to `protocol.md` v2 (2026-11-08).
Provides pseudocode for the two registered nulls and the TOST outcome criteria so
that `null.py`, `decision.py` and the conflict/placebo generator can be adapted in a
separate code PR. No Python code is modified here.

## 1. Null 1 — conflict/placebo per-episode exchange (statistic ρ, V1)

### 1.1 Purpose

Tests H0: "the listener's action on each episode does not depend on which message
perturbation (conflict or placebo) is applied". Under H0, swapping the conflict and
placebo labels per episode (independently) leaves the action distribution invariant.

### 1.2 Inputs

```
per_seed[i] for i in 1..k:
    conflict_hits[i]   # integer: #{episodes e where action_e = t'_e under conflict}
    placebo_hits[i]    # integer: #{episodes e where action_e = t''_e under placebo}
    # Both evaluated on the same 500 episodes; actions are per-episode scalars.
    # Also needed for the full exchange (not just counts):
    action_conflict[i]  # list of 500 actions under conflict
    action_placebo[i]   # list of 500 actions under placebo
    t_prime[i]          # list of 500 conflict targets t'_e (designated before run)
    t_double_prime[i]   # list of 500 placebo targets t''_e (designated before run)
N_ep = 500   # n_eval
B    = 10_000
rng  = seeded RNG (seed declared in manifest, e.g. 20261108)
```

### 1.3 Pseudocode

```python
observed_rho = mean over i of (conflict_hits[i] / N_ep - placebo_hits[i] / N_ep)

null_draws = []
for b in range(B):
    rho_null_b_seeds = []
    for i in range(k):
        conflict_null = 0
        placebo_null = 0
        for e in range(N_ep):
            if rng.random() < 0.5:
                # swap: episode e now contributes to conflict using placebo action
                # and to placebo using conflict action
                conflict_null += int(action_placebo[i][e] == t_prime[i][e])
                placebo_null  += int(action_conflict[i][e] == t_double_prime[i][e])
            else:
                # no swap
                conflict_null += int(action_conflict[i][e] == t_prime[i][e])
                placebo_null  += int(action_placebo[i][e] == t_double_prime[i][e])
        rho_null_b_seeds.append(conflict_null / N_ep - placebo_null / N_ep)
    null_draws.append(mean(rho_null_b_seeds))

p_value = (sum(d >= observed_rho for d in null_draws) + 1) / (B + 1)
```

### 1.4 Output

Report: observed ρ, null median, null P95, null P99, empirical p.

### 1.5 Note on the exchange

t′_e and t″_e are fixed before the evaluation run (pre-designated in the manifest).
They do not change under the null exchange. What changes is which action (conflict or
placebo) is counted against which target. This is valid because, under H0, the action
does not depend on the perturbation, only on the listener's own observation of the
episode.

## 2. Null 2 — arm-label permutation (statistic Δρ, V2)

### 2.1 Purpose

Tests H0: "the per-seed ρ values are exchangeable between `eng_sym` and
`learned_sym`". The 25 eng_sym seeds and 25 learned_sym seeds share the same 25
seed indices; this permutation acts within each index.

### 2.2 Inputs

```
rho_eng[j]      for j in 1..k   # per-seed rho for eng_sym
rho_learned[j]  for j in 1..k   # per-seed rho for learned_sym
# k = 25, one value per seed index per backend
B   = 10_000
rng = seeded RNG (seed declared in manifest)
```

### 2.3 Pseudocode

```python
observed_delta = mean(rho_eng) - mean(rho_learned)

null_draws = []
for b in range(B):
    eng_perm = list(rho_eng)
    lea_perm = list(rho_learned)
    for j in range(k):
        if rng.random() < 0.5:
            eng_perm[j], lea_perm[j] = lea_perm[j], eng_perm[j]
    null_draws.append(mean(eng_perm) - mean(lea_perm))

# Two-sided (V2 has no a priori direction)
p_value = (sum(abs(d) >= abs(observed_delta) for d in null_draws) + 1) / (B + 1)
```

### 2.4 Output

Report: observed Δρ, null median, null P95, null P99, two-sided empirical p.

## 3. TOST specification (δ_eq and δ_use)

### 3.1 Outcomes (from protocol.md section 7)

Given the CI [L, U] of the mean per-seed ρ (99 % for V1, 98 % for V2; level follows
α — wider than the nominal 95 % because the CI classifies outcomes, not the rejection
test, whose threshold is the permutation p-value):

```
if L >= delta_use:
    outcome = "message dominant"
elif U < delta_eq:
    outcome = "message ignored"   # TOST: upper bound below equivalence margin
else:
    outcome = "non concluant"
```

Requirement: δ_eq < δ_use, with a gap large enough so that both outcomes are
achievable for some ρ value. If δ_eq ≥ δ_use the two outcomes become mutually
inaccessible and the design is defective.

### 3.2 Setting δ_eq and δ_use (after pilot, before freeze)

Inputs from the pilot (8 `eng_sym` seeds, conflict/placebo evaluation):
- `sd_between` = sample sd of ρ_i across the 8 pilot seeds
- `se_mean`    = sd_between / sqrt(k)  with k = 25 (confirmatory seeds)
- `binom_noise` = sqrt(2 * 0.25 / N_ep)  with N_ep = 500  ≈ 0.032

Constraint from the protocol: `binom_noise` ≤ 0.022 per proportion term
(one-tailed sd of Bernoulli(0.5) at n = 500); the total per-seed sd of ρ_i is
at most sqrt(2) * 0.022 ≈ 0.032. This provides a hard upper bound on binomial noise
independent of the pilot.

Procedure:
1. Set δ_eq such that [0, δ_eq] represents "negligible use", at least 3–4× the pilot
   noise floor. Example (working, not binding): δ_eq = 0.05 if sd_between < 0.03.
2. Set δ_use such that [δ_use, 1] represents "clear use", at a detectable magnitude.
   Example: δ_use = 0.20. Both values must be declared and justified in the manifest
   before the freeze tag is pushed. No post-hoc adjustment.

## 4. Code follow-up (not in this PR)

The following code changes are needed in a separate PR, to be opened after this
protocol PR is merged. No Python file is modified here.

### 4.1 `experiments/n10d/spec.py`

- Add evaluation mode type: `EvalMode = Literal["intact", "msg_shuffle", "conflict",
  "field_t_deranged", "placebo"]`.
- Add (Arm, EvalMode) pairing annotation or dataclass for the conflict/placebo
  generator inputs.

### 4.2 `experiments/n10d/decision.py`

- Add outcome type: `Outcome = Literal["message_dominant", "message_ignored",
  "non_concluant", "non_mesurable"]`.
- Add `decide_v1(rho_seeds, delta_use, delta_eq, ci_method, ci_level=0.99) -> Outcome`
  using the CI and TOST rules of section 3 above (ci_level=0.99 for α₁=0.005).
- Add `decide_v2(rho_eng, rho_learned, delta_use, delta_eq, ci_method, ci_level=0.98)
  -> Outcome` using Δρ and the same TOST structure (ci_level=0.98 for α₂=0.01).
  Never hardcode 0.95; pass ci_level explicitly.
- Deprecate or archive v1 `decide_asym`, `combine_backends` (keep them for the
  NH-96 PC3 control).

### 4.3 `experiments/n10d/null.py`

- Add `simulate_conflict_placebo_null(action_conflict, action_placebo, t_prime,
  t_double_prime, k_seeds, B, rng_seed) -> list[float]` implementing section 1.3.
- Add `simulate_arm_permutation_null(rho_eng, rho_learned, B, rng_seed) -> list[float]`
  implementing section 2.3.
- Keep existing `simulate_null` and `simulate_level_null` for NH-96 PC3 (no α).

### 4.4 Conflict/placebo generator

New module (name TBD, analogous to A21 twin generator): given the evaluation episode
list (seed, C candidates, t), generates for each episode:
- `t_prime_e`: drawn uniformly from the C − 1 candidates ≠ t (conflict target).
- `t_double_prime_e`: drawn using a fixed seed declared in the manifest (placebo target).
Stores both in the manifest so the null exchange (section 1) can be reproduced.
Input validation: assert t′_e ≠ t_e and t″_e ≠ t_e for every episode.

### 4.5 `experiments/n10d/sigma_seed.py`

No change required for v2: `sigma_seed` and `sign_test_critical` are still used for
the pilot `nocomm_sym` (G1 calibration) and PC3 (`eng_asym` drop check).
