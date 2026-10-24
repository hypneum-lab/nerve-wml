# Erratum: L1a gate did not pass as specified (seed 0, n=1024)

Date: 2026-10-24. Scope: L1a Full-LIF sim training (PR #48, released in
v1.9.0).

## Facts

- Spec `docs/superpowers/specs/2026-05-30-lif-l1a-sim-training-design.md`
  section 5.2 defines three gate conditions: (1) final accuracy >= 20 %,
  (2) loss at step 2000 < loss at step 100, (3) untrained accuracy
  <= 10 % on the same eval batch.
- `scripts/lif_l1a_train.py` computed `gate_passed` from conditions (1)
  and (2) only. Condition (3) was never evaluated by the script.
- The recorded run `docs/superpowers/research/2026-05-30-lif-l1a-gate.json`
  (n=1024, seed 0) has `acc_untrained` = 0.130859375 (13.1 %), above the
  10 % threshold. Evaluated against all three conditions, the gate did
  NOT pass for n=1024 at seed 0. The recorded `gate_passed: true` reflects
  only conditions (1) and (2).
- The learned gain itself is large: final accuracy 89.1 % versus 13.1 %
  untrained (chance for 12 classes: 8.3 %).
- The `@slow` test `test_gate_untrained_lte_10pct` asserts condition (3)
  and fails at seed 0. CI runs `pytest -m "not slow"`, so it did not
  surface this.

## All sizes in the PR #48 sweep (seed 0)

| n_neurons | acc_untrained | acc_final | cond. 1 | cond. 2 | cond. 3 (<= 10 %) |
|-----------|---------------|-----------|---------|---------|-------------------|
| 128       | 8.4 %         | 85.5 %    | yes     | yes     | yes               |
| 256       | 7.8 %         | 85.7 %    | yes     | yes     | yes               |
| 512       | 5.7 %         | 88.7 %    | yes     | yes     | yes               |
| 1024      | 13.1 %        | 89.1 %    | yes     | yes     | **no**            |

Only n=1024, the primary size and the one recorded in the gate JSON,
exceeds 10 % untrained. Sizes 128, 256 and 512 satisfy all three
conditions on this single seed. Condition 2 values are taken from
`2026-05-30-lif-l1a-scaling.json`.

## Places that claimed the gate passed

- `CHANGELOG.md` (v1.9.0 section, "L1a Full-LIF sim training + scaling
  sweep (#48)"): "closes the L1a gate".
- `.zenodo.json` (v1.9.0): title "Full-LIF L1a" and description item (5).
- PR #48 body: "all 4 sizes pass gate" and the sizes table (PASS column,
  including n=1024 with 13.1 % untrained).
- `docs/superpowers/research/2026-05-30-lif-l1a-gate.json` and
  `...-scaling.json`: `gate_passed: true` (left unmodified as historical
  records).

## Handling

- The archived Zenodo v1.9.0 record cannot be edited; it will be
  corrected in the description of the next release.
- Historical JSON files and the v1.9.0 release are not modified.
- `scripts/lif_l1a_train.py` now requires all three conditions and writes
  each condition's boolean separately. The training was not rerun.
- `test_gate_untrained_lte_10pct` is unchanged (still `@slow`, <= 10 %).
