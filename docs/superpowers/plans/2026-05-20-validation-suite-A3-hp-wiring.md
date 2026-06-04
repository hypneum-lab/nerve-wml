# Validation Suite A.3 — Hyperparameter Wiring Implementation Plan

For agentic workers: execute under `superpowers:subagent-driven-development`.
Plain `git commit` only — never `--amend`, never `--no-verify`. Commit
subjects ≤ 50 chars.

**Goal.** Plan A.2.5 (equal-tuning protocol) honestly disclosed that
three hyperparameters (learned `lr`, vec2vec `lambda_cycle`, relrep
`n_anchors`) reach the runner signature but are NOT actually threaded
into the underlying training calls. The "tuning" therefore varies only
seed + steps. This plan wires the three hyperparameters all the way
through so each grid point really exercises a distinct configuration,
making the equal-tuning result defensible rather than purely a seed
study.

**Architecture.** Three small surgical edits to existing runners. No
new modules, no new classes. Each task touches one runner file plus a
tiny test that asserts the HP now changes the output deterministically.

**Tech Stack.** Python 3.12+, `uv`, pytest, ruff, mypy. No new
dependencies. Reuse `scripts.multi_seed.run_multi_seed` for the assert.

---

## Task A3.1 — Thread `lr` through `_train_learned`

**Files**
- Modify: `scripts/transducer_baselines_pilot.py`
- Modify: `tests/integration/test_transducer_baselines_pilot.py`

**Steps**

- [ ] Read `scripts/transducer_baselines_pilot.py:_train_learned`. It
  currently has signature `_train_learned(src_codes, dst_codes, steps)`
  and hard-codes `Adam(t.parameters(), lr=0.05)`.

- [ ] Add a failing slow test asserting that two different `lr` values
  produce two different MI scores (in `tests/integration/test_transducer_baselines_pilot.py`):

  ```python
  @pytest.mark.slow
  def test_learned_lr_changes_mi() -> None:
      from scripts.transducer_baselines_pilot import (
          _build_task, _mi_entropy_bits, _train_learned,
      )
      src, dst, *_ = _build_task(0)
      learned_a = _train_learned(src, dst, 200, lr=1e-2)
      learned_b = _train_learned(src, dst, 200, lr=5e-2)
      mi_a = _mi_entropy_bits(learned_a.forward(src, hard=True), dst)["mi_bits"]
      mi_b = _mi_entropy_bits(learned_b.forward(src, hard=True), dst)["mi_bits"]
      assert mi_a != mi_b
  ```

- [ ] Run, expect FAIL: `uv run pytest tests/integration/test_transducer_baselines_pilot.py::test_learned_lr_changes_mi -x -m slow`.
  (FAIL because `_train_learned` does not accept `lr`.)

- [ ] Modify `_train_learned` signature to
  `_train_learned(src_codes, dst_codes, steps, *, lr: float = 0.05)`
  and replace the hard-coded `lr=0.05` in the `Adam` call with the
  parameter. Default preserves backward compatibility with existing
  callers.

- [ ] Re-run the test, expect PASS.

- [ ] Run full slow suite to confirm no regression in the other
  transducer integration tests (they call `_train_learned` without
  `lr`, so the default keeps them green):
  `uv run pytest tests/integration/test_transducer_baselines_pilot.py -m slow`.

- [ ] Lint: `uv run ruff check scripts/transducer_baselines_pilot.py`.

- [ ] Commit:

  ```bash
  git add scripts/transducer_baselines_pilot.py tests/integration/test_transducer_baselines_pilot.py
  git commit -m "feat: thread lr through learned transducer"
  ```

  (43 chars.)

---

## Task A3.2 — Thread `lambda_cycle` & `n_anchors` through `run_transducer_benchmark`

**Files**
- Modify: `scripts/transducer_baselines_pilot.py`
- Modify: `tests/integration/test_transducer_baselines_pilot.py`

`run_transducer_benchmark` currently builds the two baselines with
hard-coded `lambda_cycle=10.0` (Vec2Vec) and `n_anchors=32`
(RelativeRep). Expose both as kwargs.

**Steps**

- [ ] Add the failing slow test:

  ```python
  @pytest.mark.slow
  def test_benchmark_lambda_cycle_changes_vec2vec() -> None:
      from scripts.transducer_baselines_pilot import run_transducer_benchmark
      a = run_transducer_benchmark(steps=200, seed=0, lambda_cycle=1.0)
      b = run_transducer_benchmark(steps=200, seed=0, lambda_cycle=100.0)
      assert a["vec2vec"]["mi_bits"] != b["vec2vec"]["mi_bits"]


  @pytest.mark.slow
  def test_benchmark_n_anchors_changes_relrep() -> None:
      from scripts.transducer_baselines_pilot import run_transducer_benchmark
      a = run_transducer_benchmark(steps=200, seed=0, n_anchors=8)
      b = run_transducer_benchmark(steps=200, seed=0, n_anchors=64)
      assert a["relative_rep"]["mi_bits"] != b["relative_rep"]["mi_bits"]
  ```

- [ ] Run, expect FAIL (TypeError or matching values): `uv run pytest tests/integration/test_transducer_baselines_pilot.py -k "lambda_cycle_changes or n_anchors_changes" -x -m slow`.

- [ ] Modify `run_transducer_benchmark` signature:

  ```python
  def run_transducer_benchmark(
      *, steps: int = 2000, seed: int = 0,
      lambda_cycle: float = 10.0, n_anchors: int = 32,
  ) -> dict[str, dict[str, float]]:
      ...
  ```

  Replace `Vec2VecTransducer(..., lambda_cycle=10.0, ...)` with
  `lambda_cycle=lambda_cycle`. Replace `RelativeRepTransducer(...,
  n_anchors=32, ...)` with `n_anchors=n_anchors`.

- [ ] Re-run the two new tests, expect PASS. Run the rest of the
  transducer integration tests, expect no regression.

- [ ] Lint: `uv run ruff check scripts/transducer_baselines_pilot.py`.

- [ ] Commit:

  ```bash
  git add scripts/transducer_baselines_pilot.py tests/integration/test_transducer_baselines_pilot.py
  git commit -m "feat: thread vec2vec + relrep hp through bench"
  ```

  (47 chars.)

---

## Task A3.3 — Rewire equal-tuning to use the new HP knobs

**Files**
- Modify: `scripts/equal_tuning_pilot.py`
- Modify: `docs/superpowers/research/equal-tuning-protocol.md`
- Modify: `tests/integration/test_equal_tuning_pilot.py`

The equal-tuning pilot now imports the real HPs. Remove the
"accepted-but-ignored" disclaimers and actually feed the grid through.

**Steps**

- [ ] Add a failing assertion to
  `tests/integration/test_equal_tuning_pilot.py`. The existing test
  only checks shape — extend it to assert that trials within a method
  see at least 2 distinct MI means (i.e. the HP actually moved
  something):

  ```python
  @pytest.mark.slow
  def test_equal_tuning_hp_actually_varies() -> None:
      from scripts.equal_tuning_pilot import run_equal_tuning
      out = run_equal_tuning(budget=3, seeds_per_trial=(0, 1))
      for method in ("learned", "vec2vec", "relrep"):
          means = sorted({round(t["mi_mean"], 6) for t in out[method]["trials"]})
          assert len(means) >= 2, f"{method} HP did not vary outputs"
  ```

- [ ] Run, expect FAIL on at least one method.

- [ ] Edit `scripts/equal_tuning_pilot.py`:
  - In `_learned_runner`, remove `del lr`; pass `lr=lr` to `_train_learned`:
    ```python
    learned = _train_learned(src_codes, dst_codes, steps, lr=lr)
    ```
  - In `_vec2vec_runner`, remove `del lambda_cycle`; pass through:
    ```python
    res = run_transducer_benchmark(
        steps=steps, seed=seed, lambda_cycle=lambda_cycle,
    )
    ```
  - In `_relrep_runner`, remove `del n_anchors`; pass through:
    ```python
    res = run_transducer_benchmark(
        steps=500, seed=seed, n_anchors=n_anchors,
    )
    ```

- [ ] Re-run the new test, expect PASS. Run the existing
  `test_equal_tuning_shape`, confirm PASS.

- [ ] Update
  `docs/superpowers/research/equal-tuning-protocol.md`: replace the
  "Limitations of the current pilot" section content with:

  ```markdown
  ## Limitations (as of Plan A.3, 2026-05-20)

  All three tunable axes (`lr`, `lambda_cycle`, `n_anchors`) are now
  threaded through the underlying training calls. Trial variation
  therefore reflects real hyperparameter effect plus seed noise.

  Open knobs still hard-coded inside the runner (left as future work):
  - Vec2Vec internal generator/discriminator widths (`hidden`).
  - Learned transducer Gumbel softmax `tau`.
  - GTM PSK/PAM modulation choice.
  ```

  Keep the rest of the doc intact.

- [ ] Lint: `uv run ruff check scripts/equal_tuning_pilot.py`.

- [ ] Run fast suite: `uv run pytest -m "not slow"` (expect green, no
  regression).

- [ ] Commit:

  ```bash
  git add scripts/equal_tuning_pilot.py \
          docs/superpowers/research/equal-tuning-protocol.md \
          tests/integration/test_equal_tuning_pilot.py
  git commit -m "feat: rewire equal-tuning to real hp grids"
  ```

  (44 chars.)

---

## Self-Review

- 3 tasks, all independent edits to existing files. Each ends in a
  single conventional commit ≤ 50 chars.
- No placeholders. Every code edit is shown verbatim.
- Reuses existing infrastructure (`run_multi_seed`, `bootstrap_ci`,
  the three baseline classes) — no new modules.
- Tests added are integration `@pytest.mark.slow`. Existing fast and
  slow tests must remain green.
- After landing, re-run `scripts/extended_eval.py` and
  `scripts/equal_tuning_pilot.py` to refresh the empirical reports
  with real-HP numbers (separate operational step, not part of this
  plan).

## Execution Handoff

Execute with `superpowers:subagent-driven-development` — one subagent
per task, fresh context, review checkpoint between tasks.
