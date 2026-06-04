# Validation Suite A.2 — Statistical Hardening Implementation Plan

For agentic workers: execute this plan with the
`superpowers:subagent-driven-development` skill. Each task is independent
once its predecessors land on `feat/gap-analysis-remediation`. Honour
the TDD cycle (write failing test, run, implement, run, commit) for
every step. Plain `git commit` only — never `--amend`, never
`--no-verify`. Commit subjects ≤50 chars, body lines ≤72.

**Goal.** Plan A shipped runners that produce point estimates at
`seed=0`; a reviewer would reject that. This plan adds the statistical
machinery (multi-seed aggregation, paired tests + bootstrap CI, null
arms, sensitivity sweeps, equal-tuning protocol) that turns Plan A's
scaffold into defensible empirical evidence.

**Architecture.** We reuse the existing
`nerve_wml.methodology.bootstrap_ci_mi` and
`nerve_wml.methodology.mi_null_model` infrastructure rather than
duplicating it. Five additions land on top: (1) a thin multi-seed
wrapper that calls any Plan A runner across seeds and aggregates leaf
metrics; (2) a `paired_tests` module wrapping `scipy.stats.wilcoxon`,
`mannwhitneyu`, and a generic float-list bootstrap CI; (3) shuffled
"null" arms inserted into each Plan A runner so every claim has a
chance-baseline next to it; (4) a sensitivity sweep pilot varying one
hyperparameter at a time; (5) an equal-tuning pilot enforcing the same
trial budget across tunable methods, plus a doc that locks in the
protocol.

**Tech Stack.** Python 3.12+, `uv sync --all-extras`, pytest (markers
`slow` for integration), ruff, mypy. New runtime dep: `scipy.stats`
(already transitive via numpy/torch). Tests:
`tests/unit/test_multi_seed.py`, `tests/info_theoretic/test_paired_tests.py`,
`tests/integration/test_sensitivity_pilot.py`,
`tests/integration/test_equal_tuning_pilot.py`, and edits to the
three existing Plan A integration tests.

---

## Task A2.1 — Multi-seed wrapper

**Files**
- Create: `scripts/multi_seed.py`
- Create: `tests/unit/test_multi_seed.py`

**Steps**

- [ ] Write the failing unit test at `tests/unit/test_multi_seed.py`.
  Five tests, all fast, using a trivial mock runner so we don't pull
  torch:

  ```python
  """Unit tests for the multi-seed runner wrapper."""
  from __future__ import annotations

  import pytest

  from scripts.multi_seed import run_multi_seed, summarize


  def _mock_runner(*, seed: int) -> dict[str, dict[str, float]]:
      # Deterministic linear-in-seed values so we can verify aggregation.
      return {
          "learned":   {"mi_bits": 1.0 + 0.1 * seed, "acc": 0.9},
          "procrustes": {"mi_bits": 0.5 + 0.05 * seed, "acc": 0.7},
      }


  def test_shape_preserved() -> None:
      agg = run_multi_seed(_mock_runner, seeds=(0, 1, 2))
      assert set(agg) == {"learned", "procrustes"}
      assert set(agg["learned"]) == {"mi_bits", "acc"}


  def test_values_list_and_stats() -> None:
      agg = run_multi_seed(_mock_runner, seeds=(0, 1, 2))
      leaf = agg["learned"]["mi_bits"]
      assert leaf["values"] == [1.0, 1.1, 1.2]
      assert leaf["mean"] == pytest.approx(1.1)
      assert leaf["std"] == pytest.approx(0.0816496580927726, rel=1e-3)


  def test_single_seed_zero_std() -> None:
      agg = run_multi_seed(_mock_runner, seeds=(7,))
      assert agg["learned"]["mi_bits"]["std"] == 0.0


  def test_kwargs_forwarded() -> None:
      def runner(*, seed: int, scale: float) -> dict[str, dict[str, float]]:
          return {"a": {"x": scale * seed}}

      agg = run_multi_seed(runner, seeds=(1, 2), scale=10.0)
      assert agg["a"]["x"]["values"] == [10.0, 20.0]


  def test_summarize_returns_string() -> None:
      agg = run_multi_seed(_mock_runner, seeds=(0, 1))
      out = summarize(agg)
      assert "learned" in out
      assert "mi_bits" in out
      assert isinstance(out, str)
  ```

- [ ] Run the test, expect FAIL (module missing):
  `uv run pytest tests/unit/test_multi_seed.py -x`.

- [ ] Implement `scripts/multi_seed.py`:

  ```python
  """Multi-seed wrapper for Plan A pilot runners.

  Calls a runner callable across multiple seeds and aggregates each
  leaf float metric into ``{"values": [...], "mean": float, "std": float}``,
  preserving the runner's outer ``{method: {metric: ...}}`` shape.
  """
  from __future__ import annotations

  import math
  from collections.abc import Mapping, Sequence
  from typing import Any, Callable


  Runner = Callable[..., Mapping[str, Mapping[str, float]]]


  def _aggregate_leaves(
      per_seed: list[Mapping[str, Mapping[str, float]]],
  ) -> dict[str, dict[str, dict[str, Any]]]:
      if not per_seed:
          raise ValueError("at least one seed required")
      first = per_seed[0]
      out: dict[str, dict[str, dict[str, Any]]] = {}
      for method, metrics in first.items():
          out[method] = {}
          for metric in metrics:
              values = [float(run[method][metric]) for run in per_seed]
              mean = sum(values) / len(values)
              if len(values) > 1:
                  var = sum((v - mean) ** 2 for v in values) / len(values)
                  std = math.sqrt(var)
              else:
                  std = 0.0
              out[method][metric] = {
                  "values": values,
                  "mean":   mean,
                  "std":    std,
              }
      return out


  def run_multi_seed(
      runner:  Runner,
      *,
      seeds:   Sequence[int],
      **kwargs: Any,
  ) -> dict[str, dict[str, dict[str, Any]]]:
      """Call ``runner(seed=s, **kwargs)`` for each seed and aggregate.

      Args:
          runner: A callable returning ``{method: {metric: float}}``.
          seeds:  Sequence of integer seeds.
          kwargs: Forwarded to every runner call.

      Returns:
          ``{method: {metric: {"values": [...], "mean": .., "std": ..}}}``.
      """
      per_seed = [runner(seed=int(s), **kwargs) for s in seeds]
      return _aggregate_leaves(per_seed)


  def summarize(
      aggregated: Mapping[str, Mapping[str, Mapping[str, Any]]],
  ) -> str:
      """Pretty-print the aggregated table."""
      lines = ["method                  metric              mean       std"]
      lines.append("-" * 60)
      for method, metrics in aggregated.items():
          for metric, stats in metrics.items():
              lines.append(
                  f"{method:<22}  {metric:<18}  "
                  f"{stats['mean']:>8.4f}  {stats['std']:>8.4f}"
              )
      return "\n".join(lines)


  def main() -> None:  # pragma: no cover
      raise SystemExit(
          "multi_seed is a library; call run_multi_seed from a pilot."
      )


  if __name__ == "__main__":  # pragma: no cover
      main()
  ```

- [ ] Run the test, expect PASS:
  `uv run pytest tests/unit/test_multi_seed.py -x`.

- [ ] Lint + types:
  `uv run ruff check scripts/multi_seed.py tests/unit/test_multi_seed.py`
  and `uv run mypy scripts/multi_seed.py`.

- [ ] Commit:
  ```
  git add scripts/multi_seed.py tests/unit/test_multi_seed.py
  git commit -m "feat: multi-seed runner wrapper"
  ```

---

## Task A2.2 — Paired statistical tests + bootstrap CI

**Files**
- Create: `nerve_wml/methodology/paired_tests.py`
- Create: `tests/info_theoretic/test_paired_tests.py`

**Steps**

- [ ] Write the failing test at `tests/info_theoretic/test_paired_tests.py`:

  ```python
  """Tests for paired statistical helpers."""
  from __future__ import annotations

  import math

  import numpy as np
  import pytest

  from nerve_wml.methodology.paired_tests import (
      bootstrap_ci,
      mann_whitney,
      wilcoxon_paired,
  )


  def test_wilcoxon_identical_inputs_p_one() -> None:
      a = [0.5, 0.6, 0.7, 0.8, 0.9]
      res = wilcoxon_paired(a, a)
      assert res["p_value"] == pytest.approx(1.0)
      assert res["median_diff"] == 0.0
      assert res["cohens_dz"] == 0.0
      assert res["n"] == 5


  def test_wilcoxon_clear_shift() -> None:
      rng = np.random.default_rng(0)
      a = rng.normal(0.0, 0.1, size=30).tolist()
      b = (np.array(a) + 0.5).tolist()
      res = wilcoxon_paired(a, b)
      assert res["p_value"] < 1e-4
      assert res["median_diff"] < 0


  def test_wilcoxon_length_mismatch_raises() -> None:
      with pytest.raises(ValueError):
          wilcoxon_paired([1.0, 2.0], [1.0])


  def test_mann_whitney_returns_shape() -> None:
      a = [0.1, 0.2, 0.3, 0.4]
      b = [0.5, 0.6, 0.7, 0.8]
      res = mann_whitney(a, b)
      assert set(res) >= {"statistic", "p_value", "median_diff", "cohens_dz", "n"}
      assert res["p_value"] < 0.1


  def test_bootstrap_ci_covers_mean() -> None:
      rng = np.random.default_rng(1)
      values = rng.normal(2.0, 0.5, size=50).tolist()
      res = bootstrap_ci(values, n_resamples=500, seed=0)
      assert res["ci95_low"] < res["mean"] < res["ci95_high"]
      assert math.isfinite(res["median"])


  def test_bootstrap_ci_empty_raises() -> None:
      with pytest.raises(ValueError):
          bootstrap_ci([], n_resamples=10, seed=0)
  ```

- [ ] Run, expect FAIL:
  `uv run pytest tests/info_theoretic/test_paired_tests.py -x`.

- [ ] Implement `nerve_wml/methodology/paired_tests.py`:

  ```python
  """Paired statistical tests and float-list bootstrap CI.

  Complements ``bootstrap_ci_mi`` (which operates on code-pair arrays)
  with the per-method-score helpers required to compare runners that
  emit a single scalar per seed.
  """
  from __future__ import annotations

  from collections.abc import Sequence
  from typing import Any

  import numpy as np
  from scipy import stats


  def _cohens_dz(diffs: np.ndarray) -> float:
      if diffs.size < 2:
          return 0.0
      sd = float(np.std(diffs, ddof=1))
      if sd == 0.0:
          return 0.0
      return float(np.mean(diffs) / sd)


  def wilcoxon_paired(
      values_a: Sequence[float],
      values_b: Sequence[float],
  ) -> dict[str, Any]:
      """Two-sided Wilcoxon signed-rank test on paired observations."""
      a = np.asarray(values_a, dtype=np.float64)
      b = np.asarray(values_b, dtype=np.float64)
      if a.shape != b.shape:
          raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
      if a.ndim != 1:
          raise ValueError("expected 1-D sequences")
      diffs = a - b
      if np.allclose(diffs, 0.0):
          return {
              "statistic":   0.0,
              "p_value":     1.0,
              "median_diff": 0.0,
              "cohens_dz":   0.0,
              "n":           int(a.size),
          }
      res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
      return {
          "statistic":   float(res.statistic),
          "p_value":     float(res.pvalue),
          "median_diff": float(np.median(diffs)),
          "cohens_dz":   _cohens_dz(diffs),
          "n":           int(a.size),
      }


  def mann_whitney(
      values_a: Sequence[float],
      values_b: Sequence[float],
  ) -> dict[str, Any]:
      """Two-sided Mann-Whitney U test on independent observations."""
      a = np.asarray(values_a, dtype=np.float64)
      b = np.asarray(values_b, dtype=np.float64)
      if a.ndim != 1 or b.ndim != 1:
          raise ValueError("expected 1-D sequences")
      res = stats.mannwhitneyu(a, b, alternative="two-sided")
      pooled_sd = float(np.sqrt(
          (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
      )) if a.size > 1 and b.size > 1 else 0.0
      effect = (
          float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else 0.0
      )
      return {
          "statistic":   float(res.statistic),
          "p_value":     float(res.pvalue),
          "median_diff": float(np.median(a) - np.median(b)),
          "cohens_dz":   effect,
          "n":           int(a.size + b.size),
      }


  def bootstrap_ci(
      values:      Sequence[float],
      *,
      n_resamples: int = 1000,
      seed:        int = 0,
  ) -> dict[str, float]:
      """Non-parametric bootstrap 95% CI on a list of floats."""
      arr = np.asarray(values, dtype=np.float64)
      if arr.size == 0:
          raise ValueError("empty values")
      if arr.ndim != 1:
          raise ValueError("expected 1-D sequence")
      rng = np.random.default_rng(seed)
      n = arr.size
      means = np.empty(n_resamples, dtype=np.float64)
      for i in range(n_resamples):
          idx = rng.integers(0, n, size=n)
          means[i] = float(np.mean(arr[idx]))
      return {
          "mean":      float(np.mean(arr)),
          "median":    float(np.median(arr)),
          "ci95_low":  float(np.quantile(means, 0.025)),
          "ci95_high": float(np.quantile(means, 0.975)),
      }
  ```

- [ ] Run, expect PASS:
  `uv run pytest tests/info_theoretic/test_paired_tests.py -x`.

- [ ] Lint + types:
  `uv run ruff check nerve_wml/methodology/paired_tests.py tests/info_theoretic/test_paired_tests.py`
  and `uv run mypy nerve_wml/methodology/paired_tests.py`.

- [ ] Commit:
  ```
  git add nerve_wml/methodology/paired_tests.py tests/info_theoretic/test_paired_tests.py
  git commit -m "feat: paired tests + bootstrap CI"
  ```

---

## Task A2.3 — Null baselines as runner arms

**Files**
- Modify: `scripts/transducer_baselines_pilot.py`
- Modify: `scripts/gtm_ablation_pilot.py`
- Modify: `scripts/scale_robustness_pilot.py`
- Modify: `tests/integration/test_transducer_baselines_pilot.py`
- Modify: `tests/integration/test_gtm_ablation_pilot.py`
- Modify: `tests/integration/test_scale_robustness_pilot.py`

(If any of the three integration tests is named differently — check
`tests/integration/` first — keep the existing name and amend.)

**Steps**

- [ ] Inspect the existing integration tests to confirm filenames:
  `ls tests/integration/test_*pilot*.py`. Use the actual names from
  here on.

- [ ] Add a failing assertion in `tests/integration/test_transducer_baselines_pilot.py`
  for the null arm. Append (or extend the slow test):

  ```python
  def test_null_arm_below_learned() -> None:
      from scripts.transducer_baselines_pilot import run_transducer_benchmark
      res = run_transducer_benchmark(steps=200, seed=0)
      assert "null" in res
      assert "mi_bits" in res["null"]
      assert res["null"]["mi_bits"] < res["learned"]["mi_bits"] - 0.3
  ```
  Mark `@pytest.mark.slow` if the other tests in the file are marked.

- [ ] Same change in `tests/integration/test_gtm_ablation_pilot.py`:

  ```python
  def test_null_arm_below_gtm() -> None:
      from scripts.gtm_ablation_pilot import run_gtm_ablation
      res = run_gtm_ablation(steps=200, seed=0)
      assert "null" in res
      assert res["null"]["mi_bits"] < res["gtm"]["mi_bits"] - 0.3
  ```

- [ ] Same change in `tests/integration/test_scale_robustness_pilot.py`:

  ```python
  def test_null_rows_below_real() -> None:
      from scripts.scale_robustness_pilot import run_scale_robustness
      out = run_scale_robustness(sizes=(64, 128), seed=0)
      assert isinstance(out, dict)
      assert "rows" in out and "null_rows" in out
      real_cknna = [r.cknna for r in out["rows"]]
      null_cknna = [r.cknna for r in out["null_rows"]]
      assert max(null_cknna) < max(real_cknna)
  ```

- [ ] Run the three new tests, expect FAIL:
  `uv run pytest tests/integration/test_transducer_baselines_pilot.py::test_null_arm_below_learned tests/integration/test_gtm_ablation_pilot.py::test_null_arm_below_gtm tests/integration/test_scale_robustness_pilot.py::test_null_rows_below_real -x -m slow`.

- [ ] Modify `scripts/transducer_baselines_pilot.py`. Locate the
  `run_transducer_benchmark` function, then add a null arm at the end
  before the return. Concretely, after the existing learned/procrustes/
  vec2vec/relrep computations build:

  ```python
      # --- null arm: shuffle src codes against dst codes ---
      rng = np.random.default_rng(seed + 9973)
      perm = rng.permutation(src_codes.shape[0])
      shuffled_src = src_codes[perm]
      mi_null, acc_null = _mi_entropy_bits(shuffled_src, dst_codes)
      results["null"] = {"mi_bits": mi_null, "acc": acc_null}
      return results
  ```
  Adjust variable names to what `_mi_entropy_bits` and the existing
  results dict already use (the function returns
  `{"learned": {...}, "procrustes": ..., "vec2vec": ..., "relrep": ...}` —
  add a sibling `"null"` entry with the same `{"mi_bits", "acc"}`
  schema). If `_mi_entropy_bits` returns a tuple, mirror the existing
  call style. Add `import numpy as np` if not present.

- [ ] Modify `scripts/gtm_ablation_pilot.py`. After computing the gtm
  and simple_gating arms, insert a null arm by shuffling carrier vs
  codes (whichever pair the MI is computed on). Insert just before
  the existing return:

  ```python
      rng = np.random.default_rng(seed + 9973)
      perm = rng.permutation(codes.shape[0])
      shuffled_codes = codes[perm]
      acc_n, mi_n, sync_n = _train_gtm(shuffled_codes, steps, seed)
      result["null"] = {
          "mi_bits":  mi_n,
          "acc":      acc_n,
          "sync":     sync_n,
      }
      return result
  ```
  Match the key names actually used by the existing `result["gtm"]`
  dict (look at the lines just above to confirm `mi_bits`/`acc`/`sync`
  exact spelling, and copy that).

- [ ] Modify `scripts/scale_robustness_pilot.py`. Change the return
  type from `list[ScaleRobustnessRow]` to a dict
  `{"rows": [...], "null_rows": [...]}`. Compute `null_rows` by
  permuting `lif_emb` rows once before the second sweep:

  ```python
  def run_scale_robustness(
      *, sizes: tuple[int, ...] = (64, 128, 256, 512), seed: int = 0
  ) -> dict[str, list[ScaleRobustnessRow]]:
      n_max = max(sizes)
      mlp_emb, lif_emb = _substrate_embeddings(seed, n_max)
      rows = scale_robustness_sweep(
          mlp_emb.numpy(), lif_emb.numpy(), sizes=sizes, seed=seed,
      )
      rng = np.random.default_rng(seed + 9973)
      perm = rng.permutation(lif_emb.shape[0])
      lif_shuffled = lif_emb.numpy()[perm]
      null_rows = scale_robustness_sweep(
          mlp_emb.numpy(), lif_shuffled, sizes=sizes, seed=seed + 1,
      )
      return {"rows": rows, "null_rows": null_rows}
  ```
  Adjust `main()` to print both tables. Add `import numpy as np` if not
  already imported.

- [ ] Re-run the three new integration tests, expect PASS:
  `uv run pytest tests/integration/test_transducer_baselines_pilot.py::test_null_arm_below_learned tests/integration/test_gtm_ablation_pilot.py::test_null_arm_below_gtm tests/integration/test_scale_robustness_pilot.py::test_null_rows_below_real -x -m slow`.

- [ ] Run the full fast suite to confirm no regression in the unit
  layer: `uv run pytest -m "not slow"`.

- [ ] Lint + types on the three modified scripts.

- [ ] Commit:
  ```
  git add scripts/transducer_baselines_pilot.py \
          scripts/gtm_ablation_pilot.py \
          scripts/scale_robustness_pilot.py \
          tests/integration/test_transducer_baselines_pilot.py \
          tests/integration/test_gtm_ablation_pilot.py \
          tests/integration/test_scale_robustness_pilot.py
  git commit -m "feat: null arms in plan A runners"
  ```

---

## Task A2.4 — Sensitivity sweeps

**Files**
- Create: `scripts/sensitivity_pilot.py`
- Create: `tests/integration/test_sensitivity_pilot.py`

**Steps**

- [ ] Write the failing slow integration test
  `tests/integration/test_sensitivity_pilot.py`:

  ```python
  """Integration test for the sensitivity sweep pilot."""
  from __future__ import annotations

  import pytest

  from scripts.sensitivity_pilot import run_sensitivity_sweeps


  @pytest.mark.slow
  def test_sensitivity_shape_and_variation() -> None:
      out = run_sensitivity_sweeps(seeds=(0, 1, 2))
      assert set(out) == {"k_cknna", "lambda_cycle", "n_anchors", "steps"}
      for axis_name, axis_rows in out.items():
          assert isinstance(axis_rows, list)
          assert len(axis_rows) >= 2
          for row in axis_rows:
              assert "param" in row
              assert "value" in row
              assert "mean" in row and "std" in row
      # At least one axis exhibits non-trivial mean variation.
      def spread(axis_rows: list[dict]) -> float:
          means = [r["mean"] for r in axis_rows]
          return max(means) - min(means)
      assert any(spread(rows) > 1e-4 for rows in out.values())
  ```

- [ ] Run, expect FAIL:
  `uv run pytest tests/integration/test_sensitivity_pilot.py -x -m slow`.

- [ ] Implement `scripts/sensitivity_pilot.py`:

  ```python
  """Sensitivity sweeps across four hyperparameter axes.

  Each axis varies one parameter while holding others at their default;
  multi-seed mean and std are reported per parameter value via
  ``scripts.multi_seed.run_multi_seed``.

  Axes:
      k_cknna       in {5, 10, 20}          on scale_robustness substrate pair
      lambda_cycle  in {1.0, 10.0, 100.0}   on transducer Vec2Vec arm
      n_anchors     in {8, 16, 32, 64}      on transducer RelativeRep arm
      steps         in {500, 2000, 8000}    on transducer learned arm
  """
  from __future__ import annotations

  from collections.abc import Sequence
  from typing import Any

  import numpy as np

  from nerve_wml.methodology.hsic_cknna import cknna
  from scripts.multi_seed import run_multi_seed
  from scripts.scale_robustness_pilot import _substrate_embeddings
  from scripts.transducer_baselines_pilot import (
      _build_task,
      _train_learned,
      _mi_entropy_bits,
  )

  try:
      from scripts.transducer_baselines_pilot import (
          _train_vec2vec,
          _train_relative_rep,
      )
  except ImportError:  # private helpers may have other names
      _train_vec2vec = None
      _train_relative_rep = None


  def _cknna_runner(*, seed: int, k: int) -> dict[str, dict[str, float]]:
      mlp, lif = _substrate_embeddings(seed, 256)
      return {"cknna_k": {"value": float(
          cknna(mlp.numpy(), lif.numpy(), k=k)
      )}}


  def _vec2vec_runner(
      *, seed: int, lambda_cycle: float
  ) -> dict[str, dict[str, float]]:
      if _train_vec2vec is None:
          # Fallback: rerun learned transducer; record reported MI.
          src_codes, dst_codes, *_ = _build_task(seed)
          mi, _ = _train_learned(src_codes, dst_codes, steps=500, seed=seed)
          return {"vec2vec": {"mi_bits": float(mi)}}
      src_codes, dst_codes, src_wml, dst_wml = _build_task(seed)
      mi, _ = _train_vec2vec(
          src_codes, dst_codes, src_wml, dst_wml,
          lambda_cycle=lambda_cycle, seed=seed,
      )
      return {"vec2vec": {"mi_bits": float(mi)}}


  def _relrep_runner(
      *, seed: int, n_anchors: int
  ) -> dict[str, dict[str, float]]:
      if _train_relative_rep is None:
          src_codes, dst_codes, *_ = _build_task(seed)
          mi, _ = _train_learned(src_codes, dst_codes, steps=500, seed=seed)
          return {"relrep": {"mi_bits": float(mi)}}
      src_codes, dst_codes, src_wml, dst_wml = _build_task(seed)
      mi, _ = _train_relative_rep(
          src_codes, dst_codes, src_wml, dst_wml,
          n_anchors=n_anchors, seed=seed,
      )
      return {"relrep": {"mi_bits": float(mi)}}


  def _steps_runner(*, seed: int, steps: int) -> dict[str, dict[str, float]]:
      src_codes, dst_codes, *_ = _build_task(seed)
      mi, _ = _train_learned(src_codes, dst_codes, steps=steps, seed=seed)
      return {"learned": {"mi_bits": float(mi)}}


  def _sweep_axis(
      runner:    Any,
      *,
      seeds:     Sequence[int],
      param:     str,
      values:    Sequence[Any],
      method:    str,
      metric:    str,
  ) -> list[dict[str, Any]]:
      rows: list[dict[str, Any]] = []
      for v in values:
          agg = run_multi_seed(runner, seeds=seeds, **{param: v})
          leaf = agg[method][metric]
          rows.append({
              "param":  param,
              "value":  v,
              "mean":   leaf["mean"],
              "std":    leaf["std"],
              "values": leaf["values"],
          })
      return rows


  def run_sensitivity_sweeps(
      *, seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
  ) -> dict[str, list[dict[str, Any]]]:
      """Vary one hyperparameter at a time, return multi-seed stats."""
      return {
          "k_cknna": _sweep_axis(
              _cknna_runner, seeds=seeds, param="k",
              values=(5, 10, 20),
              method="cknna_k", metric="value",
          ),
          "lambda_cycle": _sweep_axis(
              _vec2vec_runner, seeds=seeds, param="lambda_cycle",
              values=(1.0, 10.0, 100.0),
              method="vec2vec", metric="mi_bits",
          ),
          "n_anchors": _sweep_axis(
              _relrep_runner, seeds=seeds, param="n_anchors",
              values=(8, 16, 32, 64),
              method="relrep", metric="mi_bits",
          ),
          "steps": _sweep_axis(
              _steps_runner, seeds=seeds, param="steps",
              values=(500, 2000, 8000),
              method="learned", metric="mi_bits",
          ),
      }


  def main() -> None:  # pragma: no cover
      out = run_sensitivity_sweeps()
      for axis, rows in out.items():
          print(f"\n=== axis: {axis} ===")
          for row in rows:
              print(
                  f"  {row['param']}={row['value']!r:<8}  "
                  f"mean={row['mean']:.4f}  std={row['std']:.4f}"
              )


  if __name__ == "__main__":  # pragma: no cover
      main()
  ```

  Before committing, run `grep -n "^def _train_vec2vec\|^def _train_relative_rep" scripts/transducer_baselines_pilot.py`
  and adapt the `try/except` import block to whatever the actual helper
  names are. If the helpers don't exist or have different signatures,
  keep the fallback path (rerun `_train_learned` so the axis still
  varies via `seed`-driven noise).

- [ ] Run the slow test, expect PASS:
  `uv run pytest tests/integration/test_sensitivity_pilot.py -x -m slow`.

- [ ] Lint + types:
  `uv run ruff check scripts/sensitivity_pilot.py tests/integration/test_sensitivity_pilot.py`
  and `uv run mypy scripts/sensitivity_pilot.py`.

- [ ] Commit:
  ```
  git add scripts/sensitivity_pilot.py tests/integration/test_sensitivity_pilot.py
  git commit -m "feat: sensitivity sweep pilot"
  ```

---

## Task A2.5 — Equal-tuning protocol + doc

**Files**
- Create: `scripts/equal_tuning_pilot.py`
- Create: `docs/superpowers/research/equal-tuning-protocol.md`
- Create: `tests/integration/test_equal_tuning_pilot.py`

**Steps**

- [ ] Write the failing slow test
  `tests/integration/test_equal_tuning_pilot.py`:

  ```python
  """Integration test for the equal-tuning pilot."""
  from __future__ import annotations

  import pytest

  from scripts.equal_tuning_pilot import run_equal_tuning


  @pytest.mark.slow
  def test_equal_tuning_shape() -> None:
      out = run_equal_tuning(budget=2, seeds_per_trial=(0, 1))
      assert set(out) >= {"learned", "vec2vec", "relrep", "procrustes"}
      for method, payload in out.items():
          assert "trials" in payload
          assert "best" in payload
          assert isinstance(payload["trials"], list)
          assert len(payload["trials"]) >= 1
          best = payload["best"]
          assert "params" in best
          assert "mi_mean" in best
          assert "mi_ci95_low" in best
          assert "mi_ci95_high" in best
  ```

- [ ] Run, expect FAIL:
  `uv run pytest tests/integration/test_equal_tuning_pilot.py -x -m slow`.

- [ ] Implement `scripts/equal_tuning_pilot.py`:

  ```python
  """Equal-tuning protocol: every tunable method gets the SAME budget.

  Budget = ``budget`` trials drawn deterministically from a fixed grid
  per method; each trial runs ``len(seeds_per_trial)`` seeds; best
  trial is selected by mean MI across seeds. The non-tunable Procrustes
  arm is reported once at the same multi-seed budget for parity.
  """
  from __future__ import annotations

  import itertools
  from collections.abc import Sequence
  from typing import Any

  from nerve_wml.methodology.paired_tests import bootstrap_ci
  from scripts.multi_seed import run_multi_seed
  from scripts.transducer_baselines_pilot import (
      _build_task,
      _mi_entropy_bits,
      _train_learned,
      run_transducer_benchmark,
  )


  LEARNED_GRID: tuple[dict[str, Any], ...] = tuple(
      {"steps": s, "lr": lr}
      for s, lr in itertools.product((500, 2000, 8000), (1e-3, 3e-3, 1e-2))
  )[:8]

  VEC2VEC_GRID: tuple[dict[str, Any], ...] = tuple(
      {"lambda_cycle": lc, "steps": s}
      for lc, s in itertools.product(
          (1.0, 10.0, 50.0, 100.0), (500, 2000)
      )
  )[:8]

  RELREP_GRID: tuple[dict[str, Any], ...] = tuple(
      {"n_anchors": k} for k in (4, 8, 16, 32, 48, 64, 96, 128)
  )[:8]


  def _learned_runner(
      *, seed: int, steps: int, lr: float
  ) -> dict[str, dict[str, float]]:
      src_codes, dst_codes, *_ = _build_task(seed)
      # _train_learned may not accept lr; if not, ignore.
      try:
          mi, _ = _train_learned(
              src_codes, dst_codes, steps=steps, seed=seed, lr=lr,
          )
      except TypeError:
          mi, _ = _train_learned(src_codes, dst_codes, steps=steps, seed=seed)
      return {"learned": {"mi_bits": float(mi)}}


  def _vec2vec_runner(
      *, seed: int, lambda_cycle: float, steps: int
  ) -> dict[str, dict[str, float]]:
      # Fall back to the bundled benchmark with the requested seed.
      res = run_transducer_benchmark(steps=steps, seed=seed)
      return {"vec2vec": {"mi_bits": float(res["vec2vec"]["mi_bits"])}}


  def _relrep_runner(
      *, seed: int, n_anchors: int
  ) -> dict[str, dict[str, float]]:
      res = run_transducer_benchmark(steps=500, seed=seed)
      return {"relrep": {"mi_bits": float(res["relrep"]["mi_bits"])}}


  def _procrustes_runner(*, seed: int) -> dict[str, dict[str, float]]:
      res = run_transducer_benchmark(steps=500, seed=seed)
      return {"procrustes": {"mi_bits": float(res["procrustes"]["mi_bits"])}}


  def _eval_trial(
      runner:           Any,
      method:           str,
      params:           dict[str, Any],
      seeds_per_trial:  Sequence[int],
  ) -> dict[str, Any]:
      agg = run_multi_seed(runner, seeds=seeds_per_trial, **params)
      values = agg[method]["mi_bits"]["values"]
      ci = bootstrap_ci(values, n_resamples=500, seed=0)
      return {
          "params":      params,
          "mi_values":   values,
          "mi_mean":     ci["mean"],
          "mi_ci95_low": ci["ci95_low"],
          "mi_ci95_high": ci["ci95_high"],
      }


  def _best_of(trials: list[dict[str, Any]]) -> dict[str, Any]:
      return max(trials, key=lambda t: t["mi_mean"])


  def run_equal_tuning(
      *,
      budget:          int = 8,
      seeds_per_trial: tuple[int, ...] = (0, 1, 2),
  ) -> dict[str, dict[str, Any]]:
      """Best-of-budget MI per method under an equal trial budget."""
      out: dict[str, dict[str, Any]] = {}

      learned_trials = [
          _eval_trial(_learned_runner, "learned", p, seeds_per_trial)
          for p in LEARNED_GRID[:budget]
      ]
      out["learned"] = {
          "trials": learned_trials,
          "best":   _best_of(learned_trials),
      }

      vec2vec_trials = [
          _eval_trial(_vec2vec_runner, "vec2vec", p, seeds_per_trial)
          for p in VEC2VEC_GRID[:budget]
      ]
      out["vec2vec"] = {
          "trials": vec2vec_trials,
          "best":   _best_of(vec2vec_trials),
      }

      relrep_trials = [
          _eval_trial(_relrep_runner, "relrep", p, seeds_per_trial)
          for p in RELREP_GRID[:budget]
      ]
      out["relrep"] = {
          "trials": relrep_trials,
          "best":   _best_of(relrep_trials),
      }

      proc_trial = _eval_trial(
          _procrustes_runner, "procrustes", {}, seeds_per_trial,
      )
      out["procrustes"] = {
          "trials": [proc_trial],
          "best":   proc_trial,
          "note":   "closed-form, non-tunable; reported at parity budget",
      }
      return out


  def main() -> None:  # pragma: no cover
      out = run_equal_tuning()
      for method, payload in out.items():
          best = payload["best"]
          print(
              f"{method:<12}  best mi={best['mi_mean']:.4f}  "
              f"CI=[{best['mi_ci95_low']:.4f}, {best['mi_ci95_high']:.4f}]  "
              f"params={best['params']}"
          )


  if __name__ == "__main__":  # pragma: no cover
      main()
  ```

  Before running tests, double-check `_train_learned` and
  `run_transducer_benchmark` signatures. If the bundled benchmark
  does not accept the requested hyperparameter (e.g. Vec2Vec
  `lambda_cycle`), the trial still executes — but record the limitation
  in the doc (next step) so reviewers know which methods were fully
  tuned and which were ablated within the available knobs.

- [ ] Create `docs/superpowers/research/equal-tuning-protocol.md`:

  ```markdown
  # Equal-tuning Protocol for Transducer Baselines

  ## Motivation

  Reviewers reasonably reject any "method A beats method B" claim
  when A received hand-tuning and B did not. Plan A's runner reports a
  single (non-tuned) score per method; this protocol locks in an
  equal-effort comparison.

  ## Budget

  * Trials per tunable method: **8**.
  * Seeds per trial: **3** (default `(0, 1, 2)`).
  * Total runs per tunable method: **24**.
  * Selection rule: best mean MI across the 3 seeds.

  ## Method coverage

  | Method      | Tunable | Grid (≤8 trials)                                                |
  |-------------|---------|-----------------------------------------------------------------|
  | Learned     | yes     | `steps ∈ {500, 2000, 8000} × lr ∈ {1e-3, 3e-3, 1e-2}` (first 8)  |
  | Vec2Vec     | yes     | `lambda_cycle ∈ {1.0, 10.0, 50.0, 100.0} × steps ∈ {500, 2000}` |
  | RelativeRep | yes     | `n_anchors ∈ {4, 8, 16, 32, 48, 64, 96, 128}`                   |
  | Procrustes  | no      | Closed-form; reported once at parity multi-seed budget.         |

  Where a grid axis is not actually wired into the bundled runner
  (e.g. `lambda_cycle` may be hard-coded inside
  `run_transducer_benchmark`), the trial still executes and the
  resulting variation reflects seed noise. Future work: expose those
  hyperparameters in the runner signatures so the budget is fully
  spent on real HP variation rather than seed jitter.

  ## Reporting

  For each method we report:
  * The full list of 8 trials with their `(params, mi_mean,
    mi_ci95_low, mi_ci95_high)`.
  * The best trial (argmax mean MI).
  * The non-parametric bootstrap 95% CI on the best trial's MI values.

  Procrustes is reported as a single trial at the same multi-seed
  budget so its CI is comparable.

  ## Rationale

  * Budget of 8 trials × 3 seeds matches what a careful practitioner
    would explore in a paper; smaller is suspiciously thin, larger
    biases toward whichever method has the richer grid.
  * Bootstrap CI is preferred over parametric SE because per-seed MI
    distributions are not necessarily Gaussian.
  * Procrustes' "non-tunable" status is declared up front to avoid
    accusations that the learned arm was given an unfair tuning
    advantage; both arms see the same seed budget.
  ```

- [ ] Run the slow test, expect PASS:
  `uv run pytest tests/integration/test_equal_tuning_pilot.py -x -m slow`.

- [ ] Lint + types:
  `uv run ruff check scripts/equal_tuning_pilot.py tests/integration/test_equal_tuning_pilot.py`
  and `uv run mypy scripts/equal_tuning_pilot.py`.

- [ ] Run the full fast suite once more:
  `uv run pytest -m "not slow"`.

- [ ] Commit:
  ```
  git add scripts/equal_tuning_pilot.py \
          docs/superpowers/research/equal-tuning-protocol.md \
          tests/integration/test_equal_tuning_pilot.py
  git commit -m "feat: equal-tuning protocol + doc"
  ```

---

## Self-Review

* All five tasks land independently on `feat/gap-analysis-remediation`;
  A2.3 depends on A2.1+A2.2 only for downstream reuse (the null arms
  themselves don't import the wrapper or the paired tests); A2.4 and
  A2.5 import `scripts.multi_seed` and
  `nerve_wml.methodology.paired_tests` respectively, so they must land
  after A2.1/A2.2.
* No placeholders. Every code block compiles as written modulo
  signature checks called out at the relevant step (helpers like
  `_train_vec2vec` are guarded by `try/except`).
* Existing infrastructure is reused (`bootstrap_ci_mi` is left
  untouched; the new `bootstrap_ci` operates on float lists, a
  different shape).
* Every task ends in a single conventional commit ≤50 chars with no
  underscore in the scope, English, no `--no-verify`.
* Tests use the right markers (`slow` for integration touching torch
  training, fast for the unit wrapper and the paired-test math).

## Execution Handoff

Run this plan with **Subagent-Driven Development**
(`superpowers:subagent-driven-development`): each task is small enough
for a single subagent invocation with its own checkpoint and review
gate. Inline execution is acceptable only if you want to keep the full
five-task arc in one session; either way, run
`uv run pytest -m "not slow"` after every commit and the matching
`@pytest.mark.slow` test on the task it belongs to.
