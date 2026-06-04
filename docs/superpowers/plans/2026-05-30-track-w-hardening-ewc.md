# Track-W EWC Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a genuinely hard continual task (`HardSplitTask`) with shared output head that produces catastrophic forgetting (≥ 50 %) without mitigation, implement EWC (diagonal Fisher + quadratic penalty) over `wml.parameters()`, and produce a reproducible multi-seed comparison `none` / `rehearsal` / `ewc` on that task. Results traced to `docs/superpowers/research/2026-05-30-w4-ewc-comparison.json` via `scripts/factcheck_audit.py`.

**Architecture:** New sub-package `track_w/continual/` isolates continual-learning mechanisms (`ewc.py`, `rehearsal.py`). A new `run_w4_compare(method, task, steps, seed, **kw) -> dict` in `scripts/track_w_pilot.py` provides the shared training scaffold. `HardSplitTask` in `track_w/tasks/hard_split.py` wraps two sequential `HardFlowProxyTask` sub-tasks over a shared label space (classes 0..11, shared 12-way output head). Existing `run_w4_shared_head` / `run_w4_rehearsal` and all gate-w-passed tests are untouched.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.3, pytest + `@pytest.mark.slow`, uv, `scripts/factcheck_audit.py` (JSON claim tracing). No new dependencies.

---

## T1 — `track_w/tasks/hard_split.py`: HardSplitTask

**Rationale:** `SplitMnistLikeTask` uses disjoint output heads — forgetting is avoided by construction. `HardSplitTask` wraps two sequential `HardFlowProxyTask` sub-tasks with the *same* 12-class label space so catastrophic forgetting is real.

**Files:**
- **Create** `track_w/tasks/hard_split.py`
- **Create** `tests/unit/track_w/test_hard_split_task.py`

### Steps

- [ ] **Write the failing unit test** (`tests/unit/track_w/test_hard_split_task.py`):

  ```python
  """Unit tests for HardSplitTask."""
  import torch
  from track_w.tasks.hard_split import HardSplitTask


  def test_hard_split_shapes():
      task = HardSplitTask(seed=0)
      x0, y0 = task.subtasks[0].sample(batch=32)
      x1, y1 = task.subtasks[1].sample(batch=32)
      assert x0.shape == (32, 16)
      assert y0.shape == (32,)
      assert x1.shape == (32, 16)
      assert y1.shape == (32,)


  def test_hard_split_shared_label_space():
      """Both sub-tasks emit labels in 0..11 (same 12-class head)."""
      task = HardSplitTask(seed=0)
      for subtask in task.subtasks:
          _, y = subtask.sample(batch=256)
          assert y.min().item() >= 0
          assert y.max().item() <= 11


  def test_hard_split_subtasks_different_distributions():
      """The two sub-tasks must be non-identical (different centroids)."""
      task = HardSplitTask(seed=0)
      x0, _ = task.subtasks[0].sample(batch=128)
      x1, _ = task.subtasks[1].sample(batch=128)
      # Means should differ by at least 0.1 in L2 norm.
      assert (x0.mean(0) - x1.mean(0)).norm().item() > 0.1
  ```

- [ ] **Run the test; verify FAIL** (module not found):

  ```bash
  uv run pytest tests/unit/track_w/test_hard_split_task.py -v
  # Expected: ERROR ModuleNotFoundError: No module named 'track_w.tasks.hard_split'
  ```

- [ ] **Implement `track_w/tasks/hard_split.py`**:

  ```python
  """HardSplitTask — two sequential sub-tasks over a shared 12-class head.

  Both sub-tasks use HardFlowProxyTask with n_classes=12 but distinct seeds,
  so the class centroids and XOR-gating hyperplanes differ. A vanilla
  shared-head learner trained Task0 → Task1 with no mitigation forgets ≥ 50 %
  of Task0 accuracy (verified by test_w4_hard_split_baseline).
  """
  from __future__ import annotations

  from dataclasses import dataclass, field

  from .hard_flow_proxy import HardFlowProxyTask


  @dataclass
  class HardSplitTask:
      """Sequential pair of HardFlowProxyTask over a shared 12-class label space.

      Attributes
      ----------
      seed : int
          Base seed. subtasks[0] uses seed, subtasks[1] uses seed + 1.
      dim : int
          Input feature dimension (matches MlpWML default d_hidden=16).
      n_classes : int
          Number of shared output classes (12; same head for both sub-tasks).
      subtasks : list[HardFlowProxyTask]
          [subtasks[0], subtasks[1]] — train sequentially.
      """

      seed: int = 0
      dim: int = 16
      n_classes: int = 12
      subtasks: list = field(init=False)

      def __post_init__(self) -> None:
          self.subtasks = [
              HardFlowProxyTask(
                  dim=self.dim, n_classes=self.n_classes, seed=self.seed
              ),
              HardFlowProxyTask(
                  dim=self.dim, n_classes=self.n_classes, seed=self.seed + 1
              ),
          ]
  ```

- [ ] **Run the test; verify PASS**:

  ```bash
  uv run pytest tests/unit/track_w/test_hard_split_task.py -v
  # Expected: 3 passed
  ```

- [ ] **Commit**:

  ```bash
  git add track_w/tasks/hard_split.py tests/unit/track_w/test_hard_split_task.py
  git commit -m "feat(track-w): HardSplitTask shared-head 12-class"
  ```

---

## T2 — `track_w/continual/ewc.py`: EWC diagonal Fisher + penalty

**Rationale:** Implements `estimate_fisher` (squared gradients of log-likelihood) and `penalty` (quadratic EWC term) over `wml.parameters()`. Respects W-2 (penalty covers codebook) and W-1 (no mutation of another WML).

**Files:**
- **Create** `track_w/continual/__init__.py`
- **Create** `track_w/continual/ewc.py`
- **Create** `tests/unit/track_w/test_ewc.py`

### Steps

- [ ] **Write the failing unit test** (`tests/unit/track_w/test_ewc.py`):

  ```python
  """Unit tests for EWC (diagonal Fisher + quadratic penalty)."""
  import torch
  from track_w.mlp_wml import MlpWML
  from track_w.tasks.hard_split import HardSplitTask
  from track_w.continual.ewc import estimate_fisher, penalty


  def _make_loader(task_idx: int, n_batches: int = 4, batch: int = 32):
      """Return a list of (x, y) pairs from HardSplitTask subtask[task_idx]."""
      task = HardSplitTask(seed=0)
      return [task.subtasks[task_idx].sample(batch=batch) for _ in range(n_batches)]


  def test_estimate_fisher_returns_param_keys():
      torch.manual_seed(0)
      wml = MlpWML(id=0, d_hidden=16, seed=0)
      loader = _make_loader(0)
      fisher = estimate_fisher(wml, loader)
      param_names = {name for name, _ in wml.named_parameters()}
      assert set(fisher.keys()) == param_names


  def test_estimate_fisher_non_negative():
      torch.manual_seed(0)
      wml = MlpWML(id=0, d_hidden=16, seed=0)
      loader = _make_loader(0)
      fisher = estimate_fisher(wml, loader)
      for name, f in fisher.items():
          assert (f >= 0).all(), f"Fisher[{name}] has negative entries"


  def test_estimate_fisher_covers_codebook():
      """W-2: codebook must appear in the Fisher dict (penalty covers it)."""
      torch.manual_seed(0)
      wml = MlpWML(id=0, d_hidden=16, seed=0)
      loader = _make_loader(0)
      fisher = estimate_fisher(wml, loader)
      assert "codebook" in fisher


  def test_penalty_zero_at_theta_star():
      """Penalty is 0 when current params equal theta_star."""
      torch.manual_seed(0)
      wml = MlpWML(id=0, d_hidden=16, seed=0)
      loader = _make_loader(0)
      fisher = estimate_fisher(wml, loader)
      # Snapshot params *before* any update.
      theta_star = {name: p.detach().clone() for name, p in wml.named_parameters()}
      pen = penalty(wml, fisher, theta_star, lam=1.0)
      assert pen.item() < 1e-8, f"Expected ~0 penalty at theta_star, got {pen.item()}"


  def test_penalty_positive_after_update():
      """After an SGD step, penalty > 0 (params diverged from theta_star)."""
      torch.manual_seed(0)
      wml = MlpWML(id=0, d_hidden=16, seed=0)
      loader = _make_loader(0)
      fisher = estimate_fisher(wml, loader)
      theta_star = {name: p.detach().clone() for name, p in wml.named_parameters()}

      # One SGD step to move params away from theta_star.
      opt = torch.optim.SGD(wml.parameters(), lr=0.1)
      x, y = loader[0]
      logits = wml.emit_head_pi(wml.core(x))[:, :12]
      loss = torch.nn.functional.cross_entropy(logits, y)
      opt.zero_grad(); loss.backward(); opt.step()

      pen = penalty(wml, fisher, theta_star, lam=1.0)
      assert pen.item() > 0.0


  def test_penalty_scales_with_lam():
      """penalty(lam=2) == 2 * penalty(lam=1)."""
      torch.manual_seed(0)
      wml = MlpWML(id=0, d_hidden=16, seed=0)
      loader = _make_loader(0)
      fisher = estimate_fisher(wml, loader)
      theta_star = {name: p.detach().clone() for name, p in wml.named_parameters()}

      # Move params.
      opt = torch.optim.SGD(wml.parameters(), lr=0.1)
      x, y = loader[0]
      logits = wml.emit_head_pi(wml.core(x))[:, :12]
      loss = torch.nn.functional.cross_entropy(logits, y)
      opt.zero_grad(); loss.backward(); opt.step()

      p1 = penalty(wml, fisher, theta_star, lam=1.0).item()
      p2 = penalty(wml, fisher, theta_star, lam=2.0).item()
      assert abs(p2 - 2 * p1) < 1e-5
  ```

- [ ] **Run the test; verify FAIL** (module not found):

  ```bash
  uv run pytest tests/unit/track_w/test_ewc.py -v
  # Expected: ERROR ModuleNotFoundError: No module named 'track_w.continual'
  ```

- [ ] **Create `track_w/continual/__init__.py`** (empty):

  ```python
  """Continual-learning mechanisms for Track-W."""
  ```

- [ ] **Implement `track_w/continual/ewc.py`**:

  ```python
  """EWC — Elastic Weight Consolidation (diagonal Fisher approximation).

  API
  ---
  estimate_fisher(wml, data_loader) -> dict[str, Tensor]
      Compute diagonal Fisher information for each named parameter of `wml`
      from the squared gradients of the cross-entropy log-likelihood.

  penalty(wml, fisher, theta_star, lam) -> Tensor
      Return the scalar EWC penalty  lam/2 * Σ_i F_i (θ_i − θ*_i)².

  Invariants
  ----------
  W-1: No mutation of another WML's parameters — this module only reads
       and computes gradients on the wml passed in.
  W-2: penalty ranges over wml.parameters() which includes the codebook,
       so the codebook is regularised alongside the MLP weights.
  """
  from __future__ import annotations

  import torch
  import torch.nn.functional as F
  from torch import Tensor

  from track_w.mlp_wml import MlpWML


  def estimate_fisher(
      wml: MlpWML,
      data_loader: list[tuple[Tensor, Tensor]],
  ) -> dict[str, Tensor]:
      """Diagonal Fisher information from squared log-likelihood gradients.

      Parameters
      ----------
      wml : MlpWML
          The model after Task 0 training (weights = θ*).
      data_loader : list of (x, y) pairs
          Batches from Task 0. Labels must be in [0, n_classes).

      Returns
      -------
      dict mapping parameter name → non-negative diagonal Fisher tensor
          (same shape as the parameter).
      """
      n_classes = data_loader[0][1].max().item() + 1  # inferred from data
      fisher: dict[str, Tensor] = {
          name: torch.zeros_like(p) for name, p in wml.named_parameters()
      }
      wml.eval()
      n_batches = len(data_loader)
      for x, y in data_loader:
          logits = wml.emit_head_pi(wml.core(x))[:, : int(n_classes)]
          log_probs = F.log_softmax(logits, dim=-1)
          # Use predicted class (empirical Fisher) to avoid needing true labels
          # at inference time, but here we have them so we use the true label
          # log-likelihood for a tighter diagonal estimate.
          nll = F.nll_loss(log_probs, y)
          wml.zero_grad()
          nll.backward()
          for name, p in wml.named_parameters():
              if p.grad is not None:
                  fisher[name] += p.grad.detach() ** 2
      for name in fisher:
          fisher[name] /= n_batches
      wml.train()
      return fisher


  def penalty(
      wml: MlpWML,
      fisher: dict[str, Tensor],
      theta_star: dict[str, Tensor],
      lam: float,
  ) -> Tensor:
      """EWC quadratic penalty: lam/2 * Σ_i F_i (θ_i − θ*_i)².

      Parameters
      ----------
      wml : MlpWML
          Current model (θ, being trained on Task 1).
      fisher : dict[str, Tensor]
          Diagonal Fisher from estimate_fisher() — same keys as named_parameters.
      theta_star : dict[str, Tensor]
          Snapshot of wml parameters right after Task 0 (before Task 1 training).
      lam : float
          Regularisation strength (sweep and record in result JSON).

      Returns
      -------
      Scalar tensor, differentiable w.r.t. wml.parameters().
      """
      pen = torch.tensor(0.0)
      for name, p in wml.named_parameters():
          if name in fisher and name in theta_star:
              diff = p - theta_star[name]
              pen = pen + (fisher[name] * diff ** 2).sum()
      return lam / 2.0 * pen
  ```

- [ ] **Run the test; verify PASS**:

  ```bash
  uv run pytest tests/unit/track_w/test_ewc.py -v
  # Expected: 6 passed
  ```

- [ ] **Run non-regression (existing gate tests stay green)**:

  ```bash
  uv run pytest tests/integration/track_w/test_gate_w4_honest.py tests/integration/track_w/test_gate_w4.py -v
  # Expected: all passed
  ```

- [ ] **Commit**:

  ```bash
  git add track_w/continual/__init__.py track_w/continual/ewc.py tests/unit/track_w/test_ewc.py
  git commit -m "feat(track-w): EWC diagonal Fisher and penalty"
  ```

---

## T3 — `track_w/continual/rehearsal.py`: RehearsalBuffer refactor

**Rationale:** Extract the batch-mixing logic from `run_w4_rehearsal` into a reusable `RehearsalBuffer` so `run_w4_compare` can invoke all three methods through a single scaffold. Pure refactor — `run_w4_rehearsal` behaviour stays bit-stable.

**Files:**
- **Create** `track_w/continual/rehearsal.py`
- **Modify** (verify, no behaviour change) `scripts/track_w_pilot.py`
- **Test** `tests/integration/track_w/test_gate_w4_honest.py` must stay green

### Steps

- [ ] **Implement `track_w/continual/rehearsal.py`**:

  ```python
  """RehearsalBuffer — replay-based continual learning helper.

  Encapsulates the batch-mixing logic from run_w4_rehearsal so that
  run_w4_compare can use the same scaffold for all three methods
  (none / rehearsal / ewc).
  """
  from __future__ import annotations

  from dataclasses import dataclass, field

  import torch
  import torch.nn.functional as F
  from torch import Tensor


  @dataclass
  class RehearsalBuffer:
      """Fixed-replay buffer: mixes old-task samples into each Task-1 step.

      Parameters
      ----------
      rehearsal_frac : float
          Fraction of each mini-batch filled with Task-0 samples (default 0.3,
          matching run_w4_rehearsal).
      total_batch : int
          Total mini-batch size before the mix (default 64).

      Usage
      -----
      buf = RehearsalBuffer()
      buf.store(task0)          # capture the Task-0 sampler
      for _ in range(steps):
          loss = buf.mixed_loss(wml, task1, n_classes=12)
          opt.zero_grad(); loss.backward(); opt.step()
      """

      rehearsal_frac: float = 0.3
      total_batch: int = 64
      _task0: object | None = field(default=None, init=False, repr=False)

      def store(self, task0: object) -> None:
          """Register the Task-0 sampler for replay."""
          self._task0 = task0

      def mixed_loss(
          self,
          wml: object,
          task1: object,
          n_classes: int,
      ) -> Tensor:
          """Return weighted cross-entropy over a mixed Task0/Task1 mini-batch.

          Weights proportional to batch sizes (mirrors run_w4_rehearsal exactly):
              loss = (loss_new * n_new + loss_old * n_old) / total_batch
          """
          import torch.nn.functional as F  # noqa: PLC0415

          n_old = int(self.total_batch * self.rehearsal_frac)
          n_new = self.total_batch - n_old

          def _loss(task, n):
              x, y = task.sample(batch=n)
              logits = wml.emit_head_pi(wml.core(x))[:, :n_classes]
              return F.cross_entropy(logits, y)

          loss_new = _loss(task1, n_new)
          loss_old = _loss(self._task0, n_old)
          return (loss_new * n_new + loss_old * n_old) / self.total_batch
  ```

- [ ] **Run existing W4 rehearsal gate to confirm no regression**:

  ```bash
  uv run pytest tests/integration/track_w/test_gate_w4_honest.py -v
  # Expected: 1 passed (behaviour unchanged — run_w4_rehearsal untouched)
  ```

- [ ] **Commit**:

  ```bash
  git add track_w/continual/rehearsal.py
  git commit -m "refactor(track-w): extract RehearsalBuffer from pilot"
  ```

---

## T4 — `run_w4_compare` in `scripts/track_w_pilot.py`

**Rationale:** Unified training scaffold for `none` / `rehearsal` / `ewc` on any Task pair. Returns a JSON-serialisable dict with `{forgetting, acc0_before, acc0_after, acc1, method, lam}`. Existing functions remain unchanged.

**Files:**
- **Modify** `scripts/track_w_pilot.py` (append `run_w4_compare`)

### Steps

- [ ] **Write a quick smoke test** (`tests/unit/track_w/test_run_w4_compare_smoke.py`):

  ```python
  """Smoke tests for run_w4_compare — checks dict keys and numeric ranges."""
  import torch
  import pytest
  from scripts.track_w_pilot import run_w4_compare
  from track_w.tasks.hard_split import HardSplitTask


  @pytest.mark.parametrize("method", ["none", "rehearsal", "ewc"])
  def test_run_w4_compare_keys(method):
      torch.manual_seed(0)
      task = HardSplitTask(seed=0)
      result = run_w4_compare(method=method, task=task, steps=50, seed=0)
      for key in ("forgetting", "acc0_before", "acc0_after", "acc1", "method", "lam"):
          assert key in result, f"Missing key {key!r} for method={method}"

  @pytest.mark.parametrize("method", ["none", "rehearsal", "ewc"])
  def test_run_w4_compare_method_label(method):
      torch.manual_seed(0)
      task = HardSplitTask(seed=0)
      result = run_w4_compare(method=method, task=task, steps=50, seed=0)
      assert result["method"] == method
  ```

- [ ] **Run the test; verify FAIL** (function not defined yet):

  ```bash
  uv run pytest tests/unit/track_w/test_run_w4_compare_smoke.py -v
  # Expected: ERROR ImportError or AttributeError
  ```

- [ ] **Append `run_w4_compare` to `scripts/track_w_pilot.py`** (after the last top-level function):

  ```python
  def run_w4_compare(
      method: str,
      task,
      steps: int = 400,
      seed: int = 0,
      rehearsal_frac: float = 0.3,
      lam: float = 1.0,
  ) -> dict:
      """Unified W4 comparison scaffold for none / rehearsal / ewc.

      Parameters
      ----------
      method : str
          One of "none", "rehearsal", "ewc".
      task : HardSplitTask
          Provides task.subtasks[0] (Task 0) and task.subtasks[1] (Task 1).
          Must expose subtask.sample(batch) -> (x, y) and subtask.n_classes.
      steps : int
          Training steps per task.
      seed : int
          Manual seed for reproducibility.
      rehearsal_frac : float
          Fraction of Task-1 mini-batch filled with Task-0 samples (rehearsal only).
      lam : float
          EWC regularisation strength (ewc only).

      Returns
      -------
      dict with keys:
          forgetting       : float  (acc0_before - acc0_after) / acc0_before
          acc0_before      : float  Task-0 accuracy before Task-1 training
          acc0_after       : float  Task-0 accuracy after Task-1 training
          acc1             : float  Task-1 accuracy after Task-1 training
          method           : str    the method argument
          lam              : float  lam value (0.0 for none/rehearsal)
      """
      import torch
      import torch.nn.functional as F
      from track_w.mlp_wml import MlpWML
      from track_w.mock_nerve import MockNerve
      from track_w.continual.ewc import estimate_fisher, penalty as ewc_penalty
      from track_w.continual.rehearsal import RehearsalBuffer

      if method not in ("none", "rehearsal", "ewc"):
          raise ValueError(f"method must be 'none', 'rehearsal', or 'ewc'; got {method!r}")

      torch.manual_seed(seed)
      nerve = MockNerve(n_wmls=2, k=1, seed=seed)
      nerve.set_phase_active(gamma=True, theta=False)
      wml   = MlpWML(id=0, d_hidden=16, seed=seed)
      opt   = torch.optim.Adam(wml.parameters(), lr=1e-2)

      task0 = task.subtasks[0]
      task1 = task.subtasks[1]
      n_classes = task0.n_classes  # 12

      def _eval(t) -> float:
          x, y = t.sample(batch=256)
          with torch.no_grad():
              pred = wml.emit_head_pi(wml.core(x))[:, :n_classes].argmax(-1)
          return (pred == y).float().mean().item()

      def _task_loss(t, batch_size: int) -> torch.Tensor:
          x, y = t.sample(batch=batch_size)
          logits = wml.emit_head_pi(wml.core(x))[:, :n_classes]
          return F.cross_entropy(logits, y)

      # --- Task 0 training (same for all methods) ---
      for _ in range(steps):
          loss = _task_loss(task0, 64)
          opt.zero_grad(); loss.backward(); opt.step()

      acc0_before = _eval(task0)

      # --- EWC: snapshot Fisher and theta* after Task 0 ---
      fisher: dict = {}
      theta_star: dict = {}
      if method == "ewc":
          loader = [task0.sample(batch=64) for _ in range(8)]
          fisher = estimate_fisher(wml, loader)
          theta_star = {name: p.detach().clone() for name, p in wml.named_parameters()}

      # --- Rehearsal: initialise buffer ---
      buf: RehearsalBuffer | None = None
      if method == "rehearsal":
          buf = RehearsalBuffer(rehearsal_frac=rehearsal_frac, total_batch=64)
          buf.store(task0)

      # --- Task 1 training ---
      for _ in range(steps):
          if method == "none":
              loss = _task_loss(task1, 64)
          elif method == "rehearsal":
              assert buf is not None
              loss = buf.mixed_loss(wml, task1, n_classes=n_classes)
          else:  # ewc
              loss = _task_loss(task1, 64)
              if fisher:
                  loss = loss + ewc_penalty(wml, fisher, theta_star, lam=lam)
          opt.zero_grad(); loss.backward(); opt.step()

      acc0_after = _eval(task0)
      acc1       = _eval(task1)
      forgetting = (acc0_before - acc0_after) / max(acc0_before, 1e-6)

      return {
          "forgetting":  forgetting,
          "acc0_before": acc0_before,
          "acc0_after":  acc0_after,
          "acc1":        acc1,
          "method":      method,
          "lam":         lam if method == "ewc" else 0.0,
      }
  ```

- [ ] **Run the smoke test; verify PASS**:

  ```bash
  uv run pytest tests/unit/track_w/test_run_w4_compare_smoke.py -v
  # Expected: 6 passed
  ```

- [ ] **Verify no regression on all W4 gate tests**:

  ```bash
  uv run pytest tests/integration/track_w/ -k "w4" -v
  # Expected: all existing W4 tests pass
  ```

- [ ] **Commit**:

  ```bash
  git add scripts/track_w_pilot.py tests/unit/track_w/test_run_w4_compare_smoke.py
  git commit -m "feat(track-w): run_w4_compare unified scaffold"
  ```

---

## T5 — Integration tests (baseline, EWC gate, multi-seed comparison)

**Rationale:** Three integration tests exercise the full pipeline. `test_w4_hard_split_baseline` proves forgetting is real (≥ 0.50); `test_gate_w4_ewc` measures EWC and asserts it is reported (soft: beats `none`); `test_w4_method_comparison` is multi-seed (n=5, marked `slow`) and asserts both mitigations strictly improve on `none`. All thresholds traced to the research JSON.

**Files:**
- **Create** `tests/integration/track_w/test_w4_hard_split_baseline.py`
- **Create** `tests/integration/track_w/test_gate_w4_ewc.py`
- **Create** `tests/integration/track_w/test_w4_method_comparison.py`

### Steps

- [ ] **Write `tests/integration/track_w/test_w4_hard_split_baseline.py`**:

  ```python
  """Integration: HardSplitTask baseline — no mitigation forgets >= 50 %."""
  import torch
  from scripts.track_w_pilot import run_w4_compare
  from track_w.tasks.hard_split import HardSplitTask


  def test_hard_split_none_forgets_at_least_50pct():
      """Prove the task is genuinely hard: forgetting >= 0.50 without mitigation.

      Threshold 0.50 traces to docs/superpowers/research/2026-05-30-w4-ewc-comparison.json
      key "baseline_none_forgetting_threshold".
      """
      torch.manual_seed(0)
      task = HardSplitTask(seed=0)
      result = run_w4_compare(method="none", task=task, steps=400, seed=0)
      assert result["acc0_before"] > 0.30, (
          f"Task0 baseline too low ({result['acc0_before']:.3f}), "
          "adjust HardSplitTask difficulty"
      )
      assert result["forgetting"] >= 0.50, (
          f"Expected forgetting >= 0.50 but got {result['forgetting']:.3f}. "
          "The task is not hard enough — increase HardFlowProxyTask noise or overlap."
      )
  ```

- [ ] **Run the test**:

  ```bash
  uv run pytest tests/integration/track_w/test_w4_hard_split_baseline.py -v
  # Expected: PASS (if HardSplitTask produces >= 50 % forgetting without mitigation)
  # If FAIL with forgetting < 0.50: tune HardSplitTask (reduce centroids scale or
  # increase noise std in HardFlowProxyTask — do NOT change the assertion).
  ```

- [ ] **Write `tests/integration/track_w/test_gate_w4_ewc.py`**:

  ```python
  """Integration gate: EWC on HardSplitTask — measured forgetting reported honestly."""
  import torch
  from scripts.track_w_pilot import run_w4_compare
  from track_w.tasks.hard_split import HardSplitTask


  def test_ewc_forgetting_is_reported():
      """EWC forgetting is measured and returned (honest reporting, any value)."""
      torch.manual_seed(0)
      task = HardSplitTask(seed=0)
      result = run_w4_compare(method="ewc", task=task, steps=400, seed=0, lam=1.0)
      assert "forgetting" in result
      assert isinstance(result["forgetting"], float)
      # EWC result is reported even if it does not beat rehearsal.
      # The < 0.20 threshold is a target, NOT a construction (spec §13).


  def test_ewc_beats_none_baseline():
      """Soft gate: EWC forgetting < none forgetting (seed=0, lam=1.0).

      If this fails, report the honest result — do not tune lam to pass by construction.
      Threshold traces to docs/superpowers/research/2026-05-30-w4-ewc-comparison.json
      key "ewc_vs_none_seed0".
      """
      torch.manual_seed(0)
      task = HardSplitTask(seed=0)
      none_result = run_w4_compare(method="none", task=task, steps=400, seed=0)
      ewc_result  = run_w4_compare(method="ewc",  task=task, steps=400, seed=0, lam=1.0)
      assert ewc_result["forgetting"] < none_result["forgetting"], (
          f"EWC ({ewc_result['forgetting']:.3f}) did not beat none "
          f"({none_result['forgetting']:.3f}). "
          "Record this as-is in the research JSON (honest reporting)."
      )
  ```

- [ ] **Run the EWC gate test**:

  ```bash
  uv run pytest tests/integration/track_w/test_gate_w4_ewc.py -v
  # Expected: both pass. If test_ewc_beats_none_baseline FAILS, record the
  # forgetting values in the research JSON and document as honest result.
  ```

- [ ] **Write `tests/integration/track_w/test_w4_method_comparison.py`**:

  ```python
  """Multi-seed comparison: none / rehearsal / ewc on HardSplitTask (slow)."""
  import statistics
  import torch
  import pytest
  from scripts.track_w_pilot import run_w4_compare
  from track_w.tasks.hard_split import HardSplitTask


  SEEDS = list(range(5))
  LAM   = 1.0  # EWC strength — swept and documented in research JSON


  @pytest.mark.slow
  def test_w4_all_methods_multi_seed():
      """n=5 seeds: rehearsal and ewc both strictly beat none on mean forgetting.

      Results traced to docs/superpowers/research/2026-05-30-w4-ewc-comparison.json
      keys "multi_seed_none_mean", "multi_seed_rehearsal_mean", "multi_seed_ewc_mean".
      """
      forgetting: dict[str, list[float]] = {"none": [], "rehearsal": [], "ewc": []}
      for seed in SEEDS:
          task = HardSplitTask(seed=seed)
          for method in ("none", "rehearsal", "ewc"):
              kw = {"lam": LAM} if method == "ewc" else {}
              r = run_w4_compare(method=method, task=task, steps=400, seed=seed, **kw)
              forgetting[method].append(r["forgetting"])

      mean_none      = statistics.mean(forgetting["none"])
      mean_rehearsal = statistics.mean(forgetting["rehearsal"])
      mean_ewc       = statistics.mean(forgetting["ewc"])

      assert mean_rehearsal < mean_none, (
          f"rehearsal mean forgetting ({mean_rehearsal:.3f}) did not beat "
          f"none ({mean_none:.3f})"
      )
      assert mean_ewc < mean_none, (
          f"ewc mean forgetting ({mean_ewc:.3f}) did not beat "
          f"none ({mean_none:.3f})"
      )
      # Which method wins: print for tracing, no assertion (honest reporting).
      winner = "rehearsal" if mean_rehearsal <= mean_ewc else "ewc"
      print(
          f"\n[multi-seed n={len(SEEDS)}] "
          f"none={mean_none:.3f}  rehearsal={mean_rehearsal:.3f}  "
          f"ewc={mean_ewc:.3f}  winner={winner}"
      )
  ```

- [ ] **Run the multi-seed test**:

  ```bash
  uv run pytest tests/integration/track_w/test_w4_method_comparison.py -m slow -v -s
  # Expected: 1 passed, with winner printed.
  # If either mitigation does not beat none: record honestly in research JSON.
  ```

- [ ] **Run the full W4 integration suite (non-regression)**:

  ```bash
  uv run pytest tests/integration/track_w/ -v
  # Expected: all existing tests pass, 3 new tests pass.
  ```

- [ ] **Commit**:

  ```bash
  git add tests/integration/track_w/test_w4_hard_split_baseline.py \
          tests/integration/track_w/test_gate_w4_ewc.py \
          tests/integration/track_w/test_w4_method_comparison.py
  git commit -m "test(track-w): integration tests hard-split baseline, EWC gate, multi-seed"
  ```

---

## T6 — Research JSON + factcheck_audit.py hook

**Rationale:** Every numeric threshold in the integration tests must be traceable to a JSON cell (scientific protocol, CLAUDE.md). Create the research JSON and add the EWC claims to `run_audit()` in `scripts/factcheck_audit.py`.

**Files:**
- **Create** `docs/superpowers/research/2026-05-30-w4-ewc-comparison.json`
- **Modify** `scripts/factcheck_audit.py` (append EWC claims to `run_audit()`)

### Steps

- [ ] **Run the baseline + multi-seed tests to capture real values**, redirecting output to a temp file, then populate the JSON:

  ```bash
  uv run pytest tests/integration/track_w/test_w4_hard_split_baseline.py \
                tests/integration/track_w/test_gate_w4_ewc.py \
                tests/integration/track_w/test_w4_method_comparison.py \
                -m slow -v -s 2>&1 | tee /tmp/ewc_run.txt
  ```

  Inspect `/tmp/ewc_run.txt` to extract:
  - `baseline_none_forgetting_threshold`: 0.50 (spec-defined lower bound for task hardness)
  - `baseline_none_forgetting_seed0`: actual measured forgetting (method=none, seed=0)
  - `ewc_vs_none_seed0_none`: actual none forgetting
  - `ewc_vs_none_seed0_ewc`: actual ewc forgetting
  - `multi_seed_none_mean`: mean over 5 seeds (method=none)
  - `multi_seed_rehearsal_mean`: mean over 5 seeds (method=rehearsal)
  - `multi_seed_ewc_mean`: mean over 5 seeds (method=ewc)
  - `ewc_lam_used`: 1.0 (default, to be updated if a sweep finds a better value)
  - `winner`: "rehearsal" or "ewc"

- [ ] **Create `docs/superpowers/research/2026-05-30-w4-ewc-comparison.json`** with the actual measured values (replace `<VALUE>` with real numbers from the test run):

  ```json
  {
    "_session": "2026-05-30-track-w-hardening-ewc",
    "_description": "HardSplitTask baseline + EWC vs rehearsal multi-seed comparison",
    "baseline_none_forgetting_threshold": 0.50,
    "baseline_none_forgetting_seed0": "<VALUE>",
    "ewc_vs_none_seed0_none": "<VALUE>",
    "ewc_vs_none_seed0_ewc": "<VALUE>",
    "ewc_lam_used": 1.0,
    "multi_seed_n": 5,
    "multi_seed_none_mean": "<VALUE>",
    "multi_seed_rehearsal_mean": "<VALUE>",
    "multi_seed_ewc_mean": "<VALUE>",
    "winner": "<VALUE>"
  }
  ```

- [ ] **Append EWC claim checks to `run_audit()` in `scripts/factcheck_audit.py`** (at the end of the function body, before any `return`):

  ```python
  # --- EWC / HardSplitTask claims (2026-05-30-track-w-hardening-ewc) ---
  ewc_d = _maybe(RESEARCH / "2026-05-30-w4-ewc-comparison.json")
  if ewc_d is not None:
      _say("=" * 80)
      _say("CLAIM EWC-1: HardSplitTask baseline forgetting >= 0.50 (task is hard)")
      actual_baseline = ewc_d.get("baseline_none_forgetting_seed0", 0.0)
      threshold       = ewc_d.get("baseline_none_forgetting_threshold", 0.50)
      # check() uses abs(expected - computed) <= tol; encode directional test as:
      # expected = actual_baseline, computed = threshold, tol = actual_baseline - threshold
      # Simpler: just check the bool and call check() with bool values.
      check(
          "hard_split_baseline_forgetting_threshold",
          True,
          bool(actual_baseline >= threshold),
      )

      _say("=" * 80)
      _say("CLAIM EWC-2: EWC forgetting < none forgetting (seed=0)")
      ewc_f  = ewc_d.get("ewc_vs_none_seed0_ewc",  1.0)
      none_f = ewc_d.get("ewc_vs_none_seed0_none", 0.0)
      check(
          "ewc_beats_none_seed0",
          True,
          bool(ewc_f < none_f),
      )

      _say("=" * 80)
      _say("CLAIM EWC-3: multi-seed mean forgetting recorded (rehearsal)")
      orphan(
          "multi_seed_rehearsal_mean",
          f"recorded={ewc_d.get('multi_seed_rehearsal_mean', 'MISSING')} "
          "(no fixed baseline — value logged for paper table)"
      )
      _say("CLAIM EWC-4: multi-seed mean forgetting recorded (ewc)")
      orphan(
          "multi_seed_ewc_mean",
          f"recorded={ewc_d.get('multi_seed_ewc_mean', 'MISSING')} "
          "(no fixed baseline — value logged for paper table)"
      )
  else:
      orphan("ewc_hardening", "2026-05-30-w4-ewc-comparison.json not found — run T6 tests first")
  ```

- [ ] **Run factcheck_audit in CI mode**:

  ```bash
  uv run python scripts/factcheck_audit.py --ci
  # Expected: 0 DIVERGENT (EWC-1/EWC-2 are directional; EWC-3/EWC-4 are orphan-logged)
  ```

- [ ] **Run the full fast test suite to confirm no regression**:

  ```bash
  uv run pytest -m "not slow" -q
  # Expected: all existing tests pass
  ```

- [ ] **Commit**:

  ```bash
  git add docs/superpowers/research/2026-05-30-w4-ewc-comparison.json \
          scripts/factcheck_audit.py
  git commit -m "docs(track-w): EWC research JSON and factcheck claims"
  ```

---

## Invariant checklist (before PR open)

Run once after all T1–T6 steps are complete:

```bash
# W-1..W-4 non-regression
uv run pytest tests/integration/track_w/test_gate_w.py -v

# Full fast suite (L1–L2)
uv run pytest -m "not slow" -q

# Slow integration suite
uv run pytest -m slow -q

# Factcheck
uv run python scripts/factcheck_audit.py --ci

# Linting
uv run ruff check track_w/continual/ track_w/tasks/hard_split.py scripts/track_w_pilot.py
```

All must be green before the PR is opened.
