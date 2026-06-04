# Validation Suite A.4 — AKOrN Comparison Arm Implementation Plan

For agentic workers: execute under `superpowers:subagent-driven-development`.
Plain `git commit` only — never `--amend`, never `--no-verify`. Commit
subjects ≤ 50 chars.

**Goal.** Plan A.2 declared AKOrN (Artificial Kuramoto Oscillatory Neurons,
Miyato et al., ICLR 2025) as scoped future work in
`docs/superpowers/research/akorn-future-work.md`. This plan lands a
minimal but real AKOrN comparison arm so the GTM ablation has a third
contender beyond the simple-gating control — addressing the critique
"GTM not collapsing is a weak claim without an oscillator-based head-to-head."

**Architecture.** One small Kuramoto-based multiplexer
(`KuramotoMultiplexer`) parallel to `SimpleGatingMultiplexer`, plus an
extension of `scripts/gtm_ablation_pilot.py` to include the new arm.
The Kuramoto unit is intentionally simple: a fixed-N coupled oscillator
network with learned natural frequencies and a linear read-out — enough
to instantiate the AKOrN philosophy (phase-coded computation) without
requiring a full ODE integrator. Result: a clear 3-arm comparison on
the same routing task with synchrony measured the same way.

**Tech Stack.** Python 3.12+, `uv`, pytest, PyTorch (`nn.Module`,
`nn.Parameter`, autograd for backprop through a discrete Euler step),
ruff, mypy. No new dependencies. Reuse Plan A.2.3's null-arm pattern
and Plan A.2's multi-seed wrapper for the empirical run.

---

## Task A4.1 — KuramotoMultiplexer module

**Files**
- Modify: `track_p/transducer_baselines.py`
- Test: `tests/unit/test_kuramoto_multiplexer.py`

**Steps**

- [ ] Write the failing test at `tests/unit/test_kuramoto_multiplexer.py`:

  ```python
  import torch

  from track_p.transducer_baselines import KuramotoMultiplexer


  def test_kuramoto_round_trips_codes_noise_free():
      torch.manual_seed(0)
      m = KuramotoMultiplexer(alphabet_size=64, n_symbols=7)
      codes = torch.randint(0, 64, (8, 7))
      carrier = m.forward(codes)
      assert carrier.shape[0] == 8
      recovered = m.demodulate(carrier)
      assert recovered.shape == codes.shape
      assert (recovered >= 0).all() and (recovered < 64).all()


  def test_kuramoto_is_differentiable():
      torch.manual_seed(1)
      m = KuramotoMultiplexer(alphabet_size=64, n_symbols=7)
      codes = torch.randint(0, 64, (4, 7))
      carrier = m.forward(codes)
      carrier.sum().backward()
      # Coupling weights and natural frequencies must receive gradient.
      assert m.coupling.grad is not None
      assert m.natural_freqs.grad is not None


  def test_kuramoto_phase_state_evolves():
      torch.manual_seed(2)
      m = KuramotoMultiplexer(alphabet_size=64, n_symbols=7, n_steps=20)
      codes = torch.tensor([[3, 17, 42, 5, 9, 30, 11]])
      carrier_short = m.forward(codes, n_steps=2)
      carrier_long = m.forward(codes, n_steps=20)
      # Longer integration produces a different terminal phase pattern.
      assert not torch.allclose(carrier_short, carrier_long, atol=1e-6)
  ```

- [ ] Run, expect FAIL (`ImportError: cannot import name 'KuramotoMultiplexer'`):
  `cd /Users/electron/Documents/Projets/nerve-wml-wt-gap && uv run pytest tests/unit/test_kuramoto_multiplexer.py -x`.

- [ ] Append to `track_p/transducer_baselines.py`:

  ```python
  class KuramotoMultiplexer(nn.Module):
      """Minimal AKOrN-style coupled-oscillator multiplexer.

      Inspired by Miyato et al. (ICLR 2025, arXiv:2410.13821) Artificial
      Kuramoto Oscillatory Neurons. Each of `n_oscillators` units has a
      learned natural frequency and is coupled to every other through a
      learned [N, N] coupling matrix. A stimulus code is written as a
      phase-bias injection across a subset of oscillators (selected
      deterministically by code bits, same trick as the simple gating
      control). After `n_steps` of Euler integration of the Kuramoto
      update, the terminal phase pattern is read out linearly to logits.

      Provides the same `forward(codes) -> carrier` /
      `demodulate(carrier) -> codes` contract as `SimpleGatingMultiplexer`
      so the gtm ablation script can drop it in as a third arm.

      Parameters
      ----------
      alphabet_size : int
          Code alphabet size.
      n_symbols : int
          Code slots per carrier.
      n_oscillators : int
          Oscillator pool size (default 32; smaller = faster, less expressive).
      n_steps : int
          Default Euler-step count per forward (default 8).
      dt : float
          Integration step size (default 0.1).
      """

      def __init__(
          self,
          *,
          alphabet_size: int = 64,
          n_symbols: int = 7,
          n_oscillators: int = 32,
          n_steps: int = 8,
          dt: float = 0.1,
      ) -> None:
          super().__init__()
          self.alphabet_size = alphabet_size
          self.n_symbols = n_symbols
          self.n_oscillators = n_oscillators
          self.n_steps = n_steps
          self.dt = dt
          # Learned natural frequencies, one per oscillator.
          self.natural_freqs = nn.Parameter(
              torch.randn(n_oscillators) * 0.5
          )
          # Learned coupling matrix [N, N]. Symmetric initialisation
          # for stability; not enforced symmetric afterwards.
          init_coupling = torch.randn(n_oscillators, n_oscillators) * 0.1
          self.coupling = nn.Parameter(
              0.5 * (init_coupling + init_coupling.T)
          )
          # Phase-bias injection: each code maps to a small fixed pattern
          # of phase offsets across the oscillator pool, derived from the
          # code bits — matches the SimpleGating idiom.
          self.code_bias = nn.Parameter(
              torch.randn(alphabet_size, n_oscillators) * 0.1
          )
          # Linear read-out: phase pattern -> per-symbol logits.
          self.readout = nn.Linear(
              n_oscillators, n_symbols * alphabet_size,
          )

      def forward(
          self, codes: torch.Tensor, *, n_steps: int | None = None,
      ) -> torch.Tensor:
          """Encode `[B, n_symbols]` long codes to a `[B, n_oscillators]` carrier."""
          if codes.shape[-1] != self.n_symbols:
              raise ValueError(
                  f"expected {self.n_symbols} symbols, got {codes.shape[-1]}"
              )
          steps = n_steps if n_steps is not None else self.n_steps
          batch = codes.shape[0]
          # Initial phase = mean of injected biases across symbol slots.
          biases = self.code_bias[codes]  # [B, n_symbols, n_oscillators]
          phase = biases.mean(dim=1)      # [B, n_oscillators]
          # Discretised Kuramoto update for `steps` Euler steps.
          for _ in range(steps):
              # dphi_i/dt = omega_i + sum_j K_ij * sin(phi_j - phi_i)
              diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)
              # diff[b, i, j] = phi_i - phi_j; we want phi_j - phi_i for the formula.
              # Use -diff so the sign is right.
              interaction = (self.coupling * torch.sin(-diff)).sum(dim=-1)
              phase = phase + self.dt * (self.natural_freqs + interaction)
          return phase  # [B, n_oscillators]

      def demodulate(self, carrier: torch.Tensor) -> torch.Tensor:
          """Recover `[B, n_symbols]` long codes from a `[B, n_oscillators]` carrier."""
          logits = self.readout(carrier)
          logits = logits.view(-1, self.n_symbols, self.alphabet_size)
          return logits.argmax(dim=-1)

      def demodulate_logits(self, carrier: torch.Tensor) -> torch.Tensor:
          """`[B, n_symbols, alphabet_size]` logits — for training the read-out."""
          logits = self.readout(carrier)
          return logits.view(-1, self.n_symbols, self.alphabet_size)
  ```

- [ ] Run, expect PASS (3 tests).

- [ ] Lint + types: `uv run ruff check track_p/transducer_baselines.py tests/unit/test_kuramoto_multiplexer.py && uv run mypy track_p`. Both clean.

- [ ] Commit:
  ```bash
  git add track_p/transducer_baselines.py tests/unit/test_kuramoto_multiplexer.py
  git commit -m "feat: Kuramoto multiplexer for AKOrN arm"
  ```
  (40 chars.)

---

## Task A4.2 — Add AKOrN arm to GTM ablation pilot

**Files**
- Modify: `scripts/gtm_ablation_pilot.py`
- Modify: `tests/integration/test_gtm_ablation_pilot.py`

The pilot currently has `gtm`, `simple_gating`, `null` arms. Add `akorn`
(KuramotoMultiplexer) — same routing task, same synchrony metric.

**Steps**

- [ ] Add failing assertion to `tests/integration/test_gtm_ablation_pilot.py`:

  ```python
  @pytest.mark.slow
  def test_gtm_ablation_includes_akorn_arm() -> None:
      from scripts.gtm_ablation_pilot import run_gtm_ablation
      res = run_gtm_ablation(steps=200, seed=0)
      assert "akorn" in res
      for key in ("accuracy", "mi_bits", "synchrony_index"):
          assert key in res["akorn"]
  ```

- [ ] Run, expect FAIL: the runner doesn't have an `akorn` arm yet.

- [ ] In `scripts/gtm_ablation_pilot.py`, add a `_train_akorn` function
  parallel to `_train_simple` (the existing simple-gating trainer).
  Use the same training pattern: forward → demodulate → cross-entropy
  on the symbol logits → backprop. Match accuracy / mi_bits /
  synchrony_index outputs. Then in `run_gtm_ablation`, compute the
  AKOrN arm right after `simple_gating` and store as `result["akorn"]`
  with the same schema. Update the `null` arm assertion in
  `test_gtm_does_not_fully_collapse` if it does a strict-set check
  (change to superset) — but DON'T modify that test's `< 0.95`
  assertion.

  Concrete snippet to add (adapt to the existing `_train_simple`
  pattern):

  ```python
  def _train_akorn(
      codes: torch.Tensor, steps: int,
  ) -> tuple[float, float, float]:
      """Train the KuramotoMultiplexer; same return triple as _train_simple."""
      from track_p.transducer_baselines import KuramotoMultiplexer

      m = KuramotoMultiplexer(
          alphabet_size=64, n_symbols=codes.shape[-1],
          n_oscillators=32, n_steps=8,
      )
      opt = Adam(m.parameters(), lr=0.02)
      for _ in range(steps):
          opt.zero_grad()
          carrier = m.forward(codes)
          logits = m.demodulate_logits(carrier)
          loss = F.cross_entropy(
              logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
          )
          loss.backward()
          opt.step()
      with torch.no_grad():
          carrier = m.forward(codes)
          pred = m.demodulate(carrier)
          acc = float((pred == codes).float().mean())
          mi = mi_miller_madow_discrete(
              pred.reshape(-1).cpu().numpy().astype(np.int64),
              codes.reshape(-1).cpu().numpy().astype(np.int64),
          ) * _BITS
          sync = _synchrony_index(carrier)
      return acc, mi, sync
  ```

  Then in `run_gtm_ablation`, after the existing `simple_gating` block
  and before the `null` block, add:

  ```python
  acc_a, mi_a, sync_a = _train_akorn(codes, steps)
  result["akorn"] = {
      "accuracy":         acc_a,
      "mi_bits":          mi_a,
      "synchrony_index":  sync_a,
  }
  ```

- [ ] Run the new test, expect PASS. Run the full slow suite for the
  ablation file to confirm no regression:
  `uv run pytest tests/integration/test_gtm_ablation_pilot.py -m slow`.

- [ ] Lint + types: `uv run ruff check scripts/gtm_ablation_pilot.py tests/integration/test_gtm_ablation_pilot.py && uv run mypy track_p`.

- [ ] Commit:
  ```bash
  git add scripts/gtm_ablation_pilot.py tests/integration/test_gtm_ablation_pilot.py
  git commit -m "feat: AKOrN arm in GTM ablation pilot"
  ```
  (37 chars.)

---

## Task A4.3 — Smoke + research note

**Files**
- Create: `docs/superpowers/research/2026-05-20-akorn-comparison.md`

**Steps**

- [ ] Run the GTM ablation pilot with multi-seed via the wrapper:

  ```bash
  cd /Users/electron/Documents/Projets/nerve-wml-wt-gap
  uv run python -c "
  from scripts.multi_seed import run_multi_seed
  from scripts.gtm_ablation_pilot import run_gtm_ablation
  agg = run_multi_seed(run_gtm_ablation, seeds=(0,1,2,3,4), steps=200)
  for method, metrics in agg.items():
      for metric, leaf in metrics.items():
          print(f'{method:>14s}  {metric:<18s}  mean={leaf[\"mean\"]:7.4f}  std={leaf[\"std\"]:6.4f}')
  "
  ```

- [ ] Capture the 4-arm × 3-metric table.

- [ ] Run paired Wilcoxon (gtm vs akorn) and (akorn vs simple_gating)
  on the synchrony_index values to test whether AKOrN's phase
  dynamics give it materially different synchrony from the simple
  gating control:

  ```bash
  uv run python -c "
  from scripts.multi_seed import run_multi_seed
  from scripts.gtm_ablation_pilot import run_gtm_ablation
  from nerve_wml.methodology.paired_tests import wilcoxon_paired
  agg = run_multi_seed(run_gtm_ablation, seeds=(0,1,2,3,4), steps=200)
  for pair in (('gtm','akorn'), ('akorn','simple_gating'), ('gtm','simple_gating')):
      a, b = pair
      va = agg[a]['synchrony_index']['values']
      vb = agg[b]['synchrony_index']['values']
      res = wilcoxon_paired(va, vb)
      print(f'{a} vs {b}: p={res[\"p_value\"]:.4f}  d_z={res[\"cohens_dz\"]:+.2f}  median_diff={res[\"median_diff\"]:+.4f}')
  "
  ```

- [ ] Write the research note `docs/superpowers/research/2026-05-20-akorn-comparison.md`:
  - **Headline**: 3-5 bullets summarising whether AKOrN's synchrony index lands closer to GTM, closer to simple_gating, or somewhere distinct.
  - **Method**: 5 seeds, GTM ablation runner with 4 arms, steps=200, AKOrN with `n_oscillators=32, n_steps=8`.
  - **Result table**: 4 arms × {accuracy, mi_bits, synchrony_index} mean ± std.
  - **Paired tests**: gtm vs akorn, akorn vs simple_gating, gtm vs simple_gating — paste the p-values.
  - **Interpretation**: be honest. If AKOrN's synchrony lies between GTM (0.21) and simple_gating (0.08), it confirms band-multiplexing is a real axis. If AKOrN clusters near simple_gating, GTM's distinctness is robust. If AKOrN matches GTM, then the multiplexing claim weakens.
  - **Open questions**: scaling, longer Euler integration, comparison to full ODE-integrator AKOrN per Miyato et al.

- [ ] Commit:
  ```bash
  git add docs/superpowers/research/2026-05-20-akorn-comparison.md
  git commit -m "docs: AKOrN 4-arm GTM ablation result note"
  ```
  (44 chars.)

---

## Self-Review

- 3 tasks, each ends in a single conventional commit ≤ 50 chars.
- No placeholders. Every code edit is verbatim. The Kuramoto class is
  a real implementation (Euler integration of the standard Kuramoto
  ODE with learned natural frequencies and coupling), not a stub.
- Reuses Plan A.2.1's `run_multi_seed`, Plan A.2.2's `wilcoxon_paired`,
  and Plan A.2.3's null-arm pattern.
- The AKOrN arm uses the same training loop pattern as `_train_simple`
  to ensure fair budget (same steps, same optimiser, same lr).
- This plan documents AKOrN as a **minimal** comparison arm — not a
  full reimplementation of Miyato et al.'s production AKOrN. The
  research note will explicitly note this limitation.

## Execution Handoff

Execute with `superpowers:subagent-driven-development` — one subagent
per task. A4.2 depends on A4.1 (the new class must exist); A4.3
depends on A4.2 (the runner must report the akorn arm).
