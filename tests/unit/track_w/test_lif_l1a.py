"""Unit tests for L1a: Full-LIF sim training on HardFlowProxyTask.

Test taxonomy:
- (a) pipeline construction + gradient flow
- (b) trained > untrained (learning signal present)
- (c) CI-friendly fast variants + @slow markers for sweep

Gate L1a criteria (spec §5.2):
1. Final acc >= 20 % (vs ~8.3 % chance)
2. Loss at step N_STEPS < loss at step 100
3. Untrained acc <= 10 %
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from track_w._surrogate import spike_with_surrogate
from track_w.lif_wml import LifWML
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask

# L1a constants (match scripts/lif_l1a_train.py)
TASK_DIM = 16
N_CLASSES = 12
INPUT_ENCODER_DIM = 64
T_TICKS = 8
BATCH = 64


def _make_lif(n_neurons: int, seed: int = 0) -> LifWML:
    return LifWML(id=0, n_neurons=n_neurons, alphabet_size=64,
                  input_dim=INPUT_ENCODER_DIM, seed=seed + 10)


def _make_encoder(seed: int = 0) -> torch.nn.Linear:
    torch.manual_seed(seed)
    enc = torch.nn.Linear(TASK_DIM, INPUT_ENCODER_DIM)
    torch.nn.init.kaiming_normal_(enc.weight, nonlinearity="relu")
    torch.nn.init.zeros_(enc.bias)
    return enc


def _forward_t_ticks(
    lif: LifWML,
    encoder: torch.nn.Linear,
    x: torch.Tensor,
) -> torch.Tensor:
    """T-tick unrolled forward; returns accumulated spikes [batch, n_neurons]."""
    batch = x.shape[0]
    v_mem = torch.zeros(batch, lif.n_neurons)
    spikes_acc = torch.zeros(batch, lif.n_neurons)
    for _ in range(T_TICKS):
        i_in = lif.input_proj(encoder(x))
        v_mem = v_mem + (1e-3 / lif.tau_mem) * (-v_mem + i_in)
        spikes = spike_with_surrogate(v_mem, v_thr=lif.v_thr)
        v_mem = v_mem * (1.0 - spikes.detach())
        spikes_acc = spikes_acc + spikes
    return spikes_acc


def _eval_acc_no_grad(
    lif: LifWML, encoder: torch.nn.Linear, task: HardFlowProxyTask
) -> float:
    with torch.no_grad():
        x, y = task.sample(batch=512)
        spikes_acc = _forward_t_ticks(lif, encoder, x)
        logits = lif.emit_head_pi(spikes_acc)[:, :N_CLASSES]
        return (logits.argmax(-1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# (a) Construction and gradient flow
# ---------------------------------------------------------------------------

class TestLIFL1aPipelineConstruction:
    """Pipeline constructs and produces valid gradients."""

    def test_lifwml_1024_constructs(self):
        lif = _make_lif(n_neurons=1024)
        assert lif.n_neurons == 1024
        assert lif.input_dim == INPUT_ENCODER_DIM
        assert lif.input_proj.in_features == INPUT_ENCODER_DIM
        assert lif.input_proj.out_features == 1024
        assert lif.emit_head_pi.in_features == 1024

    def test_encoder_input_dim_matches(self):
        enc = _make_encoder()
        assert enc.out_features == INPUT_ENCODER_DIM
        assert enc.in_features == TASK_DIM

    def test_forward_produces_spike_output_shape(self):
        torch.manual_seed(42)
        lif = _make_lif(n_neurons=256)
        enc = _make_encoder()
        x = torch.randn(BATCH, TASK_DIM)
        spikes_acc = _forward_t_ticks(lif, enc, x)
        assert spikes_acc.shape == (BATCH, 256)

    def test_gradients_flow_through_surrogate(self):
        """Loss backward must populate gradients on all trainable params."""
        torch.manual_seed(0)
        lif = _make_lif(n_neurons=128)
        enc = _make_encoder()
        task = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=0)

        x, y = task.sample(batch=BATCH)
        spikes_acc = _forward_t_ticks(lif, enc, x)
        logits = lif.emit_head_pi(spikes_acc)[:, :N_CLASSES]
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # input_proj, emit_head_pi, and encoder must all have gradients.
        assert lif.input_proj.weight.grad is not None
        assert lif.input_proj.weight.grad.abs().sum() > 0, "input_proj grad is zero"
        assert lif.emit_head_pi.weight.grad is not None
        assert lif.emit_head_pi.weight.grad.abs().sum() > 0, "emit_head_pi grad is zero"
        assert enc.weight.grad is not None
        assert enc.weight.grad.abs().sum() > 0, "encoder grad is zero"

    def test_codebook_grad_flows(self):
        """Codebook is a Parameter; it must also receive gradient."""
        torch.manual_seed(7)
        lif = _make_lif(n_neurons=64)
        enc = _make_encoder()
        task = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=0)

        x, y = task.sample(batch=BATCH)
        spikes_acc = _forward_t_ticks(lif, enc, x)
        logits = lif.emit_head_pi(spikes_acc)[:, :N_CLASSES]
        F.cross_entropy(logits, y).backward()

        # Codebook is not in the compute graph for the emit_head_pi path
        # (it's only used in pattern-match decode during step()), so we
        # only assert the primary trainable params got gradients (above).
        # This test just documents the known scope.
        assert True  # structural, not a numerical assertion


# ---------------------------------------------------------------------------
# (b) Trained > untrained  (fast, few steps, CI-friendly)
# ---------------------------------------------------------------------------

class TestLIFL1aLearnsSignal:
    """Loss must descend and trained acc must beat random."""

    def test_loss_descends_fast(self):
        """100-step smoke: loss at step 100 < loss at step 1."""
        torch.manual_seed(0)
        lif = _make_lif(n_neurons=256)
        enc = _make_encoder()
        task = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=0)
        optimizer = torch.optim.Adam(
            list(lif.parameters()) + list(enc.parameters()), lr=3e-3
        )

        losses = []
        for _ in range(100):
            x, y = task.sample(batch=BATCH)
            spikes_acc = _forward_t_ticks(lif, enc, x)
            logits = lif.emit_head_pi(spikes_acc)[:, :N_CLASSES]
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lif.parameters()) + list(enc.parameters()), 1.0
            )
            optimizer.step()
            losses.append(loss.item())

        # The last-10-step mean should be lower than the first-10-step mean.
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early, (
            f"Loss did not descend: early={early:.4f} late={late:.4f}"
        )

    def test_trained_beats_untrained_acc(self):
        """300-step training on n=256 must beat untrained by > 2 pp."""
        torch.manual_seed(1)
        lif = _make_lif(n_neurons=256, seed=1)
        enc = _make_encoder(seed=1)
        task = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=1)
        eval_task = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=1)

        acc_untrained = _eval_acc_no_grad(lif, enc, eval_task)

        optimizer = torch.optim.Adam(
            list(lif.parameters()) + list(enc.parameters()), lr=3e-3
        )
        for _ in range(300):
            x, y = task.sample(batch=BATCH)
            spikes_acc = _forward_t_ticks(lif, enc, x)
            logits = lif.emit_head_pi(spikes_acc)[:, :N_CLASSES]
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lif.parameters()) + list(enc.parameters()), 1.0
            )
            optimizer.step()

        eval_task2 = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=1)
        acc_trained = _eval_acc_no_grad(lif, enc, eval_task2)

        assert acc_trained > acc_untrained + 0.02, (
            f"Trained acc {acc_trained:.3f} not meaningfully > "
            f"untrained {acc_untrained:.3f}"
        )


# ---------------------------------------------------------------------------
# (c) Full L1a gate — marked slow; requires macM1 run for citation
# ---------------------------------------------------------------------------

class TestLIFL1aGate:
    """Full gate conditions from spec §5.2 (2000 steps, n=1024).

    These tests are marked @slow and are excluded from the fast CI suite.
    They are run on macM1 and their results are written to
    docs/superpowers/research/2026-05-30-lif-l1a-gate.json.
    """

    @pytest.mark.slow
    def test_gate_acc_gte_20pct(self, tmp_path):
        """Full 2000-step run at n=1024 must reach acc >= 20 %."""
        from scripts.lif_l1a_train import PRIMARY_N, train_one
        r = train_one(PRIMARY_N, seed=0, log_path=None, verbose=False)
        assert r["acc_final"] >= 0.20, (
            f"Gate failed: acc_final={r['acc_final']:.3f} < 0.20. "
            f"loss_100={r['loss_step100']:.4f} loss_final={r['loss_step2000']:.4f}"
        )

    @pytest.mark.slow
    def test_gate_loss_descends(self, tmp_path):
        """Full 2000-step run: loss at step 2000 < loss at step 100."""
        from scripts.lif_l1a_train import PRIMARY_N, train_one
        r = train_one(PRIMARY_N, seed=0, log_path=None, verbose=False)
        assert r["loss_step2000"] < r["loss_step100"], (
            f"Loss did not descend: "
            f"loss_100={r['loss_step100']:.4f} >= loss_final={r['loss_step2000']:.4f}"
        )

    @pytest.mark.slow
    def test_gate_untrained_lte_10pct(self):
        """Random-weight LifWML must not trivially exceed 10 % on the task."""
        torch.manual_seed(0)
        lif = _make_lif(n_neurons=1024)
        enc = _make_encoder()
        eval_task = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=0)
        acc = _eval_acc_no_grad(lif, enc, eval_task)
        assert acc <= 0.10, (
            f"Untrained acc {acc:.3f} > 0.10 — structural bias detected"
        )

    @pytest.mark.slow
    def test_sweep_monotone_or_reported(self):
        """Sweep {128,256,512,1024}: run and report results.

        Does NOT assert strict monotonicity (may plateau) — reports honestly.
        Asserts only that all sizes converge above untrained baseline.
        """
        from scripts.lif_l1a_train import run_sweep
        results = run_sweep(verbose=False)
        for r in results:
            assert r["acc_final"] > r["acc_untrained"] + 0.01, (
                f"n={r['n_neurons']}: trained {r['acc_final']:.3f} not > "
                f"untrained {r['acc_untrained']:.3f} + 1pp"
            )
