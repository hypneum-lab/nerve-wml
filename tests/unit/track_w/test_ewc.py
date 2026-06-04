"""Unit tests for EWC (diagonal Fisher + quadratic penalty)."""
import torch

from track_w.continual.ewc import estimate_fisher, penalty
from track_w.mlp_wml import MlpWML
from track_w.tasks.hard_split import HardSplitTask


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
    opt.zero_grad()
    loss.backward()
    opt.step()

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
    opt.zero_grad()
    loss.backward()
    opt.step()

    p1 = penalty(wml, fisher, theta_star, lam=1.0).item()
    p2 = penalty(wml, fisher, theta_star, lam=2.0).item()
    assert abs(p2 - 2 * p1) < 1e-5
