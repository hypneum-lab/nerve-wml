"""Tests for bridge.kiki_nerve_advisor — never-raise contract + env gate."""
import torch

from bridge.checkpoint import save_advisor_checkpoint
from bridge.kiki_nerve_advisor import NerveWmlAdvisor
from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML


def _mk_checkpoint(tmp_path):
    pool = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)
    save_advisor_checkpoint(pool, nerve, tmp_path)
    return tmp_path


def test_advise_returns_none_when_disabled(tmp_path):
    adv = NerveWmlAdvisor(enabled=False, checkpoint_path=tmp_path)
    q = torch.randn(1, 16)
    assert adv.advise(q) is None


def test_advise_returns_none_without_checkpoint():
    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=None)
    q = torch.randn(1, 16)
    assert adv.advise(q) is None


def test_advise_returns_none_on_missing_checkpoint(tmp_path):
    missing = tmp_path / "nonexistent"
    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=missing)
    q = torch.randn(1, 16)
    assert adv.advise(q) is None


def test_advise_returns_dict_with_valid_checkpoint(tmp_path):
    path = _mk_checkpoint(tmp_path)
    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=path, n_domains=35)
    q = torch.randn(1, 16)
    result = adv.advise(q)
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) == 35
    # All weights are floats.
    for w in result.values():
        assert isinstance(w, float)
    # Softmax: sum to 1.
    assert abs(sum(result.values()) - 1.0) < 1e-4


def test_advise_returns_none_on_nan_input(tmp_path):
    path = _mk_checkpoint(tmp_path)
    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=path)
    q = torch.full((1, 16), float("nan"))
    assert adv.advise(q) is None


def test_advise_returns_none_on_shape_mismatch(tmp_path):
    path = _mk_checkpoint(tmp_path)
    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=path)
    # Wrong token dim (checkpoint uses d_hidden=16, we pass 32).
    q = torch.randn(1, 32)
    assert adv.advise(q) is None


def test_advise_never_raises_on_corrupt_checkpoint(tmp_path):
    # Write a garbage file at the checkpoint path.
    (tmp_path / "manifest.pt").write_bytes(b"not a torch file")
    (tmp_path / "pool.pt").write_bytes(b"also garbage")
    (tmp_path / "nerve.pt").write_bytes(b"more garbage")
    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=tmp_path)
    q = torch.randn(1, 16)
    # Must return None rather than raise.
    assert adv.advise(q) is None


def test_advise_is_idempotent(tmp_path):
    path = _mk_checkpoint(tmp_path)
    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=path, n_domains=35)
    q = torch.randn(1, 16)
    a = adv.advise(q)
    b = adv.advise(q)
    assert a == b
