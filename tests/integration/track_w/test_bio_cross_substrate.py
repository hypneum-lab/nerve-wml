import torch

from nerve_core.protocols import WML
from track_w.bio_clients import MockBioCultureClient
from track_w.bio_wml import BioWML
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.transformer_wml import TransformerWML


def _pool():
    return [
        MlpWML(id=0, d_hidden=16, seed=0),
        LifWML(id=1, n_neurons=16, seed=1),
        TransformerWML(id=2, d_model=16, n_tokens=4, n_heads=2, seed=2),
        BioWML(id=3, client=MockBioCultureClient(seed=3), seed=3),
    ]


def test_all_four_substrates_satisfy_wml_protocol():
    for wml in _pool():
        assert isinstance(wml, WML)


def test_heterogeneous_pool_runs_ten_ticks():
    nerve = MockNerve(n_wmls=4, k=2, seed=0)
    pool = _pool()
    for tick in range(10):
        nerve.set_phase_active(gamma=True, theta=True)
        for wml in pool:
            wml.step(nerve, t=float(tick))
        nerve.tick(dt=1e-3)
    # The bio substrate logged at least one round-trip latency.
    bio = pool[3]
    assert isinstance(bio, BioWML)
    assert bio.last_latency_ms is not None


def test_bio_wml_codebook_is_trainable_in_pool():
    bio = BioWML(id=3, client=MockBioCultureClient(seed=3), seed=3)
    opt = torch.optim.SGD(bio.parameters(), lr=0.01)
    before = bio.codebook.detach().clone()
    loss = bio.emit_head_pi(bio.codebook.mean(dim=0)).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.equal(before, bio.codebook)
