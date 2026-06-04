import torch

from nerve_core.protocols import WML
from track_w.bio_clients import MockBioCultureClient
from track_w.bio_wml import BioWML


def _mk(**kw):
    return BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0, **kw)


def test_bio_wml_has_required_attrs():
    wml = _mk()
    assert wml.id == 0
    assert wml.codebook.shape == (64, 16)


def test_bio_wml_conforms_to_wml_protocol():
    wml = _mk()
    assert isinstance(wml, WML)


def test_bio_wml_parameters_include_codebook():
    wml = _mk()
    param_ids = {id(p) for p in wml.parameters()}
    assert id(wml.codebook) in param_ids


def test_bio_wml_seed_is_local():
    torch.manual_seed(42)
    expected = torch.rand(1).item()
    torch.manual_seed(42)
    _ = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=99)
    observed = torch.rand(1).item()
    assert expected == observed


def test_bio_wml_accepts_input_dim_larger_than_d_hidden():
    wml = BioWML(
        id=0, client=MockBioCultureClient(seed=0),
        input_dim=784, d_hidden=16, seed=0,
    )
    x = torch.randn(4, 784)
    h = wml.input_proj(x)
    assert h.shape == (4, 16)


def test_bio_wml_default_input_dim_matches_d_hidden():
    wml = BioWML(
        id=0, client=MockBioCultureClient(seed=0),
        d_hidden=16, seed=0,
    )
    assert wml.input_dim == 16
