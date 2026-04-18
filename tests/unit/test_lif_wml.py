import torch

from track_w.lif_wml import LifWML


def test_lif_wml_has_required_attrs():
    wml = LifWML(id=0, n_neurons=50, seed=0)
    assert wml.id == 0
    assert wml.codebook.shape == (64, 50)
    assert wml.v_mem.shape == (50,)
    assert wml.v_thr == 1.0


def test_lif_wml_parameters_include_codebook():
    wml = LifWML(id=0, n_neurons=50, seed=0)
    param_ids = {id(p) for p in wml.parameters()}
    assert id(wml.codebook) in param_ids


def test_lif_wml_seed_is_local():
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = LifWML(id=0, n_neurons=50, seed=99)
    observed = torch.rand(1).item()

    assert expected == observed
