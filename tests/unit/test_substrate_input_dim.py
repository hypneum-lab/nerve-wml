import torch

from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.transformer_wml import TransformerWML


def test_mlp_wml_accepts_input_dim_larger_than_d_hidden():
    wml = MlpWML(id=0, input_dim=784, d_hidden=128, seed=0)
    x = torch.randn(4, 784)
    h = wml.core(x)
    assert h.shape == (4, 128)


def test_lif_wml_accepts_input_dim_larger_than_n_neurons():
    wml = LifWML(id=0, input_dim=784, n_neurons=128, seed=0)
    assert wml.input_proj.in_features == 784
    assert wml.input_proj.out_features == 128


def test_transformer_wml_accepts_input_dim():
    wml = TransformerWML(
        id=0, input_dim=784, d_model=128, n_tokens=8, n_heads=4, seed=0,
    )
    x = torch.randn(4, 784)
    h = wml.core(x)
    assert h.shape == (4, 128)


def test_mlp_wml_default_input_dim_matches_d_hidden():
    # v1.1 backward-compat: default input_dim=None falls back to d_hidden.
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    x = torch.randn(4, 16)
    h = wml.core(x)
    assert h.shape == (4, 16)


def test_lif_wml_default_input_dim_matches_n_neurons():
    wml = LifWML(id=0, n_neurons=16, seed=0)
    assert wml.input_proj.in_features == 16


def test_transformer_wml_default_input_dim_matches_d_model():
    wml = TransformerWML(id=0, d_model=16, n_tokens=4, n_heads=2, seed=0)
    x = torch.randn(4, 16)
    h = wml.core(x)
    assert h.shape == (4, 16)
