import pytest
from track_w.configs.wml_config import WmlConfig


def test_wml_config_defaults_match_v1_1():
    cfg = WmlConfig()
    assert cfg.d_hidden == 16
    assert cfg.n_neurons == 16
    assert cfg.d_model == 16
    assert cfg.alphabet_size == 64
    assert cfg.input_dim == 16


def test_wml_config_mnist_preset():
    cfg = WmlConfig.mnist()
    assert cfg.input_dim == 784
    assert cfg.d_hidden == 128
    assert cfg.n_neurons == 128
    assert cfg.d_model == 128
    assert cfg.alphabet_size == 256


def test_wml_config_validates_divisibility():
    with pytest.raises(ValueError):
        WmlConfig(d_model=15, n_heads=2)
    with pytest.raises(ValueError):
        WmlConfig(d_model=16, n_tokens=5)
