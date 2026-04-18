from typing import get_type_hints
from nerve_core.protocols import Nerve, WML


def test_nerve_has_alphabet_size_constant():
    assert Nerve.ALPHABET_SIZE == 64


def test_nerve_has_gamma_theta_constants():
    assert Nerve.GAMMA_HZ == 40.0
    assert Nerve.THETA_HZ == 6.0


def test_nerve_protocol_has_required_methods():
    required = {"send", "listen", "time", "tick", "routing_weight"}
    assert required.issubset(set(dir(Nerve)))


def test_wml_protocol_has_required_attrs():
    hints = get_type_hints(WML)
    assert "id" in hints
    assert "codebook" in hints
    assert "step" in dir(WML)
    assert "parameters" in dir(WML)
