"""Tests for bridge.query_encoder."""
import torch

from bridge.query_encoder import QueryEncoder


def test_encoder_returns_code_indices():
    codebook = torch.randn(64, 16)
    enc = QueryEncoder(token_dim=32, hidden_dim=16, codebook=codebook, seed=0)
    tokens = torch.randn(8, 32)
    codes = enc(tokens)
    assert codes.shape == (8,)
    assert codes.dtype == torch.long
    assert (codes >= 0).all()
    assert (codes < 64).all()


def test_encoder_is_deterministic():
    codebook = torch.randn(64, 16)
    enc = QueryEncoder(token_dim=32, hidden_dim=16, codebook=codebook, seed=0)
    tokens = torch.randn(8, 32)
    a = enc(tokens)
    b = enc(tokens)
    assert torch.equal(a, b)


def test_encoder_seed_isolates_global_rng():
    codebook = torch.randn(64, 16)
    torch.manual_seed(42)
    expected = torch.rand(1).item()
    torch.manual_seed(42)
    _ = QueryEncoder(token_dim=32, hidden_dim=16, codebook=codebook, seed=99)
    observed = torch.rand(1).item()
    assert expected == observed


def test_encoder_rejects_codebook_dim_mismatch():
    codebook = torch.randn(64, 16)
    try:
        QueryEncoder(token_dim=32, hidden_dim=8, codebook=codebook)
        raise AssertionError("should have raised on dim mismatch")
    except AssertionError as e:
        assert "codebook dim" in str(e) or "should have raised" not in str(e)
