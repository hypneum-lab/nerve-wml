import os

import pytest

from track_w.bio_clients import (
    BioApiKeyMissingError,
    BioCultureClient,
    CL1Adapter,
    FinalSparkAdapter,
)

_ENV = "NERVE_WML_BIO_API_KEY"


def test_cl1_adapter_raises_without_api_key(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(BioApiKeyMissingError):
        CL1Adapter()


def test_finalspark_adapter_raises_without_api_key(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(BioApiKeyMissingError):
        FinalSparkAdapter()


def test_adapter_classes_are_constructible_with_fake_key(monkeypatch):
    # A non-empty key lets the constructor succeed; no network call
    # happens at construction time.
    monkeypatch.setenv(_ENV, "fake-key-for-construction-only")
    cl1 = CL1Adapter()
    fs = FinalSparkAdapter()
    assert isinstance(cl1, BioCultureClient)
    assert isinstance(fs, BioCultureClient)
    cl1.close()
    fs.close()


@pytest.mark.slow
def test_finalspark_roundtrip_real_hardware():
    if not os.environ.get(_ENV):
        pytest.skip("NERVE_WML_BIO_API_KEY unset — real API skipped")
    client = FinalSparkAdapter()
    frame = client.roundtrip([1, 2, 3])
    assert frame.spikes.ndim >= 2
    assert frame.latency_ms > 0.0
    client.close()


@pytest.mark.slow
def test_cl1_roundtrip_real_hardware():
    if not os.environ.get(_ENV):
        pytest.skip("NERVE_WML_BIO_API_KEY unset — real API skipped")
    client = CL1Adapter()
    frame = client.roundtrip([1, 2, 3])
    assert frame.spikes.ndim >= 2
    assert frame.latency_ms > 0.0
    client.close()
