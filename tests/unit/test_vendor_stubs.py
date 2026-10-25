"""Vendor stubs must raise informative NotImplementedError without SDK."""
import pytest

from neuromorphic.akida_stub import AkidaCompiler
from neuromorphic.loihi_stub import LoihiCompiler


def test_loihi_stub_raises_informative_error():
    with pytest.raises(NotImplementedError, match="lava-nc"):
        LoihiCompiler.compile({})


def test_akida_stub_raises_informative_error():
    with pytest.raises(NotImplementedError, match="akida"):
        AkidaCompiler.compile({})


def test_loihi_stub_points_to_docs():
    with pytest.raises(NotImplementedError, match="deployment-guide"):
        LoihiCompiler.compile({})


def test_loihi_stub_states_lava_archived():
    # Phase 0 requalification (issue #64): the stub must not promise
    # a Loihi 2 path; the Lava SDK is archived upstream.
    with pytest.raises(NotImplementedError, match="archived"):
        LoihiCompiler.compile({})


def test_akida_stub_points_to_docs():
    with pytest.raises(NotImplementedError, match="deployment-guide"):
        AkidaCompiler.compile({})
