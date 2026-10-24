from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness.codes import (  # noqa: E402
    ARM_CODES,
    CONDITION_CODES,
    run_seed_sequence,
)
from experiments.procgen_multiagent_gtm_comm.harness.gtm_mlx import (  # noqa: E402
    GammaThetaMultiplexerMLX,
    GTMConfigMLX,
)

GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "n10c_gtm_k4_a16.npz"


def test_carrier_matches_torch_golden():
    g = np.load(GOLDEN)
    mux = GammaThetaMultiplexerMLX(GTMConfigMLX(), mx.array(g["constellation"]))
    out = np.array(mux(mx.array(g["codes"].astype(np.int32))))
    assert out.shape == g["carrier"].shape
    assert np.allclose(out, g["carrier"], atol=1e-5)


def test_k_above_capacity_raises():
    mux = GammaThetaMultiplexerMLX(GTMConfigMLX(), mx.zeros((16, 2)) + 1.0)
    with pytest.raises(ValueError):
        mux(mx.zeros((2, 5), dtype=mx.int32))


def test_run_seed_sequence_distinct_per_arm():
    a = run_seed_sequence(0, "gtm", "registered")
    b = run_seed_sequence(0, "mlp", "registered")
    a2 = run_seed_sequence(0, "gtm", "registered")
    assert list(a.entropy) != list(b.entropy)
    assert list(a.entropy) == list(a2.entropy)
    assert list(a.entropy) == [0, ARM_CODES["gtm"], CONDITION_CODES["registered"]]
    assert list(b.entropy) == [0, 1, 0]
