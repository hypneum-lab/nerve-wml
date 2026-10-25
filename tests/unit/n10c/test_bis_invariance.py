"""N10-C arm invariance under the N10-C-bis additions (addendum 2026-10-25,
"What is not modified": for each of the six N10-C arms the parameter count
equals the N10-C manifest table and the initial parameters for a given key
are bit-identical before and after the N10-C-bis changes; the ``make_comm``
seeding path is unchanged).

"Before" = the frozen reference vendored from tag ``n10c-freeze-20261024``
(``tests/unit/n10c/n10c_frozen/``); "after" = the live harness.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from experiments.procgen_multiagent_gtm_comm.harness import comm as live_comm  # noqa: E402
from experiments.procgen_multiagent_gtm_comm.harness.codes import (  # noqa: E402
    ARM_CODES,
    BIS_ARMS,
    N10C_ARMS,
    init_key_int,
)
from experiments.procgen_multiagent_gtm_comm.harness.policy import (  # noqa: E402
    PairPolicy,
    param_count,
)
from tests.unit.n10c.n10c_frozen import comm as frozen_comm  # noqa: E402
from tests.unit.n10c.n10c_frozen.policy import PairPolicy as FrozenPairPolicy  # noqa: E402

# N10-C stage-2 manifest, section 2 (tag n10c-freeze-20261024)
N10C_MANIFEST_PARAMS = {
    "gtm": (734_256, 8_864), "mlp": (727_520, 2_128), "atoc": (737_344, 11_952),
    "dial": (727_520, 2_128), "eng": (726_480, 1_088), "nocomm": (726_480, 1_088),
}
TRUNK, TO_MSG, FUSE_PI_V = 622_144, 16_448, 86_800


def _flat(model):
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}


@pytest.mark.parametrize("arm", N10C_ARMS)
def test_n10c_arm_parameter_counts_equal_the_manifest_table(arm):
    m = PairPolicy(arm, mx.random.key(0))
    total, comm_n = N10C_MANIFEST_PARAMS[arm]
    assert param_count(m) == total
    assert param_count(m.comm) == comm_n
    assert param_count(m.trunk) == TRUNK and param_count(m.to_msg) == TO_MSG
    assert param_count(m.fuse) + param_count(m.pi) + param_count(m.v) == FUSE_PI_V


@pytest.mark.parametrize("arm", N10C_ARMS)
@pytest.mark.parametrize("seed", [0, 131])
def test_n10c_arm_initialisation_is_bit_identical_to_the_frozen_reference(arm, seed):
    key = init_key_int(seed, arm, "registered")
    before = _flat(FrozenPairPolicy(arm, mx.random.key(key)))
    after = _flat(PairPolicy(arm, mx.random.key(key)))
    assert sorted(before) == sorted(after)  # same parameter names
    for name in before:
        assert before[name].dtype == after[name].dtype
        assert np.array_equal(before[name], after[name]), name


@pytest.mark.parametrize("arm", N10C_ARMS)
def test_n10c_arm_forward_is_identical_to_the_frozen_reference(arm):
    rng = np.random.default_rng(3)
    obs = mx.array(rng.integers(0, 256, (3, 2, 64, 64, 3)).astype(np.float32))
    summary = mx.array(rng.random((3, 2, 2)).astype(np.float32))
    eng = mx.array(rng.integers(0, 2, (3, 2, 16)).astype(np.float32))
    key = mx.random.key(init_key_int(17, arm, "closed"))
    before, after = FrozenPairPolicy(arm, key), PairPolicy(arm, key)
    for train in (False, True):
        nz = after.sample_noise(3, mx.random.key(9))
        ob = before(obs, summary, eng, train=train, key=mx.random.key(1), noise=nz)
        oa = after(obs, summary, eng, train=train, key=mx.random.key(1), noise=nz)
        for name in ("logits", "value", "hard_bits"):
            assert np.array_equal(np.array(getattr(ob, name)),
                                  np.array(getattr(oa, name))), (name, train)
        assert oa.aux_loss is None  # no auxiliary loss for an N10-C arm


def test_arm_codes_are_appended_never_renumbered():
    frozen_codes = {"gtm": 0, "mlp": 1, "atoc": 2, "dial": 3, "eng": 4, "nocomm": 5}
    assert {a: ARM_CODES[a] for a in frozen_codes} == frozen_codes
    assert {a: ARM_CODES[a] for a in BIS_ARMS} == {
        "gtm_mis": 6, "eng_gtmrx": 7, "gtm_warm": 8, "vq": 9}
    assert len(set(ARM_CODES.values())) == len(ARM_CODES)


def test_frozen_reference_is_the_tagged_source():
    # the vendored modules are the tag's: same constants and tables
    assert np.array_equal(np.array(frozen_comm.CODE_TABLE), np.array(live_comm.CODE_TABLE))
    assert (frozen_comm.K_SYMBOLS, frozen_comm.ALPHABET, frozen_comm.N_BITS,
            frozen_comm.MSG_DIM, frozen_comm.HID_DIM) == (4, 16, 16, 64, 256)
    assert not hasattr(frozen_comm, "VQComm") and hasattr(live_comm, "VQComm")
