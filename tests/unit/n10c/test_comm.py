import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from experiments.procgen_multiagent_gtm_comm.harness.comm import (  # noqa: E402
    CODE_TABLE,
    GTMComm,
    bits_to_indices,
    eng_bits_from_window,
    make_comm,
    plasticity_scale,
    onehot_from_bits,
)
from experiments.procgen_multiagent_gtm_comm.harness.gtm_mlx import (  # noqa: E402
    GammaThetaMultiplexerMLX,
    GTMConfigMLX,
)

ARMS = ["gtm", "mlp", "atoc", "dial", "eng", "nocomm"]
B = 8


def _inputs(seed=0):
    k = mx.random.key(seed)
    k1, k2 = mx.random.split(k)
    x = mx.random.normal((B, 64), key=k1)
    h = mx.random.normal((B, 256), key=k2)
    eng = mx.array(np.random.default_rng(seed).integers(0, 2, (B, 16)).astype(np.float32))
    return x, h, eng


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_eval_bits_are_binary_16(arm):
    comm = make_comm(arm, mx.random.key(1))
    x, h, eng = _inputs()
    msg, bits = comm.send(x, train=False, key=None, eng_bits=eng)
    for a in (msg, bits):
        arr = np.array(a)
        assert arr.shape == (B, 16)
        assert set(np.unique(arr)).issubset({0.0, 1.0})
    assert np.array_equal(np.array(msg), np.array(bits))
    out = comm.receive(msg, h)
    assert out.shape == (B, 64)


@pytest.mark.parametrize("arm", ["gtm", "mlp", "atoc", "eng", "nocomm"])
def test_train_messages_binary_except_dial(arm):
    comm = make_comm(arm, mx.random.key(2))
    x, h, eng = _inputs()
    msg, bits = comm.send(x, train=True, key=mx.random.key(3), eng_bits=eng)
    assert set(np.unique(np.array(msg))).issubset({0.0, 1.0})
    assert set(np.unique(np.array(bits))).issubset({0.0, 1.0})


def test_dial_train_messages_not_binary_eval_binary():
    comm = make_comm("dial", mx.random.key(4))
    x, h, _ = _inputs()
    msg, bits = comm.send(x, train=True, key=mx.random.key(5), eng_bits=None)
    arr = np.array(msg)
    assert arr.shape == (B, 16)
    assert not set(np.unique(arr)).issubset({0.0, 1.0})
    assert ((arr > 0) & (arr < 1)).all()
    assert set(np.unique(np.array(bits))).issubset({0.0, 1.0})
    msg_e, bits_e = comm.send(x, train=False, key=None, eng_bits=None)
    assert set(np.unique(np.array(msg_e))).issubset({0.0, 1.0})


def test_code_table_msb_first():
    t = np.array(CODE_TABLE)
    assert t.shape == (16, 4)
    assert t[0].tolist() == [0, 0, 0, 0]
    assert t[1].tolist() == [0, 0, 0, 1]
    assert t[8].tolist() == [1, 0, 0, 0]
    assert t[13].tolist() == [1, 1, 0, 1]


def test_gtm_bits_roundtrip_indices():
    rng = np.random.default_rng(0)
    edge = np.array([0, 1, 7, 8, 14, 15])
    idx = np.concatenate(
        [rng.integers(0, 16, (500, 4)), np.stack([edge[:4]] * 2 + [edge[2:]] * 2)]
    ).astype(np.int32)
    oh = mx.eye(16)[mx.array(idx)]
    bits = (oh @ CODE_TABLE).reshape(idx.shape[0], 16)
    assert np.array_equal(np.array(bits_to_indices(bits)), idx)
    # differentiable one-hot rebuilt from hard bits is exact
    oh2 = onehot_from_bits(bits)
    assert np.array_equal(np.array(oh2), np.array(oh))
    # all 16 values per slot, via 16^4 = 65536 enumeration
    g = np.stack(np.meshgrid(*[np.arange(16)] * 4, indexing="ij"), -1).reshape(-1, 4)
    gb = (mx.eye(16)[mx.array(g.astype(np.int32))] @ CODE_TABLE).reshape(-1, 16)
    assert np.array_equal(np.array(bits_to_indices(gb)), g)


def test_gtm_receiver_gradient_reaches_sender():
    comm = make_comm("gtm", mx.random.key(6))
    x, h, _ = _inputs()
    key = mx.random.key(7)

    def loss(params):
        comm.update(params)
        msg, _ = comm.send(x, train=True, key=key, eng_bits=None)
        return (comm.receive(msg, h) ** 2).sum()

    grads = mx.grad(loss)(comm.trainable_parameters())
    enc = grads["encode_logits"]["weight"]
    assert float(mx.abs(enc).sum()) > 0.0
    assert float(mx.abs(grads["constellation"]).sum()) > 0.0


def test_gtm_soft_carrier_matches_hard_multiplexer():
    comm = make_comm("gtm", mx.random.key(8))
    assert isinstance(comm, GTMComm)
    idx = mx.array(np.random.default_rng(1).integers(0, 16, (32, 4)).astype(np.int32))
    oh = mx.eye(16)[idx]
    mux = GammaThetaMultiplexerMLX(GTMConfigMLX(), comm.constellation)
    hard = np.array(mux(idx))
    soft = np.array(mux.carrier_from_onehot(oh))
    assert hard.shape == soft.shape
    assert np.allclose(hard, soft, atol=1e-5)


def test_gtm_constellation_init_is_psk_plus_small_noise():
    comm = make_comm("gtm", mx.random.key(9))
    c = np.array(comm.constellation)
    ang = 2 * np.pi * np.arange(16) / 16
    psk = np.stack([np.cos(ang), np.sin(ang)], -1)
    assert c.shape == (16, 2)
    assert np.abs(c - psk).max() < 0.1
    assert np.abs(c - psk).max() > 0.0


def test_make_comm_reproducible():
    a = make_comm("gtm", mx.random.key(11))
    b = make_comm("gtm", mx.random.key(11))
    assert np.array_equal(np.array(a.constellation), np.array(b.constellation))
    assert np.array_equal(np.array(a.encode_logits.weight), np.array(b.encode_logits.weight))


def test_eng_bits_layout():
    b = eng_bits_from_window(True, 20, -3)
    assert b.shape == (16,)
    assert b[0] == 1
    assert "".join(str(int(v)) for v in b[1:6]) == "10100"
    assert "".join(str(int(v)) for v in b[6:16]) == "1111111101"
    hi = eng_bits_from_window(False, 0, 600)
    assert hi[0] == 0
    assert "".join(str(int(v)) for v in hi[6:16]) == "0111111111"  # 511
    lo = eng_bits_from_window(False, 0, -600)
    assert "".join(str(int(v)) for v in lo[6:16]) == "1000000000"  # -512


def test_eng_sender_returns_eng_bits():
    comm = make_comm("eng", mx.random.key(12))
    x, h, eng = _inputs()
    msg, bits = comm.send(x, train=False, key=None, eng_bits=eng)
    assert np.array_equal(np.array(msg), np.array(eng))


def test_nocomm_sends_zeros():
    comm = make_comm("nocomm", mx.random.key(13))
    x, h, _ = _inputs()
    for train in (True, False):
        msg, bits = comm.send(x, train=train, key=mx.random.key(1), eng_bits=None)
        assert float(mx.abs(msg).sum()) == 0.0 and float(mx.abs(bits).sum()) == 0.0


def test_plasticity_scale_endpoints_and_monotone():
    assert plasticity_scale(0, 1000) == pytest.approx(1.0)
    assert plasticity_scale(1000, 1000) == pytest.approx(0.1)
    assert plasticity_scale(5000, 1000) == pytest.approx(0.1)
    vals = [plasticity_scale(u, 1000) for u in range(0, 1001, 50)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert plasticity_scale(500, 1000) == pytest.approx(0.55)
