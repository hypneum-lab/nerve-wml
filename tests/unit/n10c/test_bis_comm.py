"""N10-C-bis message modules: ``eng_gtmrx`` (engineered sender, GTM receive
path) and ``vq`` (plain VQ with GTM's K = 4 x A = 16 symbol structure)."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from experiments.procgen_multiagent_gtm_comm.harness.codes import (  # noqa: E402
    BIS_ARMS,
    N10C_ARMS,
)
from experiments.procgen_multiagent_gtm_comm.harness.comm import (  # noqa: E402
    CODE_TABLE,
    ENG_SENDER_ARMS,
    GTM_SENDER_ARMS,
    PLASTIC_ARMS,
    VQ_BETA,
    VQ_D_E,
    VQ_LAMBDA,
    EngGTMRxComm,
    GTMComm,
    VQComm,
    bits_to_indices,
    codebook_usage,
    make_comm,
)
from experiments.procgen_multiagent_gtm_comm.harness.policy import (  # noqa: E402
    PairPolicy,
    param_count,
)

B = 8


def _inputs(seed=0):
    k1, k2 = mx.random.split(mx.random.key(seed))
    x = mx.random.normal((B, 64), key=k1)
    h = mx.random.normal((B, 256), key=k2)
    eng = mx.array(np.random.default_rng(seed).integers(0, 2, (B, 16)).astype(np.float32))
    return x, h, eng


def _binary(a):
    return set(np.unique(np.array(a))) <= {0.0, 1.0}


def _names(model):
    return sorted(k for k, _ in tree_flatten(model.parameters()))


# -- arm families -------------------------------------------------------------
def test_arm_families_and_module_classes():
    assert ENG_SENDER_ARMS == ("eng", "eng_gtmrx")
    assert PLASTIC_ARMS == ("gtm", "eng_gtmrx", "gtm_warm", "gtm_mis")
    assert GTM_SENDER_ARMS == ("gtm", "gtm_mis", "gtm_warm")
    gtm = make_comm("gtm", mx.random.key(1))
    for arm in ("gtm_mis", "gtm_warm"):  # the unchanged GTMComm module
        c = make_comm(arm, mx.random.key(1))
        assert type(c) is GTMComm
        assert np.array_equal(np.array(c.constellation), np.array(gtm.constellation))
        assert np.array_equal(np.array(c.encode_logits.weight),
                              np.array(gtm.encode_logits.weight))
    assert isinstance(make_comm("eng_gtmrx", mx.random.key(1)), EngGTMRxComm)
    assert isinstance(make_comm("vq", mx.random.key(1)), VQComm)
    with pytest.raises(ValueError):
        make_comm("ib_vq", mx.random.key(1))


# -- eng_gtmrx ----------------------------------------------------------------
def test_eng_gtmrx_sends_engineered_bits_and_receives_through_the_gtm_path():
    comm = make_comm("eng_gtmrx", mx.random.key(1))
    assert isinstance(comm, GTMComm)
    x, h, eng = _inputs()
    for train in (True, False):
        msg, bits = comm.send(x, train=train, key=mx.random.key(2), eng_bits=eng)
        assert np.array_equal(np.array(msg), np.array(eng))
        assert np.array_equal(np.array(bits), np.array(eng))
    out = comm.receive(bits, h)
    assert out.shape == (B, 64)
    # the GTM receive path (same parameters) reading the same 16 bits
    ref = make_comm("gtm", mx.random.key(1))
    ref.update(comm.parameters())
    np.testing.assert_allclose(np.array(ref.receive(bits, h)), np.array(out), atol=1e-6)
    with pytest.raises(ValueError, match="eng_bits"):
        comm.send(x, train=False, key=None, eng_bits=None)
    assert comm.noise_dim == 0 and comm.sample_noise(4, mx.random.key(0)) is None


def test_eng_gtmrx_has_exactly_the_gtm_parameter_set():
    a = PairPolicy("eng_gtmrx", mx.random.key(0))
    b = PairPolicy("gtm", mx.random.key(0))
    assert _names(a) == _names(b)
    assert param_count(a) == param_count(b) == 734_256
    assert param_count(a.comm) == 8_864
    # the untrained sender head and to_msg are what gtm_warm excludes
    for name in ("comm.encode_logits.weight", "comm.encode_logits.bias",
                 "to_msg.weight", "to_msg.bias"):
        assert name in _names(a)


def test_eng_gtmrx_gradient_reaches_constellation_and_decoder_not_sender():
    comm = make_comm("eng_gtmrx", mx.random.key(3))
    x, h, eng = _inputs()

    def loss(params):
        comm.update(params)
        msg, _ = comm.send(x, train=True, key=mx.random.key(4), eng_bits=eng)
        return (comm.receive(msg, h) ** 2).sum()

    g = mx.grad(loss)(comm.trainable_parameters())
    assert float(mx.abs(g["constellation"]).sum()) > 0.0
    assert float(mx.abs(g["decode"]["weight"]).sum()) > 0.0
    assert float(mx.abs(g["encode_logits"]["weight"]).sum()) == 0.0


# -- vq -----------------------------------------------------------------------
def test_vq_settings_are_the_pi_decisions():
    assert (VQ_D_E, VQ_BETA, VQ_LAMBDA) == (8, 0.25, 1.0)
    comm = make_comm("vq", mx.random.key(0))
    assert comm.codebook.shape == (16, 8)  # one codebook shared by the 4 slots
    assert comm.encode.weight.shape == (32, 64) and comm.decode.weight.shape == (64, 32)
    assert comm.noise_dim == 0 and comm.sample_noise(2, mx.random.key(0)) is None
    assert not hasattr(comm, "_cfg")  # no gamma/theta carrier


def test_vq_sends_exactly_16_hard_bits_in_training_and_evaluation():
    comm = make_comm("vq", mx.random.key(5))
    x, h, _ = _inputs()
    msg_e, bits_e = comm.send(x, train=False, key=None, eng_bits=None)
    assert bits_e.shape == (B, 16) and _binary(bits_e)
    assert np.array_equal(np.array(msg_e), np.array(bits_e))
    msg_t, bits_t = comm.send(x, train=True, key=mx.random.key(6), eng_bits=None)
    assert bits_t.shape == (B, 16) and _binary(bits_t)
    assert np.array_equal(np.array(bits_t), np.array(bits_e))  # deterministic sender
    assert msg_t.shape == (B, 4 * VQ_D_E)
    idx = np.array(bits_to_indices(bits_t))
    assert idx.shape == (B, 4) and idx.min() >= 0 and idx.max() < 16
    # nibble mapping (ruling R2) and nearest-codebook quantisation
    expect = np.array(mx.eye(16)[mx.array(idx)] @ CODE_TABLE).reshape(B, 16)
    assert np.array_equal(expect, np.array(bits_t))
    z = np.array(comm.encode(x)).reshape(B, 4, VQ_D_E)
    cb = np.array(comm.codebook)
    d = ((z[:, :, None, :] - cb[None, None]) ** 2).sum(-1)
    assert np.array_equal(d.argmin(-1), idx)
    # the training message's value is exactly the codebook rows (STE)
    assert np.array_equal(np.array(msg_t).reshape(B, 4, VQ_D_E), cb[idx])


def test_vq_eval_receive_from_bits_equals_train_receive_from_st_rows():
    comm = make_comm("vq", mx.random.key(7))
    x, h, _ = _inputs(1)
    msg_t, bits_t = comm.send(x, train=True, key=mx.random.key(8), eng_bits=None)
    a = np.array(comm.receive(msg_t, h))
    b = np.array(comm.receive(bits_t, h))
    assert a.shape == (B, 64)
    np.testing.assert_allclose(a, b, atol=1e-7)


def test_vq_straight_through_gradient_reaches_the_encoder_only():
    comm = make_comm("vq", mx.random.key(9))
    x, h, _ = _inputs(2)

    def loss(params, train):
        comm.update(params)
        msg, _ = comm.send(x, train=train, key=mx.random.key(1), eng_bits=None)
        return (comm.receive(msg, h) ** 2).sum()

    g = mx.grad(loss)(comm.trainable_parameters(), True)
    assert float(mx.abs(g["encode"]["weight"]).sum()) > 0.0  # STE to the encoder
    assert float(mx.abs(g["decode"]["weight"]).sum()) > 0.0
    assert float(mx.abs(g["codebook"]).sum()) == 0.0  # not through the task loss
    g_eval = mx.grad(loss)(comm.trainable_parameters(), False)
    assert float(mx.abs(g_eval["encode"]["weight"]).sum()) == 0.0  # hard bits


def test_vq_aux_loss_is_positive_decomposes_and_decreases_under_training():
    comm = make_comm("vq", mx.random.key(10))
    x = 3.0 * mx.random.normal((B, 64), key=mx.random.key(11))
    l0 = float(comm.aux_loss(x))
    assert l0 > 0.0
    z, idx, q = comm._quantize(x)
    sq = float(((z - q) ** 2).sum(-1).mean())
    assert l0 == pytest.approx(VQ_LAMBDA * (sq + VQ_BETA * sq), rel=1e-5)
    g = mx.grad(lambda p: (comm.update(p), comm.aux_loss(x))[1])(
        comm.trainable_parameters())
    assert float(mx.abs(g["codebook"]).sum()) > 0.0  # codebook loss
    assert float(mx.abs(g["encode"]["weight"]).sum()) > 0.0  # commitment loss
    assert float(mx.abs(g["decode"]["weight"]).sum()) == 0.0
    opt = optim.Adam(learning_rate=1e-2)
    vg = nn.value_and_grad(comm, lambda m: m.aux_loss(x))
    losses = [l0]
    for _ in range(30):
        val, grads = vg(comm)
        opt.update(comm, grads)
        mx.eval(comm.parameters(), opt.state)
        losses.append(float(val))
    assert losses[-1] < 0.5 * losses[0]
    assert all(np.isfinite(losses))


def test_vq_parameter_sizes_for_the_manifest():
    m = PairPolicy("vq", mx.random.key(0))
    assert param_count(m.comm.encode) == 64 * 32 + 32 == 2_080
    assert m.comm.codebook.size == 16 * 8 == 128
    assert param_count(m.comm.decode) == 32 * 64 + 64 == 2_112
    assert param_count(m.comm) == 4_320
    assert param_count(m) == 729_712


def test_codebook_usage_counts_symbol_indices():
    idx = np.array([[0, 1, 2, 3], [15, 15, 15, 15]], dtype=np.int32)
    bits = np.array(mx.eye(16)[mx.array(idx)] @ CODE_TABLE).reshape(2, 16)
    use = codebook_usage(bits)
    assert use.shape == (16,) and use.sum() == 8
    assert use[:4].tolist() == [1, 1, 1, 1] and use[15] == 4 and use[4:15].sum() == 0
    assert codebook_usage(np.zeros((3, 2, 16)))[0] == 3 * 2 * 4


# -- policy-level auxiliary loss ---------------------------------------------
@pytest.mark.parametrize("arm", N10C_ARMS + BIS_ARMS)
def test_policy_aux_loss_only_for_vq_in_training(arm):
    rng = np.random.default_rng(0)
    obs = mx.array(rng.integers(0, 256, (3, 2, 64, 64, 3)).astype(np.float32))
    summary = mx.array(rng.random((3, 2, 2)).astype(np.float32))
    eng = mx.array(rng.integers(0, 2, (3, 2, 16)).astype(np.float32))
    pol = PairPolicy(arm, mx.random.key(0))
    nz = pol.sample_noise(3, mx.random.key(2))
    tr = pol(obs, summary, eng, train=True, key=mx.random.key(1), noise=nz)
    ev = pol(obs, summary, eng, train=False, key=mx.random.key(1))
    assert ev.aux_loss is None
    if arm == "vq":
        assert tr.aux_loss is not None and float(tr.aux_loss) > 0.0
    else:
        assert tr.aux_loss is None
    assert tr.hard_bits.shape == (3, 2, 16) and _binary(tr.hard_bits)


def test_bis_arm_parameter_table():
    """Sizes of the four control arms (stage-2 manifest table, PR body)."""
    rows = {}
    for arm in BIS_ARMS:
        m = PairPolicy(arm, mx.random.key(0))
        rows[arm] = (param_count(m), param_count(m.comm), param_count(m.trunk),
                     param_count(m.to_msg),
                     param_count(m.fuse) + param_count(m.pi) + param_count(m.v))
    assert rows == {
        "gtm_mis": (734_256, 8_864, 622_144, 16_448, 86_800),
        "eng_gtmrx": (734_256, 8_864, 622_144, 16_448, 86_800),
        "gtm_warm": (734_256, 8_864, 622_144, 16_448, 86_800),
        "vq": (729_712, 4_320, 622_144, 16_448, 86_800),
    }
    print("\n| arm | total | comm | trunk | to_msg | fuse+pi+v |")
    for arm, r in rows.items():
        print(f"| {arm} | " + " | ".join(f"{v:,}" for v in r) + " |")
