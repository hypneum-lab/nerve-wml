import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from experiments.procgen_multiagent_gtm_comm.harness.policy import (  # noqa: E402
    PairPolicy,
    param_count,
)

ARMS = ["gtm", "mlp", "atoc", "dial", "eng", "nocomm"]
B = 4


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    obs = mx.array(rng.integers(0, 256, (B, 2, 64, 64, 3)).astype(np.float32))
    summary = mx.array(rng.random((B, 2, 2)).astype(np.float32))
    eng = mx.array(rng.integers(0, 2, (B, 2, 16)).astype(np.float32))
    return obs, summary, eng


def _policy(arm):
    return PairPolicy(arm, mx.random.key(0))


@pytest.mark.parametrize("arm", ARMS)
def test_output_shapes_all_arms(arm):
    obs, summary, eng = _inputs()
    pol = _policy(arm)
    for train in (False, True):
        out = pol(obs, summary, eng, train=train, key=mx.random.key(1))
        assert out.logits.shape == (B, 2, 15)
        assert out.value.shape == (B, 2)
        assert out.hard_bits.shape == (B, 2, 16)
        mx.eval(out.logits, out.value, out.hard_bits)
        hb = np.array(out.hard_bits)
        assert set(np.unique(hb)) <= {0.0, 1.0}


@pytest.mark.parametrize("arm", ARMS)
def test_partner_swap_in_policy(arm):
    obs, summary, eng = _inputs()
    pol = _policy(arm)
    # at init the bit thresholds are dominated by biases; make the sender
    # input-sensitive so a changed image flips message bits
    if hasattr(pol.comm, "encode"):
        pol.comm.encode.bias = mx.zeros_like(pol.comm.encode.bias)
        pol.comm.encode.weight = pol.comm.encode.weight * 30.0
    if hasattr(pol.comm, "encode_logits"):
        pol.comm.encode_logits.weight = pol.comm.encode_logits.weight * 30.0
    pol.to_msg.bias = mx.zeros_like(pol.to_msg.bias)
    base = pol(obs, summary, eng, train=False, key=mx.random.key(1))
    # random noise images give near-identical features at init; use a strong,
    # structured change (per-sample constant brightness) so message bits flip
    levels = mx.array([0.0, 85.0, 170.0, 255.0]).reshape(B, 1, 1, 1)
    new1 = mx.broadcast_to(levels, (B, 64, 64, 3))[:, None]
    obs_p = mx.concatenate([obs[:, :1], new1], axis=1)  # perturb agent 1 image
    eng_p = eng
    if arm == "eng":  # eng messages come from eng_bits, not the image
        eng_p = mx.concatenate([eng[:, :1], 1.0 - eng[:, 1:]], axis=1)
    pert = pol(obs_p, summary, eng_p, train=False, key=mx.random.key(1))
    mx.eval(base.logits, pert.logits)
    if arm not in ("nocomm", "eng"):
        assert not np.array_equal(np.array(base.hard_bits[:, 1]), np.array(pert.hard_bits[:, 1]))
    a = np.array(base.logits[:, 0])
    b = np.array(pert.logits[:, 0])
    if arm == "nocomm":
        np.testing.assert_allclose(a, b, atol=1e-5)
    else:
        assert not np.allclose(a, b, atol=1e-5)
    # agent 1's own logits do change with its own image (sanity)
    assert not np.allclose(np.array(base.logits[:, 1]), np.array(pert.logits[:, 1]), atol=1e-5)


@pytest.mark.parametrize("arm", ["gtm", "nocomm"])
def test_summary_ignored_when_zero_closed(arm):
    obs, _, eng = _inputs()
    zeros = mx.zeros((B, 2, 2), dtype=mx.float32)
    pol = _policy(arm)
    o1 = pol(obs, zeros, eng, train=False, key=mx.random.key(1))
    o2 = pol(obs, zeros, eng, train=False, key=mx.random.key(2))
    mx.eval(o1.logits, o2.logits, o1.value, o2.value)
    np.testing.assert_array_equal(np.array(o1.logits), np.array(o2.logits))
    np.testing.assert_array_equal(np.array(o1.value), np.array(o2.value))
    # a non-zero summary does change the output (registered condition)
    _, summary, _ = _inputs()
    o3 = pol(obs, summary, eng, train=False, key=mx.random.key(1))
    mx.eval(o3.logits)
    assert not np.allclose(np.array(o1.logits), np.array(o3.logits), atol=1e-6)


@pytest.mark.parametrize("arm", ARMS)
def test_param_count_reported(arm):
    n = param_count(_policy(arm))
    assert isinstance(n, int) and n > 0


@pytest.mark.parametrize("arm", ["gtm", "dial"])
def test_stored_noise_replays_message(arm):
    obs, summary, eng = _inputs()
    pol = _policy(arm)
    nz = pol.sample_noise(B, mx.random.key(5))
    assert nz.shape == (B, 2, pol.noise_dim)
    a = pol(obs, summary, eng, train=True, key=mx.random.key(1), noise=nz)
    b = pol(obs, summary, eng, train=True, key=mx.random.key(2), noise=nz)
    c = pol(obs, summary, eng, train=True, key=mx.random.key(1))
    for x, y in ((a, b),):
        np.testing.assert_allclose(np.array(x.logits), np.array(y.logits), atol=1e-6)
        np.testing.assert_array_equal(np.array(x.hard_bits), np.array(y.hard_bits))
    # noise=None keeps the key-driven behaviour (differs from the stored draw)
    assert not np.allclose(np.array(a.logits), np.array(c.logits))


@pytest.mark.parametrize("arm", ["mlp", "atoc", "eng", "nocomm"])
def test_deterministic_arms_have_no_noise(arm):
    assert _policy(arm).sample_noise(B, mx.random.key(0)) is None


@pytest.mark.parametrize("arm", ["mlp", "gtm"])
def test_recv_bits_override(arm):
    obs, summary, eng = _inputs()
    pol = _policy(arm)
    key = mx.random.key(1)
    base = pol(obs, summary, eng, train=False, key=key)
    swap = mx.stack([base.hard_bits[:, 1], base.hard_bits[:, 0]], axis=1)
    same = pol(obs, summary, eng, train=False, key=key, recv_bits=swap)
    np.testing.assert_array_equal(np.array(base.logits), np.array(same.logits))
    other = pol(obs, summary, eng, train=False, key=key,
                recv_bits=1.0 - swap)
    assert not np.allclose(np.array(base.logits), np.array(other.logits))
    # the sender side is unaffected by what is received
    np.testing.assert_array_equal(np.array(base.hard_bits),
                                  np.array(other.hard_bits))
