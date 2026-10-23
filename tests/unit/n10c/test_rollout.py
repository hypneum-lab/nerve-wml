import numpy as np
import pytest

from experiments.procgen_multiagent_gtm_comm.harness.rollout import (
    RolloutBuffer,
    compute_gae,
)


def _ref_gae(rewards, values, dones, last_value, gamma, lam):
    t_len, n = rewards.shape
    adv = np.zeros((t_len, n, 2), dtype=np.float64)
    for p in range(n):
        for a in range(2):
            gae = 0.0
            for t in reversed(range(t_len)):
                nv = last_value[p, a] if t == t_len - 1 else values[t + 1, p, a]
                nonterm = 0.0 if dones[t, p] else 1.0
                delta = rewards[t, p] + gamma * nv * nonterm - values[t, p, a]
                gae = delta + gamma * lam * nonterm * gae
                adv[t, p, a] = gae
    return adv, adv + values


def test_gae_no_bootstrap_across_done():
    t_len, n = 3, 1
    r = np.zeros((t_len, n), np.float32)
    r[1, 0] = 1.0
    d = np.zeros((t_len, n), bool)
    d[1, 0] = True
    v = np.full((t_len, n, 2), 0.3, np.float32)
    v[2] = 99.0  # must not leak across the episode boundary
    last = np.full((n, 2), 50.0, np.float32)
    adv, ret = compute_gae(r, v, d, last)
    np.testing.assert_allclose(adv[1, 0], 1.0 - 0.3, atol=1e-6)
    np.testing.assert_allclose(ret[1, 0], 1.0, atol=1e-6)
    # t=0 does not see step 2's value/advantage through the done at t=1
    assert np.all(adv[0, 0] < 5.0)


def test_gae_matches_reference_loop():
    rng = np.random.default_rng(0)
    t_len, n = 16, 5
    r = rng.normal(size=(t_len, n)).astype(np.float32)
    v = rng.normal(size=(t_len, n, 2)).astype(np.float32)
    d = rng.random((t_len, n)) < 0.2
    last = rng.normal(size=(n, 2)).astype(np.float32)
    adv, ret = compute_gae(r, v, d, last, gamma=0.999, lam=0.95)
    radv, rret = _ref_gae(r, v, d, last, 0.999, 0.95)
    assert adv.shape == (t_len, n, 2) and ret.shape == (t_len, n, 2)
    np.testing.assert_allclose(adv, radv, atol=1e-6)
    np.testing.assert_allclose(ret, rret, atol=1e-6)


def _filled(t_len=4, n=3, seed=0):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(n_pairs=n, horizon=t_len)
    for _t in range(t_len):
        buf.add(
            obs=rng.integers(0, 256, (n, 2, 64, 64, 3), dtype=np.uint8),
            summary=rng.normal(size=(n, 2, 2)).astype(np.float32),
            eng_bits=rng.integers(0, 2, (n, 2, 16)).astype(np.float32),
            actions=rng.integers(0, 15, (n, 2)),
            logp=rng.normal(size=(n, 2)).astype(np.float32),
            value=rng.normal(size=(n, 2)).astype(np.float32),
            team_reward=rng.normal(size=n).astype(np.float32),
            done=rng.random(n) < 0.1,
        )
    return buf


def test_buffer_full_and_gae_stored():
    buf = _filled()
    with pytest.raises(RuntimeError):
        list(buf.minibatches(np.random.default_rng(0), size_pairs=4))
    buf.finish(np.zeros((3, 2), np.float32))
    assert buf.advantages.shape == (4, 3, 2)
    with pytest.raises(RuntimeError):
        buf.add(**{k: 0 for k in (
            "obs", "summary", "eng_bits", "actions", "logp", "value",
            "team_reward", "done")})


def test_minibatches_cover_all_pairs_once():
    mx = pytest.importorskip("mlx.core")
    t_len, n = 4, 3
    buf = _filled(t_len, n)
    buf.finish(np.zeros((n, 2), np.float32))
    seen = []
    for mb in buf.minibatches(np.random.default_rng(1), size_pairs=5):
        assert isinstance(mb["obs"], mx.array)
        assert mb["obs"].dtype == mx.uint8
        k = mb["obs"].shape[0]
        assert mb["obs"].shape == (k, 2, 64, 64, 3)
        assert mb["summary"].shape == (k, 2, 2)
        assert mb["eng_bits"].shape == (k, 2, 16)
        for key in ("actions", "logp", "value", "advantages", "returns"):
            assert mb[key].shape == (k, 2)
        seen.append(np.array(mb["index"]))
    idx = np.concatenate(seen)
    assert sorted(idx.tolist()) == list(range(t_len * n))
    assert len(seen) == 3  # 12 pair-steps / 5 -> 5,5,2
    # content matches storage at the flattened index
    flat_val = buf.value.reshape(t_len * n, 2)
    mb = next(iter(buf.minibatches(np.random.default_rng(2), size_pairs=12)))
    np.testing.assert_array_equal(
        np.array(mb["value"]), flat_val[np.array(mb["index"])]
    )


def test_noise_is_stored_and_shuffled_with_samples():
    pytest.importorskip("mlx.core")
    t_len, n, dim = 3, 2, 5
    buf = RolloutBuffer(n_pairs=n, horizon=t_len, noise_dim=dim)
    rng = np.random.default_rng(0)
    for _t in range(t_len):
        buf.add(
            obs=np.zeros((n, 2, 64, 64, 3), np.uint8),
            summary=np.zeros((n, 2, 2), np.float32),
            eng_bits=np.zeros((n, 2, 16), np.float32),
            actions=np.zeros((n, 2), np.int64),
            logp=np.zeros((n, 2), np.float32),
            value=np.zeros((n, 2), np.float32),
            team_reward=np.zeros(n, np.float32),
            done=np.zeros(n, bool),
            noise=rng.normal(size=(n, 2, dim)).astype(np.float32),
        )
    buf.finish(np.zeros((n, 2), np.float32))
    mb = next(iter(buf.minibatches(np.random.default_rng(3), size_pairs=t_len * n)))
    flat = buf.noise.reshape(t_len * n, 2, dim)
    np.testing.assert_array_equal(np.array(mb["noise"]), flat[np.array(mb["index"])])
