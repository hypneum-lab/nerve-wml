"""N10-C-bis ``gtm_mis`` delivery: pair derangement, bus and rollout buffer."""
import numpy as np
import pytest

from experiments.procgen_multiagent_gtm_comm.harness.bus import (
    MessageBus,
    MessageBusViolation,
    is_derangement,
)
from experiments.procgen_multiagent_gtm_comm.harness.codes import run_seed_sequence
from experiments.procgen_multiagent_gtm_comm.harness.rollout import (
    RolloutBuffer,
    deranged_recv,
    derangement,
)


def _bits(n, seed=0):
    return np.random.default_rng(seed).integers(0, 2, (n, 2, 16)).astype(np.float32)


@pytest.mark.parametrize("n", [2, 3, 64])
def test_derangement_is_a_permutation_without_fixed_points(n):
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = derangement(n, rng)
        assert is_derangement(s, n)
        assert sorted(s.tolist()) == list(range(n))
        assert not (s == np.arange(n)).any()


def test_derangement_stream_is_the_fifth_child_and_leaves_the_first_four():
    ss = run_seed_sequence(0, "gtm_mis", "registered")
    four = [c.generate_state(4).tolist() for c in ss.spawn(4)]
    five = run_seed_sequence(0, "gtm_mis", "registered").spawn(5)
    assert [c.generate_state(4).tolist() for c in five[:4]] == four
    a = derangement(64, np.random.default_rng(five[4]))
    b = derangement(64, np.random.default_rng(
        run_seed_sequence(0, "gtm_mis", "registered").spawn(5)[4]))
    assert np.array_equal(a, b)  # deterministic from the run's stream
    c = derangement(64, np.random.default_rng(
        run_seed_sequence(17, "gtm_mis", "registered").spawn(5)[4]))
    assert not np.array_equal(a, c)


def test_deranged_recv_gives_agent_i_the_other_role_of_pair_sigma():
    n = 5
    b = _bits(n)
    sigma = derangement(n, np.random.default_rng(2))
    out = deranged_recv(b, sigma)
    assert out.shape == b.shape
    for e in range(n):
        for i in range(2):
            np.testing.assert_array_equal(out[e, i], b[sigma[e], 1 - i])
            assert not np.array_equal(out[e, i], b[e, 1 - i]) or np.array_equal(
                b[sigma[e], 1 - i], b[e, 1 - i])


def test_bus_deranged_delivery_keeps_assertions_and_bit_count():
    bus = MessageBus("gtm_mis")
    n = 8
    b = _bits(n, 3)
    sigma = derangement(n, np.random.default_rng(4))
    out = bus.deliver(b, phase="train", pairs=sigma)
    assert out.shape == (n, 2, 16) and set(np.unique(out)) <= {0.0, 1.0}
    np.testing.assert_array_equal(out, deranged_recv(b, sigma))
    assert bus.bits_logged == n * 2 * 16  # unchanged: 16 bits per agent per step
    # default delivery of the same bus is the plain partner swap
    plain = bus.deliver(b, phase="train")
    np.testing.assert_array_equal(plain[:, 0], b[:, 1])
    assert bus.bits_logged == 2 * n * 2 * 16


def test_bus_deranged_delivery_rejects_shape_values_and_non_derangements():
    bus = MessageBus("gtm_mis")
    n = 6
    b = _bits(n, 5)
    sigma = derangement(n, np.random.default_rng(6))
    bad = b.copy()
    bad[0, 0, 0] = 0.5
    with pytest.raises(MessageBusViolation, match="non-binary"):
        bus.deliver(bad, phase="train", pairs=sigma)
    with pytest.raises(MessageBusViolation, match="shape"):
        bus.deliver(b[:, :, :15], phase="train", pairs=sigma)
    with pytest.raises(MessageBusViolation, match="derangement"):
        bus.deliver(b, phase="train", pairs=np.arange(n))  # identity
    with pytest.raises(MessageBusViolation, match="derangement"):
        bus.deliver(b, phase="train", pairs=np.array([1, 0, 2, 4, 5, 3]))  # fixed pt
    with pytest.raises(MessageBusViolation, match="derangement"):
        bus.deliver(b, phase="train", pairs=np.array([1, 1, 0, 5, 4, 3]))  # not perm
    with pytest.raises(MessageBusViolation, match="derangement"):
        bus.deliver(b, phase="train", pairs=sigma[:-1])
    assert bus.bits_logged == 0  # nothing was delivered


def test_rollout_buffer_stores_and_yields_recv_bits():
    pytest.importorskip("mlx.core")
    t_len, n = 3, 4
    buf = RolloutBuffer(n_pairs=n, horizon=t_len)
    rng = np.random.default_rng(0)
    recvs = []
    for _t in range(t_len):
        r = rng.integers(0, 2, (n, 2, 16)).astype(np.float32)
        recvs.append(r)
        buf.add(
            obs=np.zeros((n, 2, 64, 64, 3), np.uint8),
            summary=np.zeros((n, 2, 2), np.float32),
            eng_bits=np.zeros((n, 2, 16), np.float32),
            actions=np.zeros((n, 2), np.int64),
            logp=np.zeros((n, 2), np.float32),
            value=np.zeros((n, 2), np.float32),
            team_reward=np.zeros(n, np.float32),
            done=np.zeros(n, bool),
            recv_bits=r,
        )
    np.testing.assert_array_equal(buf.recv_bits, np.stack(recvs))
    buf.finish(np.zeros((n, 2), np.float32))
    mb = next(iter(buf.minibatches(np.random.default_rng(1), size_pairs=t_len * n)))
    flat = buf.recv_bits.reshape(t_len * n, 2, 16)
    np.testing.assert_array_equal(np.array(mb["recv_bits"]), flat[np.array(mb["index"])])
    # arms that do not store received bits leave zeros (N10-C behaviour)
    other = RolloutBuffer(n_pairs=2, horizon=1)
    assert other.recv_bits.shape == (1, 2, 2, 16) and not other.recv_bits.any()
