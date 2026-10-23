import numpy as np
import pytest

from experiments.procgen_multiagent_gtm_comm.harness.bus import (
    MessageBus,
    MessageBusViolation,
)


def _bits(n=3, seed=0):
    return np.random.default_rng(seed).integers(0, 2, (n, 2, 16)).astype(np.float32)


def test_bus_swaps_partners():
    bus = MessageBus("gtm")
    b = _bits()
    out = bus.deliver(b, phase="eval")
    assert np.array_equal(out[:, 0], b[:, 1])
    assert np.array_equal(out[:, 1], b[:, 0])


def test_bus_counts_bits():
    bus = MessageBus("mlp")
    bus.deliver(_bits(3), phase="train")
    assert bus.bits_logged == 3 * 2 * 16


def test_bus_rejects_nonbinary_eval():
    for arm in ("gtm", "dial", "nocomm"):
        bus = MessageBus(arm)
        b = _bits()
        b[0, 0, 0] = 0.5
        with pytest.raises(MessageBusViolation):
            bus.deliver(b, phase="eval")


def test_bus_rejects_nonbinary_train_except_dial():
    b = _bits()
    b[1, 1, 3] = 0.25
    with pytest.raises(MessageBusViolation):
        MessageBus("gtm").deliver(b, phase="train")


def test_bus_allows_dial_train_soft():
    b = np.random.default_rng(1).uniform(0, 1, (3, 2, 16)).astype(np.float32)
    out = MessageBus("dial").deliver(b, phase="train")
    assert out.shape == (3, 2, 16)


def test_bus_rejects_bad_shape():
    with pytest.raises(MessageBusViolation):
        MessageBus("gtm").deliver(np.zeros((3, 2, 15), np.float32), phase="eval")
    with pytest.raises(MessageBusViolation):
        MessageBus("gtm").deliver(np.zeros((3, 16), np.float32), phase="eval")
    with pytest.raises(MessageBusViolation):
        MessageBus("dial").deliver(np.zeros((3, 2, 15), np.float32), phase="train")


def test_bus_rejects_unknown_phase_and_arm():
    with pytest.raises(ValueError):
        MessageBus("gtm").deliver(_bits(), phase="warmup")
    with pytest.raises(ValueError):
        MessageBus("bogus")


def test_frozen_agent_message_still_delivered():
    # the bus takes only bits; it has no notion of `finished`, so a finished
    # agent's (frozen-observation) message is delivered like any other.
    bus = MessageBus("gtm")
    b = _bits()
    b[:, 0] = 1.0  # agent 0 "finished", still sending its frozen message
    out = bus.deliver(b, phase="eval")
    assert (out[:, 1] == 1.0).all()
    assert bus.bits_logged == 3 * 2 * 16
