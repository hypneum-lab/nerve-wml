import time

import numpy as np

from track_w.bio_clients import (
    ActivityFrame,  # noqa: F401  # part of the public surface under test
    BioCultureClient,
    MockBioCultureClient,
    StimulusFrame,
)


def test_mock_client_satisfies_protocol():
    client = MockBioCultureClient(seed=0)
    assert isinstance(client, BioCultureClient)


def test_encode_stimulus_shapes_and_range():
    client = MockBioCultureClient(seed=0)
    frame = client.encode_stimulus([3, 17, 42])
    assert isinstance(frame, StimulusFrame)
    assert frame.codes == (3, 17, 42)
    assert frame.channels.shape == (3, client.n_stim_channels)
    assert frame.channels.min() >= 0.0
    assert frame.channels.max() <= 1.0


def test_roundtrip_is_deterministic_under_seed():
    a = MockBioCultureClient(seed=7).roundtrip([1, 2, 3])
    b = MockBioCultureClient(seed=7).roundtrip([1, 2, 3])
    assert np.array_equal(a.spikes, b.spikes)
    assert a.latency_ms == b.latency_ms


def test_roundtrip_differs_across_seeds():
    a = MockBioCultureClient(seed=1).roundtrip([1, 2, 3])
    b = MockBioCultureClient(seed=2).roundtrip([1, 2, 3])
    assert not np.array_equal(a.spikes, b.spikes)


def test_roundtrip_latency_has_jitter_within_bounds():
    client = MockBioCultureClient(
        seed=0, base_latency_ms=10.0, jitter_ms=4.0,
    )
    lats = [client.roundtrip([5]).latency_ms for _ in range(50)]
    assert all(6.0 <= x <= 14.0 for x in lats)
    assert len(set(lats)) > 1  # jitter actually varies


def test_roundtrip_sleeps_when_simulate_wall_clock():
    client = MockBioCultureClient(
        seed=0, base_latency_ms=20.0, jitter_ms=0.0,
        simulate_wall_clock=True,
    )
    t0 = time.perf_counter()
    client.roundtrip([1])
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    assert elapsed_ms >= 18.0  # actually slept ~20 ms


def test_decode_activity_returns_codes_in_alphabet():
    client = MockBioCultureClient(seed=0)
    frame = client.roundtrip([9, 9, 9])
    codes = client.decode_activity(frame)
    assert all(0 <= c < 64 for c in codes)
    assert len(codes) == 3


def test_stimulating_a_code_biases_decode_toward_that_code():
    # The mock is a noisy channel, not random: a strong repeated
    # stimulus should decode back to itself most of the time.
    client = MockBioCultureClient(seed=0, noise=0.05)
    hits = 0
    for _ in range(40):
        frame = client.roundtrip([11])
        if client.decode_activity(frame)[0] == 11:
            hits += 1
    assert hits >= 28  # >= 70 % round-trip fidelity at low noise


def test_close_is_idempotent():
    client = MockBioCultureClient(seed=0)
    client.close()
    client.close()  # must not raise
