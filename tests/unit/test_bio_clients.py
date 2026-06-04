import numpy as np

from track_w.bio_clients import (
    ActivityFrame,
    BioApiKeyMissingError,
    BioCultureClient,
    StimulusFrame,
)


def test_stimulus_frame_is_immutable_and_carries_codes():
    frame = StimulusFrame(
        codes=(3, 17, 42),
        channels=np.zeros((3, 8), dtype=np.float32),
    )
    assert frame.codes == (3, 17, 42)
    assert frame.channels.shape == (3, 8)


def test_activity_frame_records_spikes_and_latency():
    frame = ActivityFrame(
        spikes=np.zeros((8, 32), dtype=np.float32),
        latency_ms=12.5,
    )
    assert frame.spikes.shape == (8, 32)
    assert frame.latency_ms == 12.5


def test_bio_culture_client_is_a_runtime_checkable_protocol():
    # A bare object is not a BioCultureClient.
    assert not isinstance(object(), BioCultureClient)


def test_bio_api_key_missing_is_a_runtime_error():
    assert issubclass(BioApiKeyMissingError, RuntimeError)
