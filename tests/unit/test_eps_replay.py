"""Tests for bridge.eps_replay save/load round-trip."""
import numpy as np

from bridge.eps_replay import load_eps_replay, save_eps_replay


def test_save_load_round_trip(tmp_path):
    trace = np.array([[0, 1, 3, 100], [2, 1, 7, 101]], dtype=np.int32)
    metadata = {
        "commit_sha": "abc123",
        "seed": 0,
        "n_wmls": 4,
        "schema_version": "v0",
    }
    save_eps_replay(trace, metadata, tmp_path)
    loaded_trace, loaded_meta = load_eps_replay(tmp_path)

    np.testing.assert_array_equal(loaded_trace, trace)
    assert loaded_meta == metadata


def test_save_load_empty_trace(tmp_path):
    trace = np.zeros((0, 4), dtype=np.int32)
    metadata = {"schema_version": "v0", "n_wmls": 4}
    save_eps_replay(trace, metadata, tmp_path)
    loaded_trace, loaded_meta = load_eps_replay(tmp_path)

    assert loaded_trace.shape == (0, 4)
    assert loaded_meta == metadata
