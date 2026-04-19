"""Tests for bridge.checkpoint save/load."""
import torch

from bridge.checkpoint import load_advisor_checkpoint, save_advisor_checkpoint
from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML


def test_save_load_round_trip(tmp_path):
    pool = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)

    save_advisor_checkpoint(pool, nerve, tmp_path)
    loaded = load_advisor_checkpoint(tmp_path)

    assert loaded["manifest"]["n_wmls"] == 2
    assert loaded["manifest"]["k"] == 1
    assert loaded["manifest"]["alphabet_size"] == 64
    assert loaded["manifest"]["schema_version"] == "v0"

    # Pool has 2 entries with class name preserved.
    assert len(loaded["pool_state"]) == 2
    assert loaded["pool_state"][0]["class_name"] == "MlpWML"
    assert loaded["pool_state"][0]["id"] == 0


def test_loaded_state_reconstructs_wml(tmp_path):
    """After load, a fresh MlpWML must accept the state_dict."""
    pool = [MlpWML(id=0, d_hidden=16, seed=0)]
    nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)

    save_advisor_checkpoint(pool, nerve, tmp_path)
    loaded = load_advisor_checkpoint(tmp_path)

    # Snapshot an original param.
    original_cb = pool[0].codebook.data.clone()

    # Reconstruct.
    new_wml = MlpWML(id=0, d_hidden=16, seed=42)  # different seed
    new_wml.load_state_dict(loaded["pool_state"][0]["state_dict"])
    assert torch.allclose(new_wml.codebook.data, original_cb)


def test_loaded_nerve_reconstructs(tmp_path):
    pool = [MlpWML(id=0, d_hidden=16, seed=0)]
    nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)

    save_advisor_checkpoint(pool, nerve, tmp_path)
    loaded = load_advisor_checkpoint(tmp_path)

    # Rebuild nerve and load state.
    new_nerve = SimNerveAdapter(n_wmls=2, k=1, seed=99)  # different seed
    new_nerve.load_state_dict(loaded["nerve_state"])

    # Router logits should match.
    assert torch.allclose(new_nerve.router.logits.data, nerve.router.logits.data)
