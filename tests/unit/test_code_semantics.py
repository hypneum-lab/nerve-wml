import torch

from interpret.code_semantics import build_semantics_table
from track_w.mlp_wml import MlpWML


def test_semantics_table_has_one_entry_per_code():
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    inputs = torch.randn(512, 16)
    table = build_semantics_table(wml, inputs, top_k_inputs=3)
    assert set(table.keys()) == set(range(64))


def test_semantics_table_fields_per_code():
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    inputs = torch.randn(512, 16)
    table = build_semantics_table(wml, inputs, top_k_inputs=3)
    for code, entry in table.items():
        assert "top_inputs" in entry
        assert "activation_centroid" in entry
        assert "next_codes_distribution" in entry
        assert "n_samples_mapped" in entry
        assert entry["activation_centroid"].shape == (16,)
        assert entry["next_codes_distribution"].shape == (64,)
        # Distribution sums to 1 (softmax).
        assert abs(entry["next_codes_distribution"].sum().item() - 1.0) < 1e-4


def test_semantics_table_is_idempotent():
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    inputs = torch.randn(512, 16)
    a = build_semantics_table(wml, inputs, top_k_inputs=3)
    b = build_semantics_table(wml, inputs, top_k_inputs=3)
    for code in a:
        assert torch.allclose(a[code]["activation_centroid"],
                              b[code]["activation_centroid"])
        assert a[code]["n_samples_mapped"] == b[code]["n_samples_mapped"]


def test_semantics_table_top_inputs_summary_format():
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    inputs = torch.randn(512, 16)
    table = build_semantics_table(wml, inputs, top_k_inputs=3)
    # Find a code with at least one mapped sample.
    mapped = [c for c in table if table[c]["n_samples_mapped"] > 0]
    assert len(mapped) > 0
    code = mapped[0]
    for summary in table[code]["top_inputs"]:
        assert "mean" in summary
        assert "norm" in summary
        assert "argmax_dim" in summary
